from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from omicau.cli import _fixed_plan_for_aligned
from omicau.config import OmicauConfig
from omicau.data.alignment import load_and_align
from omicau.data.benchmark_data import write_mock_dataset
from omicau.models.base import make_cv_splitter, safe_n_splits
from omicau.models.split_plan import (
    SplitValidationError,
    public_manifest_identity_binding,
    validate_split_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _case_config(path: Path) -> OmicauConfig:
    return OmicauConfig.from_file(path / "config.json")


def _write_fixed_manifest(config_path: Path) -> Path:
    config = OmicauConfig.from_file(config_path)
    aligned = load_and_align(config)
    y = aligned.y.to_numpy()
    groups = aligned.groups.to_numpy()
    outer_splitter = make_cv_splitter(
        aligned.task,
        safe_n_splits(aligned.task, y, groups, config.cv.n_splits),
        config.seed,
        True,
        groups,
    )
    outer_folds = []
    for train, assessment in outer_splitter.split(np.zeros(len(y)), y, groups):
        inner_splitter = make_cv_splitter(
            aligned.task,
            safe_n_splits(aligned.task, y[train], groups[train], config.cv.inner_splits),
            config.seed,
            True,
            groups[train],
        )
        outer_folds.append(
            {
                "train": train.astype(int).tolist(),
                "assessment": assessment.astype(int).tolist(),
                "inner_folds": [
                    {
                        "train": train[inner_train].astype(int).tolist(),
                        "assessment": train[inner_assessment].astype(int).tolist(),
                    }
                    for inner_train, inner_assessment in inner_splitter.split(
                        np.zeros(len(train)), y[train], groups[train]
                    )
                ],
            }
        )
    path = config_path.parent / "fixed_splits.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "omicau_public_split_manifest_v2",
                "aligned_provenance_sha256": aligned.provenance_hash,
                **public_manifest_identity_binding(groups=groups, permutation_strata=None),
                "outer_folds": outer_folds,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def regression_case(tmp_path: Path) -> Path:
    root = write_mock_dataset(
        tmp_path / "case",
        task="regression",
        n_samples=48,
        seed=211,
        signal_features=4,
        redundant_features=3,
        confounded_features=3,
        noise_features=2,
    )
    config_path = root / "config.json"
    clinical_path = root / "clinical.csv"
    clinical = pd.read_csv(clinical_path)
    clinical["unit"] = clinical["sample_id"]
    clinical.to_csv(clinical_path, index=False)
    config = OmicauConfig.from_file(config_path)
    config.clinical.group = "unit"
    config.cv.n_splits = 2
    config.cv.inner_splits = 2
    config.controls.enabled = False
    config.neural.enabled = False
    config.xai.enabled = False
    config.to_json(config_path)
    manifest = _write_fixed_manifest(config_path)
    config = OmicauConfig.from_file(config_path)
    config.cv.split_manifest = manifest.name
    config.to_json(config_path)
    return config_path


@pytest.mark.parametrize("hash_seed", [1, 17, 101, 999, 4099])
def test_regression_preflight_is_hash_seed_stable(regression_case: Path, hash_seed: int):
    config = OmicauConfig.from_file(regression_case)
    aligned = load_and_align(config)
    plan, _ = _fixed_plan_for_aligned(config, aligned)
    expected = str(plan.receipt()["support_summary"]["minimum_realized_assessment_variance"])
    code = (
        "from omicau.cli import _fixed_plan_for_aligned; "
        "from omicau.config import OmicauConfig; "
        "from omicau.data.alignment import load_and_align; "
        f"c=OmicauConfig.from_file({str(regression_case)!r}); "
        "a=load_and_align(c); p,_=_fixed_plan_for_aligned(c,a); "
        "print(p.receipt()['support_summary']['minimum_realized_assessment_variance'])"
    )
    environment = {**os.environ, "PYTHONHASHSEED": str(hash_seed)}
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected


def test_regression_variance_gate_rejects_a_real_boundary_drift(regression_case: Path):
    config = OmicauConfig.from_file(regression_case)
    aligned = load_and_align(config)
    plan, _ = _fixed_plan_for_aligned(config, aligned)
    threshold = plan.receipt()["support_summary"]["minimum_realized_assessment_variance"]
    envelope = json.loads(Path(config.cv.split_manifest).read_text(encoding="utf-8"))
    with pytest.raises(SplitValidationError, match="c06_metric_support_insufficient"):
        validate_split_manifest(
            {"outer_folds": envelope["outer_folds"]},
            n_samples=aligned.n_samples,
            groups=aligned.groups.to_numpy(),
            task="regression",
            requested_outer_k=config.cv.n_splits,
            requested_inner_k=config.cv.inner_splits,
            minimum_training_groups=2,
            minimum_assessment_groups=2,
            y=aligned.y.to_numpy(),
            minimum_regression_assessment_groups=2,
            minimum_regression_assessment_variance=np.nextafter(threshold, np.inf),
            public_manifest=True,
        )


def test_outcome_drift_is_rejected_before_model_fitting(regression_case: Path, tmp_path: Path):
    copied = tmp_path / "copied"
    shutil.copytree(regression_case.parent, copied)
    clinical = pd.read_csv(copied / "clinical.csv")
    clinical.loc[0, "label"] = float(clinical.loc[0, "label"]) + 0.25
    clinical.to_csv(copied / "clinical.csv", index=False)
    config = _case_config(copied)
    config.clinical.path = str(copied / "clinical.csv")
    config.cv.split_manifest = str(copied / "fixed_splits.json")
    for modality in config.modalities:
        modality.path = str(copied / f"{modality.name}.csv")
    aligned = load_and_align(config)
    with pytest.raises(ValueError, match="aligned input provenance"):
        _fixed_plan_for_aligned(config, aligned)
