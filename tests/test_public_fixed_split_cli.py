from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omicau.config import OmicauConfig
from omicau.data.alignment import load_and_align
from omicau.data.benchmark_data import write_mock_dataset
from omicau.models.base import make_cv_splitter, safe_n_splits
from omicau.models.split_plan import public_manifest_identity_binding


def _write_fixed_manifest(config_path: Path) -> Path:
    config = OmicauConfig.from_file(config_path)
    aligned = load_and_align(config)
    y = aligned.y.to_numpy()
    groups = aligned.groups.to_numpy()
    outer_splitter = make_cv_splitter(
        aligned.task,
        safe_n_splits(aligned.task, y, groups, 2),
        config.seed,
        True,
        groups,
    )
    outer_folds = []
    for train, assessment in outer_splitter.split(np.zeros(len(y)), y, groups):
        inner_y, inner_groups = y[train], groups[train]
        inner_splitter = make_cv_splitter(
            aligned.task,
            safe_n_splits(aligned.task, inner_y, inner_groups, 2),
            config.seed,
            True,
            inner_groups,
        )
        inner_folds = [
            {
                "train": train[inner_train].astype(int).tolist(),
                "assessment": train[inner_assessment].astype(int).tolist(),
            }
            for inner_train, inner_assessment in inner_splitter.split(
                np.zeros(len(train)), inner_y, inner_groups
            )
        ]
        outer_folds.append(
            {
                "train": train.astype(int).tolist(),
                "assessment": assessment.astype(int).tolist(),
                "inner_folds": inner_folds,
            }
        )
    path = config_path.parent / "fixed_splits.json"
    binding = public_manifest_identity_binding(
        groups=aligned.groups.to_numpy(),
        permutation_strata=(
            None if aligned.permutation_strata is None else aligned.permutation_strata.to_numpy()
        ),
    )
    path.write_text(
        json.dumps(
            {
                "schema_version": "omicau_public_split_manifest_v2",
                "aligned_provenance_sha256": aligned.provenance_hash,
                **binding,
                "outer_folds": outer_folds,
            }
        ),
        encoding="utf-8",
    )
    return path


def _fixed_config(tmp_path: Path) -> Path:
    dataset = write_mock_dataset(
        tmp_path / "dataset",
        n_samples=48,
        seed=73,
        signal_features=4,
        redundant_features=3,
        confounded_features=3,
        noise_features=2,
    )
    config_path = dataset / "config.json"
    clinical_path = dataset / "clinical.csv"
    clinical = pd.read_csv(clinical_path)
    clinical["unit"] = clinical["sample_id"]
    clinical.to_csv(clinical_path, index=False)
    initial = OmicauConfig.from_file(config_path)
    initial.clinical.group = "unit"
    initial.clinical.permutation_strata = "batch"
    initial.to_json(config_path)
    manifest = _write_fixed_manifest(config_path)
    config = OmicauConfig.from_file(config_path)
    config.cv.n_splits = 2
    config.cv.inner_splits = 2
    config.cv.split_manifest = manifest.name
    config.cv.n_bootstrap = 20
    config.classical.models = ["linear"]
    config.classical.max_features = None
    config.neural.enabled = False
    config.xai.enabled = False
    config.compute.cores = 1
    config.reporting.html = False
    config.clinical.permutation_strata = "batch"
    config.to_json(config_path)
    return config_path


def _cli(config_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from omicau.cli import main; main()",
            "run",
            "--config",
            str(config_path),
            "--cores",
            "1",
            "--device",
            "cpu",
            "--no-llm",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_routes_a_public_fixed_manifest_through_classical_models(tmp_path):
    config_path = _fixed_config(tmp_path)
    completed = _cli(config_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    audit = json.loads((config_path.parent / "run" / "audit.json").read_text(encoding="utf-8"))
    records = audit["models"]["classical"] + audit["models"]["controls"]
    assert records
    assert all(record["split_plan_status"] == "validated_public_exact_splits" for record in records)
    receipt = records[0]["split_plan_receipt"]
    assert receipt["decision"] == "validated"
    assert receipt["split_manifest_status"] == "validated_public_exact_manifest"
    assert isinstance(receipt["split_manifest_sha256"], str)
    assert not any("index" in key or "indices" in key for key in receipt)
    target = next(record for record in records if record["name"] == "control::group_permuted_target")
    assert target["control_execution_receipt"]["decision"] == "eligible"
    assert target["control_execution_receipt"]["permutation_registry_status"] == "not_required_public_contract"
    assert target["control_execution_receipt"]["expected_null_scope"] == "conditional"
    assert target["control_execution_receipt"]["global_chance_eligible"] is False
    assert all("patient" not in json.dumps(record) for record in records)


@pytest.mark.parametrize("defect", ["provenance", "fold_count", "group", "strata"])
def test_cli_rejects_fixed_manifest_drift_before_fitting(tmp_path, defect):
    config_path = _fixed_config(tmp_path)
    if defect == "provenance":
        manifest_path = config_path.parent / "fixed_splits.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["aligned_provenance_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif defect == "fold_count":
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload["cv"]["n_splits"] = 3
        config_path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        clinical_path = config_path.parent / "clinical.csv"
        clinical = pd.read_csv(clinical_path)
        if defect == "group":
            clinical["unit"] = [f"changed-{index}" for index in range(len(clinical))]
        else:
            clinical["batch"] = clinical["batch"].iloc[::-1].to_numpy()
        clinical.to_csv(clinical_path, index=False)
    completed = _cli(config_path)
    assert completed.returncode != 0
    assert "fixed split manifest" in completed.stdout + completed.stderr
    assert not (config_path.parent / "run" / "audit.json").exists()


def test_fixed_manifest_requires_explicit_design_strata(tmp_path):
    config_path = _fixed_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["clinical"].pop("permutation_strata")
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    completed = _cli(config_path)
    assert completed.returncode != 0
    assert "permutation_strata" in completed.stdout + completed.stderr
