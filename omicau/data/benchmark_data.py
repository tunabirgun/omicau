"""Intentionally-biased synthetic multi-omic datasets.

These matrices are engineered with known ground truth so the audit can be
validated against it: a *signal* modality that genuinely predicts the target, a
*redundant* modality carrying the same latent factor (tests redundancy
detection), a *confounded* modality tied to batch rather than outcome (tests
batch-effect and confounding diagnostics), and a *pure-noise* modality (a
negative control). Missingness is injected both completely-at-random and
target-dependent (MNAR) so the missingness-bias tests have signal to find.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from omicau.config import OmicauConfig


@dataclass
class MockBundle:
    """In-memory synthetic study with recorded ground-truth modality roles."""

    modalities: dict[str, pd.DataFrame]
    clinical: pd.DataFrame
    truth: dict[str, str]
    task: str


def _loadings(rng: np.random.Generator, n_features: int, scale: float = 1.0) -> np.ndarray:
    return rng.normal(0.0, scale, size=n_features)


def make_mock_dataset(
    *,
    task: str = "classification",
    n_samples: int = 140,
    seed: int = 42,
    n_batches: int = 3,
    signal_features: int = 30,
    redundant_features: int = 20,
    confounded_features: int = 25,
    noise_features: int = 15,
    missing_mcar: float = 0.04,
    missing_mnar: float = 0.18,
) -> MockBundle:
    """Generate a controlled multi-omic bundle with known structure.

    Parameters mirror the biases under test. The signal and redundant modalities
    share a latent factor ``z`` that drives the target; the confounded modality
    is driven by a batch factor orthogonal to ``z``; noise is i.i.d. Gaussian.
    """
    rng = np.random.default_rng(seed)

    # -- latent factors ---------------------------------------------------- #
    z = rng.normal(0.0, 1.0, size=n_samples)  # outcome-driving factor
    batch = rng.integers(0, n_batches, size=n_samples)
    batch_factor = batch.astype(float) - batch.mean()  # confounding factor

    # -- target ------------------------------------------------------------ #
    if task == "classification":
        logits = 1.8 * z + rng.normal(0.0, 0.6, size=n_samples)
        prob = 1.0 / (1.0 + np.exp(-logits))
        y = (prob > 0.5).astype(int)
        y_values = np.where(y == 1, "responder", "non_responder")
    elif task == "regression":
        y = 2.5 * z + rng.normal(0.0, 0.8, size=n_samples)
        y_values = y.astype(float)
    else:
        raise ValueError(f"Unknown task '{task}' (use 'classification' or 'regression').")

    sample_ids = [f"S{ i+1 :04d}" for i in range(n_samples)]

    # -- groups: some patients contribute >1 sample (leakage risk) --------- #
    n_patients = int(n_samples * 0.75)
    patient_ids = rng.integers(0, n_patients, size=n_samples)
    patient_labels = [f"P{p:04d}" for p in patient_ids]

    def _modality(factor: np.ndarray, n_features: int, prefix: str, snr: float) -> pd.DataFrame:
        load = _loadings(rng, n_features, scale=snr)
        signal = np.outer(factor, load)
        noise = rng.normal(0.0, 1.0, size=(n_samples, n_features))
        mat = signal + noise
        cols = [f"{prefix}_g{j+1:03d}" for j in range(n_features)]
        return pd.DataFrame(mat, index=sample_ids, columns=cols)

    modalities: dict[str, pd.DataFrame] = {}
    modalities["signal"] = _modality(z, signal_features, "SIG", snr=1.2)
    # Redundant: same latent z, so it duplicates the signal's information.
    modalities["redundant"] = _modality(z, redundant_features, "RED", snr=1.0)
    # Confounded: driven by batch, not by the outcome.
    modalities["confounded"] = _modality(batch_factor, confounded_features, "CONF", snr=1.5)
    # Pure noise negative control.
    modalities["noise"] = pd.DataFrame(
        rng.normal(0.0, 1.0, size=(n_samples, noise_features)),
        index=sample_ids,
        columns=[f"NOISE_g{j+1:03d}" for j in range(noise_features)],
    )

    truth = {
        "signal": "predictive",
        "redundant": "redundant_with_signal",
        "confounded": "batch_confounded",
        "noise": "negative_control",
    }

    # -- inject missingness ------------------------------------------------ #
    for name, frame in modalities.items():
        arr = np.array(frame.to_numpy(dtype=float), copy=True)  # writable (CoW-safe)
        mcar = rng.random(arr.shape) < missing_mcar
        arr[mcar] = np.nan
        if name == "signal" and task == "classification":
            # MNAR: positive-class rows lose signal entries more often.
            pos_rows = np.where(y == 1)[0]
            mnar = rng.random((len(pos_rows), arr.shape[1])) < missing_mnar
            block = arr[pos_rows]
            block[mnar] = np.nan
            arr[pos_rows] = block
        modalities[name] = pd.DataFrame(arr, index=frame.index, columns=frame.columns)

    clinical = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": y_values,
            "patient_id": patient_labels,
            "batch": [f"batch{b+1}" for b in batch],
        }
    )

    return MockBundle(modalities=modalities, clinical=clinical, truth=truth, task=task)


def mock_config(
    output_dir: str | Path = "omicau_output",
    *,
    task: str = "classification",
    seed: int = 42,
) -> OmicauConfig:
    """Build an :class:`OmicauConfig` matching an in-memory mock bundle."""
    target = "label"
    cfg = OmicauConfig.from_dict(
        {
            "run_name": "mock_audit",
            "output_dir": str(output_dir),
            "seed": seed,
            "modalities": [
                {"name": "signal", "description": "Synthetic predictive modality"},
                {"name": "redundant", "description": "Redundant with signal"},
                {"name": "confounded", "description": "Batch-confounded modality"},
                {"name": "noise", "description": "Pure-noise negative control"},
            ],
            "clinical": {
                "target": target,
                "sample_id": "sample_id",
                "group": "patient_id",
                "batch": "batch",
                "task": task,
                "positive_label": "responder" if task == "classification" else None,
            },
            "cv": {"n_splits": 3, "seed": seed},
            "neural": {"enabled": True, "epochs": 15, "batch_size": 16, "hidden_dim": 32},
            "classical": {"enabled": True, "models": ["linear", "random_forest"]},
            "xai": {"enabled": True, "permutation_repeats": 4, "top_k_features": 15},
        }
    )
    return cfg


def write_mock_dataset(
    out_dir: str | Path,
    *,
    task: str = "classification",
    seed: int = 42,
    **kwargs,
) -> Path:
    """Write a mock bundle to CSVs plus a ready-to-run config; return the dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = make_mock_dataset(task=task, seed=seed, **kwargs)

    paths: dict[str, str] = {}
    for name, frame in bundle.modalities.items():
        p = out / f"{name}.csv"
        frame.to_csv(p, index=True, index_label="sample_id", lineterminator="\n")
        paths[name] = str(p.name)
    clin_path = out / "clinical.csv"
    bundle.clinical.to_csv(clin_path, index=False, lineterminator="\n")

    cfg = mock_config(output_dir="run", task=task, seed=seed)  # relative to config dir
    for spec in cfg.modalities:
        spec.path = paths[spec.name]
    cfg.clinical.path = clin_path.name
    cfg.to_json(out / "config.json")
    return out
