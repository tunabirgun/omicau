"""Intentionally-biased synthetic multi-omic datasets.

These matrices are engineered with known ground truth so the audit can be
validated against it: two noisy modalities sharing one latent factor, an
optional independent complementary factor, a batch-structured modality, and a
pure-noise modality. Batch-outcome confounding and target-dependent missingness
are opt-in mechanisms; the global-null scenario excludes both from observed
model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
    oracle_truth: dict[str, object] = field(default_factory=dict)


def _loadings(rng: np.random.Generator, n_features: int, scale: float = 1.0) -> np.ndarray:
    return rng.normal(0.0, scale, size=n_features)


def validate_global_null_observed_masks(
    bundle: MockBundle,
    *,
    max_abs_missing_rate_difference: float = 0.08,
    max_abs_missing_outcome_correlation: float = 0.20,
) -> None:
    """Reject target-dependent observed masks in a generated global-null bundle.

    This development gate evaluates the actual missing-value masks as well as
    the declared mechanism. It is intentionally a coarse contamination check,
    not a statistical test of exact missing-at-random behavior.
    """
    oracle = bundle.oracle_truth
    if oracle.get("scenario") != "global_null":
        raise ValueError("Observed-mask validation is defined only for global_null bundles.")
    missingness = oracle.get("missingness", {})
    if missingness.get("mnar_probability") != 0 or missingness.get("conditional_on_target"):
        raise ValueError("Global-null oracle declares target-dependent missingness.")
    mask = np.concatenate([frame.isna().to_numpy(dtype=float) for frame in bundle.modalities.values()], axis=1)
    label = bundle.clinical["label"].to_numpy()
    if bundle.task == "classification":
        positive = label == "responder"
        if positive.sum() == 0 or positive.sum() == len(positive):
            raise ValueError("Global-null classification bundle has only one class.")
        delta = np.abs(mask[positive].mean(axis=0) - mask[~positive].mean(axis=0)).max()
        if delta > max_abs_missing_rate_difference:
            raise ValueError(f"Observed masks differ by class (maximum rate difference {delta:.3f}).")
    elif bundle.task == "regression":
        outcome = np.asarray(label, dtype=float)
        correlations = [abs(np.corrcoef(outcome, mask[:, j])[0, 1])
                        for j in range(mask.shape[1]) if mask[:, j].std() > 0]
        if correlations and max(correlations) > max_abs_missing_outcome_correlation:
            raise ValueError("Observed masks correlate too strongly with the global-null outcome.")


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
    missing_mnar: float | None = None,
    scenario: Literal["alternative", "global_null", "conditional_null"] = "alternative",
    signal_strength: float = 1.0,
    complementary_strength: float = 0.0,
    batch_outcome_confounding: float = 0.0,
    group_dependence: float = 0.0,
    group_mode: Literal["legacy_random", "independent", "repeated"] = "legacy_random",
    feature_naming: Literal["legacy", "generic"] = "legacy",
    include_complementary: bool | None = None,
    complementary_features: int | None = None,
) -> MockBundle:
    """Generate a controlled multi-omic bundle with known structure.

    Parameters mirror the biases under test. Defaults reproduce the original
    four-layer alternative. Special scenarios are explicit development tools:
    ``global_null`` has no target-linked latent factor, while
    ``conditional_null`` retains shared target signal but assigns zero
    incremental target effect to a present complementary layer. ``oracle_truth``
    records the data-generating mechanism separately and is never written into
    model inputs by this module.
    """
    if task not in {"classification", "regression", "survival"}:
        raise ValueError("Unknown task; use 'classification', 'regression', or 'survival'.")
    if scenario not in {"alternative", "global_null", "conditional_null"}:
        raise ValueError("Unknown scenario; use 'alternative', 'global_null', or 'conditional_null'.")
    if group_mode not in {"legacy_random", "independent", "repeated"}:
        raise ValueError("Unknown group_mode; use 'legacy_random', 'independent', or 'repeated'.")
    if feature_naming not in {"legacy", "generic"}:
        raise ValueError("Unknown feature_naming; use 'legacy' or 'generic'.")
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    if n_batches < 1:
        raise ValueError("n_batches must be positive.")
    if any(count < 1 for count in (signal_features, redundant_features, confounded_features, noise_features)):
        raise ValueError("Each base modality must have at least one feature.")
    if missing_mnar is None:
        missing_mnar = 0.0 if scenario == "global_null" else 0.18
    if any(value < 0 for value in (missing_mcar, missing_mnar, signal_strength,
                                   complementary_strength, batch_outcome_confounding,
                                   group_dependence)):
        raise ValueError("Missingness and strength parameters must be non-negative.")
    if missing_mcar > 1 or missing_mnar > 1:
        raise ValueError("Missingness probabilities must not exceed 1.")
    if scenario == "global_null" and missing_mnar != 0:
        raise ValueError("global_null requires target-independent missingness (missing_mnar=0).")
    if task == "survival" and (scenario != "alternative" or signal_strength != 1.0
                               or complementary_strength != 0.0
                               or batch_outcome_confounding != 0.0):
        raise ValueError("Special target mechanisms are currently defined for classification and regression only.")
    if scenario == "global_null" and any(value != 0 for value in
                                          (complementary_strength, batch_outcome_confounding)):
        raise ValueError("global_null fixes complementary and batch-outcome effects at zero.")
    if scenario == "conditional_null" and complementary_strength != 0:
        raise ValueError("conditional_null fixes the complementary effect at zero.")
    if group_mode == "repeated" and group_dependence <= 0:
        raise ValueError("repeated group_mode requires positive group_dependence.")
    if group_mode != "repeated" and group_dependence != 0:
        raise ValueError("group_dependence requires group_mode='repeated'.")

    inferred_complementary = scenario == "conditional_null" or complementary_strength > 0
    if include_complementary is None:
        include_complementary = inferred_complementary
    if scenario == "conditional_null" and not include_complementary:
        raise ValueError("conditional_null requires a complementary modality.")
    if complementary_features is None:
        complementary_features = signal_features
    if include_complementary and complementary_features < 1:
        raise ValueError("complementary_features must be positive when included.")

    rng = np.random.default_rng(seed)

    # The default branch preserves the original random-number draw order.
    if group_mode == "repeated":
        n_patients = max(2, int(np.ceil(n_samples / 2)))
        patient_ids = np.arange(n_samples) % n_patients
        rng.shuffle(patient_ids)
        group_factor_by_patient = rng.normal(0.0, 1.0, size=n_patients)
        group_component = group_dependence * group_factor_by_patient[patient_ids]
        patient_labels = [f"P{p:04d}" for p in patient_ids]
    else:
        patient_ids = None
        patient_labels = None
        group_component = np.zeros(n_samples, dtype=float)

    # -- latent factors ---------------------------------------------------- #
    z = rng.normal(0.0, 1.0, size=n_samples) + group_component
    batch = rng.integers(0, n_batches, size=n_samples)
    batch_factor = batch.astype(float) - batch.mean()
    z_complementary = rng.normal(0.0, 1.0, size=n_samples) if include_complementary else None

    base_effect = 1.8 if task == "classification" else 2.5
    shared_effect = base_effect * signal_strength
    complementary_effect = base_effect * complementary_strength
    batch_effect = base_effect * batch_outcome_confounding
    if scenario == "global_null":
        shared_effect = complementary_effect = batch_effect = 0.0
    elif scenario == "conditional_null":
        complementary_effect = 0.0

    # -- target ------------------------------------------------------------ #
    if task == "classification":
        if scenario == "global_null":
            y = rng.integers(0, 2, size=n_samples)
        else:
            logits = shared_effect * z + batch_effect * batch_factor
            if z_complementary is not None:
                logits = logits + complementary_effect * z_complementary
            logits = logits + rng.normal(0.0, 0.6, size=n_samples)
            prob = 1.0 / (1.0 + np.exp(-logits))
            y = (prob > 0.5).astype(int)
        y_values = np.where(y == 1, "responder", "non_responder")
    elif task == "regression":
        y = shared_effect * z + batch_effect * batch_factor
        if z_complementary is not None:
            y = y + complementary_effect * z_complementary
        y = y + rng.normal(0.0, 0.8, size=n_samples)
        y_values = y.astype(float)
    else:
        risk = 1.6 * z + rng.normal(0.0, 0.5, size=n_samples)   # z drives the hazard
        base = rng.exponential(np.exp(-risk))                    # higher risk -> shorter time
        cens = rng.exponential(float(np.median(base)) * 2.0, size=n_samples)
        surv_time = np.minimum(base, cens) + 0.05
        surv_event = (base <= cens).astype(int)                  # 1 = event, 0 = censored
        y_values = surv_time.astype(float)
    surv_event = surv_event if task == "survival" else None

    sample_ids = [f"S{ i+1 :04d}" for i in range(n_samples)]

    # -- groups: legacy identifiers remain available only for compatibility. -- #
    if patient_ids is None:
        if group_mode == "independent":
            patient_ids = np.arange(n_samples)
        else:
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

    names = ({"signal": "layer_a", "redundant": "layer_b", "complementary": "layer_c",
              "confounded": "layer_d", "noise": "layer_e"}
             if feature_naming == "generic" else
             {"signal": "signal", "redundant": "redundant", "complementary": "complementary",
              "confounded": "confounded", "noise": "noise"})
    prefixes = ({"signal": "L1", "redundant": "L2", "complementary": "L3",
                 "confounded": "L4", "noise": "L5"}
                if feature_naming == "generic" else
                {"signal": "SIG", "redundant": "RED", "complementary": "COMP",
                 "confounded": "CONF", "noise": "NOISE"})
    modalities: dict[str, pd.DataFrame] = {}
    modalities[names["signal"]] = _modality(z, signal_features, prefixes["signal"], snr=1.2)
    modalities[names["redundant"]] = _modality(z, redundant_features, prefixes["redundant"], snr=1.0)
    if include_complementary:
        assert z_complementary is not None
        modalities[names["complementary"]] = _modality(
            z_complementary, complementary_features, prefixes["complementary"], snr=1.2
        )
    # This layer is batch-structured. It is outcome-confounded only when the
    # explicit batch_outcome_confounding parameter is non-zero.
    modalities[names["confounded"]] = _modality(
        batch_factor, confounded_features, prefixes["confounded"], snr=1.5
    )
    # Pure noise negative control.
    modalities[names["noise"]] = pd.DataFrame(
        rng.normal(0.0, 1.0, size=(n_samples, noise_features)),
        index=sample_ids,
        columns=[f"{prefixes['noise']}_g{j+1:03d}" for j in range(noise_features)],
    )

    truth = {
        names["signal"]: "shared_signal_measurement" if shared_effect else "nonpredictive_shared_structure",
        names["redundant"]: "second_noisy_shared_measurement" if shared_effect else "nonpredictive_shared_structure",
        names["confounded"]: "batch_outcome_confounded" if batch_effect else "batch_structured_only",
        names["noise"]: "negative_control",
    }
    if include_complementary:
        truth[names["complementary"]] = (
            "complementary_predictive" if complementary_effect else "conditional_null_candidate"
        )

    # -- inject missingness ------------------------------------------------ #
    for name, frame in modalities.items():
        arr = np.array(frame.to_numpy(dtype=float), copy=True)  # writable (CoW-safe)
        mcar = rng.random(arr.shape) < missing_mcar
        arr[mcar] = np.nan
        if name == names["signal"] and task == "classification" and scenario != "global_null":
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
    if task == "survival":
        clinical["time"] = y_values
        clinical["event"] = surv_event

    dimensions = {
        names["signal"]: {"target_linked": shared_effect != 0, "shared_information_factor": True,
                          "batch_structured": False, "incremental_target_effect": None},
        names["redundant"]: {"target_linked": shared_effect != 0, "shared_information_factor": True,
                             "batch_structured": False, "incremental_target_effect": None},
        names["confounded"]: {"target_linked": batch_effect != 0, "shared_information_factor": False,
                              "batch_structured": True, "incremental_target_effect": None},
        names["noise"]: {"target_linked": False, "shared_information_factor": False,
                         "batch_structured": False, "incremental_target_effect": None},
    }
    if include_complementary:
        dimensions[names["complementary"]] = {
            "target_linked": complementary_effect != 0,
            "shared_information_factor": False,
            "batch_structured": False,
            "incremental_target_effect": complementary_effect != 0,
        }
    oracle_truth: dict[str, object] = {
        "scenario": scenario,
        "target_coefficients": {
            "shared": float(shared_effect),
            "complementary": float(complementary_effect),
            "batch": float(batch_effect),
        },
        "modality_dimensions": dimensions,
        "feature_sets": {name: list(frame.columns) for name, frame in modalities.items()},
        "roles_to_input_names": names,
        "groups": {"mode": group_mode, "dependence": float(group_dependence),
                   "n_units": int(len(set(patient_ids)))},
        "missingness": {"mcar_probability": float(missing_mcar),
                        "mnar_probability": float(missing_mnar),
                        "conditional_on_target": scenario != "global_null" and task == "classification"},
        "latent_factors": {
            "shared": z.copy(), "batch": batch_factor.copy(),
            "group_component": group_component.copy(),
            **({"complementary": z_complementary.copy()} if z_complementary is not None else {}),
        },
    }
    return MockBundle(modalities=modalities, clinical=clinical, truth=truth,
                      oracle_truth=oracle_truth, task=task)


def mock_config(
    output_dir: str | Path = "omicau_output",
    *,
    task: str = "classification",
    seed: int = 42,
    include_complementary: bool = False,
    modality_names: list[str] | None = None,
) -> OmicauConfig:
    """Build an :class:`OmicauConfig` matching an in-memory mock bundle."""
    survival = task == "survival"
    modality_specs = (
        [{"name": "signal", "description": "Synthetic predictive modality"},
         {"name": "redundant", "description": "Redundant with signal"},
         *([{"name": "complementary", "description": "Synthetic complementary modality"}]
           if include_complementary else []),
         {"name": "confounded", "description": "Batch-confounded modality"},
         {"name": "noise", "description": "Pure-noise negative control"}]
        if modality_names is None else
        [{"name": name, "description": "Synthetic modality"} for name in modality_names]
    )
    cfg = OmicauConfig.from_dict(
        {
            "run_name": "mock_audit",
            "output_dir": str(output_dir),
            "seed": seed,
            "modalities": modality_specs,
            "clinical": {
                "target": "time" if survival else "label",
                "sample_id": "sample_id",
                "group": "patient_id",
                "batch": "batch",
                "task": task,
                "time": "time" if survival else None,
                "event": "event" if survival else None,
                "time_unit": "months" if survival else "",
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

    cfg = mock_config(output_dir="run", task=task, seed=seed,
                      include_complementary="complementary" in bundle.modalities,
                      modality_names=list(bundle.modalities))
    for spec in cfg.modalities:
        spec.path = paths[spec.name]
    cfg.clinical.path = clin_path.name
    cfg.to_json(out / "config.json")
    return out
