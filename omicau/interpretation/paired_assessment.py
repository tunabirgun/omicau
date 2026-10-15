"""Private paired-OOF validation and public-safe assessment summaries."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score

FAMILY_ALPHA = 0.05
ORDINARY_QUANTILES = (0.025, 0.975)
PRIMARY_METRIC = {"classification": "auroc", "regression": "r2"}


class PairingError(ValueError):
    """Paired out-of-fold records cannot support a valid contrast."""


def _as_1d(value: Any, field: str) -> np.ndarray:
    if value is None:
        raise PairingError("missing_" + field)
    array = np.asarray(value)
    if array.ndim != 1:
        raise PairingError("invalid_" + field)
    return array


def _finite(value: Any, field: str, n: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim not in (1, 2) or array.shape[0] != n or not np.isfinite(array).all():
        raise PairingError("invalid_" + field)
    return array


def _identity_matches(first: Any, second: Any) -> bool:
    if first is None or second is None:
        return False
    try:
        return bool(first == second)
    except (TypeError, ValueError):
        return False


def _arrays_equal(first: np.ndarray, second: np.ndarray) -> bool:
    if first.shape != second.shape:
        return False
    if np.issubdtype(first.dtype, np.number) and np.issubdtype(second.dtype, np.number):
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


def _invalid_identifier(value: Any) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return True
    try:
        scalar = np.asarray(value)
        if scalar.ndim == 0 and np.issubdtype(scalar.dtype, np.number):
            if not np.isfinite(scalar.item()):
                return True
    except (TypeError, ValueError):
        return True
    try:
        unequal = value != value
        if bool(unequal):
            return True
    except (TypeError, ValueError):
        return True
    try:
        hash(value)
    except TypeError:
        return True
    return False


def _expected_preprocessing_fits(split_identity: Any) -> tuple[tuple[int, ...], ...]:
    try:
        pairs = tuple(split_identity)
        expected = tuple(tuple(int(value) for value in pair[0]) for pair in pairs)
        if not pairs or any(len(pair) != 2 for pair in pairs):
            raise ValueError
        return expected
    except (TypeError, ValueError, IndexError):
        raise PairingError("invalid_split_identity") from None


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def _interval(values: np.ndarray, quantiles: tuple[float, float]) -> dict[str, Any]:
    return {
        "lower": _quantile(values, quantiles[0]),
        "upper": _quantile(values, quantiles[1]),
        "quantile_method": "linear",
    }


def _metric(y: np.ndarray, score: np.ndarray, pred: np.ndarray, task: str) -> float:
    try:
        if task == "regression":
            return float(r2_score(y, pred)) if np.ptp(y) > 0 else float("nan")
        classes = np.unique(y)
        if len(classes) == 2:
            values = score if score.ndim == 1 else score[:, 1]
            return float(roc_auc_score(y, values))
        if score.ndim == 2 and len(classes) > 2:
            return float(roc_auc_score(y, score, multi_class="ovr", average="macro"))
    except ValueError:
        pass
    return float("nan")


def _validate_result(result: Any, task: str) -> dict[str, Any]:
    if result is None:
        raise PairingError("missing_result")
    if getattr(result, "task", None) != task:
        raise PairingError("task_mismatch")
    y = _as_1d(getattr(result, "oof_true", None), "out_of_fold_targets")
    n = len(y)
    if n == 0 or not np.isfinite(y).all():
        raise PairingError("invalid_out_of_fold_targets")
    row_keys = _as_1d(getattr(result, "oof_row_keys", None), "row_keys")
    groups = getattr(result, "oof_groups", None)
    folds = _as_1d(getattr(result, "oof_fold_ids", None), "fold_ids")
    if (
        len(row_keys) != n
        or len(folds) != n
        or any(_invalid_identifier(value) for value in row_keys.tolist())
        or len(set(row_keys.tolist())) != n
    ):
        raise PairingError("invalid_row_key_or_fold_identity")
    if not np.issubdtype(folds.dtype, np.integer) or np.any(folds < 0):
        raise PairingError("invalid_fold_ids")
    resampling_unit = getattr(result, "resampling_unit", None)
    sample_independence_declared = bool(
        getattr(result, "sample_independence_declared", False)
    )
    if groups is None:
        if resampling_unit != "sample" or not sample_independence_declared:
            raise PairingError("sample_independence_not_declared")
    else:
        groups = _as_1d(groups, "groups")
        if (
            len(groups) != n
            or any(_invalid_identifier(value) for value in groups.tolist())
        ):
            raise PairingError("invalid_group_ids")
        if len(set(groups.tolist())) < 2:
            raise PairingError("insufficient_independent_groups")
        if resampling_unit != "group" or sample_independence_declared:
            raise PairingError("invalid_resampling_contract")
    split_identity = getattr(result, "split_identity", None)
    if split_identity is None:
        raise PairingError("missing_split_identity")
    split_receipt_identity = getattr(result, "split_receipt_identity", None)
    if split_receipt_identity is None:
        raise PairingError("missing_split_receipt_identity")
    if getattr(result, "preprocessing_scope", None) != "fold_local_pipeline":
        raise PairingError("invalid_preprocessing_scope")
    if not _identity_matches(
        getattr(result, "preprocessing_fit_identity", None),
        _expected_preprocessing_fits(split_identity),
    ):
        raise PairingError("missing_fold_local_preprocessing_evidence")
    score_class_order = None
    score_positive_class = None
    score_class_fit_identity = None
    score = _finite(getattr(result, "oof_score", None), "out_of_fold_scores", n)
    if task == "classification":
        outcome_labels = tuple(np.unique(y).tolist())
        score_class_order = getattr(result, "score_class_order", None)
        score_positive_class = getattr(result, "score_positive_class", None)
        if tuple(score_class_order or ()) != outcome_labels:
            raise PairingError("score_class_order_outcome_mismatch")
        if len(outcome_labels) == 2:
            if score.ndim != 1 or score_positive_class != outcome_labels[-1]:
                raise PairingError("binary_score_positive_class_mismatch")
        elif score.ndim != 2 or score.shape[1] != len(outcome_labels) or score_positive_class is not None:
            raise PairingError("multiclass_score_column_contract_invalid")
        score_class_fit_identity = getattr(result, "score_class_fit_identity", None)
        expected_fit_order = tuple(score_class_order for _ in sorted(set(folds.tolist())))
        if not _identity_matches(score_class_fit_identity, expected_fit_order):
            raise PairingError("score_class_fit_mapping_invalid")
    return {
        "y": y,
        "score": score,
        "pred": _finite(getattr(result, "oof_pred", None), "out_of_fold_predictions", n),
        "row_keys": row_keys,
        "groups": groups,
        "folds": folds,
        "split_identity": split_identity,
        "split_receipt_identity": split_receipt_identity,
        "resampling_unit": resampling_unit,
        "sample_independence_declared": sample_independence_declared,
        "score_class_order": score_class_order,
        "score_positive_class": score_positive_class,
        "score_class_fit_identity": score_class_fit_identity,
    }


def validate_paired_results(full: Any, reduced: Any, task: str, candidate: str) -> dict[str, Any]:
    """Validate private row-level bindings before calculating a paired delta."""
    left = _validate_result(full, task)
    right = _validate_result(reduced, task)
    if getattr(full, "model_role", None) != "full":
        raise PairingError("full_model_role_mismatch")
    if getattr(reduced, "model_role", None) != "reduced:" + candidate:
        raise PairingError("reduced_model_role_mismatch")
    for key, reason in (
        ("y", "unmatched_out_of_fold_targets"),
        ("row_keys", "unmatched_row_keys"),
        ("folds", "unmatched_fold_ids"),
    ):
        if not _arrays_equal(left[key], right[key]):
            raise PairingError(reason)
    if (left["groups"] is None) != (right["groups"] is None):
        raise PairingError("unmatched_group_metadata")
    if left["groups"] is not None and not _arrays_equal(left["groups"], right["groups"]):
        raise PairingError("unmatched_group_metadata")
    if not _identity_matches(left["split_identity"], right["split_identity"]):
        raise PairingError("unmatched_split_identity")
    if not _identity_matches(
        left["split_receipt_identity"],
        right["split_receipt_identity"],
    ):
        raise PairingError("unmatched_split_receipt_identity")
    if left["resampling_unit"] != right["resampling_unit"]:
        raise PairingError("unmatched_resampling_unit")
    if (
        left["sample_independence_declared"]
        != right["sample_independence_declared"]
    ):
        raise PairingError("unmatched_sample_independence_contract")
    if task == "classification":
        if not _identity_matches(left["score_class_order"], right["score_class_order"]):
            raise PairingError("unmatched_score_class_order")
        if left["score_positive_class"] != right["score_positive_class"]:
            raise PairingError("unmatched_score_positive_class")
        if not _identity_matches(
            left["score_class_fit_identity"],
            right["score_class_fit_identity"],
        ):
            raise PairingError("unmatched_score_class_fit_mapping")
    return left | {"reduced_score": right["score"], "reduced_pred": right["pred"]}


def _bootstrap_indices(groups: np.ndarray | None, n: int, rng: np.random.Generator) -> np.ndarray:
    if groups is None:
        return rng.integers(0, n, size=n)
    levels = np.unique(groups)
    positions = [np.flatnonzero(groups == level) for level in levels]
    chosen = rng.integers(0, len(levels), size=len(levels))
    return np.concatenate([positions[index] for index in chosen])


def _bootstrap_delta(
    paired: dict[str, Any], task: str, family_size: int, seed: int
) -> tuple[np.ndarray, int, int]:
    if family_size < 1:
        raise PairingError("invalid_multiplicity_family")
    target = int(np.ceil(4000 * family_size))
    attempts_max = 2 * target
    y = paired["y"]
    class_count = len(np.unique(y)) if task == "classification" else 0
    rng = np.random.default_rng(seed)
    values: list[float] = []
    attempts = 0
    while len(values) < target and attempts < attempts_max:
        attempts += 1
        index = _bootstrap_indices(paired["groups"], len(y), rng)
        if task == "classification" and len(np.unique(y[index])) < class_count:
            continue
        full = _metric(y[index], paired["score"][index], paired["pred"][index], task)
        reduced = _metric(
            y[index], paired["reduced_score"][index], paired["reduced_pred"][index], task
        )
        if np.isfinite(full) and np.isfinite(reduced):
            values.append(full - reduced)
    return np.asarray(values, dtype=float), attempts, target


def assess_incremental(
    full: Any,
    reduced: Any,
    *,
    task: str,
    candidate: str,
    family_size: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Return the protocol-defined paired held-out performance assessment."""
    valid_family_size = (
        not isinstance(family_size, bool)
        and isinstance(family_size, Integral)
        and int(family_size) >= 0
    )
    family_value = int(family_size) if valid_family_size else None
    target = 0 if family_value in (None, 0) else int(np.ceil(4000 * family_value))
    decision_quantiles = (
        None
        if family_value in (None, 0)
        else (FAMILY_ALPHA / 2 / family_value, 1 - FAMILY_ALPHA / 2 / family_value)
    )
    common = {
        "multiplicity_family": {
            "family_size": family_value,
            "family_alpha": FAMILY_ALPHA,
            "scope": "cohort_reference_ledger",
            "decision_quantiles": (
                None if decision_quantiles is None else list(decision_quantiles)
            ),
        },
        "bootstrap": {
            "seed": int(seed),
            "target_valid_resamples": target,
            "max_attempts": 2 * target,
            "n_invalid_resamples": 0,
            "quantile_method": "linear",
        },
        "resampling_unit": None,
        "resampling_contract": None,
        "n_valid_resamples": 0,
        "n_attempts": 0,
        "reason": None,
    }
    if not valid_family_size:
        return {
            "status": "unavailable",
            "estimate": None,
            "ordinary_95_ci": None,
            "decision_ci": None,
            **common,
            "reason": "invalid_multiplicity_family",
        }
    if family_value == 0:
        return {
            "status": "not_applicable",
            "estimate": None,
            "ordinary_95_ci": None,
            "decision_ci": None,
            **common,
            "reason": "single_assay_ledger",
        }
    if task not in {"classification", "regression"}:
        return {
            "status": "unavailable",
            "estimate": None,
            "ordinary_95_ci": None,
            "decision_ci": None,
            **common,
            "reason": "task_not_supported_for_paired_metric",
        }
    try:
        paired = validate_paired_results(full, reduced, task, candidate)
        estimate = _metric(paired["y"], paired["score"], paired["pred"], task) - _metric(
            paired["y"], paired["reduced_score"], paired["reduced_pred"], task
        )
        if not np.isfinite(estimate):
            raise PairingError("nonfinite_pooled_delta")
        values, attempts, target = _bootstrap_delta(paired, task, family_value, seed)
    except PairingError as error:
        return {
            "status": "unavailable",
            "estimate": None,
            "ordinary_95_ci": None,
            "decision_ci": None,
            **common,
            "reason": str(error),
        }
    unit = "group" if paired["groups"] is not None else "sample"
    base = {
        **common,
        "estimate": float(estimate),
        "resampling_unit": unit,
        "resampling_contract": {
            "unit": unit,
            "sample_independence_declared": bool(
                paired["sample_independence_declared"]
            ),
        },
        "n_valid_resamples": int(len(values)),
        "n_attempts": int(attempts),
    }
    base["bootstrap"]["n_invalid_resamples"] = int(attempts - len(values))
    if len(values) != target:
        return {
            "status": "unavailable",
            "ordinary_95_ci": None,
            "decision_ci": None,
            **base,
            "reason": "insufficient_valid_resamples",
        }
    decision = _interval(values, decision_quantiles)
    if decision["lower"] > 0:
        status = "increased_held_out_performance"
    elif decision["upper"] < 0:
        status = "reduced_held_out_performance"
    else:
        status = "unresolved"
    return {
        "status": status,
        "ordinary_95_ci": _interval(values, ORDINARY_QUANTILES),
        "decision_ci": decision,
        **base,
    }


def assess_standalone(result: Any, *, task: str, seed: int = 42, n_boot: int = 1000) -> dict[str, Any]:
    """Return a descriptive, non-decision standalone metric and bootstrap interval."""
    try:
        checked = _validate_result(result, task)
        estimate = _metric(checked["y"], checked["score"], checked["pred"], task)
        if not np.isfinite(estimate):
            raise PairingError("nonfinite_standalone_metric")
    except PairingError as error:
        return {
            "status": "unavailable", "estimate": None, "ordinary_95_ci": None,
            "resampling_unit": None, "n_valid_resamples": 0, "n_attempts": 0,
            "resampling_contract": None,
            "bootstrap": {
                "seed": int(seed), "target_valid_resamples": int(n_boot),
                "max_attempts": 2 * int(n_boot), "n_invalid_resamples": 0,
                "quantile_method": "linear",
            },
            "reason": str(error),
        }
    rng = np.random.default_rng(seed)
    target = int(n_boot)
    attempts = 0
    values: list[float] = []
    class_count = len(np.unique(checked["y"])) if task == "classification" else 0
    while len(values) < target and attempts < 2 * target:
        attempts += 1
        index = _bootstrap_indices(checked["groups"], len(checked["y"]), rng)
        if task == "classification" and len(np.unique(checked["y"][index])) < class_count:
            continue
        value = _metric(checked["y"][index], checked["score"][index], checked["pred"][index], task)
        if np.isfinite(value):
            values.append(value)
    if len(values) != target:
        return {
            "status": "unavailable", "estimate": float(estimate), "ordinary_95_ci": None,
            "resampling_unit": "group" if checked["groups"] is not None else "sample",
            "resampling_contract": {
                "unit": checked["resampling_unit"],
                "sample_independence_declared": bool(
                    checked["sample_independence_declared"]
                ),
            },
            "n_valid_resamples": int(len(values)), "n_attempts": int(attempts),
            "bootstrap": {
                "seed": int(seed), "target_valid_resamples": target,
                "max_attempts": 2 * target, "n_invalid_resamples": int(attempts - len(values)),
                "quantile_method": "linear",
            },
            "reason": "insufficient_valid_resamples",
        }
    return {
        "status": "descriptive", "estimate": float(estimate),
        "ordinary_95_ci": _interval(np.asarray(values, dtype=float), ORDINARY_QUANTILES),
        "resampling_unit": "group" if checked["groups"] is not None else "sample",
        "resampling_contract": {
            "unit": checked["resampling_unit"],
            "sample_independence_declared": bool(
                checked["sample_independence_declared"]
            ),
        },
        "n_valid_resamples": int(len(values)), "n_attempts": int(attempts),
        "bootstrap": {
            "seed": int(seed), "target_valid_resamples": target,
            "max_attempts": 2 * target, "n_invalid_resamples": int(attempts - len(values)),
            "quantile_method": "linear",
        },
        "reason": None,
    }
