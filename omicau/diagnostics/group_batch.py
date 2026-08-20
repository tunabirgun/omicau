"""Development-only group-level batch diagnostic primitives.

The functions in this module are not wired into production diagnostics. Every
method choice and Monte Carlo parameter is supplied by the caller so the final
benchmark contract can be frozen prospectively.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Literal
import warnings

import numpy as np
from scipy import stats

from omicau.diagnostics.group_missingness import (
    categorical_delta_distance,
    continuous_midrank_distance,
    holm_adjust,
    plus_one_pvalue,
)


EndpointKind = Literal["classification", "regression", "survival"]
Reducer = Literal["mean", "median"]
Standardization = Literal["global_zscore", "global_median_iqr"]
RepresentationDistance = Literal["normalized_euclidean"]
_PRIMARY_COMPONENT_STATUSES = frozenset({"tested", "not_applicable"})
_PRIMARY_COMPONENT_IDS = frozenset({"structure", "outcome", "event", "censoring"})


def _integer(value: int, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _complete_scalar(value: Any, name: str) -> None:
    if not isinstance(value, Hashable):
        raise TypeError(f"{name} values must be hashable scalars")
    try:
        missing = np.asarray(value is None or value != value)
    except (TypeError, ValueError):
        missing = np.asarray(False)
    if missing.ndim != 0 or bool(missing):
        raise ValueError(f"{name} must not contain missing values")


def _group_codes(group_ids: Sequence[Any], n_rows: int) -> tuple[np.ndarray, int]:
    groups = np.asarray(group_ids, dtype=object)
    if groups.ndim != 1 or len(groups) != n_rows:
        raise ValueError("group_ids must have one value per representation row")
    mapping: dict[Any, int] = {}
    codes = np.empty(n_rows, dtype=int)
    for row, value in enumerate(groups.tolist()):
        _complete_scalar(value, "group_ids")
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("boolean group identifiers are not supported")
        if value not in mapping:
            mapping[value] = len(mapping)
        codes[row] = mapping[value]
    if len(mapping) < 5:
        raise ValueError("at least five groups are required")
    return codes, len(mapping)


def _numeric_matrix(values: Any, name: str) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if matrix.dtype.kind == "b":
        matrix = matrix.astype(float)
    elif matrix.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be numeric")
    else:
        matrix = matrix.astype(float, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    return matrix


def collapse_group_representation(
    row_representation: Any,
    group_ids: Sequence[Any],
    *,
    group_reducer: Reducer,
) -> np.ndarray:
    """Collapse rows to one equal-weight diagnostic vector per group."""
    matrix = _numeric_matrix(row_representation, "row_representation")
    codes, n_groups = _group_codes(group_ids, matrix.shape[0])
    if group_reducer not in ("mean", "median"):
        raise ValueError("group_reducer must be 'mean' or 'median'")
    collapsed = np.vstack(
        [
            matrix[codes == group].mean(axis=0)
            if group_reducer == "mean"
            else np.median(matrix[codes == group], axis=0)
            for group in range(n_groups)
        ]
    )
    if not np.isfinite(collapsed).all():
        raise ValueError("group-level diagnostic representation must be finite")
    return collapsed


def collapse_pure_group_values(
    row_values: Sequence[Any],
    group_ids: Sequence[Any],
    *,
    value_kind: Literal["categorical", "continuous", "binary_event"],
) -> np.ndarray:
    """Collapse a complete row field that must be constant within every group."""
    values = np.asarray(row_values, dtype=object)
    if values.ndim != 1:
        raise ValueError("group field must be one-dimensional")
    codes, n_groups = _group_codes(group_ids, len(values))
    collapsed: list[Any] = []
    for group in range(n_groups):
        group_values = values[codes == group].tolist()
        for value in group_values:
            _complete_scalar(value, "group field")
        first = group_values[0]
        if any(value != first for value in group_values[1:]):
            raise ValueError("group field must be constant within each group")
        collapsed.append(first)
    if value_kind == "categorical":
        if len({type(value) for value in collapsed}) != 1:
            raise TypeError("categorical group field must use one consistent type")
        result = np.asarray(collapsed, dtype=object)
    elif value_kind in ("continuous", "binary_event"):
        if any(isinstance(value, (bool, np.bool_)) for value in collapsed):
            raise TypeError("numeric group field must not contain booleans")
        try:
            result = np.asarray(collapsed, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("numeric group field must be numeric") from exc
        if not np.isfinite(result).all():
            raise ValueError("numeric group field must be finite")
        if value_kind == "binary_event" and not np.isin(result, (0.0, 1.0)).all():
            raise ValueError("event indicator must contain only zero and one")
    else:
        raise ValueError("unsupported value_kind")
    if value_kind != "binary_event" and len({(type(v), v) for v in result.tolist()}) < 2:
        raise ValueError("group field must contain at least two values")
    return result


def standardize_group_representation(
    group_representation: Any,
    *,
    standardization: Standardization,
) -> np.ndarray:
    """Apply an explicitly selected global column standardization."""
    matrix = _numeric_matrix(group_representation, "group_representation")
    if standardization == "global_zscore":
        center = matrix.mean(axis=0)
        scale = matrix.std(axis=0, ddof=0)
    elif standardization == "global_median_iqr":
        center = np.median(matrix, axis=0)
        q25, q75 = np.percentile(matrix, [25, 75], axis=0)
        scale = q75 - q25
    else:
        raise ValueError(
            "standardization must be 'global_zscore' or 'global_median_iqr'"
        )
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("every registered representation feature must have positive global scale")
    standardized = (matrix - center) / scale
    if not np.isfinite(standardized).all() or not np.any(np.ptp(standardized, axis=0) > 0):
        raise ValueError("standardized representation is degenerate")
    return standardized


def full_representation_distance(
    standardized_representation: Any,
    *,
    representation_distance: RepresentationDistance,
) -> np.ndarray:
    """Construct the registered distance from every representation feature."""
    matrix = _numeric_matrix(standardized_representation, "standardized_representation")
    if matrix.shape[0] < 5:
        raise ValueError("at least five group-level representation rows are required")
    if representation_distance != "normalized_euclidean":
        raise ValueError("representation_distance must be 'normalized_euclidean'")
    delta = matrix[:, None, :] - matrix[None, :, :]
    distance = np.sqrt(np.mean(delta * delta, axis=2))
    np.fill_diagonal(distance, 0.0)
    if not np.any(distance > 0):
        raise ValueError("representation distance is degenerate")
    return distance


def _validate_batch(batch: np.ndarray, minimum_groups_per_batch: int) -> np.ndarray:
    minimum = _integer(minimum_groups_per_batch, "minimum_groups_per_batch", minimum=1)
    levels, counts = np.unique(batch, return_counts=True)
    if len(levels) < 2 or np.any(counts < minimum):
        raise ValueError("batch support is insufficient")
    return batch


def _validate_monte_carlo(reps: int, seed: int) -> tuple[int, int]:
    reps = _integer(reps, "reps", minimum=1)
    seed = _integer(seed, "seed", minimum=0)
    if seed > np.iinfo(np.uint32).max:
        raise ValueError("seed must fit in an unsigned 32-bit integer")
    return reps, seed


def _mgc_precomputed(
    x_distance: np.ndarray,
    y_distance: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> tuple[float, float]:
    if x_distance.shape != y_distance.shape or x_distance.shape[0] < 5:
        raise ValueError("distance matrices must be square, aligned, and contain five groups")
    for distance in (x_distance, y_distance):
        if (
            not np.isfinite(distance).all()
            or np.any(distance < 0)
            or not np.allclose(distance, distance.T, rtol=0, atol=1e-12)
            or not np.allclose(np.diag(distance), 0.0, rtol=0, atol=1e-12)
            or not np.any(distance > 0)
        ):
            raise ValueError("distance matrix is invalid or degenerate")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of replications is low.*",
            category=RuntimeWarning,
        )
        result = stats.multiscale_graphcorr(
            x_distance,
            y_distance,
            compute_distance=None,
            reps=reps,
            workers=1,
            random_state=seed,
        )
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if not np.isfinite(statistic) or not np.isfinite(p_value):
        raise ValueError("primary statistic is not estimable")
    return statistic, p_value


def structure_batch_association(
    group_representation: Any,
    group_batch: Sequence[Any],
    *,
    standardization: Standardization,
    representation_distance: RepresentationDistance,
    reps: int,
    seed: int,
    minimum_groups_per_batch: int,
) -> dict[str, Any]:
    """Test full-representation batch structure by precomputed-distance MGC."""
    reps, seed = _validate_monte_carlo(reps, seed)
    matrix = standardize_group_representation(
        group_representation, standardization=standardization
    )
    batch = np.asarray(group_batch, dtype=object)
    if batch.ndim != 1 or len(batch) != len(matrix):
        raise ValueError("group_batch must have one value per group")
    for value in batch.tolist():
        _complete_scalar(value, "batch")
    if len({type(value) for value in batch.tolist()}) != 1:
        raise TypeError("batch must use one consistent categorical type")
    batch = _validate_batch(batch, minimum_groups_per_batch)
    x_distance = full_representation_distance(
        matrix, representation_distance=representation_distance
    )
    y_distance = categorical_delta_distance(batch)
    statistic, p_value = _mgc_precomputed(
        x_distance, y_distance, reps=reps, seed=seed
    )
    return {
        "method_id": "full_representation_precomputed_mgc",
        "statistic": statistic,
        "p_value": p_value,
        "permutation_reps": reps,
    }


def _pearson_fixed_margin_statistic(batch: np.ndarray, outcome: np.ndarray) -> float:
    _, batch_codes = np.unique(batch, return_inverse=True)
    _, outcome_codes = np.unique(outcome, return_inverse=True)
    table = np.zeros((batch_codes.max() + 1, outcome_codes.max() + 1), dtype=float)
    np.add.at(table, (batch_codes, outcome_codes), 1.0)
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / table.sum()
    if np.any(expected <= 0):
        raise ValueError("fixed-margin table is not estimable")
    return float(np.sum((table - expected) ** 2 / expected))


def classification_batch_association(
    group_batch: Sequence[Any],
    group_outcome: Sequence[Any],
    *,
    reps: int,
    seed: int,
    minimum_groups_per_batch: int,
) -> dict[str, Any]:
    """Whole-group fixed-margin permutation Pearson association."""
    reps, seed = _validate_monte_carlo(reps, seed)
    batch = np.asarray(group_batch, dtype=object)
    outcome = np.asarray(group_outcome, dtype=object)
    if batch.ndim != 1 or outcome.ndim != 1 or len(batch) != len(outcome) or len(batch) < 5:
        raise ValueError("batch and outcome must be aligned group-level vectors")
    for name, values in (("batch", batch), ("outcome", outcome)):
        for value in values.tolist():
            _complete_scalar(value, name)
        if len({type(value) for value in values.tolist()}) != 1:
            raise TypeError(f"{name} must use one consistent categorical type")
        if len(set(values.tolist())) < 2:
            raise ValueError(f"{name} must contain at least two levels")
    batch = _validate_batch(batch, minimum_groups_per_batch)
    observed = _pearson_fixed_margin_statistic(batch, outcome)
    rng = np.random.default_rng(seed)
    null = np.asarray(
        [_pearson_fixed_margin_statistic(batch, rng.permutation(outcome)) for _ in range(reps)]
    )
    denominator = len(batch) * (min(len(set(batch)), len(set(outcome))) - 1)
    return {
        "method_id": "whole_group_fixed_margin_permutation_pearson",
        "statistic": observed,
        "p_value": plus_one_pvalue(observed, null),
        "effect": float(np.sqrt(observed / denominator)),
        "effect_id": "cramers_v",
        "permutation_reps": reps,
    }


def regression_batch_association(
    group_batch: Sequence[Any],
    group_outcome: Sequence[float],
    *,
    reps: int,
    seed: int,
    minimum_groups_per_batch: int,
) -> dict[str, Any]:
    """MGC between batch delta and normalized-midrank outcome distance."""
    reps, seed = _validate_monte_carlo(reps, seed)
    batch = np.asarray(group_batch, dtype=object)
    outcome = np.asarray(group_outcome)
    if batch.ndim != 1 or outcome.ndim != 1 or len(batch) != len(outcome) or len(batch) < 5:
        raise ValueError("batch and outcome must be aligned group-level vectors")
    for value in batch.tolist():
        _complete_scalar(value, "batch")
    if len({type(value) for value in batch.tolist()}) != 1:
        raise TypeError("batch must use one consistent categorical type")
    batch = _validate_batch(batch, minimum_groups_per_batch)
    outcome_distance = continuous_midrank_distance(outcome)
    statistic, p_value = _mgc_precomputed(
        categorical_delta_distance(batch), outcome_distance, reps=reps, seed=seed
    )
    return {
        "method_id": "batch_delta_outcome_midrank_precomputed_mgc",
        "statistic": statistic,
        "p_value": p_value,
        "permutation_reps": reps,
    }


def logrank_score_statistic(
    time: Sequence[float], event: Sequence[int], batch: Sequence[Any]
) -> float:
    """Reference-free k-sample log-rank score statistic for observed events."""
    time_array = np.asarray(time)
    event_array = np.asarray(event)
    batch_array = np.asarray(batch, dtype=object)
    if time_array.ndim != 1 or event_array.ndim != 1 or batch_array.ndim != 1:
        raise ValueError("survival fields must be one-dimensional")
    if not (len(time_array) == len(event_array) == len(batch_array)) or len(time_array) < 5:
        raise ValueError("survival fields must be aligned group-level vectors")
    if any(isinstance(value, (bool, np.bool_)) for value in time_array.tolist()):
        raise TypeError("survival time must be numeric, not boolean")
    try:
        time_array = time_array.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError("survival time must be numeric") from exc
    if not np.isfinite(time_array).all() or np.any(time_array <= 0):
        raise ValueError("survival time must be finite and positive")
    if any(isinstance(value, (bool, np.bool_)) for value in event_array.tolist()):
        raise TypeError("event indicator must be numeric, not boolean")
    try:
        event_array = event_array.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError("event indicator must be numeric") from exc
    if not np.isin(event_array, (0.0, 1.0)).all() or not np.any(event_array == 1):
        raise ValueError("event process has no event information")
    for value in batch_array.tolist():
        _complete_scalar(value, "batch")
    levels, batch_codes = np.unique(batch_array, return_inverse=True)
    if len(levels) < 2:
        raise ValueError("batch must contain at least two levels")
    observed = np.zeros(len(levels), dtype=float)
    expected = np.zeros(len(levels), dtype=float)
    covariance = np.zeros((len(levels), len(levels)), dtype=float)
    informative_risk_sets = 0
    for event_time in np.unique(time_array[event_array == 1]):
        at_risk = time_array >= event_time
        events = (time_array == event_time) & (event_array == 1)
        risk_counts = np.bincount(batch_codes[at_risk], minlength=len(levels)).astype(float)
        event_counts = np.bincount(batch_codes[events], minlength=len(levels)).astype(float)
        risk_total = risk_counts.sum()
        event_total = event_counts.sum()
        if risk_total <= 1 or event_total <= 0:
            continue
        probabilities = risk_counts / risk_total
        observed += event_counts
        expected += event_total * probabilities
        factor = event_total * (risk_total - event_total) / (risk_total - 1.0)
        covariance += factor * (np.diag(probabilities) - np.outer(probabilities, probabilities))
        if np.count_nonzero(risk_counts) >= 2 and factor > 0:
            informative_risk_sets += 1
    contrast = observed - expected
    rank = int(np.linalg.matrix_rank(covariance, tol=1e-12))
    if informative_risk_sets == 0 or rank == 0:
        raise ValueError("event process has no cross-batch risk-set information")
    statistic = float(contrast @ np.linalg.pinv(covariance, rcond=1e-12) @ contrast)
    if not np.isfinite(statistic) or statistic < -1e-12:
        raise ValueError("log-rank statistic is not estimable")
    return max(0.0, statistic)


def _permutation_logrank(
    time: np.ndarray,
    event: np.ndarray,
    batch: np.ndarray,
    *,
    reps: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    observed = logrank_score_statistic(time, event, batch)
    null_values: list[float] = []
    for _ in range(reps):
        try:
            null_values.append(logrank_score_statistic(time, event, rng.permutation(batch)))
        except ValueError as exc:
            if str(exc) != "event process has no cross-batch risk-set information":
                raise
            null_values.append(0.0)
    null = np.asarray(null_values)
    return observed, plus_one_pvalue(observed, null)


def survival_batch_association(
    group_batch: Sequence[Any],
    group_time: Sequence[float],
    group_event: Sequence[int],
    *,
    reps: int,
    seed: int,
    minimum_groups_per_batch: int,
) -> dict[str, Any]:
    """Separate observed event and censoring-process batch associations."""
    reps, seed = _validate_monte_carlo(reps, seed)
    batch = np.asarray(group_batch, dtype=object)
    time = np.asarray(group_time)
    event = np.asarray(group_event)
    if batch.ndim != 1 or time.ndim != 1 or event.ndim != 1 or not (
        len(batch) == len(time) == len(event)
    ):
        raise ValueError("batch, time, and event must be aligned group-level vectors")
    batch = _validate_batch(batch, minimum_groups_per_batch)
    rng = np.random.default_rng(seed)
    event_statistic, event_p = _permutation_logrank(
        time, event, batch, reps=reps, rng=rng
    )
    result: dict[str, Any] = {
        "event": {
            "status": "tested",
            "method_id": "k_sample_logrank_observed_event_process",
            "statistic": event_statistic,
            "p_value": event_p,
            "permutation_reps": reps,
            "scope": "registered_proportional_hazards_alternative",
        },
        "censoring": {
            "status": "not_applicable",
            "reason": "no_observed_censoring_events",
            "method_id": "k_sample_logrank_observed_censoring_process",
        },
    }
    event_numeric = np.asarray(event, dtype=float)
    if np.any(event_numeric == 0):
        censor_statistic, censor_p = _permutation_logrank(
            np.asarray(time), 1.0 - event_numeric, batch, reps=reps, rng=rng
        )
        result["censoring"] = {
            "status": "tested",
            "method_id": "k_sample_logrank_observed_censoring_process",
            "statistic": censor_statistic,
            "p_value": censor_p,
            "permutation_reps": reps,
            "scope": "observed_censoring_process_association_only",
        }
    return result


def apply_primary_holm(
    primary_components: Mapping[str, Mapping[str, Any]], *, alpha: float
) -> dict[str, dict[str, Any]]:
    """Apply one Holm family to exactly the supplied warning-producing components."""
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, (int, float, np.number)):
        raise TypeError("alpha must be numeric")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    names: list[str] = []
    p_values: list[float] = []
    statuses: dict[str, str] = {}
    for name, component in primary_components.items():
        if type(name) is not str or name not in _PRIMARY_COMPONENT_IDS:
            raise ValueError("primary component name must be a registered public token")
        status = component.get("status", "tested")
        if not isinstance(status, str) or status not in _PRIMARY_COMPONENT_STATUSES:
            raise ValueError("primary component status must be a registered public token")
        statuses[name] = status
        if status != "tested":
            continue
        p_value = component.get("p_value")
        if isinstance(p_value, (bool, np.bool_)):
            raise TypeError("primary p-values must be numeric")
        try:
            p_value = float(p_value)
        except (TypeError, ValueError) as exc:
            raise TypeError("primary p-values must be numeric") from exc
        if not np.isfinite(p_value) or not 0 <= p_value <= 1:
            raise ValueError("primary p-values must lie in [0, 1]")
        names.append(name)
        p_values.append(p_value)
    if not p_values:
        raise ValueError("at least one primary component must be tested")
    adjusted = holm_adjust(p_values)
    output: dict[str, dict[str, Any]] = {}
    tested = dict(zip(names, adjusted, strict=True))
    for name, component in primary_components.items():
        if name in tested:
            output[name] = {
                "status": "tested",
                "raw_p_value": float(component["p_value"]),
                "holm_p_value": float(tested[name]),
                "warning": bool(tested[name] <= alpha),
            }
        else:
            output[name] = {
                "status": statuses[name],
                "warning": False,
            }
    return output


def group_batch_diagnostics(
    row_representation: Any,
    group_ids: Sequence[Any],
    batch: Sequence[Any],
    endpoint: Sequence[Any],
    *,
    endpoint_kind: EndpointKind,
    group_reducer: Reducer,
    standardization: Standardization,
    representation_distance: RepresentationDistance,
    reps: int,
    seed: int,
    minimum_groups_per_batch: int,
    alpha: float,
    event: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build one aggregate, public-safe C05 development result for one modality."""
    group_representation = collapse_group_representation(
        row_representation, group_ids, group_reducer=group_reducer
    )
    group_batch = collapse_pure_group_values(
        batch, group_ids, value_kind="categorical"
    )
    structure = structure_batch_association(
        group_representation,
        group_batch,
        standardization=standardization,
        representation_distance=representation_distance,
        reps=reps,
        seed=seed,
        minimum_groups_per_batch=minimum_groups_per_batch,
    )
    components: dict[str, dict[str, Any]] = {
        "structure": {"status": "tested", **structure}
    }
    if endpoint_kind == "classification":
        group_endpoint = collapse_pure_group_values(
            endpoint, group_ids, value_kind="categorical"
        )
        components["outcome"] = {
            "status": "tested",
            **classification_batch_association(
                group_batch,
                group_endpoint,
                reps=reps,
                seed=seed,
                minimum_groups_per_batch=minimum_groups_per_batch,
            ),
        }
    elif endpoint_kind == "regression":
        group_endpoint = collapse_pure_group_values(
            endpoint, group_ids, value_kind="continuous"
        )
        components["outcome"] = {
            "status": "tested",
            **regression_batch_association(
                group_batch,
                group_endpoint,
                reps=reps,
                seed=seed,
                minimum_groups_per_batch=minimum_groups_per_batch,
            ),
        }
    elif endpoint_kind == "survival":
        if event is None:
            raise ValueError("survival endpoint requires an event indicator")
        group_time = collapse_pure_group_values(
            endpoint, group_ids, value_kind="continuous"
        )
        group_event = collapse_pure_group_values(
            event, group_ids, value_kind="binary_event"
        )
        survival = survival_batch_association(
            group_batch,
            group_time,
            group_event,
            reps=reps,
            seed=seed,
            minimum_groups_per_batch=minimum_groups_per_batch,
        )
        components["event"] = survival["event"]
        components["censoring"] = survival["censoring"]
    else:
        raise ValueError("endpoint_kind must be classification, regression, or survival")
    decisions = apply_primary_holm(components, alpha=alpha)
    method_ids = sorted(
        str(component["method_id"])
        for component in components.values()
        if component.get("status", "tested") == "tested"
    )
    effect_summaries = {
        name: {
            key: value
            for key, value in component.items()
            if key in {"statistic", "effect", "effect_id", "scope"}
        }
        for name, component in components.items()
    }
    return {
        "status": "development_only_not_production_wired",
        "claim_id": "C05",
        "group_count": int(len(group_representation)),
        "modality_count": 1,
        "component_decisions": decisions,
        "effect_summaries": effect_summaries,
        "eligibility_reasons": [],
        "method_ids": method_ids,
        "multiplicity_family_id": "c05_primary_all_warning_components_holm",
        "oracle_status": "not_run_development_scaffold",
        "permutation_registry_sha256": None,
        "permutation_registry_status": "unavailable_pending_frozen_registry",
    }
