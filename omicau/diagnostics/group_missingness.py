"""Development-only group-level missingness association primitives.

This module is intentionally not wired into the production diagnostics. It
requires explicit permutation counts and seeds so that calibration and protocol
freezing can determine them prospectively.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Literal

import numpy as np
from scipy import stats


EndpointKind = Literal["categorical", "continuous"]
LabelName = Literal["endpoint", "batch"]


def _integer_parameter(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _complete_scalar(value: Any, name: str) -> None:
    if not isinstance(value, Hashable):
        raise TypeError(f"{name} values must be hashable scalars")
    missing = np.asarray(value is None or _safe_isna(value))
    if missing.ndim != 0 or bool(missing):
        raise ValueError(f"{name} must not contain missing values")


def _safe_isna(value: Any) -> bool:
    try:
        result = np.asarray(value != value)
    except (TypeError, ValueError):
        return False
    if result.ndim != 0:
        return False
    try:
        return bool(result)
    except TypeError:
        return True


def _validate_categorical_types(values: Sequence[Any], name: str) -> None:
    types = {type(value) for value in values}
    if len(types) > 1:
        raise TypeError(f"{name} categorical values must have one consistent type")


def _validate_label_name(name: Any) -> LabelName:
    if not isinstance(name, str):
        raise TypeError("name must be a supported label role")
    if name not in ("endpoint", "batch"):
        raise ValueError("name must be 'endpoint' or 'batch'")
    return name


def _group_codes(group_ids: Sequence[Any], n_rows: int) -> tuple[np.ndarray, int]:
    groups = np.asarray(group_ids, dtype=object)
    if groups.ndim != 1 or len(groups) != n_rows:
        raise ValueError("group_ids must be one-dimensional with one value per row")
    mapping: dict[Any, int] = {}
    codes = np.empty(n_rows, dtype=int)
    for row, group in enumerate(groups.tolist()):
        _complete_scalar(group, "group_ids")
        if isinstance(group, (bool, np.bool_)):
            raise TypeError("boolean group identifiers are not supported")
        if group not in mapping:
            mapping[group] = len(mapping)
        codes[row] = mapping[group]
    if len(mapping) < 3:
        raise ValueError("at least three groups are required")
    return codes, len(mapping)


def _binary_missingness(row_missingness: Any) -> np.ndarray:
    values = np.asarray(row_missingness)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("row_missingness must be a non-empty two-dimensional matrix")
    if values.dtype.kind not in "biuf":
        raise TypeError("row_missingness must be boolean or numeric")
    numeric = values.astype(float, copy=False)
    if not np.isfinite(numeric).all():
        raise ValueError("row_missingness must be finite")
    if not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError("row_missingness must contain only binary values")
    return numeric


def _collapse_group_missingness_with_codes(
    row_missingness: Any,
    group_ids: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    missingness = _binary_missingness(row_missingness)
    codes, n_groups = _group_codes(group_ids, missingness.shape[0])
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    collapsed = np.zeros((n_groups, missingness.shape[1]), dtype=float)
    np.add.at(collapsed, codes, missingness)
    collapsed /= counts[:, None]
    return collapsed, codes


def collapse_group_missingness(
    row_missingness: Any,
    group_ids: Sequence[Any],
) -> np.ndarray:
    """Return equal-weight group-feature missingness proportions."""
    collapsed, _ = _collapse_group_missingness_with_codes(row_missingness, group_ids)
    return collapsed


def collapse_pure_group_labels(
    row_labels: Sequence[Any],
    group_ids: Sequence[Any],
    *,
    name: LabelName,
    kind: EndpointKind,
) -> np.ndarray:
    """Collapse row labels after verifying one complete label per group."""
    name = _validate_label_name(name)
    labels = np.asarray(row_labels, dtype=object)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional with one value per row")
    group_codes, _ = _group_codes(group_ids, len(labels))
    return _collapse_pure_group_labels(labels, group_codes, name=name, kind=kind)


def _collapse_pure_group_labels(
    labels: np.ndarray,
    group_codes: np.ndarray,
    *,
    name: LabelName,
    kind: EndpointKind,
) -> np.ndarray:
    name = _validate_label_name(name)
    if labels.ndim != 1 or len(labels) != len(group_codes):
        raise ValueError(f"{name} must be one-dimensional with one value per row")
    if kind not in ("categorical", "continuous"):
        raise ValueError("kind must be 'categorical' or 'continuous'")
    n_groups = int(group_codes.max()) + 1
    collapsed: list[Any] = []
    for group in range(n_groups):
        values = labels[group_codes == group].tolist()
        for value in values:
            _complete_scalar(value, name)
        first = values[0]
        if any(value != first for value in values[1:]):
            raise ValueError(f"{name} must be constant within each group")
        collapsed.append(first)
    if kind == "continuous":
        if any(isinstance(value, (bool, np.bool_)) for value in collapsed):
            raise TypeError(f"{name} must be numeric, not boolean")
        try:
            result = np.asarray(collapsed, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be numeric for a continuous endpoint") from exc
        if not np.isfinite(result).all():
            raise ValueError(f"{name} must be finite")
    else:
        _validate_categorical_types(collapsed, name)
        result = np.asarray(collapsed, dtype=object)
    if len({(type(value), value) for value in result.tolist()}) < 2:
        raise ValueError(f"{name} must contain at least two group-level values")
    return result


def _distance_checks(values: Any, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have at least three rows and one column")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    return matrix


def normalized_l1_distance(group_missingness: Any) -> np.ndarray:
    """Pairwise mean absolute feature difference between group profiles."""
    matrix = _distance_checks(group_missingness, "group_missingness")
    if ((matrix < 0) | (matrix > 1)).any():
        raise ValueError("group_missingness proportions must lie in [0, 1]")
    distance = np.abs(matrix[:, None, :] - matrix[None, :, :]).mean(axis=2)
    np.fill_diagonal(distance, 0.0)
    return distance


def categorical_delta_distance(labels: Sequence[Any]) -> np.ndarray:
    values = np.asarray(labels, dtype=object)
    if values.ndim != 1 or len(values) < 3:
        raise ValueError("labels must contain at least three group-level values")
    for value in values.tolist():
        _complete_scalar(value, "labels")
    _validate_categorical_types(values.tolist(), "labels")
    distance = (values[:, None] != values[None, :]).astype(float)
    np.fill_diagonal(distance, 0.0)
    return distance


def continuous_midrank_distance(values: Sequence[float]) -> np.ndarray:
    numeric = np.asarray(values)
    if numeric.ndim != 1 or len(numeric) < 3:
        raise ValueError("values must contain at least three group-level values")
    if numeric.dtype.kind == "b" or any(
        isinstance(value, (bool, np.bool_)) for value in numeric.tolist()
    ):
        raise TypeError("continuous values must be numeric, not boolean")
    try:
        numeric = numeric.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError("continuous values must be numeric") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("continuous values must be finite")
    ranks = stats.rankdata(numeric, method="average")
    scaled = (ranks - 1.0) / (len(ranks) - 1.0)
    distance = np.abs(scaled[:, None] - scaled[None, :])
    np.fill_diagonal(distance, 0.0)
    return distance


def _validate_distance(distance: Any, name: str) -> np.ndarray:
    matrix = np.asarray(distance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 5 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square distance matrix for at least five groups")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    if (matrix < 0).any() or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
        raise ValueError(f"{name} must be non-negative and symmetric")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0, atol=1e-12):
        raise ValueError(f"{name} must have a zero diagonal")
    if not np.any(matrix > 0):
        raise ValueError(f"{name} is degenerate")
    return matrix


def mgc_precomputed(
    x_distance: Any,
    y_distance: Any,
    *,
    reps: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Run MGC on validated precomputed distances with one worker."""
    reps = _integer_parameter(reps, "reps", minimum=1)
    seed = _integer_parameter(seed, "seed")
    if seed > np.iinfo(np.uint32).max:
        raise ValueError("seed must fit in an unsigned 32-bit integer")
    x = _validate_distance(x_distance, "x_distance")
    y = _validate_distance(y_distance, "y_distance")
    if x.shape != y.shape:
        raise ValueError("distance matrices must have identical shapes")
    result = stats.multiscale_graphcorr(
        x,
        y,
        compute_distance=None,
        reps=reps,
        workers=1,
        random_state=seed,
    )
    return {
        "test": "multiscale_graphcorr",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "reps": reps,
        "workers": 1,
    }


def plus_one_pvalue(observed: float, permuted: Sequence[float]) -> float:
    observed = float(observed)
    null = np.asarray(permuted, dtype=float)
    if not np.isfinite(observed) or null.ndim != 1 or len(null) == 0 or not np.isfinite(null).all():
        raise ValueError("observed and permuted statistics must be finite and non-empty")
    return float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))


def _kruskal_statistic(values: np.ndarray, labels: np.ndarray) -> float:
    classes: list[Any] = []
    for label in labels.tolist():
        if label not in classes:
            classes.append(label)
    samples = [values[labels == label] for label in classes]
    if np.ptp(values) == 0:
        return 0.0
    return float(stats.kruskal(*samples).statistic)


def _absolute_spearman(values: np.ndarray, endpoint: np.ndarray) -> float:
    if np.ptp(values) == 0 or np.ptp(endpoint) == 0:
        return 0.0
    statistic = float(stats.spearmanr(values, endpoint).statistic)
    return abs(statistic) if np.isfinite(statistic) else 0.0


def secondary_feature_statistics(
    group_missingness: Any,
    group_endpoint: Sequence[Any],
    *,
    endpoint_kind: EndpointKind,
    reps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return featurewise statistics and whole-group permutation p-values."""
    matrix = _distance_checks(group_missingness, "group_missingness")
    if ((matrix < 0) | (matrix > 1)).any():
        raise ValueError("group_missingness proportions must lie in [0, 1]")
    reps = _integer_parameter(reps, "reps", minimum=1)
    seed = _integer_parameter(seed, "seed")
    endpoint = np.asarray(group_endpoint, dtype=object)
    if endpoint.ndim != 1 or len(endpoint) != matrix.shape[0]:
        raise ValueError("group_endpoint must have one value per group")
    for value in endpoint.tolist():
        _complete_scalar(value, "group_endpoint")
    if endpoint_kind == "continuous":
        if any(isinstance(value, (bool, np.bool_)) for value in endpoint.tolist()):
            raise TypeError("continuous endpoint values must be numeric, not boolean")
        try:
            endpoint = endpoint.astype(float)
        except (TypeError, ValueError) as exc:
            raise TypeError("continuous endpoint values must be numeric") from exc
        statistic_function = _absolute_spearman
    elif endpoint_kind == "categorical":
        _validate_categorical_types(endpoint.tolist(), "group_endpoint")
        statistic_function = _kruskal_statistic
    else:
        raise ValueError("endpoint_kind must be 'categorical' or 'continuous'")
    if len({(type(value), value) for value in endpoint.tolist()}) < 2:
        raise ValueError("group_endpoint must contain at least two values")
    rng = np.random.default_rng(seed)
    permutations = np.stack([rng.permutation(len(endpoint)) for _ in range(reps)])
    statistics = np.empty(matrix.shape[1], dtype=float)
    p_values = np.empty(matrix.shape[1], dtype=float)
    for feature, values in enumerate(matrix.T):
        observed = statistic_function(values, endpoint)
        null = [statistic_function(values, endpoint[index]) for index in permutations]
        statistics[feature] = observed
        p_values[feature] = plus_one_pvalue(observed, null)
    return statistics, p_values


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = _valid_pvalues(p_values)
    order = np.argsort(p, kind="stable")
    ranked = p[order] * np.arange(len(p), 0, -1)
    ranked = np.maximum.accumulate(ranked)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def benjamini_yekutieli_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = _valid_pvalues(p_values)
    order = np.argsort(p, kind="stable")
    harmonic = np.sum(1.0 / np.arange(1, len(p) + 1))
    ranked = p[order] * len(p) * harmonic / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def _valid_pvalues(p_values: Sequence[float]) -> np.ndarray:
    if any(isinstance(value, (bool, np.bool_)) for value in p_values):
        raise TypeError("p_values must be numeric, not boolean")
    p = np.asarray(p_values)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("p_values must be a non-empty numeric vector")
    try:
        p = p.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError("p_values must be numeric") from exc
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p_values must be finite and lie in [0, 1]")
    return p


def _association_summary(
    group_missingness: np.ndarray,
    group_endpoint: np.ndarray,
    *,
    endpoint_kind: EndpointKind,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    missing_distance = normalized_l1_distance(group_missingness)
    endpoint_distance = (
        categorical_delta_distance(group_endpoint)
        if endpoint_kind == "categorical"
        else continuous_midrank_distance(group_endpoint)
    )
    primary = mgc_precomputed(missing_distance, endpoint_distance, reps=reps, seed=seed)
    aggregate_statistics, aggregate_p_values = secondary_feature_statistics(
        group_missingness.mean(axis=1, keepdims=True),
        group_endpoint,
        endpoint_kind=endpoint_kind,
        reps=reps,
        seed=seed,
    )
    feature_statistics, feature_p_values = secondary_feature_statistics(
        group_missingness,
        group_endpoint,
        endpoint_kind=endpoint_kind,
        reps=reps,
        seed=seed,
    )
    return {
        "kind": endpoint_kind,
        "primary": primary,
        "secondary": {
            "aggregate_burden": {
                "test": (
                    "permutation_kruskal_wallis"
                    if endpoint_kind == "categorical"
                    else "permutation_absolute_spearman"
                ),
                "statistic": float(aggregate_statistics[0]),
                "p_value": float(aggregate_p_values[0]),
                "reps": reps,
            },
            "per_feature": {
                "test": (
                    "permutation_kruskal_wallis"
                    if endpoint_kind == "categorical"
                    else "permutation_absolute_spearman"
                ),
                "n_features_tested": int(len(feature_statistics)),
                "maximum_statistic": float(feature_statistics.max()),
                "minimum_raw_p_value": float(feature_p_values.min()),
                "reps": reps,
            },
        },
    }


def group_missingness_diagnostics(
    row_missingness: Any,
    group_ids: Sequence[Any],
    endpoint: Sequence[Any],
    *,
    endpoint_kind: EndpointKind,
    reps: int,
    seed: int,
    batch: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Build a fixed public-safe development summary without row-level fields."""
    reps = _integer_parameter(reps, "reps", minimum=1)
    seed = _integer_parameter(seed, "seed")
    group_missingness, codes = _collapse_group_missingness_with_codes(row_missingness, group_ids)
    group_endpoint = _collapse_pure_group_labels(
        np.asarray(endpoint, dtype=object),
        codes,
        name="endpoint",
        kind=endpoint_kind,
    )
    group_batch = None
    if batch is not None:
        group_batch = _collapse_pure_group_labels(
            np.asarray(batch, dtype=object), codes, name="batch", kind="categorical"
        )
    output: dict[str, Any] = {
        "status": "development_only_not_production_wired",
        "n_rows": int(len(codes)),
        "n_groups": int(group_missingness.shape[0]),
        "n_features": int(group_missingness.shape[1]),
        "permutation_reps": reps,
        "permutation_registry_sha256": None,
        "permutation_registry_status": "unavailable_pending_frozen_registry",
        "aggregate_missing_fraction": float(group_missingness.mean()),
        "maximum_group_feature_missing_fraction": float(group_missingness.max()),
        "endpoint_association": _association_summary(
            group_missingness,
            group_endpoint,
            endpoint_kind=endpoint_kind,
            reps=reps,
            seed=seed,
        ),
        "batch_association": None,
    }
    if group_batch is not None:
        output["batch_association"] = _association_summary(
            group_missingness,
            group_batch,
            endpoint_kind="categorical",
            reps=reps,
            seed=seed,
        )
    return output
