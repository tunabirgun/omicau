"""Fail-closed validation for exact nested group-aware split manifests."""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping, Sequence
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any

import numpy as np


_PARTITION_EVIDENCE_TOKEN = object()
_SPLIT_MANIFEST_STATUS = "unavailable_pending_frozen_public_manifest"


class SplitValidationError(ValueError):
    """Public-safe C06 split-validation failure."""

    _ALLOWED_CODES = {
        "c06_group_id_missing",
        "c06_group_outcome_mixed",
        "c06_metric_support_insufficient",
        "c06_split_manifest_invalid",
    }

    def __init__(self, code: str, invariant: str):
        if code not in self._ALLOWED_CODES:
            raise ValueError("unsupported_c06_refusal_code")
        self.code = code
        self.invariant = invariant
        super().__init__(code)


class SplitPartitionEvidence:
    """Process-local partition evidence bound to one validated split plan."""

    __slots__ = (
        "__inner",
        "__inner_index_digests",
        "__locked",
        "__outer",
        "__outer_index_digests",
        "__split_digest",
        "__token",
    )

    def __init__(
        self,
        outer: Sequence[tuple[frozenset[Hashable], frozenset[Hashable]]],
        inner: Sequence[
            Sequence[tuple[frozenset[Hashable], frozenset[Hashable]]]
        ],
        outer_index_digests: Sequence[tuple[str, str]],
        inner_index_digests: Sequence[Sequence[tuple[str, str]]],
        split_digest: str,
        *,
        _token: object,
    ) -> None:
        if _token is not _PARTITION_EVIDENCE_TOKEN:
            raise TypeError("split_partition_evidence_requires_validation")
        object.__setattr__(self, "_SplitPartitionEvidence__outer", tuple(outer))
        object.__setattr__(
            self,
            "_SplitPartitionEvidence__inner",
            tuple(tuple(folds) for folds in inner),
        )
        object.__setattr__(
            self,
            "_SplitPartitionEvidence__outer_index_digests",
            tuple(outer_index_digests),
        )
        object.__setattr__(
            self,
            "_SplitPartitionEvidence__inner_index_digests",
            tuple(tuple(folds) for folds in inner_index_digests),
        )
        object.__setattr__(self, "_SplitPartitionEvidence__split_digest", split_digest)
        object.__setattr__(self, "_SplitPartitionEvidence__token", _token)
        object.__setattr__(self, "_SplitPartitionEvidence__locked", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_SplitPartitionEvidence__locked", False):
            raise AttributeError("split_partition_evidence_immutable")
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return "SplitPartitionEvidence(trusted_process_development_mechanics=True)"

    def __copy__(self) -> None:
        raise TypeError("split_partition_evidence_copy_forbidden")

    def __deepcopy__(self, memo: Any) -> None:
        raise TypeError("split_partition_evidence_copy_forbidden")

    def __reduce_ex__(self, protocol: int) -> None:
        raise TypeError("split_partition_evidence_serialization_forbidden")

    def __getstate__(self) -> None:
        raise TypeError("split_partition_evidence_serialization_forbidden")


def _partition_evidence_values(
    evidence: SplitPartitionEvidence,
) -> tuple[
    str,
    tuple[tuple[frozenset[Hashable], frozenset[Hashable]], ...],
    tuple[
        tuple[tuple[frozenset[Hashable], frozenset[Hashable]], ...], ...
    ],
    tuple[tuple[str, str], ...],
    tuple[tuple[tuple[str, str], ...], ...],
]:
    """Return process-local partition values to the fit-trace verifier."""
    if (
        type(evidence) is not SplitPartitionEvidence
        or evidence._SplitPartitionEvidence__token is not _PARTITION_EVIDENCE_TOKEN
    ):
        raise TypeError("split_partition_evidence_invalid")
    return (
        evidence._SplitPartitionEvidence__split_digest,
        evidence._SplitPartitionEvidence__outer,
        evidence._SplitPartitionEvidence__inner,
        evidence._SplitPartitionEvidence__outer_index_digests,
        evidence._SplitPartitionEvidence__inner_index_digests,
    )


def _build_validated_split_plan_type() -> tuple[type, Any]:
    from weakref import WeakKeyDictionary

    # This process-local indirection reduces accidental disclosure only. It does
    # not authenticate values against arbitrary same-process Python introspection.
    core_by_handle: WeakKeyDictionary[Any, tuple[Any, ...]] = WeakKeyDictionary()

    def resolve(handle: Any) -> tuple[Any, ...]:
        try:
            return core_by_handle[handle]
        except KeyError:
            raise TypeError("validated_split_plan_handle_invalid") from None

    class OpaqueValidatedSplitPlan:
        """Private trusted-process state; only receipt() is public-safe."""

        __slots__ = ("__weakref__",)

        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            raise TypeError("validated_split_plan_requires_validation")

        def __setattr__(self, name: str, value: Any) -> None:
            raise AttributeError("validated_split_plan_immutable")

        def __iter__(self) -> None:
            raise TypeError("validated_split_plan_private_access_forbidden")

        def __getitem__(self, key: Any) -> None:
            raise TypeError("validated_split_plan_private_access_forbidden")

        def __len__(self) -> int:
            raise TypeError("validated_split_plan_private_access_forbidden")

        def __repr__(self) -> str:
            resolve(self)
            return "ValidatedSplitPlan(trusted_process_development_mechanics=True)"

        def __getstate__(self) -> None:
            raise TypeError("validated_split_plan_private_serialization_forbidden")

        def __copy__(self) -> None:
            raise TypeError("validated_split_plan_private_copy_forbidden")

        def __deepcopy__(self, memo: Any) -> None:
            raise TypeError("validated_split_plan_private_copy_forbidden")

        def __reduce_ex__(self, protocol: int) -> None:
            raise TypeError("validated_split_plan_private_serialization_forbidden")

        def _private_outer_splits(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
            """Yield detached outer splits for trusted-process integration."""
            outer = resolve(self)[0]
            for train, assessment in outer:
                train_array = np.asarray(train, dtype=np.int64)
                assessment_array = np.asarray(assessment, dtype=np.int64)
                train_array.flags.writeable = False
                assessment_array.flags.writeable = False
                yield train_array, assessment_array

        def _private_inner_splits(
            self, outer_fold: int
        ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
            """Yield detached inner splits for trusted-process integration."""
            if isinstance(outer_fold, (bool, np.bool_)) or not isinstance(
                outer_fold, Integral
            ):
                raise TypeError("outer_fold_type")
            fold = int(outer_fold)
            inner = resolve(self)[1]
            if fold < 0 or fold >= len(inner):
                raise IndexError("outer_fold_range")
            for train, assessment in inner[fold]:
                train_array = np.asarray(train, dtype=np.int64)
                assessment_array = np.asarray(assessment, dtype=np.int64)
                train_array.flags.writeable = False
                assessment_array.flags.writeable = False
                yield train_array, assessment_array

        def receipt(self) -> dict[str, Any]:
            """Return the sole public-safe, detached aggregate receipt."""
            return json.loads(resolve(self)[4])

        def _private_partition_evidence(self) -> SplitPartitionEvidence:
            """Return partition evidence for trusted-process verification."""
            outer, inner, sample_groups, split_digest, _ = resolve(self)
            manifest = {
                "outer_folds": [
                    {
                        "assessment": list(assessment),
                        "inner_folds": [
                            {
                                "assessment": list(child_assessment),
                                "train": list(child_train),
                            }
                            for child_train, child_assessment in inner[index]
                        ],
                        "train": list(train),
                    }
                    for index, (train, assessment) in enumerate(outer)
                ]
            }
            if canonical_split_manifest_sha256(manifest) != split_digest:
                raise TypeError("validated_split_plan_manifest_binding_invalid")
            outer_groups = tuple(
                (
                    frozenset(sample_groups[index] for index in train),
                    frozenset(sample_groups[index] for index in assessment),
                )
                for train, assessment in outer
            )
            inner_groups = tuple(
                tuple(
                    (
                        frozenset(sample_groups[index] for index in train),
                        frozenset(sample_groups[index] for index in assessment),
                    )
                    for train, assessment in folds
                )
                for folds in inner
            )
            return SplitPartitionEvidence(
                outer_groups,
                inner_groups,
                tuple(
                    (
                        _canonical_index_sha256(train),
                        _canonical_index_sha256(assessment),
                    )
                    for train, assessment in outer
                ),
                tuple(
                    tuple(
                        (
                            _canonical_index_sha256(train),
                            _canonical_index_sha256(assessment),
                        )
                        for train, assessment in folds
                    )
                    for folds in inner
                ),
                split_digest,
                _token=_PARTITION_EVIDENCE_TOKEN,
            )

    OpaqueValidatedSplitPlan.__name__ = "ValidatedSplitPlan"
    OpaqueValidatedSplitPlan.__qualname__ = "ValidatedSplitPlan"

    def construct(
        outer: Sequence[tuple[np.ndarray, np.ndarray]],
        inner: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
        sample_groups: Sequence[Hashable],
        split_digest: str,
        receipt: Mapping[str, Any],
    ) -> Any:
        def frozen_pair(
            train: np.ndarray, assessment: np.ndarray
        ) -> tuple[tuple[int, ...], tuple[int, ...]]:
            return (
                tuple(int(value) for value in np.asarray(train, dtype=np.int64)),
                tuple(int(value) for value in np.asarray(assessment, dtype=np.int64)),
            )

        frozen_outer = tuple(frozen_pair(train, assessment) for train, assessment in outer)
        frozen_inner = tuple(
            tuple(frozen_pair(train, assessment) for train, assessment in folds)
            for folds in inner
        )
        receipt_json = json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        handle = object.__new__(OpaqueValidatedSplitPlan)
        core_by_handle[handle] = (
            frozen_outer,
            frozen_inner,
            tuple(sample_groups),
            split_digest,
            receipt_json,
        )
        return handle

    return OpaqueValidatedSplitPlan, construct


ValidatedSplitPlan, _validated_plan_constructor = _build_validated_split_plan_type()
del _build_validated_split_plan_type


def _fail(invariant: str, code: str = "c06_split_manifest_invalid") -> None:
    raise SplitValidationError(code, invariant)


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        _fail(f"{name}_type")
    result = int(value)
    if result < minimum:
        _fail(f"{name}_range")
    return result


def _real(value: Any, name: str, minimum: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        _fail(f"{name}_type")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        _fail(f"{name}_range")
    return result


def _missing(value: Any) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return True


def _vector(values: Sequence[Any], name: str, n_samples: int) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) != n_samples:
        _fail(f"{name}_shape")
    return array


def _index_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list):
        _fail(f"{name}_type")
    result: list[int] = []
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral):
            _fail(f"{name}_integer")
        result.append(int(item))
    if len(result) != len(set(result)):
        _fail(f"{name}_unique")
    return result


def _normalized_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, Mapping) or set(manifest) != {"outer_folds"}:
        _fail("manifest_schema")
    outer = manifest["outer_folds"]
    if not isinstance(outer, list):
        _fail("outer_folds_type")
    normalized_outer = []
    for fold in outer:
        if not isinstance(fold, Mapping) or set(fold) != {"train", "assessment", "inner_folds"}:
            _fail("outer_fold_schema")
        inner = fold["inner_folds"]
        if not isinstance(inner, list):
            _fail("inner_folds_type")
        normalized_inner = []
        for child in inner:
            if not isinstance(child, Mapping) or set(child) != {"train", "assessment"}:
                _fail("inner_fold_schema")
            normalized_inner.append({
                "assessment": sorted(_index_list(child["assessment"], "inner_assessment")),
                "train": sorted(_index_list(child["train"], "inner_train")),
            })
        normalized_outer.append({
            "assessment": sorted(_index_list(fold["assessment"], "outer_assessment")),
            "inner_folds": normalized_inner,
            "train": sorted(_index_list(fold["train"], "outer_train")),
        })
    return {"outer_folds": normalized_outer}


def canonical_split_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash the strict manifest schema using canonical JSON without exposing it."""
    normalized = _normalized_manifest(manifest)
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _canonical_index_sha256(indices: Sequence[int] | np.ndarray) -> str:
    """Digest one ordered private row-index partition under a fixed schema."""
    values = np.asarray(indices)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise TypeError("private_index_partition_schema")
    normalized = [int(value) for value in values]
    if len(normalized) != len(set(normalized)) or any(value < 0 for value in normalized):
        raise ValueError("private_index_partition_invalid")
    payload = json.dumps(
        {
            "indices": sorted(normalized),
            "schema_version": "c06_private_row_index_partition_v1",
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _partition(
    train: list[int], assessment: list[int], parent: set[int], n_samples: int, level: str
) -> tuple[set[int], set[int]]:
    train_set = set(train)
    assessment_set = set(assessment)
    if not train_set or not assessment_set:
        _fail(f"{level}_nonempty")
    if any(index < 0 or index >= n_samples for index in train_set | assessment_set):
        _fail(f"{level}_index_range")
    if not train_set <= parent or not assessment_set <= parent:
        _fail(f"{level}_parent_containment")
    if train_set & assessment_set:
        _fail(f"{level}_index_disjoint")
    if train_set | assessment_set != parent:
        _fail(f"{level}_parent_coverage")
    return train_set, assessment_set


def _group_set(groups: np.ndarray, indices: set[int]) -> set[Hashable]:
    return {groups[index] for index in indices}


def _check_group_disjoint(
    groups: np.ndarray, train: set[int], assessment: set[int], level: str
) -> None:
    if _group_set(groups, train) & _group_set(groups, assessment):
        _fail(f"{level}_group_disjoint")


def _group_values(groups: np.ndarray, values: np.ndarray, indices: set[int]) -> dict[Hashable, list[Any]]:
    result: dict[Hashable, list[Any]] = {}
    for index in indices:
        result.setdefault(groups[index], []).append(values[index])
    return result


def _classification_support(
    groups: np.ndarray,
    labels: np.ndarray,
    train: set[int],
    assessment: set[int],
    classes: set[Hashable],
    minimum_train: int,
    minimum_assessment: int,
) -> tuple[int, int]:
    train_labels = {group: values[0] for group, values in _group_values(groups, labels, train).items()}
    assessment_labels = {
        group: values[0] for group, values in _group_values(groups, labels, assessment).items()
    }
    train_counts = {label: sum(value == label for value in train_labels.values()) for label in classes}
    assessment_counts = {
        label: sum(value == label for value in assessment_labels.values()) for label in classes
    }
    train_min = min(train_counts.values())
    assessment_min = min(assessment_counts.values())
    if train_min < minimum_train or assessment_min < minimum_assessment:
        _fail("classification_support", "c06_metric_support_insufficient")
    return train_min, assessment_min


def _regression_support(
    groups: np.ndarray,
    outcome: np.ndarray,
    assessment: set[int],
    minimum_count: int,
    minimum_variance: float,
) -> tuple[int, float]:
    by_group = _group_values(groups, outcome, assessment)
    values = np.asarray([group_values[0] for group_values in by_group.values()], dtype=float)
    count = len(values)
    variance = float(np.var(values)) if count else float("nan")
    if count < minimum_count or not math.isfinite(variance) or variance < minimum_variance:
        _fail("regression_assessment_support", "c06_metric_support_insufficient")
    return count, variance


def harrell_comparable_group_pair_count(
    time: Sequence[float], event: Sequence[int], groups: Sequence[Hashable], indices: Sequence[int]
) -> int:
    """Count distinct group pairs orderable by an observed earlier event."""
    time_array = np.asarray(time, dtype=float)
    event_array = np.asarray(event, dtype=int)
    group_array = np.asarray(groups, dtype=object)
    comparable: set[frozenset[Hashable]] = set()
    for i in indices:
        if event_array[i] != 1:
            continue
        for j in indices:
            if group_array[i] != group_array[j] and time_array[j] > time_array[i]:
                comparable.add(frozenset((group_array[i], group_array[j])))
    return len(comparable)


def _survival_support(
    groups: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    train: set[int],
    assessment: set[int],
    minimum_training_events: int,
    minimum_comparable_pairs: int,
) -> tuple[int, int]:
    training_event_groups = len({groups[index] for index in train if event[index] == 1})
    comparable_pairs = harrell_comparable_group_pair_count(time, event, groups, sorted(assessment))
    if training_event_groups < minimum_training_events:
        _fail("survival_training_event_support", "c06_metric_support_insufficient")
    if comparable_pairs < minimum_comparable_pairs:
        _fail("survival_assessment_pair_support", "c06_metric_support_insufficient")
    return training_event_groups, comparable_pairs


def _validate_split_manifest_impl(
    _constructor: Any,
    manifest: Mapping[str, Any],
    *,
    n_samples: int,
    groups: Sequence[Hashable],
    task: str,
    requested_outer_k: int,
    requested_inner_k: int,
    minimum_training_groups: int,
    minimum_assessment_groups: int,
    y: Sequence[Any] | None = None,
    time: Sequence[float] | None = None,
    event: Sequence[int] | None = None,
    minimum_training_groups_per_class: int | None = None,
    minimum_assessment_groups_per_class: int | None = None,
    minimum_regression_assessment_groups: int | None = None,
    minimum_regression_assessment_variance: float | None = None,
    minimum_survival_training_event_groups: int | None = None,
    minimum_survival_assessment_comparable_pairs: int | None = None,
) -> ValidatedSplitPlan:
    """Verify exact realized nested folds and return a private runtime plan."""
    n = _integer(n_samples, "n_samples", 2)
    outer_k = _integer(requested_outer_k, "requested_outer_k", 2)
    inner_k = _integer(requested_inner_k, "requested_inner_k", 2)
    minimum_train_groups = _integer(
        minimum_training_groups, "minimum_training_groups", 1
    )
    minimum_assess_groups = _integer(
        minimum_assessment_groups, "minimum_assessment_groups", 1
    )
    if task not in {"classification", "regression", "survival"}:
        _fail("task")

    group_array = _vector(groups, "groups", n)
    if any(_missing(value) or not isinstance(value, Hashable) for value in group_array):
        _fail("group_id_missing_or_invalid", "c06_group_id_missing")
    group_count = len(set(group_array))

    labels = outcome = time_array = event_array = None
    support_parameters: tuple[int | float, ...]
    if task == "classification":
        if y is None:
            _fail("classification_outcome_missing")
        labels = _vector(y, "y", n)
        if any(_missing(value) or not isinstance(value, Hashable) for value in labels):
            _fail("classification_outcome_invalid")
        for values in _group_values(group_array, labels, set(range(n))).values():
            if len(set(values)) != 1:
                _fail("classification_group_outcome_mixed", "c06_group_outcome_mixed")
        classes = set(labels)
        if len(classes) < 2:
            _fail("classification_class_count", "c06_metric_support_insufficient")
        class_train = _integer(
            minimum_training_groups_per_class, "minimum_training_groups_per_class", 1
        )
        class_assessment = _integer(
            minimum_assessment_groups_per_class, "minimum_assessment_groups_per_class", 1
        )
        support_parameters = (class_train, class_assessment)
    elif task == "regression":
        if y is None:
            _fail("regression_outcome_missing")
        try:
            outcome = np.asarray(y, dtype=float)
        except (TypeError, ValueError):
            _fail("regression_outcome_type")
        if outcome.ndim != 1 or len(outcome) != n or not np.isfinite(outcome).all():
            _fail("regression_outcome_invalid")
        for values in _group_values(group_array, outcome, set(range(n))).values():
            if len(set(values)) != 1:
                _fail("regression_group_outcome_mixed", "c06_group_outcome_mixed")
        regression_count = _integer(
            minimum_regression_assessment_groups, "minimum_regression_assessment_groups", 1
        )
        regression_variance = _real(
            minimum_regression_assessment_variance,
            "minimum_regression_assessment_variance",
            0.0,
        )
        support_parameters = (regression_count, regression_variance)
    else:
        if time is None or event is None:
            _fail("survival_outcome_missing")
        try:
            time_array = np.asarray(time, dtype=float)
        except (TypeError, ValueError):
            _fail("survival_time_type")
        raw_event = _vector(event, "event", n)
        if time_array.ndim != 1 or len(time_array) != n:
            _fail("survival_time_shape")
        if not np.isfinite(time_array).all() or (time_array <= 0).any():
            _fail("survival_time_invalid")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or int(value) not in {0, 1}
            for value in raw_event
        ):
            _fail("survival_event_invalid")
        event_array = np.asarray(raw_event, dtype=int)
        survival_outcome = np.empty(n, dtype=object)
        survival_outcome[:] = list(zip(time_array, event_array))
        for values in _group_values(group_array, survival_outcome, set(range(n))).values():
            if len(set(values)) != 1:
                _fail("survival_group_outcome_mixed", "c06_group_outcome_mixed")
        survival_events = _integer(
            minimum_survival_training_event_groups,
            "minimum_survival_training_event_groups",
            1,
        )
        survival_pairs = _integer(
            minimum_survival_assessment_comparable_pairs,
            "minimum_survival_assessment_comparable_pairs",
            1,
        )
        support_parameters = (survival_events, survival_pairs)

    normalized = _normalized_manifest(manifest)
    outer_folds = normalized["outer_folds"]
    if len(outer_folds) != outer_k:
        _fail("outer_fold_count_exact")

    universe = set(range(n))
    outer_assessments: list[set[int]] = []
    observed: list[tuple[int | float, int | float]] = []
    observed_group_counts: list[tuple[int, int]] = []
    validated_outer: list[tuple[np.ndarray, np.ndarray]] = []
    validated_inner: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for fold in outer_folds:
        outer_train, outer_assessment = _partition(
            fold["train"], fold["assessment"], universe, n, "outer"
        )
        _check_group_disjoint(group_array, outer_train, outer_assessment, "outer")
        outer_assessments.append(outer_assessment)
        validated_outer.append(
            (
                np.asarray(sorted(outer_train), dtype=np.int64),
                np.asarray(sorted(outer_assessment), dtype=np.int64),
            )
        )
        if len(fold["inner_folds"]) != inner_k:
            _fail("inner_fold_count_exact")

        partitions = [(outer_train, outer_assessment)]
        inner_assessments: list[set[int]] = []
        current_inner: list[tuple[np.ndarray, np.ndarray]] = []
        for inner in fold["inner_folds"]:
            inner_train, inner_assessment = _partition(
                inner["train"], inner["assessment"], outer_train, n, "inner"
            )
            _check_group_disjoint(group_array, inner_train, inner_assessment, "inner")
            inner_assessments.append(inner_assessment)
            partitions.append((inner_train, inner_assessment))
            current_inner.append(
                (
                    np.asarray(sorted(inner_train), dtype=np.int64),
                    np.asarray(sorted(inner_assessment), dtype=np.int64),
                )
            )
        validated_inner.append(current_inner)
        if set().union(*inner_assessments) != outer_train or sum(
            len(indices) for indices in inner_assessments
        ) != len(outer_train):
            _fail("inner_assessment_exact_coverage")

        for train_indices, assessment_indices in partitions:
            train_group_count = len(_group_set(group_array, train_indices))
            assessment_group_count = len(_group_set(group_array, assessment_indices))
            if (
                train_group_count < minimum_train_groups
                or assessment_group_count < minimum_assess_groups
            ):
                _fail("partition_group_support", "c06_metric_support_insufficient")
            observed_group_counts.append((train_group_count, assessment_group_count))
            if task == "classification":
                observed.append(
                    _classification_support(
                        group_array,
                        labels,
                        train_indices,
                        assessment_indices,
                        classes,
                        support_parameters[0],
                        support_parameters[1],
                    )
                )
            elif task == "regression":
                observed.append(
                    _regression_support(
                        group_array,
                        outcome,
                        assessment_indices,
                        support_parameters[0],
                        support_parameters[1],
                    )
                )
            else:
                observed.append(
                    _survival_support(
                        group_array,
                        time_array,
                        event_array,
                        train_indices,
                        assessment_indices,
                        support_parameters[0],
                        support_parameters[1],
                    )
                )

    if set().union(*outer_assessments) != universe or sum(
        len(indices) for indices in outer_assessments
    ) != n:
        _fail("outer_assessment_exact_coverage")

    if task == "classification":
        support_summary = {
            "minimum_realized_assessment_groups_per_class": int(min(value[1] for value in observed)),
            "minimum_realized_training_groups_per_class": int(min(value[0] for value in observed)),
        }
    elif task == "regression":
        support_summary = {
            "minimum_realized_assessment_group_count": int(min(value[0] for value in observed)),
            "minimum_realized_assessment_variance": float(min(value[1] for value in observed)),
        }
    else:
        support_summary = {
            "minimum_realized_assessment_comparable_pairs": int(min(value[1] for value in observed)),
            "minimum_realized_training_event_groups": int(min(value[0] for value in observed)),
        }

    internal_split_digest = canonical_split_manifest_sha256(manifest)
    receipt = {
        "claim_id": "C06",
        "decision": "development_only",
        "eligibility_reason": "trusted_process_development_mechanics",
        "group_count": group_count,
        "inner_fold_count": inner_k,
        "outer_fold_count": outer_k,
        "split_manifest_sha256": None,
        "split_manifest_status": _SPLIT_MANIFEST_STATUS,
        "support_summary": {
            **support_summary,
            "minimum_realized_assessment_group_count": min(
                value[1] for value in observed_group_counts
            ),
            "minimum_realized_training_group_count": min(
                value[0] for value in observed_group_counts
            ),
        },
        "verifier_status": (
            "trusted_process_development_mechanics_pending_frozen_public_manifest"
        ),
    }
    return _constructor(
        validated_outer,
        validated_inner,
        tuple(group_array),
        internal_split_digest,
        receipt,
    )


def _bind_split_validator(constructor: Any, implementation: Any) -> Any:
    from inspect import signature

    def validate_split_manifest(*args: Any, **kwargs: Any) -> Any:
        return implementation(constructor, *args, **kwargs)

    validate_split_manifest.__name__ = "validate_split_manifest"
    validate_split_manifest.__qualname__ = "validate_split_manifest"
    validate_split_manifest.__doc__ = implementation.__doc__
    parameters = tuple(signature(implementation).parameters.values())[1:]
    validate_split_manifest.__signature__ = signature(implementation).replace(
        parameters=parameters
    )
    return validate_split_manifest


validate_split_manifest = _bind_split_validator(
    _validated_plan_constructor, _validate_split_manifest_impl
)
del _bind_split_validator, _validated_plan_constructor, _validate_split_manifest_impl
