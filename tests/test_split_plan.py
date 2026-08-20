from __future__ import annotations

from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import asdict
import hashlib
from itertools import product
import json
import pickle

import numpy as np
import pytest

import omicau.models.split_plan as split_plan_module
from omicau.models.split_plan import (
    SplitPartitionEvidence,
    SplitValidationError,
    ValidatedSplitPlan,
    _canonical_runtime_universe_sha256,
    _partition_evidence_values,
    canonical_split_manifest_sha256,
    harrell_comparable_group_pair_count,
    validate_split_manifest,
)


@contextmanager
def _failure(invariant: str, code: str = "c06_split_manifest_invalid"):
    with pytest.raises(SplitValidationError) as caught:
        yield caught
    error = caught.value
    assert error.code == code
    assert error.invariant == invariant
    assert str(error) == code
    assert error.args == (code,)


def _manifest(n_groups: int = 8) -> dict:
    outer = []
    universe = set(range(n_groups))
    for outer_fold in range(2):
        assessment = {index for index in universe if index % 2 == outer_fold}
        train = universe - assessment
        inner = []
        for inner_fold in range(2):
            inner_assessment = {index for index in train if (index // 2) % 2 == inner_fold}
            inner.append({
                "train": sorted(train - inner_assessment),
                "assessment": sorted(inner_assessment),
            })
        outer.append({
            "train": sorted(train),
            "assessment": sorted(assessment),
            "inner_folds": inner,
        })
    return {"outer_folds": outer}


def _expanded_manifest() -> dict:
    manifest = deepcopy(_manifest())
    for fold in manifest["outer_folds"]:
        fold["train"] = [row for index in fold["train"] for row in (2 * index, 2 * index + 1)]
        fold["assessment"] = [
            row for index in fold["assessment"] for row in (2 * index, 2 * index + 1)
        ]
        for inner in fold["inner_folds"]:
            inner["train"] = [
                row for index in inner["train"] for row in (2 * index, 2 * index + 1)
            ]
            inner["assessment"] = [
                row for index in inner["assessment"] for row in (2 * index, 2 * index + 1)
            ]
    return manifest


def _repeat_rows(kwargs: dict) -> dict:
    expanded = deepcopy(kwargs)
    expanded["n_samples"] = 16
    expanded["groups"] = [group for group in kwargs["groups"] for _ in range(2)]
    for field in ("y", "time", "event"):
        if field in kwargs:
            expanded[field] = [value for value in kwargs[field] for _ in range(2)]
    return expanded


def _classification_kwargs() -> dict:
    return {
        "n_samples": 8,
        "groups": [f"g{index}" for index in range(8)],
        "task": "classification",
        "y": [0, 0, 0, 0, 1, 1, 1, 1],
        "requested_outer_k": 2,
        "requested_inner_k": 2,
        "minimum_training_groups": 2,
        "minimum_assessment_groups": 2,
        "minimum_training_groups_per_class": 1,
        "minimum_assessment_groups_per_class": 1,
    }


def _regression_kwargs() -> dict:
    return {
        "n_samples": 8,
        "groups": list(range(8)),
        "task": "regression",
        "y": [0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0],
        "requested_outer_k": 2,
        "requested_inner_k": 2,
        "minimum_training_groups": 2,
        "minimum_assessment_groups": 2,
        "minimum_regression_assessment_groups": 2,
        "minimum_regression_assessment_variance": 0.2,
    }


def _survival_kwargs() -> dict:
    return {
        "n_samples": 8,
        "groups": list(range(8)),
        "task": "survival",
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "event": [1, 1, 1, 1, 1, 1, 0, 0],
        "requested_outer_k": 2,
        "requested_inner_k": 2,
        "minimum_training_groups": 2,
        "minimum_assessment_groups": 2,
        "minimum_survival_training_event_groups": 1,
        "minimum_survival_assessment_comparable_pairs": 1,
    }


def _runtime_universe_kwargs(kwargs: dict) -> dict:
    return {
        "event": kwargs.get("event"),
        "groups": kwargs["groups"],
        "task": kwargs["task"],
        "time": kwargs.get("time"),
        "y": kwargs.get("y"),
    }


@pytest.mark.parametrize("kwargs", [_classification_kwargs(), _regression_kwargs(), _survival_kwargs()])
def test_valid_exact_nested_plan_emits_only_public_safe_aggregates(kwargs: dict) -> None:
    manifest = _manifest()
    plan = validate_split_manifest(manifest, **kwargs)
    assert isinstance(plan, ValidatedSplitPlan)
    receipt = plan.receipt()
    assert set(receipt) == {
        "claim_id", "decision", "eligibility_reason", "group_count", "inner_fold_count",
        "outer_fold_count", "split_manifest_sha256", "split_manifest_status",
        "support_summary", "verifier_status",
    }
    assert receipt["decision"] == "development_only"
    assert receipt["outer_fold_count"] == receipt["inner_fold_count"] == 2
    assert receipt["split_manifest_sha256"] is None
    assert receipt["split_manifest_status"] == "unavailable_pending_frozen_public_manifest"
    assert canonical_split_manifest_sha256(manifest) not in repr(receipt)
    serialized = json.dumps(receipt, sort_keys=True)
    for forbidden in ("g0", '"train":', '"assessment":', "indices", "labels", "subject", "path"):
        assert forbidden not in serialized


def test_hash_is_canonical_for_mapping_and_index_order() -> None:
    manifest = _manifest()
    reordered = {"outer_folds": []}
    for fold in manifest["outer_folds"]:
        reordered["outer_folds"].append({
            "inner_folds": [
                {"assessment": list(reversed(child["assessment"])), "train": list(reversed(child["train"]))}
                for child in fold["inner_folds"]
            ],
            "assessment": list(reversed(fold["assessment"])),
            "train": list(reversed(fold["train"])),
        })
    assert canonical_split_manifest_sha256(manifest) == canonical_split_manifest_sha256(reordered)


def test_candidate_manifests_have_no_public_pre_freeze_commitment() -> None:
    first = _manifest()
    second = deepcopy(first)
    second["outer_folds"].reverse()
    assert canonical_split_manifest_sha256(first) != canonical_split_manifest_sha256(second)
    first_receipt = validate_split_manifest(first, **_classification_kwargs()).receipt()
    second_receipt = validate_split_manifest(second, **_classification_kwargs()).receipt()
    assert first_receipt == second_receipt
    assert first_receipt["split_manifest_sha256"] is None
    assert canonical_split_manifest_sha256(first) not in repr(first_receipt)
    assert canonical_split_manifest_sha256(second) not in repr(second_receipt)


def _exhaustive_classification_witness(labels: list[int], outer_k: int, inner_k: int) -> dict | None:
    universe = set(range(len(labels)))

    def supported(indices: set[int], assignment: tuple[int, ...], k: int) -> bool:
        return all(
            {labels[index] for index in indices if assignment[index] == fold} == {0, 1}
            and {labels[index] for index in indices if assignment[index] != fold} == {0, 1}
            for fold in range(k)
        )

    for outer_assignment in product(range(outer_k), repeat=len(labels)):
        if set(outer_assignment) != set(range(outer_k)) or not supported(universe, outer_assignment, outer_k):
            continue
        outer_folds = []
        for outer_fold in range(outer_k):
            assessment = {index for index in universe if outer_assignment[index] == outer_fold}
            train = universe - assessment
            inner_folds = None
            for compact_assignment in product(range(inner_k), repeat=len(train)):
                inner_assignment = dict(zip(sorted(train), compact_assignment))
                if set(compact_assignment) != set(range(inner_k)):
                    continue
                if all(
                    {labels[index] for index in train if inner_assignment[index] == fold} == {0, 1}
                    and {labels[index] for index in train if inner_assignment[index] != fold} == {0, 1}
                    for fold in range(inner_k)
                ):
                    inner_folds = [
                        {
                            "train": sorted(index for index in train if inner_assignment[index] != fold),
                            "assessment": sorted(index for index in train if inner_assignment[index] == fold),
                        }
                        for fold in range(inner_k)
                    ]
                    break
            if inner_folds is None:
                break
            outer_folds.append({
                "train": sorted(train),
                "assessment": sorted(assessment),
                "inner_folds": inner_folds,
            })
        if len(outer_folds) == outer_k:
            return {"outer_folds": outer_folds}
    return None


def test_small_fixture_exhaustive_assignment_oracle_agrees_on_feasibility() -> None:
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    witness = _exhaustive_classification_witness(labels, 2, 2)
    assert witness is not None
    kwargs = _classification_kwargs()
    kwargs["groups"] = list(range(8))
    assert validate_split_manifest(witness, **kwargs).receipt()["verifier_status"] == (
        "trusted_process_development_mechanics_pending_frozen_public_manifest"
    )
    assert _exhaustive_classification_witness(labels, 3, 3) is None


@pytest.mark.parametrize(
    ("mutate", "invariant"),
    [
        (lambda m: m["outer_folds"][0]["assessment"].append(m["outer_folds"][0]["train"][0]), "outer_index_disjoint"),
        (lambda m: m["outer_folds"][0]["train"].pop(), "outer_parent_coverage"),
        (lambda m: m["outer_folds"][0]["assessment"].append(m["outer_folds"][0]["assessment"][0]), "outer_assessment_unique"),
        (lambda m: m["outer_folds"][0]["inner_folds"][0]["assessment"].append(m["outer_folds"][0]["assessment"][0]), "inner_parent_containment"),
        (lambda m: m["outer_folds"].pop(), "outer_fold_count_exact"),
        (lambda m: m["outer_folds"].append(deepcopy(m["outer_folds"][0])), "outer_fold_count_exact"),
        (lambda m: m["outer_folds"][0]["inner_folds"].pop(), "inner_fold_count_exact"),
        (lambda m: m["outer_folds"].__setitem__(1, deepcopy(m["outer_folds"][0])), "outer_assessment_exact_coverage"),
        (lambda m: m["outer_folds"][0]["inner_folds"].__setitem__(1, deepcopy(m["outer_folds"][0]["inner_folds"][0])), "inner_assessment_exact_coverage"),
    ],
)
def test_watched_manifest_defects_fail(mutate, invariant: str) -> None:
    manifest = _manifest()
    mutate(manifest)
    with _failure(invariant):
        validate_split_manifest(manifest, **_classification_kwargs())


def test_group_overlap_and_missing_group_fail_without_identifier_disclosure() -> None:
    kwargs = _classification_kwargs()
    kwargs["groups"] = ["private-a", "private-a", *kwargs["groups"][2:]]
    with _failure("outer_group_disjoint") as caught:
        validate_split_manifest(_manifest(), **kwargs)
    assert "private" not in str(caught.value)

    kwargs = _classification_kwargs()
    kwargs["groups"][0] = None
    with _failure("group_id_missing_or_invalid", "c06_group_id_missing"):
        validate_split_manifest(_manifest(), **kwargs)

    kwargs = _classification_kwargs()
    kwargs["groups"][0] = np.nan
    with _failure("group_id_missing_or_invalid", "c06_group_id_missing"):
        validate_split_manifest(_manifest(), **kwargs)


def test_mixed_class_label_within_group_fails() -> None:
    manifest = _manifest(8)
    kwargs = _classification_kwargs()
    kwargs.update(n_samples=9, groups=["mixed", "mixed", *range(2, 9)], y=[0, 1, 0, 0, 1, 1, 0, 0, 1])
    with _failure("classification_group_outcome_mixed", "c06_group_outcome_mixed"):
        validate_split_manifest(manifest, **kwargs)


@pytest.mark.parametrize("kwargs", [_classification_kwargs(), _regression_kwargs(), _survival_kwargs()])
def test_valid_repeated_row_groups_preserve_one_outcome_object(kwargs: dict) -> None:
    receipt = validate_split_manifest(_expanded_manifest(), **_repeat_rows(kwargs)).receipt()
    assert receipt["verifier_status"] == (
        "trusted_process_development_mechanics_pending_frozen_public_manifest"
    )
    assert receipt["group_count"] == 8


def test_classification_support_is_required_and_watched() -> None:
    kwargs = _classification_kwargs()
    kwargs["minimum_assessment_groups_per_class"] = 3
    with _failure("classification_support", "c06_metric_support_insufficient"):
        validate_split_manifest(_manifest(), **kwargs)
    kwargs = _classification_kwargs()
    kwargs["minimum_training_groups_per_class"] = None
    with _failure("minimum_training_groups_per_class_type"):
        validate_split_manifest(_manifest(), **kwargs)


@pytest.mark.parametrize("bad_y", [[0, 1, 2, np.inf, 4, 5, 6, 7], [0, 1, 2, np.nan, 4, 5, 6, 7]])
def test_regression_requires_finite_outcomes(bad_y: list[float]) -> None:
    kwargs = _regression_kwargs()
    kwargs["y"] = bad_y
    with _failure("regression_outcome_invalid"):
        validate_split_manifest(_manifest(), **kwargs)


def test_regression_count_and_variance_thresholds_are_watched() -> None:
    kwargs = _regression_kwargs()
    kwargs["minimum_regression_assessment_groups"] = 3
    with _failure("regression_assessment_support", "c06_metric_support_insufficient"):
        validate_split_manifest(_manifest(), **kwargs)
    kwargs = _regression_kwargs()
    kwargs["minimum_regression_assessment_variance"] = 100.0
    with _failure("regression_assessment_support", "c06_metric_support_insufficient"):
        validate_split_manifest(_manifest(), **kwargs)


def test_mixed_regression_outcome_within_group_fails_before_support() -> None:
    kwargs = _repeat_rows(_regression_kwargs())
    kwargs["y"][1] += 0.5
    with _failure("regression_group_outcome_mixed", "c06_group_outcome_mixed"):
        validate_split_manifest(_expanded_manifest(), **kwargs)


@pytest.mark.parametrize(
    ("field", "value", "invariant"),
    [
        ("time", [1, 2, 3, 4, 5, 6, 7, 0], "survival_time_invalid"),
        ("time", [1, 2, 3, 4, 5, 6, 7, np.inf], "survival_time_invalid"),
        ("event", [1, 1, 1, 1, 1, 1, 0, 2], "survival_event_invalid"),
        ("event", [True, 1, 1, 1, 1, 1, 0, 0], "survival_event_invalid"),
    ],
)
def test_survival_outcome_validation(field: str, value: list, invariant: str) -> None:
    kwargs = _survival_kwargs()
    kwargs[field] = value
    with _failure(invariant):
        validate_split_manifest(_manifest(), **kwargs)


@pytest.mark.parametrize("field", ["time", "event"])
def test_mixed_survival_outcome_tuple_within_group_fails(field: str) -> None:
    kwargs = _repeat_rows(_survival_kwargs())
    kwargs[field][1] = 1 - kwargs[field][1] if field == "event" else kwargs[field][1] + 1
    with _failure("survival_group_outcome_mixed", "c06_group_outcome_mixed"):
        validate_split_manifest(_expanded_manifest(), **kwargs)


def test_survival_training_events_and_no_comparable_pairs_fail() -> None:
    kwargs = _survival_kwargs()
    kwargs["minimum_survival_training_event_groups"] = 4
    with _failure("survival_training_event_support", "c06_metric_support_insufficient"):
        validate_split_manifest(_manifest(), **kwargs)
    kwargs = _survival_kwargs()
    kwargs["event"] = [0] * 8
    with _failure("survival_training_event_support", "c06_metric_support_insufficient"):
        validate_split_manifest(_manifest(), **kwargs)
    kwargs = _survival_kwargs()
    kwargs["time"] = [1.0] * 8
    with _failure("survival_assessment_pair_support", "c06_metric_support_insufficient"):
        validate_split_manifest(_manifest(), **kwargs)
    kwargs = _survival_kwargs()
    kwargs["minimum_survival_assessment_comparable_pairs"] = 2
    with _failure("survival_assessment_pair_support", "c06_metric_support_insufficient"):
        validate_split_manifest(_manifest(), **kwargs)
    assert harrell_comparable_group_pair_count([1, 1], [1, 1], [0, 1], [0, 1]) == 0


def test_independent_comparable_pair_oracle_agrees() -> None:
    time = [1.0, 4.0, 2.0, 5.0]
    event = [1, 0, 1, 1]
    groups = ["a", "b", "c", "d"]
    oracle = {
        frozenset((groups[i], groups[j]))
        for i in range(4)
        for j in range(i + 1, 4)
        if (event[i] == 1 and time[i] < time[j]) or (event[j] == 1 and time[j] < time[i])
    }
    assert harrell_comparable_group_pair_count(time, event, groups, range(4)) == len(oracle)


@pytest.mark.parametrize(
    ("field", "value", "invariant"),
    [
        ("requested_outer_k", True, "requested_outer_k_type"),
        ("requested_inner_k", 1, "requested_inner_k_range"),
        ("n_samples", 8.0, "n_samples_type"),
    ],
)
def test_boolean_and_noninteger_parameters_fail(field: str, value, invariant: str) -> None:
    kwargs = _classification_kwargs()
    kwargs[field] = value
    with _failure(invariant):
        validate_split_manifest(_manifest(), **kwargs)


@pytest.mark.parametrize("bad_index", [True, 0.0, "0"])
def test_boolean_and_noninteger_indices_fail(bad_index) -> None:
    manifest = _manifest()
    manifest["outer_folds"][0]["train"][0] = bad_index
    with _failure("outer_train_integer"):
        validate_split_manifest(manifest, **_classification_kwargs())


@pytest.mark.parametrize("bad_index", [-1, 8])
def test_out_of_range_indices_fail(bad_index: int) -> None:
    manifest = _manifest()
    manifest["outer_folds"][0]["train"][0] = bad_index
    with _failure("outer_index_range"):
        validate_split_manifest(manifest, **_classification_kwargs())


def test_unknown_manifest_fields_are_rejected_before_receipt() -> None:
    manifest = _manifest()
    manifest["subject_ids"] = ["private"]
    with _failure("manifest_schema") as caught:
        validate_split_manifest(manifest, **_classification_kwargs())
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("kind", ["group", "outcome", "manifest"])
def test_hostile_values_never_enter_public_error_text(kind: str) -> None:
    hostile = r"C:\private\subject-991\labels.tsv"
    manifest = _expanded_manifest() if kind == "outcome" else _manifest()
    kwargs = _repeat_rows(_classification_kwargs()) if kind == "outcome" else _classification_kwargs()
    if kind == "group":
        kwargs["groups"][0] = {"subject_path": hostile}
        invariant, code = "group_id_missing_or_invalid", "c06_group_id_missing"
    elif kind == "outcome":
        kwargs["y"][1] = hostile
        invariant, code = "classification_group_outcome_mixed", "c06_group_outcome_mixed"
    else:
        manifest[hostile] = ["private-value"]
        invariant, code = "manifest_schema", "c06_split_manifest_invalid"
    with _failure(invariant, code) as caught:
        validate_split_manifest(manifest, **kwargs)
    assert hostile not in str(caught.value)
    assert hostile not in repr(caught.value)


def test_error_code_allowlist_is_closed() -> None:
    with pytest.raises(ValueError, match="unsupported_c06_refusal_code"):
        SplitValidationError("hostile_code", "manifest_schema")


def test_runtime_plan_iterators_are_copy_safe_and_fold_scoped() -> None:
    plan = validate_split_manifest(_manifest(), **_classification_kwargs())
    outer = list(plan._private_outer_splits())
    assert [(train.tolist(), assessment.tolist()) for train, assessment in outer] == [
        ([1, 3, 5, 7], [0, 2, 4, 6]),
        ([0, 2, 4, 6], [1, 3, 5, 7]),
    ]
    assert not outer[0][0].flags.writeable
    outer[0][0].flags.writeable = True
    outer[0][0][0] = 0
    assert next(plan._private_outer_splits())[0].tolist() == [1, 3, 5, 7]

    inner = list(plan._private_inner_splits(0))
    assert [(train.tolist(), assessment.tolist()) for train, assessment in inner] == [
        ([3, 7], [1, 5]),
        ([1, 5], [3, 7]),
    ]
    assert not inner[0][1].flags.writeable
    inner[0][1].flags.writeable = True
    inner[0][1][0] = 7
    assert next(plan._private_inner_splits(0))[1].tolist() == [1, 5]


def test_runtime_plan_is_immutable_and_private_serialization_fails() -> None:
    plan = validate_split_manifest(_manifest(), **_classification_kwargs())
    with pytest.raises(AttributeError, match="validated_split_plan_immutable"):
        plan.extra = "private"  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        json.dumps(plan)
    with pytest.raises(TypeError, match="validated_split_plan_private_serialization_forbidden"):
        pickle.dumps(plan)
    with pytest.raises(TypeError, match="validated_split_plan_private_copy_forbidden"):
        copy(plan)
    with pytest.raises(TypeError, match="validated_split_plan_private_copy_forbidden"):
        deepcopy(plan)
    with pytest.raises(TypeError):
        asdict(plan)
    with pytest.raises(TypeError):
        vars(plan)
    assert not hasattr(plan, "__dict__")
    assert repr(plan) == "ValidatedSplitPlan(trusted_process_development_mechanics=True)"


def test_plan_public_surface_exposes_only_aggregate_receipt() -> None:
    plan = validate_split_manifest(_manifest(), **_classification_kwargs())
    assert [name for name in dir(plan) if not name.startswith("_")] == ["receipt"]
    for old_name in ("outer_splits", "inner_splits", "partition_evidence"):
        assert not hasattr(plan, old_name)
        with pytest.raises(AttributeError):
            getattr(plan, old_name)
    assert "only receipt() is public-safe" in (ValidatedSplitPlan.__doc__ or "")
    public = json.dumps(plan.receipt(), sort_keys=True)
    for group in (f"g{index}" for index in range(8)):
        assert group not in public
        assert hashlib.sha256(group.encode("ascii")).hexdigest() not in public
    assert canonical_split_manifest_sha256(_manifest()) not in public


@pytest.mark.parametrize(
    "kwargs", [_classification_kwargs(), _regression_kwargs(), _survival_kwargs()]
)
def test_private_runtime_universe_accepts_only_exact_ordered_rows(kwargs: dict) -> None:
    plan = validate_split_manifest(_manifest(), **kwargs)
    assert plan._private_validate_runtime_universe(
        **_runtime_universe_kwargs(kwargs)
    ) is None

    shortened = deepcopy(kwargs)
    shortened["groups"] = shortened["groups"][:-1]
    for field in ("y", "time", "event"):
        if field in shortened:
            shortened[field] = shortened[field][:-1]
    with pytest.raises(
        TypeError, match="^validated_split_plan_runtime_universe_mismatch$"
    ):
        plan._private_validate_runtime_universe(
            **_runtime_universe_kwargs(shortened)
        )

    changed_task = deepcopy(kwargs)
    changed_task["task"] = (
        "regression" if kwargs["task"] == "classification" else "classification"
    )
    with pytest.raises(
        TypeError, match="^validated_split_plan_runtime_universe_mismatch$"
    ):
        plan._private_validate_runtime_universe(
            **_runtime_universe_kwargs(changed_task)
        )


def test_private_runtime_universe_detects_class_and_group_reassignments() -> None:
    kwargs = _classification_kwargs()
    plan = validate_split_manifest(_manifest(), **kwargs)

    class_swap = deepcopy(kwargs)
    class_swap["y"][0], class_swap["y"][4] = class_swap["y"][4], class_swap["y"][0]
    group_relabel = deepcopy(kwargs)
    group_relabel["groups"][0] = "runtime-private-marker"
    group_reassignment = deepcopy(kwargs)
    group_reassignment["groups"][0], group_reassignment["groups"][1] = (
        group_reassignment["groups"][1],
        group_reassignment["groups"][0],
    )
    for changed in (class_swap, group_relabel, group_reassignment):
        with pytest.raises(
            TypeError, match="^validated_split_plan_runtime_universe_mismatch$"
        ) as error:
            plan._private_validate_runtime_universe(
                **_runtime_universe_kwargs(changed)
            )
        assert "runtime-private-marker" not in str(error.value)


def test_private_runtime_universe_detects_regression_shift() -> None:
    kwargs = _regression_kwargs()
    plan = validate_split_manifest(_manifest(), **kwargs)
    shifted = deepcopy(kwargs)
    shifted["y"] = [value + 10.0 for value in shifted["y"]]
    with pytest.raises(
        TypeError, match="^validated_split_plan_runtime_universe_mismatch$"
    ):
        plan._private_validate_runtime_universe(**_runtime_universe_kwargs(shifted))


@pytest.mark.parametrize("field", ["time", "event"])
def test_private_runtime_universe_detects_survival_changes(field: str) -> None:
    kwargs = _survival_kwargs()
    plan = validate_split_manifest(_manifest(), **kwargs)
    changed = deepcopy(kwargs)
    changed[field][6] = 1 if field == "event" else changed[field][6] + 0.25
    with pytest.raises(
        TypeError, match="^validated_split_plan_runtime_universe_mismatch$"
    ):
        plan._private_validate_runtime_universe(**_runtime_universe_kwargs(changed))


def test_private_runtime_universe_detects_repeated_group_row_reordering() -> None:
    kwargs = _repeat_rows(_classification_kwargs())
    plan = validate_split_manifest(_expanded_manifest(), **kwargs)
    reordered = deepcopy(kwargs)
    order = [0, 2, 1, *range(3, len(kwargs["groups"]))]
    reordered["groups"] = [kwargs["groups"][index] for index in order]
    reordered["y"] = [kwargs["y"][index] for index in order]
    with pytest.raises(
        TypeError, match="^validated_split_plan_runtime_universe_mismatch$"
    ):
        plan._private_validate_runtime_universe(**_runtime_universe_kwargs(reordered))


def test_private_runtime_universe_identity_never_enters_public_outputs() -> None:
    kwargs = _classification_kwargs()
    plan = validate_split_manifest(_manifest(), **kwargs)
    identity = _canonical_runtime_universe_sha256(
        **_runtime_universe_kwargs(kwargs)
    )
    changed = deepcopy(kwargs)
    changed["groups"][0] = "runtime-private-marker"
    with pytest.raises(TypeError) as caught:
        plan._private_validate_runtime_universe(**_runtime_universe_kwargs(changed))
    public = repr(plan) + json.dumps(plan.receipt(), sort_keys=True)
    assert identity not in public
    assert "runtime_universe" not in public
    assert identity not in str(caught.value)
    assert "runtime-private-marker" not in str(caught.value)


def test_runtime_plan_cannot_be_constructed_without_validation() -> None:
    with pytest.raises(TypeError, match="validated_split_plan_requires_validation"):
        ValidatedSplitPlan([], [], [], "0" * 64, {}, _validation_token=object())


def test_validated_plan_is_opaque_noncontainer_without_writable_state() -> None:
    plan = validate_split_manifest(_manifest(), **_classification_kwargs())
    assert not isinstance(plan, tuple)
    for operation in (
        lambda: tuple.__getitem__(plan, 0),
        lambda: tuple.__iter__(plan),
        lambda: tuple.__repr__(plan),
        lambda: iter(plan),
        lambda: len(plan),
        lambda: plan[0],
    ):
        with pytest.raises(TypeError):
            operation()
    with pytest.raises(AttributeError):
        object.__getattribute__(plan, "_ValidatedSplitPlan__inner")
    for name, value in (
        ("_ValidatedSplitPlan__inner", (((1,), (2,)),)),
        ("_ValidatedSplitPlan__split_digest", "0" * 64),
        ("_ValidatedSplitPlan__core", ((((1,), (2,)),), "0" * 64)),
    ):
        with pytest.raises(AttributeError):
            object.__setattr__(plan, name, value)
    assert plan.receipt()["inner_fold_count"] == 2


def test_ordinary_module_api_exposes_no_plan_registry_or_constructor_binding() -> None:
    forbidden_names = {
        "_validated_plan_constructor",
        "_bind_split_validator",
        "_validate_split_manifest_impl",
        "core_by_handle",
    }
    assert forbidden_names.isdisjoint(vars(split_plan_module))
    with pytest.raises(TypeError, match="validated_split_plan_requires_validation"):
        ValidatedSplitPlan()
    forged_core = ((((), ()),), (), (), "0" * 64, '{"inner_fold_count":999}')
    with pytest.raises(TypeError):
        tuple.__new__(ValidatedSplitPlan, forged_core)


def test_partition_evidence_is_opaque_fixed_and_nonserializable() -> None:
    plan = validate_split_manifest(_manifest(), **_classification_kwargs())
    evidence = plan._private_partition_evidence()
    assert isinstance(evidence, SplitPartitionEvidence)
    assert repr(evidence) == (
        "SplitPartitionEvidence(trusted_process_development_mechanics=True)"
    )
    assert "g0" not in repr(evidence)
    with pytest.raises(TypeError, match="split_partition_evidence_copy_forbidden"):
        copy(evidence)
    with pytest.raises(TypeError, match="split_partition_evidence_copy_forbidden"):
        deepcopy(evidence)
    with pytest.raises(TypeError, match="split_partition_evidence_serialization_forbidden"):
        pickle.dumps(evidence)
    with pytest.raises(TypeError):
        asdict(evidence)
    with pytest.raises(TypeError, match="split_partition_evidence_requires_validation"):
        SplitPartitionEvidence([], [], [], [], "0" * 64, _token=object())
    with pytest.raises(AttributeError, match="split_partition_evidence_immutable"):
        evidence._SplitPartitionEvidence__outer = ()


def test_partition_evidence_detaches_from_mutated_group_input() -> None:
    kwargs = _classification_kwargs()
    groups = kwargs["groups"]
    plan = validate_split_manifest(_manifest(), **kwargs)
    groups[:] = [f"changed{index}" for index in range(len(groups))]
    evidence = plan._private_partition_evidence()
    assert repr(evidence) == (
        "SplitPartitionEvidence(trusted_process_development_mechanics=True)"
    )
    _, outer, inner, _, _ = _partition_evidence_values(evidence)
    assert outer[0] == (
        frozenset({"g1", "g3", "g5", "g7"}),
        frozenset({"g0", "g2", "g4", "g6"}),
    )
    assert inner[0][0] == (frozenset({"g3", "g7"}), frozenset({"g1", "g5"}))
    serialized = json.dumps(plan.receipt(), sort_keys=True)
    assert "changed" not in serialized


def test_runtime_plan_rejects_invalid_outer_fold_selector() -> None:
    plan = validate_split_manifest(_manifest(), **_classification_kwargs())
    with pytest.raises(TypeError, match="outer_fold_type"):
        list(plan._private_inner_splits(True))
    with pytest.raises(IndexError, match="outer_fold_range"):
        list(plan._private_inner_splits(2))


def test_generic_partition_group_minima_are_explicit_and_watched() -> None:
    kwargs = _classification_kwargs()
    kwargs["minimum_training_groups"] = 3
    with _failure("partition_group_support", "c06_metric_support_insufficient"):
        validate_split_manifest(_manifest(), **kwargs)

    kwargs = _classification_kwargs()
    kwargs["minimum_assessment_groups"] = True
    with _failure("minimum_assessment_groups_type"):
        validate_split_manifest(_manifest(), **kwargs)

    kwargs = _classification_kwargs()
    del kwargs["minimum_training_groups"]
    with pytest.raises(TypeError, match="minimum_training_groups"):
        validate_split_manifest(_manifest(), **kwargs)


def test_aggregate_receipt_is_exact_deterministic_and_detached() -> None:
    plan = validate_split_manifest(_manifest(), **_classification_kwargs())
    expected = {
        "claim_id": "C06",
        "decision": "development_only",
        "eligibility_reason": "trusted_process_development_mechanics",
        "group_count": 8,
        "inner_fold_count": 2,
        "outer_fold_count": 2,
        "split_manifest_sha256": None,
        "split_manifest_status": "unavailable_pending_frozen_public_manifest",
        "support_summary": {
            "minimum_realized_assessment_group_count": 2,
            "minimum_realized_assessment_groups_per_class": 1,
            "minimum_realized_training_group_count": 2,
            "minimum_realized_training_groups_per_class": 1,
        },
        "verifier_status": (
            "trusted_process_development_mechanics_pending_frozen_public_manifest"
        ),
    }
    assert plan.receipt() == expected
    assert plan.receipt() == validate_split_manifest(
        _manifest(), **_classification_kwargs()
    ).receipt()
    detached = plan.receipt()
    detached["support_summary"]["minimum_realized_training_group_count"] = 999
    assert plan.receipt() == expected
