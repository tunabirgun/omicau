from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from itertools import product
import json

import numpy as np
import pytest

from omicau.models.split_plan import (
    SplitValidationError,
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
        "minimum_survival_training_event_groups": 1,
        "minimum_survival_assessment_comparable_pairs": 1,
    }


@pytest.mark.parametrize("kwargs", [_classification_kwargs(), _regression_kwargs(), _survival_kwargs()])
def test_valid_exact_nested_plan_emits_only_public_safe_aggregates(kwargs: dict) -> None:
    manifest = _manifest()
    receipt = validate_split_manifest(manifest, **kwargs)
    assert set(receipt) == {
        "claim_id", "decision", "eligibility_reason", "group_count", "inner_fold_count",
        "outer_fold_count", "split_manifest_sha256", "support_summary", "verifier_status",
    }
    assert receipt["decision"] == "eligible"
    assert receipt["outer_fold_count"] == receipt["inner_fold_count"] == 2
    assert receipt["split_manifest_sha256"] == canonical_split_manifest_sha256(manifest)
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
    assert validate_split_manifest(witness, **kwargs)["verifier_status"] == "verified"
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
    receipt = validate_split_manifest(_expanded_manifest(), **_repeat_rows(kwargs))
    assert receipt["verifier_status"] == "verified"
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
