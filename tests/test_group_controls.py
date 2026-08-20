from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from omicau.models.group_controls import (
    GroupControlError,
    execute_group_endpoint_permutation,
    proxy_risk_receipt,
    scan_registered_exact_copy,
    validate_group_endpoint_permutation,
    validate_target_fit_ancestry,
)


SCHEMA = {"name": "site", "version": "1", "fields": ["site"], "target_derived": False}


def _private_registry(purpose: str) -> dict:
    return {
        "schema_version": "c07_private_registry_binding_v1",
        "purpose": purpose,
        "nonce_hex": hashlib.sha256(f"synthetic nonce for {purpose}".encode()).hexdigest(),
        "registry_id": f"synthetic_{purpose}",
        "artifact": {
            "artifact_id": "synthetic_fixture",
            "sha256": hashlib.sha256(b"synthetic artifact").hexdigest(),
        },
        "entries": [{
            "entry_id": "synthetic_entry",
            "artifact_sha256": hashlib.sha256(b"synthetic entry").hexdigest(),
        }],
    }


REGISTRY = _private_registry("group_endpoint_permutation")
CONTRACT = {"unit": "highest_exchangeable_group", "scope": "outer_train_only"}


def _base(task: str = "classification") -> dict:
    values = {
        "groups": [f"g{i // 2}" for i in range(12)],
        "strata": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
        "strata_schema": SCHEMA,
        "permutation_registry": REGISTRY,
        "exchangeability_contract": CONTRACT,
        "task": task,
        "outer_train_indices": list(range(8)),
        "minimum_distinct_nonidentity_assignments": 1,
    }
    if task == "classification":
        values["y"] = [0, 0, 1, 1, 2, 2, 3, 3, 7, 7, 8, 8]
    elif task == "regression":
        values["y"] = [float(i // 2) for i in range(12)]
    else:
        values["time"] = [float(i // 2 + 1) for i in range(12)]
        values["event"] = [i // 2 % 2 for i in range(12)]
    return values


def _run(values: dict, seed: int = 31, consume=None) -> tuple[dict, dict]:
    captured = {}
    if consume is None:
        consume = lambda result: captured.update({k: v.copy() for k, v in result.items()})
    receipt = execute_group_endpoint_permutation(**values, seed=seed, consume=consume)
    return receipt, captured


@pytest.mark.parametrize("task", ["classification", "regression", "survival"])
def test_permutation_is_deterministic_groupwise_and_training_only(task: str) -> None:
    values = _base(task)
    receipt, first = _run(values)
    _, second = _run(values)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert receipt["attainable_permutation_count"] == 3
    assert receipt["outer_train_group_count"] == 4
    assert receipt["permutation_registry_sha256"] is None
    assert receipt["permutation_registry_status"] == "unavailable_pending_frozen_registry"
    assert receipt["transform_role"] == "stress_control_only_not_inferential_randomization"
    for array in first.values():
        assert len(array) == 8
        assert all(array[index] == array[index + 1] for index in range(0, 8, 2))
    serialized = json.dumps(receipt, sort_keys=True)
    for forbidden in ("g0", "g1", "seed", "indices", "labels", "mapping", "subject", "path"):
        assert forbidden not in serialized.lower()


def test_independent_enumeration_oracle_accepts_generated_assignment() -> None:
    values = _base()
    _, captured = _run(values)
    observed = tuple(captured["y"].tolist())
    possible = {
        tuple(left + right)
        for left in ([0, 0, 1, 1], [1, 1, 0, 0])
        for right in ([2, 2, 3, 3], [3, 3, 2, 2])
    }
    possible.remove(tuple(values["y"][:8]))
    assert observed in possible


@pytest.mark.parametrize("task", ["classification", "regression", "survival"])
def test_duplicate_endpoint_objects_use_exact_multiset_count(task: str) -> None:
    values = _base(task)
    values["groups"] = [f"g{i}" for i in range(8)]
    values["strata"] = ["a"] * 3 + ["b"] * 3 + ["c"] * 2
    values["outer_train_indices"] = list(range(6))
    if task == "classification":
        values["y"] = [0, 0, 1, 2, 2, 3, 9, 9]
    elif task == "regression":
        values["y"] = [0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 9.0, 9.0]
    else:
        values["time"] = [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 9.0, 9.0]
        values["event"] = [0, 0, 1, 1, 1, 0, 0, 0]
    receipt, captured = _run(values)
    assert receipt["attainable_permutation_count"] == 8
    original = values.get("y", values.get("time"))[:6]
    candidate = captured.get("y", captured.get("time")).tolist()
    assert candidate != original
    if task == "survival":
        from collections import Counter

        before = Counter(zip(values["time"][:6], values["event"][:6]))
        after = Counter(zip(captured["time"].tolist(), captured["event"].tolist()))
        assert after == before


def test_independent_combinatorial_oracle_with_duplicate_values() -> None:
    from collections import Counter
    from math import factorial

    values = _base()
    values.update(
        groups=[f"g{i}" for i in range(8)],
        strata=["a"] * 4 + ["b"] * 3 + ["c"],
        outer_train_indices=list(range(7)),
        y=[0, 0, 1, 2, 3, 3, 4, 9],
    )
    receipt, _ = _run(values)
    block_values = ([0, 0, 1, 2], [3, 3, 4])
    oracle = 1
    for block in block_values:
        oracle *= factorial(len(block)) // __import__("math").prod(
            factorial(count) for count in Counter(block).values()
        )
    assert receipt["attainable_permutation_count"] == oracle - 1


def test_identity_value_only_and_configured_minimum_fail() -> None:
    values = _base()
    values["y"] = [0] * 8 + [7, 7, 8, 8]
    with pytest.raises(GroupControlError, match="c07_no_nontrivial_group_permutation"):
        _run(values)
    values = _base()
    values["minimum_distinct_nonidentity_assignments"] = 4
    with pytest.raises(GroupControlError, match="c07_no_nontrivial_group_permutation"):
        _run(values)


def test_attainable_count_is_python_integer_beyond_int64() -> None:
    count = 21
    values = _base()
    values.update(
        groups=[f"g{i}" for i in range(count + 1)],
        strata=["a"] * count + ["assessment"],
        outer_train_indices=list(range(count)),
        y=list(range(count)) + [99],
    )
    receipt, _ = _run(values)
    assert receipt["attainable_permutation_count"] == __import__("math").factorial(count) - 1
    assert receipt["attainable_permutation_count"] > np.iinfo(np.int64).max


def test_unequal_group_sizes_broadcast_whole_objects() -> None:
    values = _base()
    values.update(groups=["a", "a", "a", "b", "b", "c", "c", "c", "c", "d", "e", "e"])
    values.update(strata=["x"] * 5 + ["y"] * 7, y=[0] * 3 + [1] * 2 + [2] * 4 + [3] + [9] * 2)
    values["outer_train_indices"] = list(range(10))
    _, captured = _run(values)
    assert len(set(captured["y"][:3])) == len(set(captured["y"][3:5])) == 1
    assert len(set(captured["y"][5:9])) == 1


def test_watched_row_permutation_fails() -> None:
    values = _base()
    candidate = {"y": [0, 1, 0, 1, 2, 3, 2, 3]}
    with pytest.raises(GroupControlError, match="c07_permutation_scope_invalid") as caught:
        validate_group_endpoint_permutation(**values, candidate=candidate)
    assert caught.value.invariant == "candidate_rowwise_not_groupwise"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda x: x["y"].__setitem__(1, 9), "c07_group_outcome_mixed"),
        (lambda x: x.update(strata_schema={**SCHEMA, "target_derived": True}), "c07_stratum_target_derived"),
        (lambda x: x.update(strata=[str(i // 2) for i in range(12)]), "c07_no_nontrivial_group_permutation"),
        (lambda x: x.update(outer_train_indices=[0, 2, 3, 4, 5, 6, 7]), "c07_permutation_scope_invalid"),
        (lambda x: x.update(seed=True), "c07_contract_invalid"),
        (lambda x: x.update(minimum_distinct_nonidentity_assignments=True), "c07_contract_invalid"),
        (lambda x: x.update(minimum_distinct_nonidentity_assignments=1.0), "c07_contract_invalid"),
        (lambda x: x.update(minimum_distinct_nonidentity_assignments=0), "c07_contract_invalid"),
        (lambda x: x["y"].__setitem__(0, True), "c07_endpoint_invalid"),
    ],
)
def test_watched_endpoint_and_contract_failures(mutation, code: str) -> None:
    values = _base()
    values["seed"] = 3
    mutation(values)
    with pytest.raises(GroupControlError, match=code):
        execute_group_endpoint_permutation(**values, consume=lambda _: None)


def test_assessment_mutation_is_detected() -> None:
    values = _base()
    def mutate_assessment(_: dict) -> None:
        values["y"][10] = 99
    with pytest.raises(GroupControlError, match="c07_permutation_scope_invalid") as caught:
        _run(values, consume=mutate_assessment)
    assert caught.value.invariant == "assessment_outcomes_unchanged"


def _fit_registry() -> dict:
    return {"nodes": [
        {"node_id": "x", "parents": [], "target_dependent": False},
        {"node_id": "p", "parents": [], "target_dependent": False},
        {"node_id": "select", "parents": ["x", "p"], "target_dependent": True},
        {"node_id": "fit", "parents": ["select"], "target_dependent": True},
    ]}


def test_fit_ancestry_passes_and_emits_only_aggregates() -> None:
    receipt = validate_target_fit_ancestry(_fit_registry(), permuted_target_node="p", contract=CONTRACT)
    assert receipt["target_dependent_node_count"] == 2
    assert "select" not in json.dumps(receipt)


@pytest.mark.parametrize("defect", ["reuse", "unregistered", "cycle", "bool"])
def test_watched_fit_ancestry_failures(defect: str) -> None:
    registry = _fit_registry()
    if defect == "reuse":
        registry["nodes"][2]["parents"] = ["x"]
    elif defect == "unregistered":
        registry["nodes"][2]["parents"] = ["missing"]
    elif defect == "cycle":
        registry["nodes"][0]["parents"] = ["fit"]
    else:
        registry["nodes"][2]["target_dependent"] = 1
    with pytest.raises(GroupControlError, match="c07_fit_ancestry_invalid"):
        validate_target_fit_ancestry(registry, permuted_target_node="p", contract=CONTRACT)


def _scan(task: str, relation: str, candidate, **endpoint) -> dict:
    registry = {"task": task, "relations": [relation]}
    if relation == "frozen_risk_order_inversion":
        registry["minimum_comparable_pairs"] = 2
    return scan_registered_exact_copy(
        task=task,
        relation=relation,
        candidate=candidate,
        relation_registry=registry,
        contract={"scope": "registered_only"},
        **endpoint,
    )


@pytest.mark.parametrize(
    ("task", "relation", "candidate", "endpoint"),
    [
        ("classification", "equality", [0, 1, 0, 1], {"y": [0, 1, 0, 1]}),
        ("classification", "bijection", ["b", "a", "b", "a"], {"y": [0, 1, 0, 1]}),
        ("regression", "equality", [1.0, 2.0, 4.0], {"y": [1.0, 2.0, 4.0]}),
        ("regression", "nonzero_affine", [3.0, 5.0, 9.0], {"y": [1.0, 2.0, 4.0]}),
        ("survival", "joint_time_event_tuple", {"time": [1.0, 2.0], "event": [1, 0]}, {"time": [1.0, 2.0], "event": [1, 0]}),
        ("survival", "frozen_risk_order_inversion", [3.0, 2.0, 1.0], {"time": [1.0, 2.0, 3.0], "event": [1, 0, 1]}),
    ],
)
def test_registered_exact_copy_modes(task, relation, candidate, endpoint) -> None:
    receipt = _scan(task, relation, candidate, **endpoint)
    assert receipt["matched"] is True
    assert receipt["finding_kind"] == "exact_copy"


def test_unregistered_and_nonfinite_transforms_fail() -> None:
    with pytest.raises(GroupControlError, match="c07_unregistered_transform"):
        _scan("regression", "square", [1.0, 4.0], y=[1.0, 2.0])
    with pytest.raises(GroupControlError, match="c07_exact_copy_input_invalid"):
        _scan("regression", "equality", [1.0, np.inf], y=[1.0, 2.0])


@pytest.mark.parametrize(
    ("time", "risk"),
    [
        ([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]),
        ([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]),
        ([1.0, 2.0], [2.0, 1.0]),
        ([1.0, 1.0, 2.0], [3.0, 2.0, 1.0]),
    ],
)
def test_watched_noninformative_or_under_supported_risk_order_returns_not_detected(
    time: list[float], risk: list[float]
) -> None:
    receipt = _scan(
        "survival",
        "frozen_risk_order_inversion",
        risk,
        time=time,
        event=[1] * len(time),
    )
    assert receipt["matched"] is False
    assert receipt["decision"] == "not_detected"


def test_tied_but_informative_risk_order_agrees_with_independent_pairwise_oracle() -> None:
    time = [1.0, 1.0, 2.0, 3.0]
    risk = [3.0, 3.0, 2.0, 1.0]

    def oracle() -> tuple[bool, int]:
        comparisons = []
        for left in range(len(time)):
            for right in range(left + 1, len(time)):
                if time[left] == time[right]:
                    comparisons.append(risk[left] == risk[right])
                else:
                    comparisons.append((time[left] < time[right]) == (risk[left] > risk[right]))
        comparable = sum(
            time[left] != time[right]
            for left in range(len(time))
            for right in range(left + 1, len(time))
        )
        return all(comparisons), comparable

    oracle_match, comparable = oracle()
    receipt = _scan(
        "survival",
        "frozen_risk_order_inversion",
        risk,
        time=time,
        event=[1, 0, 1, 0],
    )
    assert oracle_match is True and comparable == 5
    assert receipt["matched"] is True


def test_risk_order_minimum_must_be_explicit_and_well_typed() -> None:
    base = {
        "task": "survival",
        "relation": "frozen_risk_order_inversion",
        "candidate": [3.0, 2.0, 1.0],
        "contract": {"scope": "registered_only"},
        "time": [1.0, 2.0, 3.0],
        "event": [1, 1, 1],
    }
    for minimum in (None, True, 0, 1.0):
        registry = {"task": "survival", "relations": ["frozen_risk_order_inversion"]}
        if minimum is not None:
            registry["minimum_comparable_pairs"] = minimum
        with pytest.raises(GroupControlError, match="c07_unregistered_transform"):
            scan_registered_exact_copy(relation_registry=registry, **base)


def test_proxy_alarm_is_never_called_leakage() -> None:
    receipt = proxy_risk_receipt(
        recoverability_alarm=True,
        calibration_registry=_private_registry("proxy_risk_calibration"),
        contract={"conclusion": "proxy_risk_only"},
    )
    assert receipt["finding_kind"] == "proxy_risk"
    assert receipt["leakage_conclusion"] is False
    assert "leakage" not in receipt["decision"]
    assert receipt["calibration_registry_commitment_status"] == "unavailable_pending_frozen_registry"
    assert not any("commitment_sha256" in key for key in receipt)


@pytest.mark.parametrize(
    "registry",
    [
        {"seed": 31},
        {"sha256": "ab" * 32},
        {},
        {"schema_version": "c07_private_registry_binding_v1"},
    ],
)
def test_proxy_registry_seed_only_digest_only_and_malformed_fail(registry: dict) -> None:
    with pytest.raises(GroupControlError, match="c07_contract_invalid"):
        proxy_risk_receipt(
            recoverability_alarm=True,
            calibration_registry=registry,
            contract={"conclusion": "proxy_risk_only"},
        )


@pytest.mark.parametrize("nonce", [None, "00" * 32, "ab" * 32, "01", "not-hex" * 8])
def test_weak_missing_or_malformed_private_nonce_fails(nonce: str | None) -> None:
    registry = _private_registry("proxy_risk_calibration")
    if nonce is None:
        registry.pop("nonce_hex")
    else:
        registry["nonce_hex"] = nonce
    with pytest.raises(GroupControlError, match="c07_contract_invalid"):
        proxy_risk_receipt(
            recoverability_alarm=False,
            calibration_registry=registry,
            contract={"conclusion": "proxy_risk_only"},
        )


def test_seed_bearing_public_contract_and_permutation_registry_fail() -> None:
    with pytest.raises(GroupControlError, match="c07_contract_invalid"):
        proxy_risk_receipt(
            recoverability_alarm=True,
            calibration_registry=_private_registry("proxy_risk_calibration"),
            contract={"seed": 31},
        )
    values = _base()
    values["permutation_registry"] = {"seed": 31}
    with pytest.raises(GroupControlError, match="c07_contract_invalid"):
        _run(values)


def test_contract_hashes_are_order_invariant_and_nonfinite_fails() -> None:
    first = _base()
    second = deepcopy(first)
    second["permutation_registry"] = dict(reversed(list(REGISTRY.items())))
    assert _run(first)[0]["permutation_registry_sha256"] is None
    assert _run(second)[0]["permutation_registry_sha256"] is None
    assert _run(first)[0]["permutation_registry_status"] == _run(second)[0]["permutation_registry_status"]
    first["exchangeability_contract"] = {"bad": np.nan}
    with pytest.raises(GroupControlError, match="c07_contract_invalid"):
        _run(first)
