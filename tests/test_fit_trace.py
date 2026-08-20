from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

import omicau.models.fit_trace as fit_trace_module
from omicau.models.fit_trace import (
    FitTraceError,
    StackingPredictionEvidence,
    _make_private_fit_node,
    canonical_state_sha256,
    validate_fit_trace,
    verify_poison_result,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _seed_registry(
    *, seed: int = 31, nonce: bytes | None = None
) -> dict[str, object]:
    return {
        "artifact_sha256": _digest("frozen-seed-registry"),
        "registry_nonce": (
            hashlib.sha256(b"private-registry-nonce").digest() if nonce is None else nonce
        ),
        "entries": [
            {
                "engine": "numpy",
                "purpose": "model_initialization",
                "seed": seed,
                "seed_id": "primary_rng",
            }
        ],
        "schema_version": "c08_private_seed_registry_v1",
    }


def _registries(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(record["node_digest"]): _seed_registry() for record in records}


def _private_node_fields(record: dict[str, object]) -> dict[str, object]:
    fields = copy.deepcopy(record["node"])
    fields.pop("seed_registry_status")
    fields["private_seed_registry"] = _seed_registry()
    return fields


def _node(
    callsite: str,
    learned_state_digest: str,
    *,
    fit_digest: str | None = None,
    parents: list[str] | None = None,
) -> dict[str, object]:
    return _make_private_fit_node(
        assessment_digest=_digest("assessment"),
        callsite=callsite,
        component=callsite.split(":")[-1],
        component_version="1.0",
        fit_digest=fit_digest or _digest("training"),
        fold="outer-0.inner-0",
        input_schema_digest=_digest("schema"),
        learned_state_digest=learned_state_digest,
        output_support={"feature_count": 3, "group_count": 4},
        parameters_digest=_digest("parameters"),
        parent_node_digests=[] if parents is None else parents,
        parent_split_digest=_digest("split"),
        private_seed_registry=_seed_registry(),
        stage="inner-training",
        target_use_flag=False,
        validation_digest=_digest("validation"),
    )


def _independent_states(train: np.ndarray) -> tuple[dict[str, object], ...]:
    imputer = {"feature_means": np.mean(train, axis=0)}
    imputed = np.where(np.isnan(train), imputer["feature_means"], train)
    scaler = {
        "feature_means": np.mean(imputed, axis=0),
        "feature_scales": np.std(imputed, axis=0),
    }
    standardized = (imputed - scaler["feature_means"]) / scaler["feature_scales"]
    support = {
        "feature_support": np.sum(np.isfinite(train), axis=0).astype(np.int64),
        "minimum_support": 2,
    }
    _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    pca = {"basis": vt[:2], "singular_values": singular_values[:2]}
    return imputer, scaler, support, pca


def _trace_fixture() -> tuple[
    list[dict[str, object]], dict[str, set[str]], dict[str, set[str]]
]:
    train = np.array(
        [[1.0, 2.0, 4.0], [2.0, 5.0, 3.0], [4.0, 8.0, 2.0], [5.0, 11.0, 1.0]]
    )
    states = _independent_states(train)
    callsites = [
        "omicau.models:imputer",
        "omicau.models:scaler",
        "omicau.models:support",
        "omicau.models:pca",
    ]
    records: list[dict[str, object]] = []
    parent: list[str] = []
    for callsite, state in zip(callsites, states, strict=True):
        contract = {"basis": 0} if callsite.endswith(":pca") else None
        record = _node(
            callsite,
            canonical_state_sha256(state, pca_basis_contract=contract),
            parents=parent,
        )
        records.append(record)
        parent = [str(record["node_digest"])]
    fit = {str(record["node_digest"]): {"train-a", "train-b"} for record in records}
    assessment = {str(record["node_digest"]): {"held-out"} for record in records}
    return records, fit, assessment


def test_independent_refits_trace_and_poison_semantics() -> None:
    records, fit, assessment = _trace_fixture()
    calls = {str(record["node"]["callsite"]) for record in records}
    receipt = validate_fit_trace(
        records,
        static_callsites=calls,
        fit_ancestry=fit,
        assessment_ancestry=assessment,
        private_seed_registries=_registries(records),
        cache_training_digests={
            str(records[1]["node_digest"]): str(records[1]["node"]["fit_digest"])
        },
        stacking_predictions=[
            StackingPredictionEvidence(
                str(records[-1]["node_digest"]), "held-out", frozenset({"train-a", "train-b"})
            )
        ],
    )
    assert receipt["decision"] == "development_only"
    assert receipt["seed_registry_status"] == "unavailable_pending_frozen_registry"
    assert receipt["verifier_status"] == "mechanics_verified_seed_registry_pending"
    assert receipt["node_count"] == 4
    assert set(receipt) == {
        "callsite_inventory_sha256",
        "claim_id",
        "decision",
        "fit_trace_sha256",
        "node_count",
        "seed_registry_status",
        "state_digest_count",
        "verifier_status",
    }
    assert "held-out" not in repr(receipt)

    train = np.array(
        [[1.0, 2.0, 4.0], [2.0, 5.0, 3.0], [4.0, 8.0, 2.0], [5.0, 11.0, 1.0]]
    )
    baseline_states = _independent_states(train)
    assessment_x = np.array([[2.5, 6.0, 2.5], [3.0, 7.0, 2.0]])
    poisoned_x = assessment_x.copy()
    poisoned_x[0, 0] += 100.0
    scaler = baseline_states[1]
    baseline_predictions = (
        (assessment_x - scaler["feature_means"]) / scaler["feature_scales"]
    ).sum(axis=1)
    poisoned_predictions = (
        (poisoned_x - scaler["feature_means"]) / scaler["feature_scales"]
    ).sum(axis=1)
    state_digest = canonical_state_sha256(scaler)
    assert not np.array_equal(baseline_predictions, poisoned_predictions)
    assert (
        verify_poison_result(
            "assessment_feature",
            baseline_state_digest=state_digest,
            poisoned_state_digest=canonical_state_sha256(_independent_states(train)[1]),
            baseline_predictions=baseline_predictions,
            poisoned_predictions=poisoned_predictions,
            baseline_sentinel_predictions=baseline_predictions[1:],
            poisoned_sentinel_predictions=poisoned_predictions[1:],
        )
        == "verified"
    )
    assert (
        verify_poison_result(
            "assessment_outcome",
            baseline_state_digest=state_digest,
            poisoned_state_digest=state_digest,
            baseline_predictions=baseline_predictions,
            poisoned_predictions=baseline_predictions.copy(),
        )
        == "verified"
    )


def test_state_digest_is_mapping_order_independent_and_exact() -> None:
    first = {"offset": np.array([0.0, -0.0]), "support": np.int64(4)}
    second = {"support": 4, "offset": np.array([0.0, -0.0])}
    changed = {"offset": np.array([-0.0, 0.0]), "support": 4}
    assert canonical_state_sha256(first) == canonical_state_sha256(second)
    assert canonical_state_sha256(first) != canonical_state_sha256(changed)


def test_pca_basis_sign_contract_is_explicit() -> None:
    basis = np.array([[-0.8, 0.6], [0.2, -0.9]])
    state = {"basis": basis, "variance": np.array([2.0, 1.0])}
    flipped = {"basis": -basis, "variance": np.array([2.0, 1.0])}
    assert canonical_state_sha256(state) != canonical_state_sha256(flipped)
    assert canonical_state_sha256(
        state, pca_basis_contract={"basis": 0}
    ) == canonical_state_sha256(flipped, pca_basis_contract={"basis": 0})


@pytest.mark.parametrize(
    "state",
    [
        {"value": True},
        {"value": np.array([1.0, np.nan])},
        {"value": np.array([1.0, np.inf])},
        {"value": np.array([True, False])},
        {"value": [1.0, 2.0]},
    ],
)
def test_uncanonicalizable_state_watched_failures(state: dict[str, object]) -> None:
    with pytest.raises(FitTraceError) as error:
        canonical_state_sha256(state)
    assert error.value.code == "c08_state_uncanonicalizable"


def test_static_runtime_inventory_missing_callsite_fails() -> None:
    records, fit, assessment = _trace_fixture()
    with pytest.raises(FitTraceError) as error:
        validate_fit_trace(
            records,
            static_callsites={"omicau.models:imputer"},
            fit_ancestry=fit,
            assessment_ancestry=assessment,
            private_seed_registries=_registries(records),
        )
    assert error.value.code == "c08_static_runtime_inventory_mismatch"


def test_empty_trace_fails() -> None:
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete"):
        validate_fit_trace(
            [],
            static_callsites={"omicau.models:imputer"},
            fit_ancestry={},
            assessment_ancestry={},
            private_seed_registries={},
        )


def test_assessment_group_in_fit_ancestry_fails() -> None:
    records, fit, assessment = _trace_fixture()
    fit[str(records[0]["node_digest"])].add("held-out")
    with pytest.raises(FitTraceError) as error:
        validate_fit_trace(
            records,
            static_callsites={str(record["node"]["callsite"]) for record in records},
            fit_ancestry=fit,
            assessment_ancestry=assessment,
            private_seed_registries=_registries(records),
        )
    assert error.value.code == "c08_assessment_ancestry_detected"


def test_cached_state_wrong_training_digest_fails() -> None:
    records, fit, assessment = _trace_fixture()
    with pytest.raises(FitTraceError) as error:
        validate_fit_trace(
            records,
            static_callsites={str(record["node"]["callsite"]) for record in records},
            fit_ancestry=fit,
            assessment_ancestry=assessment,
            private_seed_registries=_registries(records),
            cache_training_digests={str(records[0]["node_digest"]): _digest("wrong-train")},
        )
    assert error.value.code == "c08_cache_training_digest_mismatch"


def test_duplicate_node_fails() -> None:
    records, _, _ = _trace_fixture()
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete") as error:
        validate_fit_trace(
            [records[0], copy.deepcopy(records[0])],
            static_callsites={str(records[0]["node"]["callsite"])},
            fit_ancestry={},
            assessment_ancestry={},
            private_seed_registries={},
        )
    assert error.value.invariant == "node_digest_unique"


def test_cycle_fails_before_untrusted_digest_acceptance() -> None:
    state = canonical_state_sha256({"value": 1.0})
    first = _node("omicau.models:model", state)
    second = _node("omicau.models:threshold", state)
    first["node_digest"] = "a" * 64
    second["node_digest"] = "b" * 64
    first["node"]["parent_node_digests"] = ["b" * 64]
    second["node"]["parent_node_digests"] = ["a" * 64]
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete") as error:
        validate_fit_trace(
            [first, second],
            static_callsites={"omicau.models:model", "omicau.models:threshold"},
            fit_ancestry={},
            assessment_ancestry={},
            private_seed_registries={},
        )
    assert error.value.invariant == "trace_cycle"


def test_raw_private_field_seed_and_path_disclosure_fail() -> None:
    valid = _node("omicau.models:imputer", canonical_state_sha256({"value": 1.0}))
    private = _private_node_fields(valid)
    private["labels"] = [0, 1]
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete"):
        _make_private_fit_node(**private)

    raw_seed = _private_node_fields(valid)
    raw_seed["seed_state_digest"] = 42
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete"):
        _make_private_fit_node(**raw_seed)

    local_path = _private_node_fields(valid)
    local_path["callsite"] = r"C:\private\fit.py"
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete"):
        _make_private_fit_node(**local_path)


@pytest.mark.parametrize(
    ("field", "hostile"),
    [
        ("callsite", "C:private"),
        ("callsite", "omicau.models:endpoint"),
        ("callsite", "omicau.models:password-secret"),
        ("component", "batch-label"),
        ("stage", "group-a"),
    ],
)
def test_hostile_or_unregistered_node_identifier_fails(field: str, hostile: str) -> None:
    record = _node("omicau.models:imputer", canonical_state_sha256({"value": 1.0}))
    fields = _private_node_fields(record)
    fields[field] = hostile
    with pytest.raises(FitTraceError, match="c08_"):
        _make_private_fit_node(**fields)


@pytest.mark.parametrize("hostile", ["endpoint", "batch_label", "password_secret", "C:private"])
def test_hostile_or_unregistered_output_support_key_fails(hostile: str) -> None:
    record = _node("omicau.models:imputer", canonical_state_sha256({"value": 1.0}))
    fields = _private_node_fields(record)
    fields["output_support"] = {hostile: 1}
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete"):
        _make_private_fit_node(**fields)


def test_raw_node_builder_is_not_public_api() -> None:
    assert not hasattr(fit_trace_module, "make_fit_node")


def test_arbitrary_seed_digest_cannot_be_supplied() -> None:
    original = _node("omicau.models:imputer", canonical_state_sha256({"value": 1.0}))
    fields = _private_node_fields(original)
    fields["seed_state_digest"] = _digest("31")
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete") as error:
        _make_private_fit_node(**fields)
    assert error.value.invariant == "private_node_input_schema"


@pytest.mark.parametrize(
    "mutation",
    [
        {"registry_nonce": b"short"},
        {"schema_version": "c08_private_seed_registry_v0"},
        {"entries": []},
    ],
)
def test_malformed_private_seed_registry_fails(mutation: dict[str, object]) -> None:
    registry = _seed_registry()
    registry.update(mutation)
    fields = _private_node_fields(
        _node("omicau.models:imputer", canonical_state_sha256({"value": 1.0}))
    )
    fields["private_seed_registry"] = registry
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete"):
        _make_private_fit_node(**fields)


def test_private_seed_and_nonce_do_not_affect_public_receipt() -> None:
    state = canonical_state_sha256({"value": 1.0})
    first_registry = _seed_registry(seed=123456789012345678, nonce=b"x" * 32)
    second_registry = _seed_registry(seed=987654321098765432, nonce=b"y" * 32)
    first_fields = _private_node_fields(_node("omicau.models:imputer", state))
    second_fields = copy.deepcopy(first_fields)
    first_fields["private_seed_registry"] = first_registry
    second_fields["private_seed_registry"] = second_registry
    first = _make_private_fit_node(**first_fields)
    second = _make_private_fit_node(**second_fields)
    assert first == second

    digest = str(first["node_digest"])
    common = {
        "static_callsites": {"omicau.models:imputer"},
        "fit_ancestry": {digest: {"train"}},
        "assessment_ancestry": {digest: {"held-out"}},
    }
    first_receipt = validate_fit_trace(
        [first], private_seed_registries={digest: first_registry}, **common
    )
    second_receipt = validate_fit_trace(
        [second], private_seed_registries={digest: second_registry}, **common
    )
    assert first_receipt == second_receipt
    public = repr({"record": first, "receipt": first_receipt})
    assert "123456789012345678" not in public
    assert "987654321098765432" not in public
    assert first_registry["registry_nonce"].hex() not in public
    assert second_registry["registry_nonce"].hex() not in public


def test_bool_output_support_is_rejected() -> None:
    node = _node("omicau.models:imputer", canonical_state_sha256({"value": 1.0}))
    fields = _private_node_fields(node)
    fields["output_support"] = {"feature_count": True}
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete"):
        _make_private_fit_node(**fields)


def test_feature_poison_changed_sentinel_is_watched_failure() -> None:
    state = canonical_state_sha256({"value": 1.0})
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete") as error:
        verify_poison_result(
            "assessment_feature",
            baseline_state_digest=state,
            poisoned_state_digest=state,
            baseline_predictions=np.array([1.0, 2.0]),
            poisoned_predictions=np.array([3.0, 4.0]),
            baseline_sentinel_predictions=np.array([2.0]),
            poisoned_sentinel_predictions=np.array([4.0]),
        )
    assert error.value.invariant == "assessment_feature_sentinel_prediction_invariance"


def test_outcome_poison_prediction_or_state_drift_fails() -> None:
    state = canonical_state_sha256({"value": 1.0})
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete") as prediction_error:
        verify_poison_result(
            "assessment_outcome",
            baseline_state_digest=state,
            poisoned_state_digest=state,
            baseline_predictions=np.array([1.0]),
            poisoned_predictions=np.array([2.0]),
        )
    assert prediction_error.value.invariant == "assessment_outcome_prediction_invariance"
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete") as state_error:
        verify_poison_result(
            "assessment_outcome",
            baseline_state_digest=state,
            poisoned_state_digest=_digest("changed"),
            baseline_predictions=np.array([1.0]),
            poisoned_predictions=np.array([1.0]),
        )
    assert state_error.value.invariant == "poison_learned_state_invariance"


def test_stacking_predicted_group_must_be_excluded() -> None:
    records, fit, assessment = _trace_fixture()
    with pytest.raises(FitTraceError) as error:
        validate_fit_trace(
            records,
            static_callsites={str(record["node"]["callsite"]) for record in records},
            fit_ancestry=fit,
            assessment_ancestry=assessment,
            private_seed_registries=_registries(records),
            stacking_predictions=[
                StackingPredictionEvidence(
                    str(records[-1]["node_digest"]),
                    "held-out",
                    frozenset({"train-a", "held-out"}),
                )
            ],
        )
    assert error.value.code == "c08_stacking_in_group_fit"
