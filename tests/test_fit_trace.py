from __future__ import annotations

import copy
from collections import Counter
from dataclasses import asdict
import hashlib
import pickle

import numpy as np
import pytest

import omicau.models.fit_trace as fit_trace_module
from omicau.models.fit_trace import (
    CacheUseEvidence,
    FitTraceError,
    StackingPredictionEvidence,
    _cache_identity,
    _make_private_fit_node,
    canonical_state_sha256,
    validate_fit_trace,
    verify_cache_use,
    verify_stacking_prediction,
)
from omicau.models.split_plan import (
    ValidatedSplitPlan,
    _canonical_index_sha256,
    canonical_split_manifest_sha256,
    validate_split_manifest,
)


_CALLSITE_COMPONENTS = {
    "base.calibrator": "calibrator",
    "base.feature_selector": "feature_selector",
    "base.imputer": "imputer",
    "base.latent_factor": "latent_factor",
    "base.model": "model",
    "base.pca": "pca",
    "base.scaler": "scaler",
    "base.support": "support",
    "base.threshold": "threshold",
    "base.variance_filter": "variance_filter",
    "batch.adjuster": "batch_adjuster",
    "neural.optimizer": "optimizer",
    "neural.scaler": "scaler",
    "stacking.stacker": "stacker",
    "survival.cox": "cox",
    "survival.imputer": "imputer",
    "survival.pca": "pca",
    "survival.scaler": "scaler",
}
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


def _split_manifest(*, alternate: bool = False) -> dict[str, object]:
    manifest = {
        "outer_folds": [
            {
                "assessment": [0, 1, 2, 3],
                "inner_folds": [
                    {"assessment": [4, 5], "train": [6, 7]},
                    {"assessment": [6, 7], "train": [4, 5]},
                ],
                "train": [4, 5, 6, 7],
            },
            {
                "assessment": [4, 5, 6, 7],
                "inner_folds": [
                    {"assessment": [0, 1], "train": [2, 3]},
                    {"assessment": [2, 3], "train": [0, 1]},
                ],
                "train": [0, 1, 2, 3],
            },
        ]
    }
    if alternate:
        manifest = {
            "outer_folds": [
                {
                    "assessment": [0, 1, 4, 5],
                    "inner_folds": [
                        {"assessment": [2, 3], "train": [6, 7]},
                        {"assessment": [6, 7], "train": [2, 3]},
                    ],
                    "train": [2, 3, 6, 7],
                },
                {
                    "assessment": [2, 3, 6, 7],
                    "inner_folds": [
                        {"assessment": [0, 1], "train": [4, 5]},
                        {"assessment": [4, 5], "train": [0, 1]},
                    ],
                    "train": [0, 1, 4, 5],
                },
            ]
        }
    return manifest


def _split_plan(*, alternate: bool = False) -> ValidatedSplitPlan:
    return validate_split_manifest(
        _split_manifest(alternate=alternate),
        **_split_validation_spec(),
    )


def _split_validation_spec() -> dict[str, object]:
    return {
        "event": None,
        "groups": [f"g{index}" for index in range(8)],
        "minimum_assessment_groups": 2,
        "minimum_assessment_groups_per_class": 1,
        "minimum_regression_assessment_groups": None,
        "minimum_regression_assessment_variance": None,
        "minimum_survival_assessment_comparable_pairs": None,
        "minimum_survival_training_event_groups": None,
        "minimum_training_groups": 2,
        "minimum_training_groups_per_class": 1,
        "n_samples": 8,
        "requested_inner_k": 2,
        "requested_outer_k": 2,
        "task": "classification",
        "time": None,
        "y": [0, 1, 0, 1, 0, 1, 0, 1],
    }


def _split_digest(*, alternate: bool = False) -> str:
    return canonical_split_manifest_sha256(_split_manifest(alternate=alternate))


def _poison_cases() -> dict[str, dict[str, object]]:
    state = _digest("poison-state")
    return {
        "assessment_feature": {
            "baseline_predictions": np.array([1.0, 2.0]),
            "baseline_sentinel_predictions": np.array([2.0]),
            "baseline_state_digest": state,
            "poisoned_predictions": np.array([3.0, 2.0]),
            "poisoned_sentinel_predictions": np.array([2.0]),
            "poisoned_state_digest": state,
        },
        "assessment_outcome": {
            "baseline_predictions": np.array([1.0, 2.0]),
            "baseline_state_digest": state,
            "poisoned_predictions": np.array([1.0, 2.0]),
            "poisoned_state_digest": state,
        },
    }


def _private_node_fields(record: dict[str, object]) -> dict[str, object]:
    fields = copy.deepcopy(record["node"])
    fields.pop("seed_registry_status")
    fields["private_seed_registry"] = _seed_registry()
    return fields


def _partition_digests(
    fold: str, *, alternate: bool = False
) -> tuple[str, str]:
    outer_text, *inner_text = fold.split(".inner-")
    outer_index = int(outer_text.removeprefix("outer-"))
    outer = _split_manifest(alternate=alternate)["outer_folds"][outer_index]
    partition = outer if not inner_text else outer["inner_folds"][int(inner_text[0])]
    return (
        _canonical_index_sha256(partition["train"]),
        _canonical_index_sha256(partition["assessment"]),
    )


def _node(
    callsite_id: str,
    learned_state_digest: str,
    *,
    alternate_split: bool = False,
    assessment_digest: str | None = None,
    fit_digest: str | None = None,
    fold: str = "outer-0.inner-0",
    parents: list[str] | None = None,
    stage: str = "inner-training",
) -> dict[str, object]:
    if fit_digest is not None and assessment_digest is not None:
        expected_fit, expected_assessment = fit_digest, assessment_digest
    else:
        expected_fit, expected_assessment = _partition_digests(
            fold, alternate=alternate_split
        )
    return _make_private_fit_node(
        assessment_digest=assessment_digest or expected_assessment,
        callsite_id=callsite_id,
        code_digest=_digest("source-code"),
        component=_CALLSITE_COMPONENTS[callsite_id],
        component_version="1.0",
        environment_digest=_digest("environment-lock"),
        fit_digest=fit_digest or expected_fit,
        fold=fold,
        input_schema_digest=_digest("schema"),
        learned_state_digest=learned_state_digest,
        output_support={"feature_count": 3, "group_count": 4},
        parameters_digest=_digest("parameters"),
        parent_node_digests=[] if parents is None else parents,
        parent_split_digest=_split_digest(alternate=alternate_split),
        private_seed_registry=_seed_registry(),
        stage=stage,
        state_schema_digest=_digest("state-schema"),
        target_use_flag=False,
        validation_digest=_digest("validation"),
    )


def _execution_profile(records: list[dict[str, object]]) -> dict[str, object]:
    counts = Counter(str(record["node"]["callsite_id"]) for record in records)
    return {
        "planned_counts": {
            callsite_id: counts[callsite_id]
            for callsite_id in sorted(_CALLSITE_COMPONENTS)
        },
        "schema_version": "c08_development_execution_profile_v1",
    }


def _unused_cache(records: list[dict[str, object]]) -> dict[str, CacheUseEvidence]:
    return {
        str(record["node_digest"]): verify_cache_use(
            "unused", node=record["node"]
        )
        for record in records
    }


def _validation_kwargs(
    records: list[dict[str, object]],
    fit: dict[str, set[str]],
    assessment: dict[str, set[str]],
) -> dict[str, object]:
    return {
        "cache_evidence": _unused_cache(records),
        "execution_profile": _execution_profile(records),
        "poison_cases": _poison_cases(),
        "private_seed_registries": _registries(records),
        "split_manifest": _split_manifest(),
        "split_plan": _split_plan(),
        "split_validation_spec": _split_validation_spec(),
    }


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
    callsite_ids = [
        "base.imputer",
        "base.scaler",
        "base.support",
        "base.pca",
    ]
    records: list[dict[str, object]] = []
    parent: list[str] = []
    for callsite_id, state in zip(callsite_ids, states, strict=True):
        contract = {"basis": 0} if callsite_id.endswith(".pca") else None
        record = _node(
            callsite_id,
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
    receipt = validate_fit_trace(
        records,
        **_validation_kwargs(records, fit, assessment),
    )
    assert receipt["decision"] == "development_only"
    assert receipt["seed_registry_status"] == "unavailable_pending_frozen_registry"
    assert receipt["verifier_status"] == (
        "trusted_process_development_mechanics_pending_production_inventory"
    )
    assert receipt["node_count"] == 4
    assert set(receipt) == {
        "callsite_inventory_sha256",
        "claim_id",
        "decision",
        "redacted_fit_trace_sha256",
        "node_count",
        "poison_case_status",
        "seed_registry_status",
        "split_manifest_sha256",
        "split_manifest_status",
        "state_digest_count",
        "verifier_status",
    }
    assert receipt["poison_case_status"] == (
        "mechanics_checked_from_supplied_cases_pending_production_binding"
    )
    assert receipt["split_manifest_sha256"] is None
    assert receipt["split_manifest_status"] == (
        "unavailable_pending_frozen_public_manifest"
    )
    assert _split_digest() not in repr(receipt)
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
    poison_cases = {
        "assessment_feature": {
            "baseline_predictions": baseline_predictions,
            "baseline_sentinel_predictions": baseline_predictions[1:],
            "baseline_state_digest": state_digest,
            "poisoned_predictions": poisoned_predictions,
            "poisoned_sentinel_predictions": poisoned_predictions[1:],
            "poisoned_state_digest": canonical_state_sha256(
                _independent_states(train)[1]
            ),
        },
        "assessment_outcome": {
            "baseline_predictions": baseline_predictions,
            "baseline_state_digest": state_digest,
            "poisoned_predictions": baseline_predictions.copy(),
            "poisoned_state_digest": state_digest,
        },
    }
    kwargs = _validation_kwargs(records, fit, assessment)
    kwargs["poison_cases"] = poison_cases
    assert validate_fit_trace(records, **kwargs)["decision"] == "development_only"


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
    profile = _execution_profile(records)
    del profile["planned_counts"]["base.pca"]
    with pytest.raises(FitTraceError) as error:
        validate_fit_trace(
            records,
            **{
                **_validation_kwargs(records, fit, assessment),
                "execution_profile": profile,
            },
        )
    assert error.value.code == "c08_static_runtime_inventory_mismatch"


def test_execution_profile_cannot_change_authoritative_dispositions() -> None:
    records, fit, assessment = _trace_fixture()
    profile = _execution_profile(records)
    profile["dispositions"] = {"base.pca": "excluded"}
    with pytest.raises(FitTraceError) as error:
        validate_fit_trace(
            records,
            **{
                **_validation_kwargs(records, fit, assessment),
                "execution_profile": profile,
            },
        )
    assert error.value.code == "c08_static_runtime_inventory_mismatch"


def test_duplicate_runtime_callsite_and_planned_count_drift_fail() -> None:
    records, fit, assessment = _trace_fixture()
    duplicate = _node(
        "base.imputer", canonical_state_sha256({"value": 2.0})
    )
    duplicate_digest = str(duplicate["node_digest"])
    duplicated_records = [*records, duplicate]
    fit[duplicate_digest] = {"train-a", "train-b"}
    assessment[duplicate_digest] = {"held-out"}
    kwargs = _validation_kwargs(duplicated_records, fit, assessment)
    kwargs["execution_profile"] = _execution_profile(records)
    with pytest.raises(FitTraceError) as duplicate_error:
        validate_fit_trace(duplicated_records, **kwargs)
    assert duplicate_error.value.code == "c08_static_runtime_inventory_mismatch"

    _, fresh_fit, fresh_assessment = _trace_fixture()
    kwargs = _validation_kwargs(records, fresh_fit, fresh_assessment)
    profile = copy.deepcopy(kwargs["execution_profile"])
    profile["planned_counts"]["base.imputer"] = 2
    kwargs["execution_profile"] = profile
    with pytest.raises(FitTraceError) as count_error:
        validate_fit_trace(records, **kwargs)
    assert count_error.value.code == "c08_static_runtime_inventory_mismatch"


def test_empty_trace_fails() -> None:
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete"):
        validate_fit_trace([], **{
            "cache_evidence": {},
            "execution_profile": _execution_profile([]),
            "poison_cases": _poison_cases(),
            "private_seed_registries": {},
            "split_manifest": _split_manifest(),
            "split_plan": _split_plan(),
            "split_validation_spec": _split_validation_spec(),
        })


def test_caller_supplied_ancestry_is_rejected() -> None:
    records, fit, assessment = _trace_fixture()
    kwargs = _validation_kwargs(records, fit, assessment)
    kwargs["fit_ancestry"] = {str(records[0]["node_digest"]): {"forged"}}
    with pytest.raises(TypeError, match="fit_ancestry"):
        validate_fit_trace(
            records,
            **kwargs,
        )
    kwargs = _validation_kwargs(records, fit, assessment)
    kwargs["assessment_ancestry"] = {
        str(records[0]["node_digest"]): {"forged"}
    }
    with pytest.raises(TypeError, match="assessment_ancestry"):
        validate_fit_trace(records, **kwargs)


def test_cached_state_wrong_training_digest_fails() -> None:
    records, fit, assessment = _trace_fixture()
    identity = _cache_identity(records[0]["node"])
    identity["fit_digest"] = _digest("wrong-train")
    with pytest.raises(FitTraceError) as error:
        verify_cache_use("loaded", node=records[0]["node"], cached_identity=identity)
    assert error.value.code == "c08_cache_training_digest_mismatch"


def test_cache_use_requires_exact_coverage_and_status_identity_contract() -> None:
    records, fit, assessment = _trace_fixture()
    first_digest = str(records[0]["node_digest"])
    kwargs = _validation_kwargs(records, fit, assessment)
    del kwargs["cache_evidence"][first_digest]
    with pytest.raises(FitTraceError) as missing_error:
        validate_fit_trace(records, **kwargs)
    assert missing_error.value.invariant == "cache_use_node_coverage"

    for status, identity, invariant in [
        (
            "unused",
            _cache_identity(records[0]["node"]),
            "unused_cache_identity_forbidden",
        ),
        ("loaded", None, "used_cache_identity_required"),
        ("caller_verified", None, "cache_use_schema"),
    ]:
        with pytest.raises(FitTraceError) as error:
            verify_cache_use(
                status, node=records[0]["node"], cached_identity=identity
            )
        assert error.value.invariant == invariant

    kwargs = _validation_kwargs(records, fit, assessment)
    kwargs["cache_evidence"][first_digest] = verify_cache_use(
        "reused",
        node=records[0]["node"],
        cached_identity=_cache_identity(records[0]["node"]),
    )
    assert validate_fit_trace(records, **kwargs)["decision"] == "development_only"


def test_private_evidence_objects_fail_closed_under_object_protocols() -> None:
    record = _node("base.imputer", canonical_state_sha256({"value": 1.0}))
    cache = verify_cache_use("unused", node=record["node"])
    stacking = verify_stacking_prediction(
        str(record["node_digest"]),
        "private-predicted-group",
        frozenset({"private-fit-group"}),
    )
    for evidence in (cache, stacking):
        rendered = repr(evidence)
        assert "private-" not in rendered
        assert _digest("training") not in rendered
        for operation in (
            copy.copy,
            copy.deepcopy,
            asdict,
            pickle.dumps,
        ):
            with pytest.raises(TypeError) as error:
                operation(evidence)
            assert "private-" not in str(error.value)


def test_cache_evidence_is_immutable_and_remains_valid_after_attacks() -> None:
    record = _node("base.imputer", canonical_state_sha256({"value": 1.0}))
    digest = str(record["node_digest"])
    kwargs = _validation_kwargs([record], {digest: {"unused"}}, {digest: {"unused"}})
    evidence = kwargs["cache_evidence"][digest]
    marker = "credential-private-marker"
    for name in ("_CacheUseEvidence__status", "extra"):
        with pytest.raises(AttributeError, match="cache_use_evidence_immutable") as error:
            setattr(evidence, name, marker)
        assert marker not in str(error.value)
    assert validate_fit_trace([record], **kwargs)["decision"] == "development_only"


def test_private_evidence_constructors_and_hostile_cache_identity_fail_closed() -> None:
    marker = "credential-private-marker"
    with pytest.raises(TypeError) as stacking_error:
        StackingPredictionEvidence(
            _digest("node"), marker, frozenset({marker}), _token=object()
        )
    assert marker not in str(stacking_error.value)
    with pytest.raises(TypeError) as cache_error:
        CacheUseEvidence("loaded", {"identity": marker}, _token=object())
    assert marker not in str(cache_error.value)

    record = _node("base.imputer", canonical_state_sha256({"value": 1.0}))
    with pytest.raises(FitTraceError) as identity_error:
        verify_cache_use(
            "loaded", node=record["node"], cached_identity={"identity": marker}
        )
    assert marker not in str(identity_error.value)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("assessment_digest", _digest("changed-assessment")),
        ("callsite_id", "base.scaler"),
        ("code_digest", _digest("changed-code")),
        ("component", "scaler"),
        ("component_version", "2.0"),
        ("environment_digest", _digest("changed-environment")),
        ("fit_digest", _digest("changed-fit")),
        ("fold", "outer-1.inner-0"),
        ("input_schema_digest", _digest("changed-input-schema")),
        ("learned_state_digest", _digest("changed-state")),
        ("output_support", {"feature_count": 9}),
        ("parameters_digest", _digest("changed-parameters")),
        ("parent_node_digests", [_digest("changed-parent")]),
        ("parent_split_digest", _digest("changed-split")),
        ("seed_registry_status", "registry_available"),
        ("stage", "outer-training"),
        ("state_schema_digest", _digest("changed-state-schema")),
        ("target_use_flag", True),
        ("validation_digest", _digest("changed-validation")),
    ],
)
def test_cache_identity_field_drift_fails(field: str, changed: object) -> None:
    records, fit, assessment = _trace_fixture()
    identity = _cache_identity(records[0]["node"])
    identity[field] = changed
    with pytest.raises(FitTraceError) as error:
        verify_cache_use("reused", node=records[0]["node"], cached_identity=identity)
    assert error.value.code == "c08_cache_training_digest_mismatch"


@pytest.mark.parametrize(
    "omitted",
    ["split_manifest", "split_plan", "split_validation_spec", "poison_cases"],
)
def test_split_or_poison_receipt_input_omission_fails(omitted: str) -> None:
    records, fit, assessment = _trace_fixture()
    kwargs = _validation_kwargs(records, fit, assessment)
    kwargs.pop(omitted)
    with pytest.raises(TypeError):
        validate_fit_trace(records, **kwargs)


def test_split_manifest_drift_fails() -> None:
    records, fit, assessment = _trace_fixture()
    kwargs = _validation_kwargs(records, fit, assessment)
    kwargs["split_plan"] = _split_plan(alternate=True)
    with pytest.raises(FitTraceError) as error:
        validate_fit_trace(records, **kwargs)
    assert error.value.invariant == "caller_split_partitions_exact"


def test_raw_split_manifest_and_validation_spec_must_match_caller_plan() -> None:
    records, fit, assessment = _trace_fixture()
    kwargs = _validation_kwargs(records, fit, assessment)
    kwargs["split_manifest"] = _split_manifest(alternate=True)
    with pytest.raises(FitTraceError) as manifest_error:
        validate_fit_trace(records, **kwargs)
    assert manifest_error.value.invariant == "caller_split_partitions_exact"

    kwargs = _validation_kwargs(records, fit, assessment)
    kwargs["split_validation_spec"]["groups"] = list(
        reversed(kwargs["split_validation_spec"]["groups"])
    )
    with pytest.raises(FitTraceError) as spec_mismatch_error:
        validate_fit_trace(records, **kwargs)
    assert spec_mismatch_error.value.invariant == "caller_split_partitions_exact"

    kwargs = _validation_kwargs(records, fit, assessment)
    kwargs["split_validation_spec"]["unexpected"] = None
    with pytest.raises(FitTraceError) as spec_error:
        validate_fit_trace(records, **kwargs)
    assert spec_error.value.invariant == "split_validation_spec_exact"


def test_candidate_manifests_leave_no_public_trace_commitment() -> None:
    state = canonical_state_sha256({"value": 1.0})
    first = _node("base.imputer", state)
    second = _node("base.imputer", state, alternate_split=True)
    first_digest = str(first["node_digest"])
    second_digest = str(second["node_digest"])
    first_receipt = validate_fit_trace(
        [first],
        **_validation_kwargs(
            [first], {first_digest: {"unused"}}, {first_digest: {"unused"}}
        ),
    )
    second_kwargs = _validation_kwargs(
        [second], {second_digest: {"unused"}}, {second_digest: {"unused"}}
    )
    second_kwargs["split_manifest"] = _split_manifest(alternate=True)
    second_kwargs["split_plan"] = _split_plan(alternate=True)
    second_receipt = validate_fit_trace([second], **second_kwargs)
    assert first_receipt == second_receipt
    assert first_receipt["split_manifest_sha256"] is None
    public = repr(first_receipt)
    assert _split_digest() not in public
    assert _split_digest(alternate=True) not in public


@pytest.mark.parametrize("field", ["fit_digest", "assessment_digest"])
def test_node_index_partition_digest_mismatch_fails(field: str) -> None:
    kwargs = {field: _digest(f"wrong-{field}")}
    record = _node(
        "base.imputer",
        canonical_state_sha256({"value": 1.0}),
        **kwargs,
    )
    digest = str(record["node_digest"])
    with pytest.raises(FitTraceError) as error:
        validate_fit_trace(
            [record],
            **_validation_kwargs(
                [record], {digest: {"unused"}}, {digest: {"unused"}}
            ),
        )
    assert error.value.invariant == "node_index_partition_digest_exact"
    assert error.value.code == "c08_assessment_ancestry_detected"


def test_equal_fit_and_assessment_index_digests_fail() -> None:
    _, assessment_digest = _partition_digests("outer-0.inner-0")
    record = _node(
        "base.imputer",
        canonical_state_sha256({"value": 1.0}),
        fit_digest=assessment_digest,
        assessment_digest=assessment_digest,
    )
    digest = str(record["node_digest"])
    with pytest.raises(FitTraceError) as error:
        validate_fit_trace(
            [record],
            **_validation_kwargs(
                [record], {digest: {"unused"}}, {digest: {"unused"}}
            ),
        )
    assert error.value.invariant == "node_index_partition_digest_exact"


def test_stage_fold_selector_and_plan_ranges_are_watched() -> None:
    valid = _node("base.imputer", canonical_state_sha256({"value": 1.0}))
    fields = _private_node_fields(valid)
    fields["stage"] = "outer-training"
    with pytest.raises(FitTraceError) as grammar_error:
        _make_private_fit_node(**fields)
    assert grammar_error.value.invariant == "stage_fold_selector_exact"

    out_of_range = _node(
        "base.imputer",
        canonical_state_sha256({"value": 1.0}),
        assessment_digest=_digest("out-of-range-assessment"),
        fit_digest=_digest("out-of-range-fit"),
        fold="outer-9.inner-0",
    )
    kwargs = _validation_kwargs(
        [out_of_range],
        {str(out_of_range["node_digest"]): {"unused"}},
        {str(out_of_range["node_digest"]): {"unused"}},
    )
    with pytest.raises(FitTraceError) as range_error:
        validate_fit_trace([out_of_range], **kwargs)
    assert range_error.value.invariant == "node_outer_fold_range"
    assert range_error.value.code == "c08_assessment_ancestry_detected"


def test_swapped_private_inner_partition_with_matching_node_digests_fails() -> None:
    plan = _split_plan()
    partition = _split_manifest()["outer_folds"][0]["inner_folds"][0]
    train = tuple(partition["train"])
    assessment = tuple(partition["assessment"])
    with pytest.raises(AttributeError, match="validated_split_plan_immutable"):
        plan._ValidatedSplitPlan__inner = ((assessment, train),)
    record = _node(
        "base.imputer",
        canonical_state_sha256({"value": 1.0}),
        fit_digest=_canonical_index_sha256(assessment),
        assessment_digest=_canonical_index_sha256(train),
    )
    digest = str(record["node_digest"])
    kwargs = _validation_kwargs([record], {digest: {"unused"}}, {digest: {"unused"}})
    kwargs["split_plan"] = plan
    with pytest.raises(FitTraceError) as error:
        validate_fit_trace([record], **kwargs)
    assert error.value.invariant == "node_index_partition_digest_exact"
    assert plan.receipt()["inner_fold_count"] == 2


def test_ordinary_instance_api_rejects_coordinated_attribute_replacement() -> None:
    record = _node("base.imputer", canonical_state_sha256({"value": 1.0}))
    digest = str(record["node_digest"])
    kwargs = _validation_kwargs([record], {digest: {"unused"}}, {digest: {"unused"}})
    plan = kwargs["split_plan"]
    for name, value in (
        ("_ValidatedSplitPlan__inner", ((((4, 5), (6, 7)),),)),
        ("_ValidatedSplitPlan__split_digest", _digest("replacement")),
        ("_ValidatedSplitPlan__receipt_json", "{}"),
        (
            "_ValidatedSplitPlan__core",
            (((((4, 5), (6, 7)),),), _digest("replacement")),
        ),
    ):
        with pytest.raises(AttributeError):
            object.__setattr__(plan, name, value)
    assert validate_fit_trace([record], **kwargs)["decision"] == "development_only"


def test_closure_forged_plan_receipt_fails_independent_revalidation() -> None:
    constructor = next(
        cell.cell_contents
        for cell in validate_split_manifest.__closure__ or ()
        if getattr(cell.cell_contents, "__name__", None) == "construct"
    )
    manifest = _split_manifest()
    outer = [
        (
            np.asarray(fold["train"], dtype=np.int64),
            np.asarray(fold["assessment"], dtype=np.int64),
        )
        for fold in manifest["outer_folds"]
    ]
    inner = [
        [
            (
                np.asarray(fold["train"], dtype=np.int64),
                np.asarray(fold["assessment"], dtype=np.int64),
            )
            for fold in outer_fold["inner_folds"]
        ]
        for outer_fold in manifest["outer_folds"]
    ]
    forged_receipt = _split_plan().receipt()
    forged_receipt["group_count"] = 999
    forged_receipt["support_summary"][
        "minimum_realized_training_group_count"
    ] = 999
    forged_plan = constructor(
        outer,
        inner,
        _split_validation_spec()["groups"],
        _split_digest(),
        forged_receipt,
    )
    record = _node("base.imputer", canonical_state_sha256({"value": 1.0}))
    digest = str(record["node_digest"])
    kwargs = _validation_kwargs([record], {digest: {"unused"}}, {digest: {"unused"}})
    kwargs["split_plan"] = forged_plan
    with pytest.raises(FitTraceError) as error:
        validate_fit_trace([record], **kwargs)
    assert error.value.invariant == "caller_split_receipt_exact"


def test_raw_split_digest_and_fake_split_plan_are_rejected() -> None:
    records, fit, assessment = _trace_fixture()
    kwargs = _validation_kwargs(records, fit, assessment)
    kwargs["split_manifest_sha256"] = kwargs["split_plan"].receipt()[
        "split_manifest_sha256"
    ]
    with pytest.raises(TypeError):
        validate_fit_trace(records, **kwargs)
    kwargs = _validation_kwargs(records, fit, assessment)
    kwargs["split_plan"] = object()
    with pytest.raises(FitTraceError) as error:
        validate_fit_trace(records, **kwargs)
    assert error.value.invariant == "validated_split_plan_required"
    kwargs = _validation_kwargs(records, fit, assessment)
    kwargs["split_plan"] = object.__new__(ValidatedSplitPlan)
    with pytest.raises(FitTraceError) as unregistered_error:
        validate_fit_trace(records, **kwargs)
    assert unregistered_error.value.invariant == "validated_split_plan_receipt"


def test_poison_tokens_and_external_success_evidence_are_absent() -> None:
    assert not hasattr(fit_trace_module, "PoisonEvidence")
    assert not hasattr(fit_trace_module, "verify_poison_result")
    assert not hasattr(fit_trace_module, "_POISON_EVIDENCE_TOKEN")


def test_raw_poison_case_schema_and_coverage_are_exact() -> None:
    records, fit, assessment = _trace_fixture()
    kwargs = _validation_kwargs(records, fit, assessment)
    kwargs["poison_cases"] = ["mechanics_checked"]
    with pytest.raises(FitTraceError) as string_error:
        validate_fit_trace(records, **kwargs)
    assert string_error.value.invariant == "poison_case_coverage"

    for omitted in ("assessment_feature", "assessment_outcome"):
        kwargs = _validation_kwargs(records, fit, assessment)
        del kwargs["poison_cases"][omitted]
        with pytest.raises(FitTraceError) as incomplete_error:
            validate_fit_trace(records, **kwargs)
        assert incomplete_error.value.invariant == "poison_case_coverage"

    kwargs = _validation_kwargs(records, fit, assessment)
    kwargs["poison_cases"]["assessment_outcome"]["success_status"] = "checked"
    with pytest.raises(FitTraceError) as schema_error:
        validate_fit_trace(records, **kwargs)
    assert schema_error.value.invariant == "poison_case_schema"


def test_duplicate_node_fails() -> None:
    records, _, _ = _trace_fixture()
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete") as error:
        duplicate = [records[0], copy.deepcopy(records[0])]
        validate_fit_trace(duplicate, **{
            "cache_evidence": {},
            "execution_profile": _execution_profile(duplicate),
            "poison_cases": _poison_cases(),
            "private_seed_registries": {},
            "split_manifest": _split_manifest(),
            "split_plan": _split_plan(),
            "split_validation_spec": _split_validation_spec(),
        })
    assert error.value.invariant == "node_digest_unique"


def test_cycle_fails_before_untrusted_digest_acceptance() -> None:
    state = canonical_state_sha256({"value": 1.0})
    first = _node("base.model", state)
    second = _node("base.threshold", state)
    first["node_digest"] = "a" * 64
    second["node_digest"] = "b" * 64
    first["node"]["parent_node_digests"] = ["b" * 64]
    second["node"]["parent_node_digests"] = ["a" * 64]
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete") as error:
        validate_fit_trace([first, second], **{
            "cache_evidence": {},
            "execution_profile": _execution_profile([first, second]),
            "poison_cases": _poison_cases(),
            "private_seed_registries": {},
            "split_manifest": _split_manifest(),
            "split_plan": _split_plan(),
            "split_validation_spec": _split_validation_spec(),
        })
    assert error.value.invariant == "trace_cycle"


def test_raw_private_field_seed_and_path_disclosure_fail() -> None:
    valid = _node("base.imputer", canonical_state_sha256({"value": 1.0}))
    private = _private_node_fields(valid)
    private["labels"] = [0, 1]
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete"):
        _make_private_fit_node(**private)

    raw_seed = _private_node_fields(valid)
    raw_seed["seed_state_digest"] = 42
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete"):
        _make_private_fit_node(**raw_seed)

    local_path = _private_node_fields(valid)
    local_path["callsite_id"] = r"C:\private\fit.py"
    with pytest.raises(FitTraceError, match="c08_callsite_unregistered"):
        _make_private_fit_node(**local_path)


@pytest.mark.parametrize(
    ("field", "hostile"),
    [
        ("callsite_id", "C:private"),
        ("callsite_id", "endpoint.label"),
        ("callsite_id", "password.secret"),
        ("component", "batch-label"),
        ("stage", "group-a"),
    ],
)
def test_hostile_or_unregistered_node_identifier_fails(field: str, hostile: str) -> None:
    record = _node("base.imputer", canonical_state_sha256({"value": 1.0}))
    fields = _private_node_fields(record)
    fields[field] = hostile
    with pytest.raises(FitTraceError, match="c08_"):
        _make_private_fit_node(**fields)


@pytest.mark.parametrize(
    "hostile",
    [
        "C:private",
        r"C:\private\fit.py",
        "endpoint.label",
        "password.secret",
        "user@example.org",
    ],
)
def test_hostile_execution_profile_id_fails_without_reflection(hostile: str) -> None:
    records, fit, assessment = _trace_fixture()
    kwargs = _validation_kwargs(records, fit, assessment)
    profile = copy.deepcopy(kwargs["execution_profile"])
    profile["planned_counts"][hostile] = profile["planned_counts"].pop(
        "base.imputer"
    )
    kwargs["execution_profile"] = profile
    with pytest.raises(FitTraceError) as error:
        validate_fit_trace(records, **kwargs)
    assert error.value.code == "c08_static_runtime_inventory_mismatch"
    assert hostile not in str(error.value)


@pytest.mark.parametrize("hostile", ["endpoint", "batch_label", "password_secret", "C:private"])
def test_hostile_or_unregistered_output_support_key_fails(hostile: str) -> None:
    record = _node("base.imputer", canonical_state_sha256({"value": 1.0}))
    fields = _private_node_fields(record)
    fields["output_support"] = {hostile: 1}
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete"):
        _make_private_fit_node(**fields)


def test_raw_node_builder_is_not_public_api() -> None:
    assert not hasattr(fit_trace_module, "make_fit_node")


def test_arbitrary_seed_digest_cannot_be_supplied() -> None:
    original = _node("base.imputer", canonical_state_sha256({"value": 1.0}))
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
        _node("base.imputer", canonical_state_sha256({"value": 1.0}))
    )
    fields["private_seed_registry"] = registry
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete"):
        _make_private_fit_node(**fields)


def test_private_seed_and_nonce_do_not_affect_public_receipt() -> None:
    state = canonical_state_sha256({"value": 1.0})
    first_registry = _seed_registry(seed=123456789012345678, nonce=b"x" * 32)
    second_registry = _seed_registry(seed=987654321098765432, nonce=b"y" * 32)
    first_fields = _private_node_fields(_node("base.imputer", state))
    second_fields = copy.deepcopy(first_fields)
    first_fields["private_seed_registry"] = first_registry
    second_fields["private_seed_registry"] = second_registry
    first = _make_private_fit_node(**first_fields)
    second = _make_private_fit_node(**second_fields)
    assert first == second
    first_cache = verify_cache_use("unused", node=first["node"])
    second_cache = verify_cache_use("unused", node=second["node"])
    assert repr(first_cache) == repr(second_cache) == (
        "CacheUseEvidence(mechanics_checked=True)"
    )
    assert _cache_identity(first["node"]) == _cache_identity(second["node"])

    digest = str(first["node_digest"])
    common = {
        "cache_evidence": {digest: first_cache},
        "execution_profile": _execution_profile([first]),
        "poison_cases": _poison_cases(),
        "split_manifest": _split_manifest(),
        "split_plan": _split_plan(),
        "split_validation_spec": _split_validation_spec(),
    }
    first_receipt = validate_fit_trace(
        [first], private_seed_registries={digest: first_registry}, **common
    )
    second_common = {**common, "cache_evidence": {digest: second_cache}}
    second_receipt = validate_fit_trace(
        [second], private_seed_registries={digest: second_registry}, **second_common
    )
    assert first_receipt == second_receipt
    public = repr(first_receipt)
    assert "123456789012345678" not in public
    assert "987654321098765432" not in public
    assert first_registry["registry_nonce"].hex() not in public
    assert second_registry["registry_nonce"].hex() not in public
    assert _digest("31") not in public
    assert hashlib.sha256(b"123456789012345678").hexdigest() not in public
    assert hashlib.sha256(b"987654321098765432").hexdigest() not in public
    assert "base.imputer" not in public
    assert "held-out" not in public
    assert "train" not in public


def test_bool_output_support_is_rejected() -> None:
    node = _node("base.imputer", canonical_state_sha256({"value": 1.0}))
    fields = _private_node_fields(node)
    fields["output_support"] = {"feature_count": True}
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete"):
        _make_private_fit_node(**fields)


def test_feature_poison_changed_sentinel_is_watched_failure() -> None:
    records, fit, assessment = _trace_fixture()
    kwargs = _validation_kwargs(records, fit, assessment)
    kwargs["poison_cases"]["assessment_feature"][
        "poisoned_sentinel_predictions"
    ] = np.array([4.0])
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete") as error:
        validate_fit_trace(records, **kwargs)
    assert error.value.invariant == "assessment_feature_sentinel_prediction_invariance"


def test_outcome_poison_prediction_or_state_drift_fails() -> None:
    records, fit, assessment = _trace_fixture()
    kwargs = _validation_kwargs(records, fit, assessment)
    kwargs["poison_cases"]["assessment_outcome"]["poisoned_predictions"] = np.array(
        [2.0, 2.0]
    )
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete") as prediction_error:
        validate_fit_trace(records, **kwargs)
    assert prediction_error.value.invariant == "assessment_outcome_prediction_invariance"
    kwargs = _validation_kwargs(records, fit, assessment)
    kwargs["poison_cases"]["assessment_outcome"]["poisoned_state_digest"] = _digest(
        "changed"
    )
    with pytest.raises(FitTraceError, match="c08_fit_trace_incomplete") as state_error:
        validate_fit_trace(records, **kwargs)
    assert state_error.value.invariant == "poison_learned_state_invariance"


def test_stacking_predicted_group_must_be_excluded() -> None:
    stacker = _node(
        "stacking.stacker",
        canonical_state_sha256({"value": 1.0}),
        fold="outer-0",
        stage="stacking-training",
    )
    digest = str(stacker["node_digest"])
    fit = {digest: {"g4", "g5", "g6", "g7"}}
    assessment = {digest: {"g0", "g1", "g2", "g3"}}
    kwargs = _validation_kwargs([stacker], fit, assessment)
    with pytest.raises(FitTraceError) as error:
        validate_fit_trace(
            [stacker],
            **kwargs,
            stacking_predictions=[
                verify_stacking_prediction(
                    digest,
                    "g6",
                    frozenset({"g6", "g7"}),
                )
            ],
        )
    assert error.value.code == "c08_stacking_in_group_fit"


def test_stacking_evidence_is_mandatory_unique_and_complete() -> None:
    stacker = _node(
        "stacking.stacker",
        canonical_state_sha256({"value": 1.0}),
        fold="outer-0",
        stage="stacking-training",
    )
    digest = str(stacker["node_digest"])
    fit = {digest: {"g4", "g5", "g6", "g7"}}
    assessment = {digest: {"g0", "g1", "g2", "g3"}}
    kwargs = _validation_kwargs([stacker], fit, assessment)
    with pytest.raises(FitTraceError) as missing_error:
        validate_fit_trace([stacker], **kwargs)
    assert missing_error.value.invariant == "stacking_prediction_evidence_complete"

    complete = [
        verify_stacking_prediction(digest, group, fit_groups)
        for group, fit_groups in (
            ("g4", frozenset({"g6", "g7"})),
            ("g5", frozenset({"g6", "g7"})),
            ("g6", frozenset({"g4", "g5"})),
            ("g7", frozenset({"g4", "g5"})),
        )
    ]
    marker = "credential-private-marker"
    for name in ("_StackingPredictionEvidence__predicted_group", "extra"):
        with pytest.raises(
            AttributeError, match="stacking_prediction_evidence_immutable"
        ) as error:
            setattr(complete[0], name, marker)
        assert marker not in str(error.value)
    assert validate_fit_trace(
        [stacker], **kwargs, stacking_predictions=complete
    )["decision"] == "development_only"

    duplicate = [*complete[:3], complete[0]]
    with pytest.raises(FitTraceError) as duplicate_error:
        validate_fit_trace([stacker], **kwargs, stacking_predictions=duplicate)
    assert duplicate_error.value.invariant == "stacking_predicted_group_unique"

    with pytest.raises(FitTraceError) as incomplete_error:
        validate_fit_trace([stacker], **kwargs, stacking_predictions=complete[:-1])
    assert incomplete_error.value.invariant == "stacking_prediction_evidence_complete"

    wrong_plan = verify_stacking_prediction(
        digest, "g4", frozenset({"g5", "g6", "g7"})
    )
    with pytest.raises(FitTraceError) as wrong_plan_error:
        validate_fit_trace(
            [stacker], **kwargs, stacking_predictions=[wrong_plan, *complete[1:]]
        )
    assert wrong_plan_error.value.invariant == "stacking_fit_partition_exact"


def test_public_trace_receipt_contains_no_group_identity_or_group_hash() -> None:
    records, fit, assessment = _trace_fixture()
    receipt = validate_fit_trace(records, **_validation_kwargs(records, fit, assessment))
    public = repr(receipt)
    for group in (f"g{index}" for index in range(8)):
        assert group not in public
        assert _digest(group) not in public
    assert "outside-parent" not in public


def test_nonstacker_cannot_accept_stacking_evidence() -> None:
    records, fit, assessment = _trace_fixture()
    evidence = verify_stacking_prediction(
        str(records[0]["node_digest"]),
        "held-out",
        frozenset({"train-a", "train-b"}),
    )
    with pytest.raises(FitTraceError) as error:
        validate_fit_trace(
            records,
            **_validation_kwargs(records, fit, assessment),
            stacking_predictions=[evidence],
        )
    assert error.value.invariant == "stacking_node_missing"
