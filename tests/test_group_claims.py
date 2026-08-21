import json

import numpy as np
import pandas as pd
import pytest

from omicau.diagnostics import group_claims as gc
from omicau.data.alignment import align_modalities
from omicau.data.benchmark_data import make_mock_dataset, mock_config


def _inputs() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    groups = np.repeat([f"g{i}" for i in range(10)], 2)
    endpoint = np.repeat([0, 1] * 5, 2)
    batch = np.repeat(["x"] * 5 + ["y"] * 5, 2)
    modalities = {
        "rna": {
            "missingness": np.tile([0.01, 0.4], (20, 1)),
            "representation": np.column_stack(
                [np.arange(20.0), np.arange(20.0) ** 2]
            ),
        },
        "protein": {
            "missingness": np.tile([0.2, 0.4], (20, 1)),
            "representation": np.column_stack(
                [np.arange(20.0) + 1, (np.arange(20.0) + 1) ** 2]
            ),
        },
    }
    return modalities, groups, endpoint, batch


def _parameters(modalities: dict) -> dict:
    batch = _inputs()[3]
    return {
        "endpoint_task": "classification",
        "batch_by_modality": {name: batch.copy() for name in modalities},
        "eligible_modality_registry": {
            name: {"eligibility_status": "eligible"} for name in modalities
        },
        "event": None,
        "alpha": 0.05,
        "permutation_reps": {
            "missingness": 19,
            "c05_structure": 19,
            "c05_outcome": 19,
        },
        "permutation_seeds": {
            name: {
                "missingness": 991_000_001 + offset,
                "c05_structure": 991_000_101 + offset,
                "c05_outcome": 991_000_201 + offset,
            }
            for offset, name in enumerate(modalities)
        },
        "support": {"minimum_groups": 5, "minimum_groups_per_batch": 3},
        "group_reducer": "mean",
        "standardization": "global_zscore",
        "representation_distance": "normalized_euclidean",
    }


def _call(modalities: dict, groups: np.ndarray, endpoint: np.ndarray, **overrides):
    parameters = _parameters(modalities)
    parameters.update(overrides)
    return gc.group_claim_receipts(modalities, groups, endpoint, **parameters)


def _independent_holm(values: list[float]) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    order = np.argsort(raw, kind="stable")
    sorted_adjusted = (len(raw) - np.arange(len(raw))) * raw[order]
    sorted_adjusted = np.maximum.accumulate(sorted_adjusted)
    adjusted = np.empty_like(raw)
    adjusted[order] = np.minimum(sorted_adjusted, 1.0)
    return adjusted


def _aligned_inputs(task="classification"):
    bundle = make_mock_dataset(task=task, n_samples=48, seed=23)
    clinical = bundle.clinical.copy()
    group_order = {
        group: index for index, group in enumerate(sorted(clinical["patient_id"].unique()))
    }
    if task == "classification":
        clinical["label"] = clinical["patient_id"].map(
            lambda group: "responder" if group_order[group] % 2 else "nonresponder"
        )
    elif task == "regression":
        clinical["label"] = clinical["patient_id"].map(
            lambda group: float(group_order[group])
        )
    else:
        clinical["time"] = clinical["patient_id"].map(
            lambda group: float(group_order[group] + 1)
        )
        clinical["event"] = clinical["patient_id"].map(
            lambda group: int(group_order[group] % 2)
        )
    clinical["batch"] = clinical["patient_id"].map(
        lambda group: f"batch{group_order[group] % 3 + 1}"
    )
    config = mock_config(task=task)
    config.clinical.batch_by_modality = {
        name: "batch" for name in bundle.modalities
    }
    aligned = align_modalities(bundle.modalities, clinical, config)
    representations = {
        name: {
            "values": pd.DataFrame(
                np.nan_to_num(modality.X, nan=0.0),
                index=aligned.sample_ids,
                columns=[f"{name}::{feature}" for feature in modality.feature_names],
            ),
        }
        for name, modality in aligned.modalities.items()
    }
    parameters = _parameters(
        {
            name: {"missingness": None, "representation": None}
            for name in aligned.modalities
        }
    )
    parameters.pop("batch_by_modality")
    parameters.pop("endpoint_task")
    parameters.pop("event")
    return aligned, representations, parameters


def test_aligned_adapter_binds_exact_modalities_and_preserves_development_status(monkeypatch):
    _install_fakes(monkeypatch)
    aligned, representations, parameters = _aligned_inputs()

    result = gc.group_claim_receipts_from_aligned(
        aligned, representations, **parameters
    )

    assert result["status"] == "development_only_not_production_wired"
    assert result["modality_count"] == len(aligned.modalities)
    assert set(result["claims"]) == {"C03", "C04", "C05"}


def test_aligned_adapter_derives_missingness_without_mutating_inputs(monkeypatch):
    aligned, representations, parameters = _aligned_inputs()
    name = next(iter(aligned.modalities))
    row = aligned.modalities[name].frame.index[0]
    column = aligned.modalities[name].frame.columns[0]
    aligned.modalities[name].frame.loc[row, column] = np.nan
    before = {
        key: value["values"].copy(deep=True) for key, value in representations.items()
    }
    seen: dict[str, np.ndarray] = {}

    def fake_missingness(missingness, *args, **kwargs):
        seen["missingness"] = np.asarray(missingness).copy()
        return {
            "endpoint_association": {"primary": {"p_value": 0.5}},
            "batch_association": {"primary": {"p_value": 0.5}},
        }

    monkeypatch.setattr(gc.gm, "group_missingness_diagnostics", fake_missingness)
    monkeypatch.setattr(
        gc.gb, "structure_batch_association", lambda *a, **k: {"p_value": 0.5}
    )
    monkeypatch.setattr(
        gc.gb, "classification_batch_association", lambda *a, **k: {"p_value": 0.5}
    )

    gc.group_claim_receipts_from_aligned(aligned, representations, **parameters)

    assert seen["missingness"].dtype == np.int8
    assert int(seen["missingness"].sum()) >= 1
    assert all(
        representations[key]["values"].equals(value)
        for key, value in before.items()
    )


def test_aligned_adapter_routes_regression_without_defaults(monkeypatch):
    _install_fakes(monkeypatch)
    aligned, representations, parameters = _aligned_inputs("regression")

    result = gc.group_claim_receipts_from_aligned(
        aligned, representations, **parameters
    )

    assert result["status"] == "development_only_not_production_wired"
    assert set(result["claims"]) == {"C03", "C04", "C05"}


def test_aligned_adapter_preserves_fixed_survival_refusal():
    aligned, representations, parameters = _aligned_inputs("survival")

    with pytest.raises(
        gc.GroupClaimsRefusal, match=f"^{gc.REFUSAL_SURVIVAL_UNSUPPORTED}$"
    ):
        gc.group_claim_receipts_from_aligned(
            aligned, representations, **parameters
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_groups",
        "missing_batch_modality",
        "missing_batch_value",
        "group_order",
        "endpoint_order",
        "batch_order",
        "missing_representation",
        "nonfinite_representation",
        "row_count",
        "modality_order",
        "representation_order",
        "representation_value_order",
        "representation_modality_swap",
        "representation_extra_field",
        "duplicate_sample",
    ],
)
def test_aligned_adapter_refuses_incomplete_or_unbound_inputs(mutation, monkeypatch):
    aligned, representations, parameters = _aligned_inputs()
    hostile = r"C:\private\token=synthetic-secret-marker"
    if mutation == "missing_groups":
        aligned.groups = None
    elif mutation == "missing_batch_modality":
        aligned.batch_by_modality.pop(next(iter(aligned.batch_by_modality)))
    elif mutation == "missing_batch_value":
        name = next(iter(aligned.batch_by_modality))
        aligned.batch_by_modality[name] = aligned.batch_by_modality[name].copy()
        aligned.batch_by_modality[name].iloc[0] = pd.NA
    elif mutation == "group_order":
        aligned.groups = aligned.groups.iloc[::-1]
    elif mutation == "endpoint_order":
        aligned.y = aligned.y.iloc[::-1]
    elif mutation == "batch_order":
        name = next(iter(aligned.batch_by_modality))
        aligned.batch_by_modality[name] = aligned.batch_by_modality[name].iloc[::-1]
    elif mutation == "missing_representation":
        representations.pop(next(iter(representations)))
    elif mutation == "nonfinite_representation":
        representations[next(iter(representations))]["values"].iloc[0, 0] = np.inf
    elif mutation == "row_count":
        name = next(iter(representations))
        representations[name]["values"] = representations[name]["values"].iloc[:-1]
    elif mutation == "modality_order":
        name = next(iter(aligned.modalities))
        aligned.modalities[name].frame = aligned.modalities[name].frame.iloc[::-1]
    elif mutation == "representation_order":
        name = next(iter(representations))
        representations[name]["values"] = representations[name]["values"].iloc[::-1]
    elif mutation == "representation_value_order":
        name = next(iter(representations))
        reversed_values = representations[name]["values"].iloc[::-1].copy()
        reversed_values.index = aligned.sample_ids
        representations[name]["values"] = reversed_values
    elif mutation == "representation_modality_swap":
        first, second = list(representations)[:2]
        representations[first]["values"], representations[second]["values"] = (
            representations[second]["values"],
            representations[first]["values"],
        )
    elif mutation == "representation_extra_field":
        representations[next(iter(representations))]["extra"] = hostile
    else:
        aligned.sample_ids[1] = aligned.sample_ids[0]
    with pytest.raises(
        gc.GroupClaimsRefusal, match=f"^{gc.REFUSAL_ALIGNED_INPUT}$"
    ) as captured:
        gc.group_claim_receipts_from_aligned(
            aligned, representations, **parameters
        )
    assert hostile.lower() not in str(captured.value).lower()


def test_aligned_adapter_remains_inactive_in_cli_source():
    from pathlib import Path

    source = Path("omicau/cli.py").read_text(encoding="utf-8")
    assert "group_claim_receipts_from_aligned" not in source


def _install_fakes(monkeypatch, *, secondary_p: float = 0.5) -> None:
    def fake_missingness(row_missingness, *args, **kwargs):
        values = np.asarray(row_missingness, dtype=float)
        return {
            "endpoint_association": {
                "primary": {"p_value": float(values[0, 0])},
                "secondary": {"minimum_raw_p_value": secondary_p},
            },
            "batch_association": {
                "primary": {"p_value": float(values[0, 1])},
                "secondary": {"minimum_raw_p_value": secondary_p},
            },
        }

    monkeypatch.setattr(gc.gm, "group_missingness_diagnostics", fake_missingness)
    monkeypatch.setattr(
        gc.gb,
        "structure_batch_association",
        lambda *args, **kwargs: {"p_value": 0.02},
    )
    monkeypatch.setattr(
        gc.gb,
        "classification_batch_association",
        lambda *args, **kwargs: {"p_value": 0.04},
    )
    monkeypatch.setattr(
        gc.gb,
        "regression_batch_association",
        lambda *args, **kwargs: {"p_value": 0.04},
    )


def test_complete_claim_families_match_independent_holm_oracle(monkeypatch):
    _install_fakes(monkeypatch)
    modalities, groups, endpoint, _ = _inputs()
    result = _call(modalities, groups, endpoint)
    expected = _independent_holm([0.01, 0.2])
    c03 = result["claims"]["C03"]
    assert c03["eligible_primary_component_count"] == 2
    assert c03["tested_primary_component_count"] == 2
    assert c03["minimum_holm_p_value"] == pytest.approx(expected.min())
    assert c03["warning_count"] == int(np.count_nonzero(expected <= 0.05))
    assert result["claims"]["C04"]["eligible_primary_component_count"] == 2
    assert result["claims"]["C05"]["eligible_primary_component_count"] == 4


def test_watched_fail_per_modality_holm_cannot_replace_global_family(monkeypatch):
    _install_fakes(monkeypatch)
    modalities, groups, endpoint, _ = _inputs()
    for specification in modalities.values():
        specification["missingness"][:, 0] = 0.03
    result = _call(modalities, groups, endpoint)
    assert result["claims"]["C03"]["minimum_holm_p_value"] == pytest.approx(0.06)
    assert result["claims"]["C03"]["warning_count"] == 0
    assert result["claims"]["C05"]["minimum_holm_p_value"] == pytest.approx(0.08)
    assert result["claims"]["C05"]["warning_count"] == 0


def test_watched_fail_secondary_localization_never_triggers_primary_decision(monkeypatch):
    _install_fakes(monkeypatch, secondary_p=1e-15)
    modalities, groups, endpoint, _ = _inputs()
    for specification in modalities.values():
        specification["missingness"][:, :2] = 0.5
    result = _call(modalities, groups, endpoint)
    assert result["claims"]["C03"]["warning_count"] == 0
    assert result["claims"]["C04"]["warning_count"] == 0
    assert all(
        receipt["secondary_evidence_status"]
        == "localization_only_not_decision_triggering"
        for receipt in result["claims"].values()
    )


def test_each_modality_uses_its_own_explicit_batch_vector(monkeypatch):
    modalities, groups, endpoint, shared_batch = _inputs()
    distinct_batch = np.repeat(["x", "y"] * 5, 2)
    batches = {"rna": shared_batch.copy(), "protein": distinct_batch.copy()}
    seen: list[np.ndarray] = []

    def fake_missingness(row_missingness, *args, **kwargs):
        seen.append(np.asarray(kwargs["batch"], dtype=object).copy())
        return {
            "endpoint_association": {"primary": {"p_value": 0.5}},
            "batch_association": {"primary": {"p_value": 0.5}},
        }

    monkeypatch.setattr(gc.gm, "group_missingness_diagnostics", fake_missingness)
    monkeypatch.setattr(
        gc.gb, "structure_batch_association", lambda *args, **kwargs: {"p_value": 0.5}
    )
    monkeypatch.setattr(
        gc.gb,
        "classification_batch_association",
        lambda *args, **kwargs: {"p_value": 0.5},
    )
    _call(modalities, groups, endpoint, batch_by_modality=batches)
    assert len(seen) == 2
    assert any(np.array_equal(values, shared_batch) for values in seen)
    assert any(np.array_equal(values, distinct_batch) for values in seen)
    assert not np.array_equal(seen[0], seen[1])


def test_exact_aggregate_receipt_schema_and_pending_registry(monkeypatch):
    _install_fakes(monkeypatch)
    modalities, groups, endpoint, _ = _inputs()
    result = _call(modalities, groups, endpoint)
    assert set(result) == {
        "status",
        "scope",
        "modality_count",
        "alpha",
        "claims",
        "eligible_modality_registry_sha256",
        "eligible_modality_registry_status",
        "methods_completion_status",
        "permutation_registry_sha256",
        "permutation_registry_status",
        "oracle_status",
    }
    assert set(result["claims"]) == {"C03", "C04", "C05"}
    expected_receipt_fields = {
        "claim_id",
        "status",
        "multiplicity_family_status",
        "multiplicity_family_id",
        "eligible_primary_component_count",
        "tested_primary_component_count",
        "warning_count",
        "any_warning",
        "minimum_holm_p_value",
        "method_ids",
        "secondary_evidence_status",
        "oracle_status",
    }
    assert all(set(receipt) == expected_receipt_fields for receipt in result["claims"].values())
    assert all(
        receipt["multiplicity_family_status"]
        == "complete_against_supplied_unfrozen_registry"
        for receipt in result["claims"].values()
    )
    assert result["eligible_modality_registry_sha256"] is None
    assert (
        result["eligible_modality_registry_status"]
        == "supplied_unfrozen_pending_freeze"
    )
    assert result["permutation_registry_sha256"] is None
    assert result["permutation_registry_status"] == "unavailable_pending_frozen_registry"
    assert result["oracle_status"] == "not_run_development_scaffold"
    assert result["methods_completion_status"] == "not_assessed_development_scaffold"


def test_output_is_aggregate_and_does_not_expose_seeds_or_hostile_values(monkeypatch):
    _install_fakes(monkeypatch)
    modalities, groups, endpoint, batch = _inputs()
    groups = np.repeat([rf"C:\private\subject_{i}" for i in range(10)], 2)
    batch = np.repeat(
        [r"C:\private\batch_a.tsv"] * 5 + ["token=synthetic-secret-marker"] * 5,
        2,
    )
    result = _call(
        modalities,
        groups,
        endpoint,
        batch_by_modality={name: batch.copy() for name in modalities},
    )
    serialized = json.dumps(result).lower()
    for forbidden in (
        "subject_",
        "batch_a.tsv",
        "synthetic-secret-marker",
        "991000001",
        "991000101",
        "991000201",
        "group_ids",
        "raw_p_value",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("endpoint_task", "event", "code"),
    [
        ("survival", None, gc.REFUSAL_EVENT_REQUIRED),
        ("survival", np.ones(20), gc.REFUSAL_SURVIVAL_UNSUPPORTED),
        ("classification", np.ones(20), gc.REFUSAL_EVENT_NOT_APPLICABLE),
    ],
)
def test_event_and_survival_refusals_are_fixed(endpoint_task, event, code):
    modalities, groups, endpoint, _ = _inputs()
    with pytest.raises(gc.GroupClaimsRefusal, match=f"^{code}$"):
        _call(modalities, groups, endpoint, endpoint_task=endpoint_task, event=event)


def test_missing_batch_refusal_is_fixed():
    modalities, groups, endpoint, _ = _inputs()
    with pytest.raises(gc.GroupClaimsRefusal, match=f"^{gc.REFUSAL_BATCH_REQUIRED}$"):
        _call(modalities, groups, endpoint, batch_by_modality=None)


def test_watched_fail_mixed_group_endpoint_is_refused(monkeypatch):
    _install_fakes(monkeypatch)
    modalities, groups, endpoint, _ = _inputs()
    endpoint = endpoint.copy()
    endpoint[1] = 1 - endpoint[0]
    with pytest.raises(gc.GroupClaimsRefusal, match=f"^{gc.REFUSAL_ANALYSIS}$"):
        _call(modalities, groups, endpoint)


@pytest.mark.parametrize("field", ["missingness", "representation"])
def test_watched_fail_nan_inputs_are_refused_without_value_reflection(field, monkeypatch):
    modalities, groups, endpoint, _ = _inputs()
    modalities = {"rna": modalities["rna"]}
    if field == "representation":
        _install_fakes(monkeypatch)
    else:
        modalities["rna"]["missingness"] = np.tile([0, 1], (20, 1))
    modalities["rna"][field] = modalities["rna"][field].astype(float, copy=True)
    modalities["rna"][field][0, 0] = np.nan
    with pytest.raises(gc.GroupClaimsRefusal, match=f"^{gc.REFUSAL_ANALYSIS}$"):
        _call(modalities, groups, endpoint)


@pytest.mark.parametrize(
    "mutation",
    [
        "hostile_modality",
        "hostile_field",
        "hostile_seed_field",
        "partial_seed_registry",
        "partial_reps",
        "partial_batch_mapping",
        "extra_batch_mapping",
    ],
)
def test_hostile_or_partial_mapping_keys_fail_closed_without_reflection(mutation):
    modalities, groups, endpoint, _ = _inputs()
    parameters = _parameters(modalities)
    hostile = r"C:\private\token=synthetic-secret-marker"
    if mutation == "hostile_modality":
        modalities[hostile] = modalities.pop("rna")
        parameters = _parameters(modalities)
    elif mutation == "hostile_field":
        modalities["rna"][hostile] = np.zeros((20, 1))
    elif mutation == "hostile_seed_field":
        parameters["permutation_seeds"]["rna"][hostile] = 7
    elif mutation == "partial_seed_registry":
        parameters["permutation_seeds"].pop("rna")
    elif mutation == "partial_batch_mapping":
        parameters["batch_by_modality"].pop("rna")
    elif mutation == "extra_batch_mapping":
        parameters["batch_by_modality"]["metabolite"] = _inputs()[3]
    else:
        parameters["permutation_reps"].pop("c05_outcome")
    with pytest.raises(gc.GroupClaimsRefusal, match=f"^{gc.REFUSAL_SCHEMA}$") as captured:
        gc.group_claim_receipts(modalities, groups, endpoint, **parameters)
    message = str(captured.value).lower()
    assert "private" not in message
    assert "synthetic-secret-marker" not in message


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_modality",
        "extra_modality",
        "omitted_modality_input",
        "omitted_registry",
        "wrong_status",
        "extra_field",
    ],
)
def test_watched_eligibility_registry_failures_use_fixed_code_without_reflection(mutation):
    modalities, groups, endpoint, _ = _inputs()
    parameters = _parameters(modalities)
    registry = parameters["eligible_modality_registry"]
    hostile = r"C:\private\token=synthetic-secret-marker"
    if mutation == "missing_modality":
        registry.pop("rna")
    elif mutation == "extra_modality":
        registry["metabolite"] = {"eligibility_status": "eligible"}
    elif mutation == "omitted_modality_input":
        modalities.pop("rna")
    elif mutation == "omitted_registry":
        parameters.pop("eligible_modality_registry")
        with pytest.raises(TypeError):
            gc.group_claim_receipts(modalities, groups, endpoint, **parameters)
        return
    elif mutation == "wrong_status":
        registry["rna"]["eligibility_status"] = hostile
    else:
        registry["rna"][hostile] = "eligible"
    with pytest.raises(
        gc.GroupClaimsRefusal, match=f"^{gc.REFUSAL_ELIGIBILITY_REGISTRY}$"
    ) as captured:
        gc.group_claim_receipts(modalities, groups, endpoint, **parameters)
    message = str(captured.value).lower()
    assert "private" not in message
    assert "synthetic-secret-marker" not in message


@pytest.mark.parametrize("association", ["endpoint_association", "batch_association"])
@pytest.mark.parametrize("component_status", ["not_applicable", "not_tested", None])
def test_watched_non_tested_or_partial_missingness_family_is_refused(
    monkeypatch, component_status, association
):
    _install_fakes(monkeypatch)
    modalities, groups, endpoint, _ = _inputs()

    def fake_missingness(*args, **kwargs):
        component = {"p_value": 0.5}
        if component_status is None:
            component.pop("p_value")
        else:
            component["status"] = component_status
        result = {
            "endpoint_association": {"primary": component},
            "batch_association": {"primary": {"p_value": 0.5}},
        }
        if association == "batch_association":
            result["endpoint_association"], result["batch_association"] = (
                result["batch_association"],
                result["endpoint_association"],
            )
        return result

    monkeypatch.setattr(gc.gm, "group_missingness_diagnostics", fake_missingness)
    with pytest.raises(gc.GroupClaimsRefusal, match=f"^{gc.REFUSAL_ANALYSIS}$"):
        _call(modalities, groups, endpoint)


@pytest.mark.parametrize(
    "function_name",
    ["structure_batch_association", "classification_batch_association"],
)
def test_watched_non_tested_c05_component_is_refused(monkeypatch, function_name):
    _install_fakes(monkeypatch)
    modalities, groups, endpoint, _ = _inputs()
    monkeypatch.setattr(
        gc.gb,
        function_name,
        lambda *args, **kwargs: {"status": "not_applicable", "p_value": 0.5},
    )
    with pytest.raises(gc.GroupClaimsRefusal, match=f"^{gc.REFUSAL_ANALYSIS}$"):
        _call(modalities, groups, endpoint)


def test_seed_zero_is_valid_but_raw_seed_is_never_returned(monkeypatch):
    _install_fakes(monkeypatch)
    modalities, groups, endpoint, _ = _inputs()
    seeds = {
        name: {"missingness": 0, "c05_structure": 0, "c05_outcome": 0}
        for name in modalities
    }
    result = _call(modalities, groups, endpoint, permutation_seeds=seeds)
    assert "seed" not in json.dumps(result).lower()


def test_explicit_support_gate_is_watched(monkeypatch):
    _install_fakes(monkeypatch)
    modalities, groups, endpoint, _ = _inputs()
    with pytest.raises(gc.GroupClaimsRefusal, match=f"^{gc.REFUSAL_SUPPORT}$"):
        _call(
            modalities,
            groups,
            endpoint,
            support={"minimum_groups": 11, "minimum_groups_per_batch": 3},
        )


@pytest.mark.parametrize("missing_group", [None, np.nan])
def test_incomplete_group_identifiers_are_refused_with_fixed_code(missing_group):
    modalities, groups, endpoint, _ = _inputs()
    groups = groups.astype(object)
    groups[0] = missing_group
    with pytest.raises(gc.GroupClaimsRefusal, match=f"^{gc.REFUSAL_SCHEMA}$"):
        _call(modalities, groups, endpoint)
