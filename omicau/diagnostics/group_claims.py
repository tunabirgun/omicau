"""Development-only cross-modality receipts for group-aware claims C03-C05."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
import re
from typing import Any, Literal

import numpy as np

from omicau.diagnostics import group_batch as gb
from omicau.diagnostics import group_missingness as gm


EndpointTask = Literal["classification", "regression", "survival"]

REFUSAL_SCHEMA = "GROUP_CLAIMS_SCHEMA_REFUSED"
REFUSAL_BATCH_REQUIRED = "GROUP_CLAIMS_BATCH_REQUIRED"
REFUSAL_EVENT_REQUIRED = "GROUP_CLAIMS_EVENT_REQUIRED"
REFUSAL_EVENT_NOT_APPLICABLE = "GROUP_CLAIMS_EVENT_NOT_APPLICABLE"
REFUSAL_SURVIVAL_UNSUPPORTED = "GROUP_CLAIMS_SURVIVAL_MISSINGNESS_UNSUPPORTED"
REFUSAL_SUPPORT = "GROUP_CLAIMS_SUPPORT_REFUSED"
REFUSAL_ANALYSIS = "GROUP_CLAIMS_ANALYSIS_REFUSED"
REFUSAL_ELIGIBILITY_REGISTRY = "GROUP_CLAIMS_ELIGIBILITY_REGISTRY_REFUSED"

_MODALITY_FIELDS = frozenset({"missingness", "representation"})
_REPS_FIELDS = frozenset({"missingness", "c05_structure", "c05_outcome"})
_SUPPORT_FIELDS = frozenset({"minimum_groups", "minimum_groups_per_batch"})
_SEED_FIELDS = _REPS_FIELDS
_ELIGIBILITY_FIELDS = frozenset({"eligibility_status"})
_SAFE_ID = re.compile(r"[a-z][a-z0-9_]{0,63}")


class GroupClaimsRefusal(ValueError):
    """Fail closed with a fixed public refusal code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _refuse(code: str) -> None:
    raise GroupClaimsRefusal(code)


def _exact_mapping(value: Any, fields: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _refuse(REFUSAL_SCHEMA)
    return value


def _positive_integer(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        _refuse(REFUSAL_SCHEMA)
    result = int(value)
    if result < 1:
        _refuse(REFUSAL_SCHEMA)
    return result


def _seed(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        _refuse(REFUSAL_SCHEMA)
    result = int(value)
    if result < 0:
        _refuse(REFUSAL_SCHEMA)
    if result > np.iinfo(np.uint32).max:
        _refuse(REFUSAL_SCHEMA)
    return result


def _alpha(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        _refuse(REFUSAL_SCHEMA)
    result = float(value)
    if not np.isfinite(result) or not 0 < result < 1:
        _refuse(REFUSAL_SCHEMA)
    return result


def _complete_group_count(group_ids: Sequence[Any], n_rows: int) -> int:
    groups = np.asarray(group_ids, dtype=object)
    if groups.ndim != 1 or len(groups) != n_rows:
        _refuse(REFUSAL_SCHEMA)
    distinct: dict[Any, None] = {}
    for value in groups.tolist():
        if not isinstance(value, Hashable) or isinstance(value, (bool, np.bool_)):
            _refuse(REFUSAL_SCHEMA)
        try:
            missing = np.asarray(value is None or value != value)
        except (TypeError, ValueError):
            _refuse(REFUSAL_SCHEMA)
        try:
            is_missing = missing.ndim != 0 or bool(missing)
        except (TypeError, ValueError):
            _refuse(REFUSAL_SCHEMA)
        if is_missing:
            _refuse(REFUSAL_SCHEMA)
        distinct[value] = None
    return len(distinct)


def _p_value(component: Mapping[str, Any]) -> float:
    if not isinstance(component, Mapping):
        _refuse(REFUSAL_ANALYSIS)
    if "status" in component and component.get("status") != "tested":
        _refuse(REFUSAL_ANALYSIS)
    value = component.get("p_value")
    if isinstance(value, (bool, np.bool_)):
        _refuse(REFUSAL_ANALYSIS)
    try:
        result = float(value)
    except (TypeError, ValueError):
        _refuse(REFUSAL_ANALYSIS)
    if not np.isfinite(result) or not 0 <= result <= 1:
        _refuse(REFUSAL_ANALYSIS)
    return result


def _family_receipt(
    claim_id: str,
    p_values: Sequence[float],
    *,
    alpha: float,
    method_ids: Sequence[str],
) -> dict[str, Any]:
    adjusted = gm.holm_adjust(p_values)
    family_ids = {
        "C03": "c03_target_event_censoring_missingness_all_modalities_holm",
        "C04": "c04_batch_missingness_all_modalities_holm",
        "C05": "c05_structure_outcome_event_censoring_all_modalities_holm",
    }
    return {
        "claim_id": claim_id,
        "status": "development_only_not_production_wired",
        "multiplicity_family_status": "complete_against_supplied_unfrozen_registry",
        "multiplicity_family_id": family_ids[claim_id],
        "eligible_primary_component_count": int(len(p_values)),
        "tested_primary_component_count": int(len(p_values)),
        "warning_count": int(np.count_nonzero(adjusted <= alpha)),
        "any_warning": bool(np.any(adjusted <= alpha)),
        "minimum_holm_p_value": float(np.min(adjusted)),
        "method_ids": sorted(set(method_ids)),
        "secondary_evidence_status": "localization_only_not_decision_triggering",
        "oracle_status": "not_run_development_scaffold",
    }


def group_claim_receipts(
    modalities: Mapping[str, Mapping[str, Any]],
    group_ids: Sequence[Any],
    endpoint: Sequence[Any],
    *,
    endpoint_task: EndpointTask,
    batch_by_modality: Mapping[str, Sequence[Any]] | None,
    eligible_modality_registry: Mapping[str, Mapping[str, Any]],
    event: Sequence[int] | None,
    alpha: float,
    permutation_reps: Mapping[str, int],
    permutation_seeds: Mapping[str, Mapping[str, int]],
    support: Mapping[str, int],
    group_reducer: gb.Reducer,
    standardization: gb.Standardization,
    representation_distance: gb.RepresentationDistance,
) -> dict[str, Any]:
    """Return aggregate development receipts after global claim-level Holm tests."""
    if endpoint_task not in ("classification", "regression", "survival"):
        _refuse(REFUSAL_SCHEMA)
    if batch_by_modality is None:
        _refuse(REFUSAL_BATCH_REQUIRED)
    if endpoint_task == "survival":
        if event is None:
            _refuse(REFUSAL_EVENT_REQUIRED)
        _refuse(REFUSAL_SURVIVAL_UNSUPPORTED)
    if event is not None:
        _refuse(REFUSAL_EVENT_NOT_APPLICABLE)
    alpha_value = _alpha(alpha)
    reps_input = _exact_mapping(permutation_reps, _REPS_FIELDS)
    reps = {name: _positive_integer(reps_input[name]) for name in _REPS_FIELDS}
    support_input = _exact_mapping(support, _SUPPORT_FIELDS)
    minimum_groups = _positive_integer(support_input["minimum_groups"])
    minimum_groups_per_batch = _positive_integer(
        support_input["minimum_groups_per_batch"]
    )
    if minimum_groups < 5:
        _refuse(REFUSAL_SUPPORT)
    if group_reducer not in ("mean", "median"):
        _refuse(REFUSAL_SCHEMA)
    if standardization not in ("global_zscore", "global_median_iqr"):
        _refuse(REFUSAL_SCHEMA)
    if representation_distance != "normalized_euclidean":
        _refuse(REFUSAL_SCHEMA)
    if not isinstance(modalities, Mapping) or not modalities:
        _refuse(REFUSAL_SCHEMA)
    names = list(modalities)
    if any(type(name) is not str or _SAFE_ID.fullmatch(name) is None for name in names):
        _refuse(REFUSAL_SCHEMA)
    if (
        not isinstance(eligible_modality_registry, Mapping)
        or set(eligible_modality_registry) != set(names)
    ):
        _refuse(REFUSAL_ELIGIBILITY_REGISTRY)
    for name in names:
        try:
            eligibility = _exact_mapping(
                eligible_modality_registry[name], _ELIGIBILITY_FIELDS
            )
        except GroupClaimsRefusal:
            _refuse(REFUSAL_ELIGIBILITY_REGISTRY)
        eligibility_status = eligibility["eligibility_status"]
        if type(eligibility_status) is not str or eligibility_status != "eligible":
            _refuse(REFUSAL_ELIGIBILITY_REGISTRY)
    if not isinstance(batch_by_modality, Mapping) or set(batch_by_modality) != set(names):
        _refuse(REFUSAL_SCHEMA)
    if not isinstance(permutation_seeds, Mapping) or set(permutation_seeds) != set(names):
        _refuse(REFUSAL_SCHEMA)
    seeds: dict[str, dict[str, int]] = {}
    n_rows: int | None = None
    for name in sorted(names):
        specification = _exact_mapping(modalities[name], _MODALITY_FIELDS)
        missingness = np.asarray(specification["missingness"])
        representation = np.asarray(specification["representation"])
        if missingness.ndim != 2 or representation.ndim != 2:
            _refuse(REFUSAL_SCHEMA)
        if missingness.shape[0] == 0 or representation.shape[0] != missingness.shape[0]:
            _refuse(REFUSAL_SCHEMA)
        if n_rows is None:
            n_rows = int(missingness.shape[0])
        elif missingness.shape[0] != n_rows:
            _refuse(REFUSAL_SCHEMA)
        try:
            modality_batch = np.asarray(batch_by_modality[name], dtype=object)
        except (TypeError, ValueError):
            _refuse(REFUSAL_SCHEMA)
        if modality_batch.ndim != 1 or len(modality_batch) != missingness.shape[0]:
            _refuse(REFUSAL_SCHEMA)
        seed_input = _exact_mapping(permutation_seeds[name], _SEED_FIELDS)
        seeds[name] = {field: _seed(seed_input[field]) for field in _SEED_FIELDS}
    assert n_rows is not None
    if _complete_group_count(group_ids, n_rows) < minimum_groups:
        _refuse(REFUSAL_SUPPORT)
    if np.asarray(endpoint, dtype=object).ndim != 1 or len(endpoint) != n_rows:
        _refuse(REFUSAL_SCHEMA)
    c03: list[float] = []
    c04: list[float] = []
    c05: list[float] = []
    c03_methods: list[str] = []
    c04_methods: list[str] = []
    c05_methods: list[str] = []
    missingness_kind = "categorical" if endpoint_task == "classification" else "continuous"
    try:
        for name in sorted(names):
            specification = modalities[name]
            modality_batch = batch_by_modality[name]
            missingness_result = gm.group_missingness_diagnostics(
                specification["missingness"],
                group_ids,
                endpoint,
                endpoint_kind=missingness_kind,
                reps=reps["missingness"],
                seed=seeds[name]["missingness"],
                batch=modality_batch,
            )
            c03.append(_p_value(missingness_result["endpoint_association"]["primary"]))
            c04.append(_p_value(missingness_result["batch_association"]["primary"]))
            c03_methods.append("full_missingness_profile_precomputed_mgc")
            c04_methods.append("full_missingness_profile_precomputed_mgc")

            representation = gb.collapse_group_representation(
                specification["representation"],
                group_ids,
                group_reducer=group_reducer,
            )
            group_batch = gb.collapse_pure_group_values(
                modality_batch, group_ids, value_kind="categorical"
            )
            structure = gb.structure_batch_association(
                representation,
                group_batch,
                standardization=standardization,
                representation_distance=representation_distance,
                reps=reps["c05_structure"],
                seed=seeds[name]["c05_structure"],
                minimum_groups_per_batch=minimum_groups_per_batch,
            )
            c05.append(_p_value(structure))
            c05_methods.append("full_representation_precomputed_mgc")
            endpoint_kind = "categorical" if endpoint_task == "classification" else "continuous"
            group_endpoint = gb.collapse_pure_group_values(
                endpoint, group_ids, value_kind=endpoint_kind
            )
            if endpoint_task == "classification":
                outcome = gb.classification_batch_association(
                    group_batch,
                    group_endpoint,
                    reps=reps["c05_outcome"],
                    seed=seeds[name]["c05_outcome"],
                    minimum_groups_per_batch=minimum_groups_per_batch,
                )
                c05_methods.append("whole_group_fixed_margin_permutation_pearson")
            else:
                outcome = gb.regression_batch_association(
                    group_batch,
                    group_endpoint,
                    reps=reps["c05_outcome"],
                    seed=seeds[name]["c05_outcome"],
                    minimum_groups_per_batch=minimum_groups_per_batch,
                )
                c05_methods.append("batch_delta_outcome_midrank_precomputed_mgc")
            c05.append(_p_value(outcome))
    except GroupClaimsRefusal:
        raise
    except (KeyError, TypeError, ValueError, IndexError):
        _refuse(REFUSAL_ANALYSIS)

    expected_modalities = len(names)
    if not (
        len(c03) == expected_modalities
        and len(c04) == expected_modalities
        and len(c05) == 2 * expected_modalities
    ):
        _refuse(REFUSAL_ANALYSIS)

    return {
        "status": "development_only_not_production_wired",
        "scope": "eligible_modalities_aggregate_claim_receipts",
        "modality_count": int(len(names)),
        "alpha": alpha_value,
        "eligible_modality_registry_sha256": None,
        "eligible_modality_registry_status": "supplied_unfrozen_pending_freeze",
        "claims": {
            "C03": _family_receipt(
                "C03", c03, alpha=alpha_value, method_ids=c03_methods
            ),
            "C04": _family_receipt(
                "C04", c04, alpha=alpha_value, method_ids=c04_methods
            ),
            "C05": _family_receipt(
                "C05", c05, alpha=alpha_value, method_ids=c05_methods
            ),
        },
        "methods_completion_status": "not_assessed_development_scaffold",
        "permutation_registry_sha256": None,
        "permutation_registry_status": "unavailable_pending_frozen_registry",
        "oracle_status": "not_run_development_scaffold",
    }
