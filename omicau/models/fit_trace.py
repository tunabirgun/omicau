"""Private fit-trace validation and aggregate-only C08 receipts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
import hashlib
import json
import math
from numbers import Integral, Real
import re
from typing import Any

import numpy as np

from omicau.models.split_plan import ValidatedSplitPlan


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SEMVER_RE = re.compile(r"^[0-9]+\.[0-9]+(?:\.[0-9]+)?$")
_FOLD_RE = re.compile(r"^outer-[0-9]+(?:\.inner-[0-9]+)?$")
_COMPONENTS = {
    "batch_adjuster",
    "calibrator",
    "cox",
    "feature_selector",
    "imputer",
    "latent_factor",
    "model",
    "optimizer",
    "pca",
    "scaler",
    "stacker",
    "support",
    "threshold",
    "variance_filter",
}
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
_EXECUTION_PROFILE_FIELDS = {"planned_counts", "schema_version"}
_EXECUTION_PROFILE_VERSION = "c08_development_execution_profile_v1"
_AUTHORITATIVE_CALLSITE_CATALOG = tuple(
    {
        "callsite_id": callsite_id,
        "component": component,
        "disposition": "traced",
    }
    for callsite_id, component in sorted(_CALLSITE_COMPONENTS.items())
)
_STAGES = {
    "calibration-training",
    "cross-fit-training",
    "inner-training",
    "outer-training",
    "stacking-training",
}
_OUTPUT_SUPPORT_KEYS = {
    "class_count",
    "component_count",
    "event_count",
    "feature_count",
    "group_count",
    "output_count",
    "sample_count",
}
_SEED_REGISTRY_FIELDS = {
    "artifact_sha256",
    "entries",
    "registry_nonce",
    "schema_version",
}
_SEED_ENTRY_FIELDS = {"engine", "purpose", "seed", "seed_id"}
_SEED_ENGINES = {"numpy", "python", "scipy", "torch"}
_SEED_PURPOSES = {"model_initialization", "permutation", "resampling", "split"}
_PRIVATE_TOKENS = {
    "batch",
    "endpoint",
    "fitted_state",
    "group",
    "group_ids",
    "indices",
    "labels",
    "local_paths",
    "path",
    "raw_seed",
    "row_indices",
    "sample_ids",
    "seed",
    "secret",
    "subject_ids",
    "token",
    "credential",
    "password",
}
_SECRET_TOKENS = {"authorization", "credential", "password", "secret", "token"}
_NODE_FIELDS = {
    "assessment_digest",
    "callsite_id",
    "code_digest",
    "component",
    "component_version",
    "environment_digest",
    "fit_digest",
    "fold",
    "input_schema_digest",
    "learned_state_digest",
    "output_support",
    "parameters_digest",
    "parent_node_digests",
    "parent_split_digest",
    "seed_registry_status",
    "stage",
    "state_schema_digest",
    "target_use_flag",
    "validation_digest",
}
_RECORD_FIELDS = {"node", "node_digest"}
_PRIVATE_NODE_INPUT_FIELDS = (_NODE_FIELDS - {"seed_registry_status"}) | {
    "private_seed_registry"
}
_PENDING_SEED_REGISTRY = "unavailable_pending_frozen_registry"
_POISON_TEST_STATUS = "feature_outcome_sentinel_verified"
_POISON_KINDS = frozenset({"assessment_feature", "assessment_outcome"})
_POISON_EVIDENCE_TOKEN = object()
_CACHE_STATUSES = frozenset({"loaded", "reused", "unused"})
_CACHE_EVIDENCE_TOKEN = object()
_STACKING_EVIDENCE_TOKEN = object()
_CACHE_IDENTITY_FIELDS = {
    "assessment_digest",
    "callsite_id",
    "code_digest",
    "component",
    "component_version",
    "environment_digest",
    "fit_digest",
    "fold",
    "input_schema_digest",
    "learned_state_digest",
    "output_support",
    "parameters_digest",
    "parent_node_digests",
    "parent_split_digest",
    "seed_registry_status",
    "stage",
    "state_schema_digest",
    "target_use_flag",
    "validation_digest",
}


class FitTraceError(ValueError):
    """Fail-closed C08 trace or learned-state validation failure."""

    _ALLOWED_CODES = {
        "c08_assessment_ancestry_detected",
        "c08_cache_training_digest_mismatch",
        "c08_callsite_unregistered",
        "c08_fit_trace_incomplete",
        "c08_stacking_in_group_fit",
        "c08_state_uncanonicalizable",
        "c08_static_runtime_inventory_mismatch",
    }

    def __init__(self, code: str, invariant: str):
        if code not in self._ALLOWED_CODES:
            raise ValueError("unsupported_c08_refusal_code")
        self.code = code
        self.invariant = invariant
        super().__init__(code)


def _fail(invariant: str, code: str = "c08_fit_trace_incomplete") -> None:
    raise FitTraceError(code, invariant)


def _canonical_json(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return rendered.encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError):
        _fail("canonical_json")


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{name}_sha256")
    return value


def _label(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        _fail(f"{name}_label")
    lowered = value.lower()
    if (
        re.search(r"^[A-Za-z]:", value)
        or "/" in value
        or "\\" in value
        or "@" in value
        or "://" in value
        or any(token in re.split(r"[_.:-]+", lowered) for token in _SECRET_TOKENS)
    ):
        _fail(f"{name}_path_disclosure")
    return value


def _key(value: Any, name: str) -> str:
    result = _label(value, name)
    if _IDENTIFIER_RE.fullmatch(result) is None:
        _fail(f"{name}_identifier")
    lowered = result.lower()
    if any(token in lowered.split("_") for token in _PRIVATE_TOKENS):
        _fail(f"{name}_private")
    return result


def _component(value: Any, name: str) -> str:
    result = _label(value, name)
    if result not in _COMPONENTS:
        _fail(f"{name}_unregistered", "c08_callsite_unregistered")
    return result


def _callsite_id(value: Any, name: str) -> str:
    try:
        result = _label(value, name)
    except FitTraceError:
        _fail(f"{name}_unregistered", "c08_callsite_unregistered")
    if result not in _CALLSITE_COMPONENTS:
        _fail(f"{name}_unregistered", "c08_callsite_unregistered")
    return result


def _normalize_execution_profile(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _EXECUTION_PROFILE_FIELDS:
        _fail("execution_profile_schema", "c08_static_runtime_inventory_mismatch")
    if value["schema_version"] != _EXECUTION_PROFILE_VERSION:
        _fail("execution_profile_version", "c08_static_runtime_inventory_mismatch")
    counts = value["planned_counts"]
    if not isinstance(counts, Mapping) or set(counts) != set(_CALLSITE_COMPONENTS):
        _fail("execution_profile_callsite_coverage", "c08_static_runtime_inventory_mismatch")
    normalized: dict[str, int] = {}
    for callsite_id in sorted(_CALLSITE_COMPONENTS):
        planned = counts[callsite_id]
        if isinstance(planned, (bool, np.bool_)) or not isinstance(planned, Integral):
            _fail("execution_profile_planned_count_type")
        count = int(planned)
        if count < 0:
            _fail("execution_profile_planned_count_range")
        normalized[callsite_id] = count
    return normalized


def _version(value: Any) -> str:
    result = _label(value, "component_version")
    if _SEMVER_RE.fullmatch(result) is None:
        _fail("component_version_schema")
    return result


def _fold(value: Any) -> str:
    result = _label(value, "fold")
    if _FOLD_RE.fullmatch(result) is None:
        _fail("fold_schema")
    return result


def _stage(value: Any) -> str:
    result = _label(value, "stage")
    if result not in _STAGES:
        _fail("stage_unregistered")
    return result


def _support(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        _fail("output_support_schema")
    result: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        key = _label(raw_key, "output_support_key")
        if _IDENTIFIER_RE.fullmatch(key) is None:
            _fail("output_support_key_identifier")
        if key not in _OUTPUT_SUPPORT_KEYS:
            _fail("output_support_key_unregistered")
        if isinstance(raw_count, (bool, np.bool_)) or not isinstance(raw_count, Integral):
            _fail("output_support_count_type")
        count = int(raw_count)
        if count < 0:
            _fail("output_support_count_range")
        result[key] = count
    return dict(sorted(result.items()))


def _validate_private_seed_registry(value: Any) -> None:
    """Validate private mechanics without treating nonce material as public evidence."""
    if not isinstance(value, Mapping) or set(value) != _SEED_REGISTRY_FIELDS:
        _fail("private_seed_registry_schema")
    if value["schema_version"] != "c08_private_seed_registry_v1":
        _fail("private_seed_registry_version")
    _sha256(value["artifact_sha256"], "seed_registry_artifact")
    nonce = value["registry_nonce"]
    if not isinstance(nonce, bytes) or len(nonce) != 32:
        _fail("private_seed_registry_nonce")
    entries = value["entries"]
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
        _fail("private_seed_registry_entries")
    seen_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != _SEED_ENTRY_FIELDS:
            _fail("private_seed_registry_entry_schema")
        seed_id = _key(entry["seed_id"], "seed_id")
        if seed_id in seen_ids:
            _fail("private_seed_registry_entry_unique")
        seen_ids.add(seed_id)
        engine = entry["engine"]
        purpose = entry["purpose"]
        seed = entry["seed"]
        if engine not in _SEED_ENGINES or purpose not in _SEED_PURPOSES:
            _fail("private_seed_registry_entry_vocabulary")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
            _fail("private_seed_registry_seed_type")
        seed_value = int(seed)
        if seed_value < 0 or seed_value >= 2**64:
            _fail("private_seed_registry_seed_range")


def _normalize_node(node: Any) -> dict[str, Any]:
    if not isinstance(node, Mapping) or set(node) != _NODE_FIELDS:
        _fail("node_schema")
    parents = node["parent_node_digests"]
    if not isinstance(parents, list):
        _fail("parent_node_digests_type")
    normalized_parents = [_sha256(value, "parent_node_digest") for value in parents]
    if len(normalized_parents) != len(set(normalized_parents)):
        _fail("parent_node_digests_unique")
    target_use = node["target_use_flag"]
    if not isinstance(target_use, (bool, np.bool_)):
        _fail("target_use_flag_type")
    callsite_id = _callsite_id(node["callsite_id"], "callsite_id")
    component = _component(node["component"], "component")
    if _CALLSITE_COMPONENTS[callsite_id] != component:
        _fail("callsite_component_exact", "c08_callsite_unregistered")
    return {
        "assessment_digest": _sha256(node["assessment_digest"], "assessment_digest"),
        "callsite_id": callsite_id,
        "code_digest": _sha256(node["code_digest"], "code_digest"),
        "component": component,
        "component_version": _version(node["component_version"]),
        "environment_digest": _sha256(
            node["environment_digest"], "environment_digest"
        ),
        "fit_digest": _sha256(node["fit_digest"], "fit_digest"),
        "fold": _fold(node["fold"]),
        "input_schema_digest": _sha256(node["input_schema_digest"], "input_schema_digest"),
        "learned_state_digest": _sha256(
            node["learned_state_digest"], "learned_state_digest"
        ),
        "output_support": _support(node["output_support"]),
        "parameters_digest": _sha256(node["parameters_digest"], "parameters_digest"),
        "parent_node_digests": normalized_parents,
        "parent_split_digest": _sha256(
            node["parent_split_digest"], "parent_split_digest"
        ),
        "seed_registry_status": (
            node["seed_registry_status"]
            if node["seed_registry_status"] == _PENDING_SEED_REGISTRY
            else _fail("seed_registry_status")
        ),
        "stage": _stage(node["stage"]),
        "state_schema_digest": _sha256(
            node["state_schema_digest"], "state_schema_digest"
        ),
        "target_use_flag": bool(target_use),
        "validation_digest": _sha256(node["validation_digest"], "validation_digest"),
    }


def fit_node_sha256(node: Mapping[str, Any]) -> str:
    """Return the SHA-256 of one strict canonical C08 node."""
    normalized = _normalize_node(node)
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


def _make_private_fit_node(**fields: Any) -> dict[str, Any]:
    """Build one internal trace record from private verifier inputs."""
    if set(fields) != _PRIVATE_NODE_INPUT_FIELDS:
        _fail("private_node_input_schema")
    private_registry = fields.pop("private_seed_registry")
    _validate_private_seed_registry(private_registry)
    node = _normalize_node({**fields, "seed_registry_status": _PENDING_SEED_REGISTRY})
    return {"node": node, "node_digest": fit_node_sha256(node)}


def _canonical_array(value: np.ndarray) -> dict[str, Any]:
    if value.dtype.fields is not None or value.dtype.kind not in "iuf":
        _fail("state_array_dtype", "c08_state_uncanonicalizable")
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        _fail("state_array_nonfinite", "c08_state_uncanonicalizable")
    dtype = value.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(value.astype(dtype, copy=False))
    return {
        "data_hex": canonical.tobytes(order="C").hex(),
        "dtype": canonical.dtype.str,
        "kind": "array",
        "shape": list(canonical.shape),
    }


def _canonical_scalar(value: Any) -> dict[str, str]:
    if isinstance(value, (bool, np.bool_)):
        _fail("state_bool", "c08_state_uncanonicalizable")
    if isinstance(value, Integral):
        return {"kind": "integer", "value": str(int(value))}
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            _fail("state_scalar_nonfinite", "c08_state_uncanonicalizable")
        return {"kind": "float64", "value": number.hex()}
    _fail("state_scalar_type", "c08_state_uncanonicalizable")


def _sign_canonicalize_basis(value: np.ndarray, component_axis: int) -> np.ndarray:
    if value.ndim != 2 or value.dtype.kind != "f" or component_axis not in {0, 1}:
        _fail("pca_basis_contract", "c08_state_uncanonicalizable")
    if not np.isfinite(value).all():
        _fail("pca_basis_nonfinite", "c08_state_uncanonicalizable")
    result = value.copy()
    component_count = result.shape[component_axis]
    for index in range(component_count):
        component = np.take(result, index, axis=component_axis)
        if component.size == 0:
            _fail("pca_basis_empty_component", "c08_state_uncanonicalizable")
        pivot = component.reshape(-1)[int(np.argmax(np.abs(component)))]
        if pivot < 0 or (pivot == 0 and np.signbit(pivot)):
            if component_axis == 0:
                result[index, :] *= -1
            else:
                result[:, index] *= -1
    return result


def canonical_state_sha256(
    state: Mapping[str, Any], *, pca_basis_contract: Mapping[str, int] | None = None
) -> str:
    """Digest exact finite numeric state, optionally fixing PCA-basis signs.

    ``pca_basis_contract`` maps an explicitly named two-dimensional array to
    its component axis (0 for rows, 1 for columns). Only the digest is returned.
    """
    if not isinstance(state, Mapping) or not state:
        _fail("state_schema", "c08_state_uncanonicalizable")
    contract = {} if pca_basis_contract is None else pca_basis_contract
    if not isinstance(contract, Mapping):
        _fail("pca_basis_contract_type", "c08_state_uncanonicalizable")
    normalized_contract: dict[str, int] = {}
    for raw_key, raw_axis in contract.items():
        key = _key(raw_key, "pca_basis_key")
        if isinstance(raw_axis, (bool, np.bool_)) or not isinstance(raw_axis, Integral):
            _fail("pca_basis_axis_type", "c08_state_uncanonicalizable")
        axis = int(raw_axis)
        if axis not in {0, 1}:
            _fail("pca_basis_axis_range", "c08_state_uncanonicalizable")
        normalized_contract[key] = axis
    if not set(normalized_contract) <= set(state):
        _fail("pca_basis_key_missing", "c08_state_uncanonicalizable")

    payload: dict[str, Any] = {}
    for raw_key, raw_value in state.items():
        key = _key(raw_key, "state_key")
        if isinstance(raw_value, np.ndarray):
            value = raw_value
            if key in normalized_contract:
                value = _sign_canonicalize_basis(value, normalized_contract[key])
            payload[key] = _canonical_array(value)
        else:
            if key in normalized_contract:
                _fail("pca_basis_not_array", "c08_state_uncanonicalizable")
            payload[key] = _canonical_scalar(raw_value)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


class StackingPredictionEvidence:
    """Opaque private evidence; group identities are never rendered or serialized."""

    __slots__ = ("__fit_groups", "__node_digest", "__predicted_group", "__token")

    def __init__(
        self,
        node_digest: str,
        predicted_group: Hashable,
        fit_groups: frozenset[Hashable],
        *,
        _token: object,
    ) -> None:
        if _token is not _STACKING_EVIDENCE_TOKEN:
            raise TypeError("stacking_evidence_requires_verification")
        self.__node_digest = node_digest
        self.__predicted_group = predicted_group
        self.__fit_groups = fit_groups
        self.__token = _token

    def __repr__(self) -> str:
        return "StackingPredictionEvidence(verified=True)"

    def __copy__(self) -> None:
        raise TypeError("private_evidence_copy_forbidden")

    def __deepcopy__(self, memo: Any) -> None:
        raise TypeError("private_evidence_copy_forbidden")

    def __reduce_ex__(self, protocol: int) -> None:
        raise TypeError("private_evidence_serialization_forbidden")

    def __getstate__(self) -> None:
        raise TypeError("private_evidence_serialization_forbidden")


class CacheUseEvidence:
    """Opaque private cache-use evidence that cannot render its identity."""

    __slots__ = ("__identity", "__status", "__token")

    def __init__(
        self, status: str, identity: Mapping[str, Any] | None, *, _token: object
    ) -> None:
        if _token is not _CACHE_EVIDENCE_TOKEN:
            raise TypeError("cache_evidence_requires_verification")
        self.__status = status
        self.__identity = identity
        self.__token = _token

    def __repr__(self) -> str:
        return "CacheUseEvidence(verified=True)"

    def __copy__(self) -> None:
        raise TypeError("private_evidence_copy_forbidden")

    def __deepcopy__(self, memo: Any) -> None:
        raise TypeError("private_evidence_copy_forbidden")

    def __reduce_ex__(self, protocol: int) -> None:
        raise TypeError("private_evidence_serialization_forbidden")

    def __getstate__(self) -> None:
        raise TypeError("private_evidence_serialization_forbidden")


def verify_stacking_prediction(
    node_digest: str,
    predicted_group: Hashable,
    fit_groups: frozenset[Hashable],
) -> StackingPredictionEvidence:
    """Verify one private out-of-group stacking prediction declaration."""
    digest = _sha256(node_digest, "stacking_node_digest")
    if not isinstance(fit_groups, frozenset) or not fit_groups:
        _fail("stacking_fit_groups_schema", "c08_stacking_in_group_fit")
    try:
        hash(predicted_group)
        if predicted_group in fit_groups:
            _fail("stacking_predicted_group_exclusion", "c08_stacking_in_group_fit")
    except TypeError:
        _fail("stacking_predicted_group_hashable", "c08_stacking_in_group_fit")
    return StackingPredictionEvidence(
        digest,
        predicted_group,
        fit_groups,
        _token=_STACKING_EVIDENCE_TOKEN,
    )


def _stacking_evidence_values(
    evidence: StackingPredictionEvidence,
) -> tuple[str, Hashable, frozenset[Hashable]]:
    if evidence._StackingPredictionEvidence__token is not _STACKING_EVIDENCE_TOKEN:
        _fail("stacking_evidence_token", "c08_stacking_in_group_fit")
    return (
        evidence._StackingPredictionEvidence__node_digest,
        evidence._StackingPredictionEvidence__predicted_group,
        evidence._StackingPredictionEvidence__fit_groups,
    )


class PoisonEvidence:
    """Opaque evidence created only after one poison check succeeds."""

    __slots__ = ("__kind", "__token")

    def __init__(self, kind: str, *, _token: object) -> None:
        if _token is not _POISON_EVIDENCE_TOKEN or kind not in _POISON_KINDS:
            raise TypeError("poison_evidence_requires_verification")
        self.__kind = kind
        self.__token = _token

    def __repr__(self) -> str:
        return "PoisonEvidence(verified=True)"

    def _verified_kind(self) -> str:
        if self.__token is not _POISON_EVIDENCE_TOKEN:
            _fail("poison_evidence_token")
        return self.__kind


def _validate_dag(records: Sequence[dict[str, Any]]) -> None:
    by_digest = {record["node_digest"]: record["node"] for record in records}
    if len(by_digest) != len(records):
        _fail("node_digest_unique")
    for digest, node in by_digest.items():
        parents = node["parent_node_digests"]
        if digest in parents:
            _fail("node_self_parent")
        if not set(parents) <= set(by_digest):
            _fail("parent_node_missing")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(digest: str) -> None:
        if digest in visiting:
            _fail("trace_cycle")
        if digest in visited:
            return
        visiting.add(digest)
        for parent in by_digest[digest]["parent_node_digests"]:
            visit(parent)
        visiting.remove(digest)
        visited.add(digest)

    for digest in by_digest:
        visit(digest)


def _normalize_records(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        _fail("trace_nonempty")
    preliminary: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != _RECORD_FIELDS:
            _fail("record_schema")
        declared = _sha256(record["node_digest"], "node_digest")
        node = _normalize_node(record["node"])
        preliminary.append({"node": node, "node_digest": declared})
    _validate_dag(preliminary)
    for record in preliminary:
        if fit_node_sha256(record["node"]) != record["node_digest"]:
            _fail("node_digest_mismatch")
    return sorted(preliminary, key=lambda item: item["node_digest"])


def _private_sets(
    values: Mapping[str, Iterable[Hashable]], node_digests: set[str], name: str
) -> dict[str, set[Hashable]]:
    if not isinstance(values, Mapping) or set(values) != node_digests:
        _fail(f"{name}_coverage")
    result: dict[str, set[Hashable]] = {}
    for digest, members in values.items():
        if isinstance(members, (str, bytes)):
            _fail(f"{name}_set")
        try:
            result[digest] = set(members)
        except TypeError:
            _fail(f"{name}_hashable")
        if not result[digest]:
            _fail(f"{name}_nonempty")
    return result


def _cache_identity(node: Mapping[str, Any]) -> dict[str, Any]:
    return {key: node[key] for key in sorted(_NODE_FIELDS)}


def _validate_cache_identity(value: Any, expected: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != _CACHE_IDENTITY_FIELDS:
        _fail("cache_identity_schema", "c08_cache_training_digest_mismatch")
    try:
        supplied_bytes = _canonical_json(value)
    except FitTraceError:
        _fail("cache_identity_canonical", "c08_cache_training_digest_mismatch")
    if supplied_bytes != _canonical_json(expected):
        _fail("cache_identity_exact", "c08_cache_training_digest_mismatch")


def verify_cache_use(
    status: str,
    *,
    node: Mapping[str, Any],
    cached_identity: Mapping[str, Any] | None = None,
) -> CacheUseEvidence:
    """Verify one private cache-use declaration without exposing its identity."""
    normalized = _normalize_node(node)
    expected = _cache_identity(normalized)
    if status not in _CACHE_STATUSES:
        _fail("cache_use_schema", "c08_cache_training_digest_mismatch")
    if status == "unused":
        if cached_identity is not None:
            _fail("unused_cache_identity_forbidden", "c08_cache_training_digest_mismatch")
        identity = None
    else:
        if cached_identity is None:
            _fail("used_cache_identity_required", "c08_cache_training_digest_mismatch")
        _validate_cache_identity(cached_identity, expected)
        identity = expected
    return CacheUseEvidence(status, identity, _token=_CACHE_EVIDENCE_TOKEN)


def _cache_evidence_values(
    evidence: CacheUseEvidence,
) -> tuple[str, Mapping[str, Any] | None]:
    if evidence._CacheUseEvidence__token is not _CACHE_EVIDENCE_TOKEN:
        _fail("cache_evidence_token", "c08_cache_training_digest_mismatch")
    return (
        evidence._CacheUseEvidence__status,
        evidence._CacheUseEvidence__identity,
    )


def _validate_poison_suite(value: Any) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail("poison_evidence_suite")
    kinds: list[str] = []
    for evidence in value:
        if type(evidence) is not PoisonEvidence:
            _fail("poison_evidence_schema")
        kinds.append(evidence._verified_kind())
    if len(kinds) != len(set(kinds)) or set(kinds) != _POISON_KINDS:
        _fail("poison_evidence_complete")


def _validate_cache_evidence(
    value: Any,
    node_digests: set[str],
    by_digest: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(value, Mapping) or set(value) != node_digests:
        _fail("cache_use_node_coverage", "c08_cache_training_digest_mismatch")
    for digest, evidence in value.items():
        if type(evidence) is not CacheUseEvidence:
            _fail("cache_use_schema", "c08_cache_training_digest_mismatch")
        status, identity = _cache_evidence_values(evidence)
        if status not in _CACHE_STATUSES:
            _fail("cache_use_schema", "c08_cache_training_digest_mismatch")
        if status == "unused":
            if identity is not None:
                _fail("unused_cache_identity_forbidden", "c08_cache_training_digest_mismatch")
        else:
            if identity is None:
                _fail("used_cache_identity_required", "c08_cache_training_digest_mismatch")
            _validate_cache_identity(
                identity,
                _cache_identity(by_digest[digest]),
            )


def validate_fit_trace(
    records: Sequence[Mapping[str, Any]],
    *,
    execution_profile: Mapping[str, Any],
    fit_ancestry: Mapping[str, Iterable[Hashable]],
    assessment_ancestry: Mapping[str, Iterable[Hashable]],
    private_seed_registries: Mapping[str, Mapping[str, Any]],
    split_plan: ValidatedSplitPlan,
    poison_evidence: Sequence[PoisonEvidence],
    cache_evidence: Mapping[str, CacheUseEvidence],
    stacking_predictions: Sequence[StackingPredictionEvidence] = (),
) -> dict[str, Any]:
    """Validate a trace with private ancestry and return aggregate-only evidence."""
    normalized = _normalize_records(records)
    node_digests = {record["node_digest"] for record in normalized}
    planned_counts = _normalize_execution_profile(execution_profile)
    planned = Counter({key: count for key, count in planned_counts.items() if count})
    runtime = Counter(record["node"]["callsite_id"] for record in normalized)
    if planned != runtime:
        _fail(
            "planned_runtime_callsite_counts_exact",
            "c08_static_runtime_inventory_mismatch",
        )

    if type(split_plan) is not ValidatedSplitPlan:
        _fail("validated_split_plan_required")
    split_receipt = split_plan.receipt()
    if (
        not isinstance(split_receipt, Mapping)
        or split_receipt.get("claim_id") != "C06"
        or split_receipt.get("decision") != "eligible"
        or split_receipt.get("verifier_status") != "verified"
    ):
        _fail("validated_split_plan_receipt")
    split_digest = _sha256(
        split_receipt.get("split_manifest_sha256"), "split_manifest"
    )
    if any(
        record["node"]["parent_split_digest"] != split_digest
        for record in normalized
    ):
        _fail("node_split_manifest_exact")
    _validate_poison_suite(poison_evidence)

    fit_sets = _private_sets(fit_ancestry, node_digests, "fit_ancestry")
    assessment_sets = _private_sets(
        assessment_ancestry, node_digests, "assessment_ancestry"
    )
    if any(fit_sets[digest] & assessment_sets[digest] for digest in node_digests):
        _fail("fit_assessment_group_disjoint", "c08_assessment_ancestry_detected")

    if not isinstance(private_seed_registries, Mapping) or set(
        private_seed_registries
    ) != node_digests:
        _fail("private_seed_registry_coverage")
    by_digest = {record["node_digest"]: record["node"] for record in normalized}
    for private_registry in private_seed_registries.values():
        _validate_private_seed_registry(private_registry)

    _validate_cache_evidence(cache_evidence, node_digests, by_digest)

    if isinstance(stacking_predictions, (str, bytes)):
        _fail("stacking_prediction_evidence_type", "c08_stacking_in_group_fit")
    stacking_nodes = {
        digest: node
        for digest, node in by_digest.items()
        if node["component"] == "stacker"
    }
    evidence_by_node: dict[str, list[StackingPredictionEvidence]] = {
        digest: [] for digest in stacking_nodes
    }
    seen_predictions: set[tuple[str, Hashable]] = set()
    for evidence in stacking_predictions:
        if type(evidence) is not StackingPredictionEvidence:
            _fail("stacking_prediction_evidence_schema", "c08_stacking_in_group_fit")
        digest, predicted_group, fit_groups = _stacking_evidence_values(evidence)
        if digest not in stacking_nodes:
            _fail("stacking_node_missing", "c08_stacking_in_group_fit")
        if fit_groups != frozenset(fit_sets[digest]):
            _fail("stacking_fit_ancestry_exact", "c08_stacking_in_group_fit")
        try:
            hash(predicted_group)
            if predicted_group in fit_groups:
                _fail("stacking_predicted_group_exclusion", "c08_stacking_in_group_fit")
            evidence_key = (digest, predicted_group)
            if evidence_key in seen_predictions:
                _fail("stacking_predicted_group_unique", "c08_stacking_in_group_fit")
            seen_predictions.add(evidence_key)
        except TypeError:
            _fail("stacking_predicted_group_hashable", "c08_stacking_in_group_fit")
        evidence_by_node[digest].append(evidence)
    for digest, node in stacking_nodes.items():
        expected = node["output_support"].get("group_count")
        if expected is None or expected < 1:
            _fail("stacking_output_group_count", "c08_stacking_in_group_fit")
        if len(evidence_by_node[digest]) != expected:
            _fail("stacking_prediction_evidence_complete", "c08_stacking_in_group_fit")

    inventory_sha256 = hashlib.sha256(
        _canonical_json(
            {
                "catalog": _AUTHORITATIVE_CALLSITE_CATALOG,
                "execution_profile": {
                    "planned_counts": planned_counts,
                    "schema_version": _EXECUTION_PROFILE_VERSION,
                },
            }
        )
    ).hexdigest()
    trace_sha256 = hashlib.sha256(_canonical_json(normalized)).hexdigest()
    return {
        "callsite_inventory_sha256": inventory_sha256,
        "claim_id": "C08",
        "decision": "development_only",
        "fit_trace_sha256": trace_sha256,
        "node_count": len(normalized),
        "poison_test_status": _POISON_TEST_STATUS,
        "seed_registry_status": _PENDING_SEED_REGISTRY,
        "split_manifest_sha256": split_digest,
        "state_digest_count": len(normalized),
        "verifier_status": "development_pending_production_inventory_and_ancestry_binding",
    }


def _prediction_array(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        _fail(f"{name}_type")
    if array.dtype.kind not in "iuf" or array.dtype.kind == "f" and not np.isfinite(array).all():
        _fail(f"{name}_numeric_finite")
    return array


def verify_poison_result(
    poison_kind: str,
    *,
    baseline_state_digest: str,
    poisoned_state_digest: str,
    baseline_predictions: Any,
    poisoned_predictions: Any,
    baseline_sentinel_predictions: Any | None = None,
    poisoned_sentinel_predictions: Any | None = None,
) -> PoisonEvidence:
    """Apply the fixed C08 feature, outcome, and sentinel poison semantics."""
    baseline_state = _sha256(baseline_state_digest, "baseline_state_digest")
    poisoned_state = _sha256(poisoned_state_digest, "poisoned_state_digest")
    if baseline_state != poisoned_state:
        _fail("poison_learned_state_invariance")
    baseline = _prediction_array(baseline_predictions, "baseline_predictions")
    poisoned = _prediction_array(poisoned_predictions, "poisoned_predictions")
    if baseline.shape != poisoned.shape:
        _fail("poison_prediction_shape")

    if poison_kind == "assessment_outcome":
        if not np.array_equal(baseline, poisoned, equal_nan=False):
            _fail("assessment_outcome_prediction_invariance")
        if baseline_sentinel_predictions is not None or poisoned_sentinel_predictions is not None:
            _fail("assessment_outcome_sentinel_not_applicable")
    elif poison_kind == "assessment_feature":
        if baseline_sentinel_predictions is None or poisoned_sentinel_predictions is None:
            _fail("assessment_feature_sentinel_required")
        sentinel_before = _prediction_array(
            baseline_sentinel_predictions, "baseline_sentinel_predictions"
        )
        sentinel_after = _prediction_array(
            poisoned_sentinel_predictions, "poisoned_sentinel_predictions"
        )
        if sentinel_before.shape != sentinel_after.shape or not np.array_equal(
            sentinel_before, sentinel_after, equal_nan=False
        ):
            _fail("assessment_feature_sentinel_prediction_invariance")
    else:
        _fail("poison_kind")
    return PoisonEvidence(poison_kind, _token=_POISON_EVIDENCE_TOKEN)
