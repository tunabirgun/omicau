"""Development-only groupwise control mechanics for the C07 benchmark claim."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from numbers import Integral, Real
import random
import re
from typing import Any

import numpy as np


class GroupControlError(ValueError):
    """Public-safe refusal from a C07 development primitive."""

    _CODES = {
        "c07_contract_invalid",
        "c07_endpoint_invalid",
        "c07_exact_copy_input_invalid",
        "c07_fit_ancestry_invalid",
        "c07_group_outcome_mixed",
        "c07_no_nontrivial_group_permutation",
        "c07_permutation_scope_invalid",
        "c07_stratum_target_derived",
        "c07_unregistered_transform",
    }

    def __init__(self, code: str, invariant: str):
        if code not in self._CODES:
            raise ValueError("unsupported_c07_refusal_code")
        self.code = code
        self.invariant = invariant
        super().__init__(code)


def _fail(code: str, invariant: str) -> None:
    raise GroupControlError(code, invariant)


def _canonical_sha256(value: Any, invariant: str) -> str:
    def normalize(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool)):
            return item
        if isinstance(item, Integral) and not isinstance(item, (bool, np.bool_)):
            return int(item)
        if isinstance(item, Real) and not isinstance(item, (bool, np.bool_)):
            result = float(item)
            if not math.isfinite(result):
                _fail("c07_contract_invalid", invariant)
            return result
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                _fail("c07_contract_invalid", invariant)
            return {key: normalize(item[key]) for key in sorted(item)}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        _fail("c07_contract_invalid", invariant)

    normalized = normalize(value)
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: Any, invariant: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        _fail("c07_contract_invalid", invariant)
    try:
        raw = bytes.fromhex(value)
    except ValueError:
        _fail("c07_contract_invalid", invariant)
    if raw.hex() != value.lower():
        _fail("c07_contract_invalid", invariant)
    return value.lower()


_PENDING_REGISTRY_STATUS = "unavailable_pending_frozen_registry"
_TECHNICAL_NODE_ID_RE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")
_SENSITIVE_NODE_ID_TOKENS = (
    "api_key",
    "apikey",
    "credential",
    "passwd",
    "password",
    "secret",
    "token",
)


def _validate_private_registry_binding(value: Any, purpose: str) -> str:
    expected = {
        "artifact",
        "entries",
        "nonce_hex",
        "purpose",
        "registry_id",
        "schema_version",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        _fail("c07_contract_invalid", "private_registry_schema")
    if value["schema_version"] != "c07_private_registry_binding_v1":
        _fail("c07_contract_invalid", "private_registry_schema_version")
    if value["purpose"] != purpose:
        _fail("c07_contract_invalid", "private_registry_purpose")
    if not isinstance(value["registry_id"], str) or not value["registry_id"]:
        _fail("c07_contract_invalid", "private_registry_id")
    nonce = value["nonce_hex"]
    if not isinstance(nonce, str) or len(nonce) != 64:
        _fail("c07_contract_invalid", "private_registry_nonce")
    try:
        nonce_bytes = bytes.fromhex(nonce)
    except ValueError:
        _fail("c07_contract_invalid", "private_registry_nonce")
    if nonce_bytes.hex() != nonce.lower() or len(set(nonce_bytes)) < 8:
        _fail("c07_contract_invalid", "private_registry_nonce")
    artifact = value["artifact"]
    if not isinstance(artifact, Mapping) or set(artifact) != {"artifact_id", "sha256"}:
        _fail("c07_contract_invalid", "private_registry_artifact_schema")
    if not isinstance(artifact["artifact_id"], str) or not artifact["artifact_id"]:
        _fail("c07_contract_invalid", "private_registry_artifact_id")
    _sha256_text(artifact["sha256"], "private_registry_artifact_sha256")
    entries = value["entries"]
    if not isinstance(entries, list) or not entries:
        _fail("c07_contract_invalid", "private_registry_entries")
    entry_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"artifact_sha256", "entry_id"}:
            _fail("c07_contract_invalid", "private_registry_entry_schema")
        entry_id = entry["entry_id"]
        if not isinstance(entry_id, str) or not entry_id or entry_id in entry_ids:
            _fail("c07_contract_invalid", "private_registry_entry_id")
        entry_ids.add(entry_id)
        _sha256_text(entry["artifact_sha256"], "private_registry_entry_sha256")
    _canonical_sha256(value, "private_registry_canonical")
    return _PENDING_REGISTRY_STATUS


def _contract_sha256(value: Any, invariant: str) -> str:
    def inspect(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if isinstance(key, str) and any(
                    token in key.lower() for token in ("seed", "nonce", "random_state")
                ):
                    _fail("c07_contract_invalid", "seed_bearing_public_contract")
                inspect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                inspect(child)

    inspect(value)
    return _canonical_sha256(value, invariant)


def _technical_node_id(value: Any, invariant: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 64
        or _TECHNICAL_NODE_ID_RE.fullmatch(value) is None
        or any(
            value == token
            or value.startswith(f"{token}_")
            or value.endswith(f"_{token}")
            or f"_{token}_" in value
            for token in _SENSITIVE_NODE_ID_TOKENS
        )
    ):
        _fail("c07_fit_ancestry_invalid", invariant)
    return value


def _identifier(value: Any, invariant: str) -> str | int:
    if isinstance(value, (bool, np.bool_)):
        _fail("c07_endpoint_invalid", invariant)
    if isinstance(value, str):
        if not value or len(value) > 1024:
            _fail("c07_endpoint_invalid", invariant)
        return value
    if isinstance(value, Integral):
        return int(value)
    _fail("c07_endpoint_invalid", invariant)


def _id_key(value: str | int) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _indices(values: Sequence[Any], n_rows: int) -> list[int]:
    if isinstance(values, (str, bytes)):
        _fail("c07_permutation_scope_invalid", "outer_train_indices_type")
    result: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            _fail("c07_permutation_scope_invalid", "outer_train_index_type")
        index = int(value)
        if index < 0 or index >= n_rows:
            _fail("c07_permutation_scope_invalid", "outer_train_index_range")
        result.append(index)
    if not result or len(result) != len(set(result)):
        _fail("c07_permutation_scope_invalid", "outer_train_indices_nonempty_unique")
    return result


def _classification_value(value: Any, invariant: str) -> str | int | float:
    if isinstance(value, (bool, np.bool_)):
        _fail("c07_endpoint_invalid", invariant)
    if isinstance(value, str):
        if not value or len(value) > 4096:
            _fail("c07_endpoint_invalid", invariant)
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if math.isfinite(result):
            return result
    _fail("c07_endpoint_invalid", invariant)


def _finite_real(value: Any, invariant: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        _fail("c07_endpoint_invalid", invariant)
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        _fail("c07_endpoint_invalid", invariant)
    return result


def _endpoint_rows(
    task: str,
    n_rows: int,
    *,
    y: Sequence[Any] | None,
    time: Sequence[Any] | None,
    event: Sequence[Any] | None,
) -> tuple[list[Any], dict[str, np.ndarray]]:
    if task not in {"classification", "regression", "survival"}:
        _fail("c07_endpoint_invalid", "task")
    if task == "survival":
        if y is not None or time is None or event is None:
            _fail("c07_endpoint_invalid", "survival_fields")
        if len(time) != n_rows or len(event) != n_rows:
            _fail("c07_endpoint_invalid", "survival_shape")
        times = np.asarray(
            [_finite_real(value, "survival_time", positive=True) for value in time], dtype=float
        )
        events_list = []
        for value in event:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                _fail("c07_endpoint_invalid", "survival_event_type")
            item = int(value)
            if item not in {0, 1}:
                _fail("c07_endpoint_invalid", "survival_event_domain")
            events_list.append(item)
        events = np.asarray(events_list, dtype=np.int8)
        return list(zip(times.tolist(), events.tolist())), {"time": times, "event": events}
    if y is None or time is not None or event is not None or len(y) != n_rows:
        _fail("c07_endpoint_invalid", "endpoint_fields")
    if task == "classification":
        values = [_classification_value(value, "classification_value") for value in y]
        return values, {"y": np.asarray(values, dtype=object)}
    values = [_finite_real(value, "regression_value") for value in y]
    array = np.asarray(values, dtype=float)
    return values, {"y": array}


def _group_context(
    *,
    groups: Sequence[Any],
    strata: Sequence[Any],
    task: str,
    outer_train_indices: Sequence[Any],
    y: Sequence[Any] | None,
    time: Sequence[Any] | None,
    event: Sequence[Any] | None,
) -> dict[str, Any]:
    if isinstance(groups, (str, bytes)) or not groups:
        _fail("c07_endpoint_invalid", "groups_type")
    n_rows = len(groups)
    if len(strata) != n_rows:
        _fail("c07_endpoint_invalid", "strata_shape")
    group_rows: dict[str | int, list[int]] = {}
    normalized_groups: list[str | int] = []
    normalized_strata: list[str | int] = []
    for index, value in enumerate(groups):
        group = _identifier(value, "group_id")
        normalized_groups.append(group)
        group_rows.setdefault(group, []).append(index)
        normalized_strata.append(_identifier(strata[index], "stratum_id"))
    objects, arrays = _endpoint_rows(task, n_rows, y=y, time=time, event=event)
    group_objects: dict[str | int, Any] = {}
    group_strata: dict[str | int, str | int] = {}
    for group, rows in group_rows.items():
        first_object = objects[rows[0]]
        if any(objects[index] != first_object for index in rows[1:]):
            _fail("c07_group_outcome_mixed", "one_endpoint_object_per_group")
        first_stratum = normalized_strata[rows[0]]
        if any(normalized_strata[index] != first_stratum for index in rows[1:]):
            _fail("c07_permutation_scope_invalid", "one_design_stratum_per_group")
        group_objects[group] = first_object
        group_strata[group] = first_stratum
    train_indices = _indices(outer_train_indices, n_rows)
    train_set = set(train_indices)
    train_groups = {
        group for group, rows in group_rows.items() if any(index in train_set for index in rows)
    }
    if any(not set(group_rows[group]) <= train_set for group in train_groups):
        _fail("c07_permutation_scope_invalid", "outer_train_contains_complete_groups")
    return {
        "arrays": arrays,
        "group_objects": group_objects,
        "group_rows": group_rows,
        "group_strata": group_strata,
        "groups": normalized_groups,
        "n_rows": n_rows,
        "train_groups": train_groups,
        "train_indices": train_indices,
    }


def _bind_contracts(
    strata_schema: Mapping[str, Any],
    permutation_registry: Mapping[str, Any],
    exchangeability_contract: Mapping[str, Any],
) -> tuple[str, str, str]:
    if not isinstance(strata_schema, Mapping) or strata_schema.get("target_derived") is not False:
        _fail("c07_stratum_target_derived", "design_strata_must_be_target_independent")
    fields = strata_schema.get("fields")
    if (
        not isinstance(strata_schema.get("name"), str)
        or not isinstance(strata_schema.get("version"), str)
        or not isinstance(fields, list)
        or not fields
        or any(not isinstance(field, str) or not field for field in fields)
    ):
        _fail("c07_contract_invalid", "strata_schema")
    if not isinstance(exchangeability_contract, Mapping) or not exchangeability_contract:
        _fail("c07_contract_invalid", "exchangeability_contract")
    return (
        _canonical_sha256(strata_schema, "strata_schema_canonical"),
        _validate_private_registry_binding(
            permutation_registry, "group_endpoint_permutation"
        ),
        _contract_sha256(exchangeability_contract, "exchangeability_contract_canonical"),
    )


def _blocks(context: Mapping[str, Any]) -> list[list[str | int]]:
    by_stratum: dict[str | int, list[str | int]] = {}
    for group in context["train_groups"]:
        by_stratum.setdefault(context["group_strata"][group], []).append(group)
    return [
        sorted(by_stratum[stratum], key=_id_key)
        for stratum in sorted(by_stratum, key=_id_key)
    ]


def _object_key(value: Any) -> tuple[str, str]:
    if isinstance(value, tuple):
        return "tuple", json.dumps(value, separators=(",", ":"), allow_nan=False)
    return type(value).__name__, repr(value)


def _multiset_count(values: Sequence[Any]) -> int:
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return math.factorial(len(values)) // math.prod(
        math.factorial(count) for count in counts.values()
    )


def _multiset_rank(values: Sequence[Any]) -> int:
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    rank = 0
    for value in values:
        actual = next(candidate for candidate in counts if candidate == value)
        for candidate in sorted(counts, key=_object_key):
            if _object_key(candidate) >= _object_key(actual) or counts[candidate] == 0:
                continue
            counts[candidate] -= 1
            rank += math.factorial(sum(counts.values())) // math.prod(
                math.factorial(count) for count in counts.values()
            )
            counts[candidate] += 1
        counts[actual] -= 1
    return rank


def _unrank_multiset(values: Sequence[Any], rank: int) -> list[Any]:
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    result: list[Any] = []
    while sum(counts.values()):
        for candidate in sorted(counts, key=_object_key):
            if counts[candidate] == 0:
                continue
            counts[candidate] -= 1
            ways = math.factorial(sum(counts.values())) // math.prod(
                math.factorial(count) for count in counts.values()
            )
            if rank < ways:
                result.append(candidate)
                break
            rank -= ways
            counts[candidate] += 1
        else:
            _fail("c07_contract_invalid", "multiset_rank_range")
    return result


def _attainable(
    blocks: Sequence[Sequence[Any]], group_objects: Mapping[Any, Any], minimum: int
) -> tuple[int, list[int]]:
    radices = [
        _multiset_count([group_objects[group] for group in block]) for block in blocks
    ]
    distinct_nonidentity = math.prod(radices) - 1
    if distinct_nonidentity < minimum:
        _fail(
            "c07_no_nontrivial_group_permutation",
            "minimum_distinct_nonidentity_assignments",
        )
    return distinct_nonidentity, radices


def _observable_assignment(
    blocks: Sequence[Sequence[Any]],
    group_objects: Mapping[Any, Any],
    radices: Sequence[int],
    seed: int,
) -> dict[Any, Any]:
    identity_rank = 0
    for block, radix in zip(blocks, radices):
        identity_rank = identity_rank * radix + _multiset_rank(
            [group_objects[group] for group in block]
        )
    total = math.prod(radices)
    selected_rank = random.Random(seed).randrange(total - 1)
    if selected_rank >= identity_rank:
        selected_rank += 1
    digits = [0] * len(blocks)
    for index in range(len(blocks) - 1, -1, -1):
        digits[index] = selected_rank % radices[index]
        selected_rank //= radices[index]
    assignment: dict[Any, Any] = {}
    for block, digit in zip(blocks, digits):
        values = _unrank_multiset([group_objects[group] for group in block], digit)
        assignment.update(zip(block, values))
    return assignment


def _minimum_distinct(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        _fail("c07_contract_invalid", "minimum_distinct_nonidentity_assignments_type")
    result = int(value)
    if result < 1:
        _fail("c07_contract_invalid", "minimum_distinct_nonidentity_assignments_range")
    return result


def _candidate_objects(
    task: str, candidate: Mapping[str, Sequence[Any]], train_length: int
) -> list[Any]:
    if not isinstance(candidate, Mapping):
        _fail("c07_permutation_scope_invalid", "candidate_schema")
    expected = {"time", "event"} if task == "survival" else {"y"}
    if set(candidate) != expected:
        _fail("c07_permutation_scope_invalid", "candidate_schema")
    if task == "survival":
        return _endpoint_rows(
            task,
            train_length,
            y=None,
            time=candidate["time"],
            event=candidate["event"],
        )[0]
    return _endpoint_rows(
        task,
        train_length,
        y=candidate["y"],
        time=None,
        event=None,
    )[0]


def _validate_candidate(context: Mapping[str, Any], task: str, candidate: Mapping[str, Any]) -> None:
    objects = _candidate_objects(task, candidate, len(context["train_indices"]))
    by_row = dict(zip(context["train_indices"], objects))
    candidate_by_group: dict[Any, Any] = {}
    for group in context["train_groups"]:
        values = [by_row[index] for index in context["group_rows"][group]]
        if any(value != values[0] for value in values[1:]):
            _fail("c07_permutation_scope_invalid", "candidate_rowwise_not_groupwise")
        candidate_by_group[group] = values[0]
    changed = False
    for block in _blocks(context):
        original = sorted(repr(context["group_objects"][group]) for group in block)
        proposed = sorted(repr(candidate_by_group[group]) for group in block)
        if proposed != original:
            _fail("c07_permutation_scope_invalid", "candidate_not_blockwise_bijection")
        changed |= any(
            candidate_by_group[group] != context["group_objects"][group] for group in block
        )
    if not changed:
        _fail("c07_no_nontrivial_group_permutation", "candidate_identity")


def validate_group_endpoint_permutation(
    *,
    groups: Sequence[Any],
    strata: Sequence[Any],
    strata_schema: Mapping[str, Any],
    permutation_registry: Mapping[str, Any],
    exchangeability_contract: Mapping[str, Any],
    task: str,
    outer_train_indices: Sequence[Any],
    minimum_distinct_nonidentity_assignments: int,
    candidate: Mapping[str, Sequence[Any]],
    y: Sequence[Any] | None = None,
    time: Sequence[Any] | None = None,
    event: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Audit a supplied training-only candidate as a whole-group blockwise permutation."""
    schema_sha, registry_sha, contract_sha = _bind_contracts(
        strata_schema, permutation_registry, exchangeability_contract
    )
    context = _group_context(
        groups=groups,
        strata=strata,
        task=task,
        outer_train_indices=outer_train_indices,
        y=y,
        time=time,
        event=event,
    )
    minimum = _minimum_distinct(minimum_distinct_nonidentity_assignments)
    blocks = _blocks(context)
    attainable, _ = _attainable(blocks, context["group_objects"], minimum)
    _validate_candidate(context, task, candidate)
    return {
        "attainable_permutation_count": attainable,
        "claim_id": "C07",
        "control_kind": "group_endpoint_permutation_audit",
        "decision": "valid",
        "exchangeability_contract_sha256": contract_sha,
        "outer_train_group_count": len(context["train_groups"]),
        "minimum_distinct_nonidentity_assignments": minimum,
        "permutation_registry_sha256": None,
        "permutation_registry_status": registry_sha,
        "strata_schema_sha256": schema_sha,
        "stratum_count": len(blocks),
        "transform_role": "stress_control_only_not_inferential_randomization",
    }


def execute_group_endpoint_permutation(
    *,
    groups: Sequence[Any],
    strata: Sequence[Any],
    strata_schema: Mapping[str, Any],
    permutation_registry: Mapping[str, Any],
    exchangeability_contract: Mapping[str, Any],
    task: str,
    outer_train_indices: Sequence[Any],
    minimum_distinct_nonidentity_assignments: int,
    seed: int,
    consume: Callable[[Mapping[str, np.ndarray]], Any],
    y: Sequence[Any] | None = None,
    time: Sequence[Any] | None = None,
    event: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Generate a forced-nonidentity stress/control transform, not randomization inference."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        _fail("c07_contract_invalid", "seed_type")
    if not callable(consume):
        _fail("c07_contract_invalid", "consumer_type")
    schema_sha, registry_sha, contract_sha = _bind_contracts(
        strata_schema, permutation_registry, exchangeability_contract
    )
    context = _group_context(
        groups=groups,
        strata=strata,
        task=task,
        outer_train_indices=outer_train_indices,
        y=y,
        time=time,
        event=event,
    )
    minimum = _minimum_distinct(minimum_distinct_nonidentity_assignments)
    blocks = _blocks(context)
    attainable, radices = _attainable(blocks, context["group_objects"], minimum)
    permuted_objects = _observable_assignment(
        blocks, context["group_objects"], radices, int(seed)
    )
    train_objects = [
        permuted_objects[context["groups"][index]] for index in context["train_indices"]
    ]
    if task == "survival":
        candidate = {
            "time": np.asarray([value[0] for value in train_objects], dtype=float),
            "event": np.asarray([value[1] for value in train_objects], dtype=np.int8),
        }
    else:
        dtype = object if task == "classification" else float
        candidate = {"y": np.asarray(train_objects, dtype=dtype)}
    _validate_candidate(context, task, candidate)
    assessment = sorted(set(range(context["n_rows"])) - set(context["train_indices"]))
    before = {
        key: array[assessment].copy() for key, array in context["arrays"].items()
    }
    consume({key: value.copy() for key, value in candidate.items()})
    _, current_arrays = _endpoint_rows(
        task,
        context["n_rows"],
        y=y,
        time=time,
        event=event,
    )
    if any(not np.array_equal(before[key], current_arrays[key][assessment]) for key in before):
        _fail("c07_permutation_scope_invalid", "assessment_outcomes_unchanged")
    return {
        "attainable_permutation_count": attainable,
        "claim_id": "C07",
        "control_kind": "group_endpoint_permutation",
        "decision": "eligible",
        "exchangeability_contract_sha256": contract_sha,
        "outer_train_group_count": len(context["train_groups"]),
        "minimum_distinct_nonidentity_assignments": minimum,
        "permutation_registry_sha256": None,
        "permutation_registry_status": registry_sha,
        "strata_schema_sha256": schema_sha,
        "stratum_count": len(blocks),
        "transform_role": "stress_control_only_not_inferential_randomization",
    }


def validate_target_fit_ancestry(
    registry: Mapping[str, Any], *, permuted_target_node: str, contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Require every registered target-dependent node to descend from the permuted target."""
    if not isinstance(registry, Mapping) or set(registry) != {
        "nodes",
        "private_registry_binding",
    }:
        _fail("c07_fit_ancestry_invalid", "registry_schema")
    registry_status = _validate_private_registry_binding(
        registry["private_registry_binding"], "fit_ancestry"
    )
    permuted_target_node = _technical_node_id(permuted_target_node, "permuted_target_node")
    nodes = registry["nodes"]
    if not isinstance(nodes, list) or not nodes:
        _fail("c07_fit_ancestry_invalid", "nodes_type")
    parents: dict[str, list[str]] = {}
    dependent: set[str] = set()
    for node in nodes:
        if not isinstance(node, Mapping) or set(node) != {"node_id", "parents", "target_dependent"}:
            _fail("c07_fit_ancestry_invalid", "node_schema")
        node_id = _technical_node_id(node["node_id"], "node_id")
        if node_id in parents:
            _fail("c07_fit_ancestry_invalid", "node_id")
        if not isinstance(node["parents"], list):
            _fail("c07_fit_ancestry_invalid", "parent_schema")
        node_parents = [
            _technical_node_id(parent, "parent_schema") for parent in node["parents"]
        ]
        if type(node["target_dependent"]) is not bool:
            _fail("c07_fit_ancestry_invalid", "target_dependent_type")
        parents[node_id] = node_parents
        if node["target_dependent"]:
            dependent.add(node_id)
    if permuted_target_node not in parents:
        _fail("c07_fit_ancestry_invalid", "permuted_target_registered")
    if any(parent not in parents for values in parents.values() for parent in values):
        _fail("c07_fit_ancestry_invalid", "unregistered_parent")
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            _fail("c07_fit_ancestry_invalid", "registry_acyclic")
        if node_id in done:
            return
        visiting.add(node_id)
        for parent in parents[node_id]:
            visit(parent)
        visiting.remove(node_id)
        done.add(node_id)

    for node_id in parents:
        visit(node_id)

    descendants = {permuted_target_node}
    changed = True
    while changed:
        changed = False
        for node_id, values in parents.items():
            if node_id not in descendants and any(parent in descendants for parent in values):
                descendants.add(node_id)
                changed = True
    if not dependent <= descendants:
        _fail("c07_fit_ancestry_invalid", "target_dependent_state_reuse")
    return {
        "claim_id": "C07",
        "decision": "valid",
        "fit_ancestry_status": "all_target_dependent_nodes_descend_from_permuted_target",
        "node_count": len(parents),
        "registry_sha256": None,
        "registry_status": registry_status,
        "target_dependent_node_count": len(dependent),
        "contract_sha256": _contract_sha256(contract, "fit_contract_canonical"),
    }


def _copy_candidate(values: Sequence[Any], n_rows: int, task: str) -> list[Any] | np.ndarray:
    if isinstance(values, (str, bytes)) or len(values) != n_rows:
        _fail("c07_exact_copy_input_invalid", "candidate_shape")
    if task == "classification":
        try:
            return [_classification_value(value, "classification_candidate") for value in values]
        except GroupControlError as error:
            _fail("c07_exact_copy_input_invalid", error.invariant)
    try:
        return np.asarray(
            [_finite_real(value, "numeric_candidate") for value in values], dtype=float
        )
    except GroupControlError as error:
        _fail("c07_exact_copy_input_invalid", error.invariant)


def _classification_bijection(target: Sequence[Any], candidate: Sequence[Any]) -> bool:
    forward: dict[Any, Any] = {}
    reverse: dict[Any, Any] = {}
    for left, right in zip(target, candidate):
        if (left in forward and forward[left] != right) or (
            right in reverse and reverse[right] != left
        ):
            return False
        forward[left] = right
        reverse[right] = left
    return True


def _nonzero_affine(target: np.ndarray, candidate: np.ndarray) -> bool:
    distinct = np.flatnonzero(target != target[0])
    if not len(distinct):
        return False
    index = int(distinct[0])
    slope = (candidate[index] - candidate[0]) / (target[index] - target[0])
    if not math.isfinite(float(slope)) or slope == 0:
        return False
    intercept = candidate[0] - slope * target[0]
    expected = intercept + slope * target
    return bool(np.array_equal(expected, candidate))


def _inverse_risk_order(
    time: np.ndarray, risk: np.ndarray, minimum_comparable_pairs: int
) -> bool:
    if len(np.unique(time)) < 2 or len(np.unique(risk)) < 2:
        return False
    comparable_pairs = 0
    for left in range(len(time)):
        for right in range(left + 1, len(time)):
            time_order = int(time[left] > time[right]) - int(time[left] < time[right])
            risk_order = int(risk[left] > risk[right]) - int(risk[left] < risk[right])
            comparable_pairs += int(time_order != 0)
            if (time_order == 0 and risk_order != 0) or (
                time_order != 0 and risk_order != -time_order
            ):
                return False
    return comparable_pairs >= minimum_comparable_pairs


def scan_registered_exact_copy(
    *,
    task: str,
    relation: str,
    candidate: Sequence[Any] | Mapping[str, Sequence[Any]],
    relation_registry: Mapping[str, Any],
    contract: Mapping[str, Any],
    y: Sequence[Any] | None = None,
    time: Sequence[Any] | None = None,
    event: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Scan only the explicitly registered exact relations for the selected task."""
    known = {
        "classification": {"equality", "bijection"},
        "regression": {"equality", "nonzero_affine"},
        "survival": {"joint_time_event_tuple", "frozen_risk_order_inversion"},
    }
    if task not in known or relation not in known[task]:
        _fail("c07_unregistered_transform", "registered_relation")
    if (
        not isinstance(relation_registry, Mapping)
        or relation_registry.get("task") != task
        or not isinstance(relation_registry.get("relations"), list)
        or relation not in relation_registry["relations"]
    ):
        _fail("c07_unregistered_transform", "relation_registry")
    expected_registry_keys = {"task", "relations"}
    if relation == "frozen_risk_order_inversion":
        expected_registry_keys.add("minimum_comparable_pairs")
    if set(relation_registry) != expected_registry_keys:
        _fail("c07_unregistered_transform", "relation_registry_schema")
    candidates = candidate if isinstance(candidate, Mapping) else None
    if task == "survival":
        if time is None or event is None:
            _fail("c07_exact_copy_input_invalid", "survival_fields")
        n_rows = len(time)
    else:
        if y is None:
            _fail("c07_exact_copy_input_invalid", "endpoint_fields")
        n_rows = len(y)
    objects, arrays = _endpoint_rows(task, n_rows, y=y, time=time, event=event)
    if relation == "joint_time_event_tuple":
        if not isinstance(candidates, Mapping) or set(candidates) != {"time", "event"}:
            _fail("c07_exact_copy_input_invalid", "joint_candidate_schema")
        proposed, _ = _endpoint_rows(
            "survival", n_rows, y=None, time=candidates["time"], event=candidates["event"]
        )
        matched = proposed == objects
    elif relation == "frozen_risk_order_inversion":
        if isinstance(candidate, Mapping):
            _fail("c07_exact_copy_input_invalid", "risk_candidate_schema")
        minimum = relation_registry.get("minimum_comparable_pairs")
        if (
            isinstance(minimum, (bool, np.bool_))
            or not isinstance(minimum, Integral)
            or int(minimum) < 1
        ):
            _fail("c07_unregistered_transform", "risk_order_minimum_registry")
        risk = _copy_candidate(candidate, n_rows, "regression")
        matched = _inverse_risk_order(arrays["time"], risk, int(minimum))
    elif relation == "equality":
        if isinstance(candidate, Mapping):
            _fail("c07_exact_copy_input_invalid", "candidate_schema")
        proposed = _copy_candidate(candidate, n_rows, task)
        target = objects if task == "classification" else arrays["y"]
        matched = bool(np.array_equal(np.asarray(proposed), np.asarray(target)))
    elif relation == "bijection":
        if isinstance(candidate, Mapping):
            _fail("c07_exact_copy_input_invalid", "candidate_schema")
        proposed = _copy_candidate(candidate, n_rows, task)
        matched = _classification_bijection(objects, proposed)
    else:
        if isinstance(candidate, Mapping):
            _fail("c07_exact_copy_input_invalid", "candidate_schema")
        proposed = _copy_candidate(candidate, n_rows, task)
        matched = _nonzero_affine(arrays["y"], proposed)
    return {
        "claim_id": "C07",
        "decision": "registered_exact_copy_detected" if matched else "not_detected",
        "finding_kind": "exact_copy" if matched else "none",
        "matched": matched,
        "observation_count": n_rows,
        "relation_registry_sha256": _canonical_sha256(
            relation_registry, "relation_registry_canonical"
        ),
        "contract_sha256": _contract_sha256(contract, "copy_contract_canonical"),
    }


def proxy_risk_receipt(
    *,
    recoverability_alarm: bool,
    calibration_registry: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Label a calibrated recoverability alarm as proxy risk, never as leakage."""
    if type(recoverability_alarm) is not bool:
        _fail("c07_contract_invalid", "recoverability_alarm_type")
    return {
        "claim_id": "C07",
        "decision": "proxy_risk" if recoverability_alarm else "not_detected",
        "finding_kind": "proxy_risk" if recoverability_alarm else "none",
        "leakage_conclusion": False,
        "wording": "target-recoverability proxy risk; provenance is required for a leakage conclusion",
        "calibration_registry_commitment_status": _validate_private_registry_binding(
            calibration_registry, "proxy_risk_calibration"
        ),
        "contract_sha256": _contract_sha256(contract, "proxy_contract_canonical"),
    }
