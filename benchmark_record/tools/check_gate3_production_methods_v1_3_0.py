"""Fail-closed validator for the v1.3.0 Gate 3 production-methods draft."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any


CONTRACT_NAME = "gate3_production_methods.draft.json"
MARKDOWN_NAME = "GATE3_PRODUCTION_METHODS_DRAFT.md"
EXPECTED_CONTRACT_SHA256 = "16f63cc30dad9cc735c20d27adc9bf75eed94b1d7ff92c678d6ec4d5b483cd20"
EXPECTED_CONTRACT_SIZE = 44812
EXPECTED_MARKDOWN_SHA256 = "9432ff79f12a358275c9adf1ba9fbd64cbfe48a3cada9f25e43461e453df14ad"
EXPECTED_MARKDOWN_SIZE = 47003
SEMANTIC_PAYLOAD_SHA256 = "5d7702fa8a91f93effc6eef678a9cb517f2dbf07cc5dd3a15ef5eba81e9e8932"
SCHEMA_TYPES_SHA256 = "f20e9ab42dda7e8bf9e1639b658dbb34795f81f338565ee30ca48cb62a966fa2"
NORMATIVE_MARKDOWN_SHA256 = "dfce1a1705a7353700de102b6549510b327e3b7850e535610b5ff9d5345a2c9c"
SOURCE_BINDINGS_SHA256 = "cdf736a72bb8b2256ba6a2520ec7ceceed92b3090d4a9ed5c127f3b6b22504e3"
CLAIM_PAYLOAD_SHA256 = {
    "C03": "0e3cc6ed1392ed7edb63bd379fa2f1e590d5130e38e2e145a59cc8fb489e3a59",
    "C04": "a5776dc5cff450a1248b105d957dbb19239280bcaa64b71be156efce93c5d3c8",
    "C05": "509f94119a58372d828439699566d18a2fd06f950a2b2e8a94379de3389941ec",
    "C06": "aa922c3360a5e79cfff9880a18265ff18e42454f2ad757c019bf4e16c1f83d2d",
    "C07": "5e11e8ac8ec79a44ad9c31199439a8e2ea193c63453ad2396e5d13cc160a16b4",
    "C08": "aa7dd94f244be14a828e877b3cb28b2fe0772711dcaa9c9317e98b56eaeb2e45",
}
SUCCESS = "PASS"
FAILURE_CODES = frozenset({
    "PATH", "REPARSE", "READ", "RACE", "SIZE", "HASH", "ENCODING",
    "ASCII", "LINE_ENDING", "JSON", "DUPLICATE", "CANONICAL",
    "CONTROL", "TOKEN", "SEMANTICS", "SOURCE", "POINTER", "MARKDOWN",
    "INTERNAL",
})
BOUND_INPUTS: tuple[tuple[str, str, int], ...] = (
    ("benchmark_record/releases/v1.3.0/gate3_diagnostic_validation_contract.draft.json", "d230ba29e0e6c90edc2ea0b1289c0834964220db3310b499c39831a3f453f77c", 35928),
    ("benchmark_record/releases/v1.3.0/gate3_inference_amendment.draft.json", "2377311ae1c7193af25ba8444fc97fdb3294b431ab487972d6812c7624b6520b", 27929),
    ("benchmark_record/releases/v1.3.0/gate3_scenario_templates.draft5.json", "c3b4f4da1cf31aa4a108648daae4697e510ce5976e47b74f69caedf12e4c0df6", 215387),
    ("benchmark_record/tools/gate3_binomial_design_v1_3_0.py", "643d3c7ea7b5a3a5b1751d747bda88fd75d145edfc3cb2b43d3b48d6637e71f8", 13443),
    ("omicau/data/alignment.py", "67470dd1f695e53212ffd93ef009d55a82c110c944f8ce3306245b34aaa4da7b", 31424),
    ("omicau/diagnostics/batch.py", "ae23bc662797953c00c84eaaea5620808eac2cda6de4743de1394c6b9dd93de0", 8482),
    ("omicau/diagnostics/missingness.py", "4dbe4800eb4e5ba3f45a37188e20cdff7c41844d4409657138363982670fc31b", 6893),
    ("omicau/interpretation/utility.py", "97970d4a53f6c0e7032729e2536ce4fa14f3575e0ba358b061c455a430c5e20d", 30249),
    ("omicau/models/base.py", "9f51ece025b0a9bb6183ca43462dd6ba24eb8f84b4c136ccca2d0ea6a78dcb6c", 18542),
    ("omicau/models/classical.py", "d4ec9934de4c8845d5a58f5d9eebe9837b3d2caa469ca2d944aca6750efd172c", 12437),
    ("omicau/models/neural.py", "0b0650be83315ac9503f9c852a0540fcd16948bd8ea79ee035f5ca5ec3141f0d", 15126),
    ("omicau/models/survival.py", "7d0b0254699fdf5b4b371a68b7165948a59dade5f49fbd33f2c8aad1647139ae", 11135),
)
_TOP_KEYS = {
    "activation", "authorizations", "claims", "cross_claim_contract",
    "dependency_freeze", "evidence_boundary", "production_gap_inventory",
    "protocol_version", "public_safe_evidence_contract", "schema_version",
    "source_bindings", "status", "unresolved_blockers", "zenodo_ready",
}
_AUTHORIZATION_KEYS = {
    "calibration_execution", "candidate_data_access", "candidate_fitting",
    "candidate_performance_inspection", "definitive_validation",
    "feature_outcome_association", "freeze", "manuscript_claim_revision",
    "publication", "scenario_truth_assignment", "software_release",
    "zenodo_upload",
}
_CLAIMS = ("C03", "C04", "C05", "C06", "C07", "C08")
_CLAIM_KEYS = {
    "advertised_scope", "eligibility", "estimand", "method", "multiplicity",
    "oracle", "outputs", "production_gaps", "quantitative_freeze",
    "scenario_truth", "secondary_evidence", "watched_failures",
}
_SOURCE_ROW_KEYS = {
    "claim_ids", "path", "pointers", "sha256", "size_bytes", "source_kind",
    "symbols",
}
_SOURCE_IDS = {
    "binomial_design_tool", "diagnostic_contract", "inference_amendment",
    "production_alignment", "production_audit_batch", "production_base_models",
    "production_classical_models", "production_missingness",
    "production_neural_models", "production_survival_models",
    "production_utility", "scenario_templates",
}
_RELATIVE = re.compile(r"^[A-Za-z0-9.][A-Za-z0-9._/-]*$")
_FORBIDDEN = re.compile(
    r"(?:[A-Za-z]:[\\/]|(?:^|[\\/])(?:Users|home|tmp)(?:[\\/]|$)|file://)",
    re.IGNORECASE,
)


class MethodsError(Exception):
    """A private validation error represented publicly by one fixed code."""

    def __init__(self, code: str):
        self.code = code if code in FAILURE_CODES else "INTERNAL"
        super().__init__(self.code)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise MethodsError("INTERNAL")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _is_reparse(path: Path) -> bool:
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except AttributeError:
        attributes = 0
    return path.is_symlink() or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _reject_reparse_chain(path: Path, stop: Path | None = None) -> None:
    for current in (path, *path.parents):
        if _is_reparse(current):
            raise MethodsError("REPARSE")
        if stop is not None and current == stop:
            return
        if current == Path(current.anchor):
            return


def _checked_artifact_path(path: str | Path, expected_name: str) -> Path:
    if expected_name not in {CONTRACT_NAME, MARKDOWN_NAME}:
        raise MethodsError("INTERNAL")
    root = _repo_root()
    release = root / "benchmark_record" / "releases" / "v1.3.0"
    expected = release / expected_name
    try:
        supplied = Path(path)
        supplied = supplied if supplied.is_absolute() else Path.cwd() / supplied
        supplied = supplied.absolute()
        resolved_root = root.resolve(strict=True)
        resolved_release = release.resolve(strict=True)
        resolved_expected = expected.resolve(strict=True)
        resolved_supplied = supplied.resolve(strict=True)
    except (OSError, RuntimeError):
        raise MethodsError("PATH") from None
    if resolved_release.parent != (resolved_root / "benchmark_record" / "releases").resolve(strict=True):
        raise MethodsError("PATH")
    if resolved_expected.parent != resolved_release or resolved_expected.name != expected_name:
        raise MethodsError("PATH")
    if resolved_supplied != resolved_expected:
        raise MethodsError("PATH")
    _reject_reparse_chain(expected, root)
    _reject_reparse_chain(supplied)
    return resolved_expected


def _checked_bound_path(relative: str) -> Path:
    allowed = {path for path, _sha256, _size in BOUND_INPUTS}
    if relative not in allowed or not _RELATIVE.fullmatch(relative):
        raise MethodsError("SOURCE")
    if ".." in relative.split("/") or "\\" in relative:
        raise MethodsError("SOURCE")
    root = _repo_root()
    target = root.joinpath(*relative.split("/"))
    try:
        resolved_root = root.resolve(strict=True)
        resolved = target.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        raise MethodsError("SOURCE") from None
    _reject_reparse_chain(target, root)
    return resolved


def _read_once(path: Path) -> bytes:
    try:
        before = _identity(path)
        raw = path.read_bytes()
        after = _identity(path)
    except OSError:
        raise MethodsError("READ") from None
    if before != after or len(raw) != before[2]:
        raise MethodsError("RACE")
    return raw


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MethodsError("DUPLICATE")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise MethodsError("JSON")


def _wire_text(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise MethodsError("ENCODING")
    try:
        text = raw.decode("utf-8")
    except UnicodeError:
        raise MethodsError("ENCODING") from None
    if any(byte > 127 for byte in raw):
        raise MethodsError("ASCII")
    if b"\r" in raw or not raw.endswith(b"\n"):
        raise MethodsError("LINE_ENDING")
    for byte in raw:
        if byte != 10 and (byte < 32 or byte == 127):
            raise MethodsError("CONTROL")
    return text


def _parse_json(raw: bytes) -> dict[str, Any]:
    text = _wire_text(raw)
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_no_duplicates,
            parse_constant=_reject_json_constant,
        )
    except MethodsError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError, OverflowError):
        raise MethodsError("JSON") from None
    if not isinstance(payload, dict):
        raise MethodsError("SEMANTICS")
    canonical = (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("ascii")
    if raw != canonical:
        raise MethodsError("CANONICAL")
    return payload


def _walk_values(value: Any, path: tuple[Any, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise MethodsError("SEMANTICS")
            _walk_values(key, path + ("<key>",))
            _walk_values(child, path + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_values(child, path + (index,))
    elif isinstance(value, str):
        if any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value):
            raise MethodsError("CONTROL")
        if _FORBIDDEN.search(value):
            raise MethodsError("TOKEN")
    elif value is None or type(value) in {bool, int, float}:
        return
    else:
        raise MethodsError("SEMANTICS")


def _mapping(value: Any, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise MethodsError("SEMANTICS")
    return value


def _list(value: Any, length: int | None = None) -> list[Any]:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        raise MethodsError("SEMANTICS")
    return value


def _exact(value: Any, expected: Any) -> None:
    if type(value) is not type(expected) or value != expected:
        raise MethodsError("SEMANTICS")


def _schema_signature(value: Any) -> Any:
    if isinstance(value, dict):
        return ["dict", [[key, _schema_signature(child)] for key, child in value.items()]]
    if isinstance(value, list):
        return ["list", len(value), [_schema_signature(child) for child in value]]
    return type(value).__name__


def _compact_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _schema_hash(value: Any) -> str:
    raw = json.dumps(_schema_signature(value), separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _pointer_token(token: str) -> str:
    if re.search(r"~(?:[^01]|$)", token):
        raise MethodsError("POINTER")
    return token.replace("~1", "/").replace("~0", "~")


def _resolve_pointer(document: Any, pointer: str) -> Any:
    if not isinstance(pointer, str):
        raise MethodsError("POINTER")
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise MethodsError("POINTER")
    current = document
    for raw_token in pointer[1:].split("/"):
        token = _pointer_token(raw_token)
        if isinstance(current, dict):
            if token not in current:
                raise MethodsError("POINTER")
            current = current[token]
        elif isinstance(current, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token):
                raise MethodsError("POINTER")
            index = int(token)
            if index >= len(current):
                raise MethodsError("POINTER")
            current = current[index]
        else:
            raise MethodsError("POINTER")
    return current


def _normative_markdown_hash(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line.strip()) for line in text.splitlines()]
    normalized = "gate3-production-methods-normative-draft-1\n" + "\n".join(lines) + "\n"
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def _validate_semantics(payload: dict[str, Any], sources: dict[str, dict[str, Any]]) -> None:
    _mapping(payload, _TOP_KEYS)
    _exact(payload["protocol_version"], "1.3.0")
    _exact(payload["schema_version"], "gate3_production_methods_draft_1")
    _exact(payload["status"], "draft_methods_only_not_calibrated_not_authorized")
    _exact(payload["zenodo_ready"], False)

    activation = _mapping(payload["activation"], {"effective", "freeze_ready", "methods_complete", "reason"})
    if any(type(activation[key]) is not bool or activation[key] for key in ("effective", "freeze_ready", "methods_complete")):
        raise MethodsError("SEMANTICS")
    authorizations = _mapping(payload["authorizations"], _AUTHORIZATION_KEYS)
    if any(type(value) is not bool or value for value in authorizations.values()):
        raise MethodsError("SEMANTICS")

    evidence = _mapping(payload["evidence_boundary"], {"current_artifact_role", "does_not_establish", "may_establish_after_independent_check"})
    if evidence["current_artifact_role"] != "reviewed design candidate only":
        raise MethodsError("SEMANTICS")
    for phrase in ("positive validation", "real-data performance", "reproducible released software"):
        if phrase not in evidence["does_not_establish"]:
            raise MethodsError("SEMANTICS")

    dependencies = _mapping(payload["dependency_freeze"], {"runtime_candidates", "validation_candidates"})
    for family in dependencies.values():
        if not isinstance(family, dict) or not family:
            raise MethodsError("SEMANTICS")
        for row in family.values():
            current = _mapping(row, {"authorized", "callables_and_defaults", "exact_version", "purpose"})
            if current["authorized"] is not False or current["callables_and_defaults"] is not None or current["exact_version"] is not None:
                raise MethodsError("SEMANTICS")

    claims = _mapping(payload["claims"], set(_CLAIMS))
    for claim_id in _CLAIMS:
        claim = _mapping(claims[claim_id], _CLAIM_KEYS)
        if _compact_hash(claim) != CLAIM_PAYLOAD_SHA256[claim_id]:
            raise MethodsError("SEMANTICS")
        quantitative = claim["quantitative_freeze"]
        if not isinstance(quantitative, dict) or not quantitative or not any(value is None for value in quantitative.values()):
            raise MethodsError("SEMANTICS")
        resolved = {key: value for key, value in quantitative.items() if value is not None}
        if resolved != ({
            "solver_objective_and_tiebreak": "zero objective feasibility search; no assignment tie-break because assignment identity is not the estimand",
        } if claim_id == "C06" else {}):
            raise MethodsError("SEMANTICS")
        scenario = _mapping(claim["scenario_truth"], {"authoritative", "expected_decisions", "source_registry_pointer"})
        if scenario != {"authoritative": False, "expected_decisions": None, "source_registry_pointer": "/source_templates"}:
            raise MethodsError("SEMANTICS")
        multiplicity = _mapping(claim["multiplicity"], {"primary_family", "secondary_families", "secondary_may_trigger_primary"})
        if multiplicity["secondary_may_trigger_primary"] is not False:
            raise MethodsError("SEMANTICS")
        outputs = _mapping(claim["outputs"], {"forbidden_private_keys", "required_public_keys"})
        for key in outputs:
            values = outputs[key]
            if not isinstance(values, list) or not values or values != sorted(set(values)):
                raise MethodsError("SEMANTICS")
        if set(outputs["forbidden_private_keys"]) & set(outputs["required_public_keys"]):
            raise MethodsError("SEMANTICS")
        eligibility = _mapping(claim["eligibility"], {"group_contract", "refusal_codes", "requirements"})
        if len(eligibility["refusal_codes"]) != len(set(eligibility["refusal_codes"])) or not eligibility["requirements"]:
            raise MethodsError("SEMANTICS")
        for key in ("production_gaps", "secondary_evidence", "watched_failures"):
            values = claim[key]
            if not isinstance(values, list) or not values or len(values) != len(set(values)):
                raise MethodsError("SEMANTICS")

    _exact(claims["C03"]["method"]["non_survival_primary"]["missingness_distance"], "normalized_L1_between_per_feature_group_missingness_proportions")
    _exact(set(claims["C03"]["method"]["secondary_localization"]), {
        "aggregate_categorical", "aggregate_continuous", "categorical_feature",
        "continuous_feature", "survival_censoring_feature", "survival_event_feature",
    })
    if "one Holm family" not in claims["C03"]["multiplicity"]["primary_family"]:
        raise MethodsError("SEMANTICS")
    _exact(claims["C04"]["method"]["missingness_distance"], "normalized_L1_between_per_feature_group_missingness_proportions")
    if "modality-specific batch label" not in claims["C04"]["eligibility"]["group_contract"]:
        raise MethodsError("SEMANTICS")
    if "one Holm family" not in claims["C04"]["multiplicity"]["primary_family"]:
        raise MethodsError("SEMANTICS")
    if "structure_outcome_event_censor" not in claims["C05"]["multiplicity"]["primary_family"]:
        raise MethodsError("SEMANTICS")
    _exact(claims["C06"]["method"]["split_rule"], "joint whole-group assignment with no fold clamping skipping replacement or overlap")
    _exact(claims["C06"]["method"]["assignment_identity"], "not part of the feasibility estimand; the first independently verified feasible manifest is frozen by SHA-256 and reused exactly")
    if not any("task-specific calibrated training and assessment support" in item for item in claims["C06"]["eligibility"]["requirements"]):
        raise MethodsError("SEMANTICS")
    if "target-independent design blocks" not in claims["C07"]["eligibility"]["group_contract"]:
        raise MethodsError("SEMANTICS")
    _exact(claims["C07"]["advertised_scope"]["claim_language"], "registered negative-control behavior exact-copy detection and calibrated target-recoverability proxy risk")
    poison = claims["C08"]["method"]["poison_contract"]
    _exact(poison, {
        "assessment_feature_poison": "learned_state_unchanged_assessment_predictions_may_change",
        "assessment_outcome_poison": "learned_state_and_predictions_unchanged",
        "unchanged_sentinel": "sentinel_predictions_unchanged_when_other_assessment_features_are_poisoned",
    })
    if "callsite_component_version" not in claims["C08"]["method"]["fit_trace_node"]:
        raise MethodsError("SEMANTICS")
    if not all(item in claims["C08"]["eligibility"]["requirements"] for item in (
        "static and runtime fit-callsite inventories reconcile exactly",
        "every data-dependent operation emits one canonical trace node",
    )):
        raise MethodsError("SEMANTICS")

    cross = _mapping(payload["cross_claim_contract"], {
        "calibration_boundary", "decision_hierarchy", "group_contract",
        "non_estimable_policy", "operational_alpha_vs_validation_alpha",
        "permutation_contract", "scenario_authority", "validation_inference_binding",
    })
    if "exactly one declared primary family" not in cross["decision_hierarchy"]:
        raise MethodsError("SEMANTICS")
    if "assigns no scenario truth expected decision or observed result" not in cross["scenario_authority"]:
        raise MethodsError("SEMANTICS")
    gaps = _mapping(payload["production_gap_inventory"], set(_CLAIMS))
    if any(not isinstance(gaps[claim], str) or not gaps[claim] for claim in _CLAIMS):
        raise MethodsError("SEMANTICS")
    public = _mapping(payload["public_safe_evidence_contract"], {"forbidden", "required"})
    if any(not isinstance(values, list) or not values or len(values) != len(set(values)) for values in public.values()):
        raise MethodsError("SEMANTICS")
    if set(public["forbidden"]) & set(public["required"]):
        raise MethodsError("SEMANTICS")
    blockers = payload["unresolved_blockers"]
    if not isinstance(blockers, list) or len(blockers) != 14 or len(blockers) != len(set(blockers)):
        raise MethodsError("SEMANTICS")


def _python_symbols(raw: bytes) -> set[str]:
    try:
        tree = ast.parse(raw.decode("utf-8-sig"))
    except (UnicodeError, SyntaxError):
        raise MethodsError("SOURCE") from None
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _validate_sources(payload: dict[str, Any], source_raw: dict[str, bytes], sources: dict[str, dict[str, Any]]) -> None:
    bindings = _mapping(payload["source_bindings"], _SOURCE_IDS)
    if _compact_hash(bindings) != SOURCE_BINDINGS_SHA256:
        raise MethodsError("SOURCE")
    pins = {path: (sha256, size) for path, sha256, size in BOUND_INPUTS}
    if set(source_raw) != set(pins) or len(pins) != len(BOUND_INPUTS):
        raise MethodsError("SOURCE")
    for row in bindings.values():
        current = _mapping(row, _SOURCE_ROW_KEYS)
        path = current["path"]
        if path not in pins:
            raise MethodsError("SOURCE")
        expected_sha256, expected_size = pins[path]
        if current["sha256"] != expected_sha256 or current["size_bytes"] != expected_size:
            raise MethodsError("SOURCE")
        raw = source_raw[path]
        if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise MethodsError("SOURCE")
        claim_ids = current["claim_ids"]
        if not isinstance(claim_ids, list) or not claim_ids or claim_ids != sorted(set(claim_ids)) or not set(claim_ids) <= set(_CLAIMS):
            raise MethodsError("SOURCE")
        pointers = current["pointers"]
        symbols = current["symbols"]
        if current["source_kind"] == "canonical_json":
            if path not in sources or symbols != []:
                raise MethodsError("SOURCE")
            if not isinstance(pointers, list) or not pointers or pointers != sorted(set(pointers)):
                raise MethodsError("POINTER")
            for pointer in pointers:
                _resolve_pointer(sources[path], pointer)
                match = re.search(r"/(C0[3-8])(?:/|$)", pointer)
                if match and match.group(1) not in claim_ids:
                    raise MethodsError("POINTER")
        elif current["source_kind"] == "python":
            if pointers != [] or not isinstance(symbols, list) or not symbols or symbols != sorted(set(symbols)):
                raise MethodsError("SOURCE")
            if not set(symbols) <= _python_symbols(raw):
                raise MethodsError("SOURCE")
        else:
            raise MethodsError("SOURCE")


def _mirror_leaves(value: Any, path: str = "", result: dict[str, str] | None = None) -> dict[str, str]:
    if result is None:
        result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            token = key.replace("~", "~0").replace("/", "~1")
            _mirror_leaves(child, f"{path}/{token}", result)
    else:
        result[path] = json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return result


def _validate_markdown(text: str, payload: dict[str, Any]) -> None:
    if _normative_markdown_hash(text) != NORMATIVE_MARKDOWN_SHA256:
        raise MethodsError("MARKDOWN")
    if not text.startswith("# Gate 3 production methods draft\n\n"):
        raise MethodsError("MARKDOWN")
    observed: dict[str, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"- `(/[^`]*)`: `(.*)`", line)
        if match:
            if match.group(1) in observed:
                raise MethodsError("MARKDOWN")
            observed[match.group(1)] = match.group(2)
    if observed != _mirror_leaves(payload):
        raise MethodsError("MARKDOWN")


def _clear(error: BaseException) -> None:
    try:
        error.__traceback__ = None
        error.__cause__ = None
        error.__context__ = None
    except Exception:
        pass


def _validation_token(contract_path: str | Path, markdown_path: str | Path) -> str:
    if not __debug__:
        raise MethodsError("INTERNAL")
    contract = _checked_artifact_path(contract_path, CONTRACT_NAME)
    markdown = _checked_artifact_path(markdown_path, MARKDOWN_NAME)
    contract_raw = _read_once(contract)
    markdown_raw = _read_once(markdown)
    if len(contract_raw) != EXPECTED_CONTRACT_SIZE or len(markdown_raw) != EXPECTED_MARKDOWN_SIZE:
        raise MethodsError("SIZE")
    if hashlib.sha256(contract_raw).hexdigest() != EXPECTED_CONTRACT_SHA256:
        raise MethodsError("HASH")
    if hashlib.sha256(markdown_raw).hexdigest() != EXPECTED_MARKDOWN_SHA256:
        raise MethodsError("HASH")
    payload = _parse_json(contract_raw)
    markdown_text = _wire_text(markdown_raw)
    _walk_values(payload)
    source_raw: dict[str, bytes] = {}
    sources: dict[str, dict[str, Any]] = {}
    for relative, expected_sha256, expected_size in BOUND_INPUTS:
        raw = _read_once(_checked_bound_path(relative))
        if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise MethodsError("SOURCE")
        source_raw[relative] = raw
        if relative.endswith(".json"):
            sources[relative] = _parse_json(raw)
    if _schema_hash(payload) != SCHEMA_TYPES_SHA256:
        raise MethodsError("SEMANTICS")
    _validate_semantics(payload, sources)
    if _compact_hash(payload) != SEMANTIC_PAYLOAD_SHA256:
        raise MethodsError("SEMANTICS")
    _validate_sources(payload, source_raw, sources)
    _validate_markdown(markdown_text, payload)
    return SUCCESS


def validate_methods(contract_path: str | Path, markdown_path: str | Path) -> str:
    """Return PASS or one fixed path-free failure code."""
    try:
        token = _validation_token(contract_path, markdown_path)
        return token if token == SUCCESS else "INTERNAL"
    except MethodsError as error:
        code = error.code if error.code in FAILURE_CODES else "INTERNAL"
        _clear(error)
        return code
    except Exception as error:
        _clear(error)
        return "INTERNAL"


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(add_help=False)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--markdown", required=True)
    try:
        args = parser.parse_args(argv)
        token = validate_methods(args.contract, args.markdown)
    except MethodsError as error:
        token = error.code if error.code in FAILURE_CODES else "INTERNAL"
        _clear(error)
    except Exception as error:
        token = "INTERNAL"
        _clear(error)
    if token == SUCCESS:
        print("GATE3_PRODUCTION_METHODS:PASS")
        return 0
    if token not in FAILURE_CODES:
        token = "INTERNAL"
    print(f"GATE3_PRODUCTION_METHODS:FAIL:{token}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
