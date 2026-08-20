"""Watched-fail tests for the Gate 3 production-methods checker."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType

try:
    import pytest
except ModuleNotFoundError:
    if __name__ != "__main__":
        raise

    class _DirectMark:
        @staticmethod
        def parametrize(*_args, **_kwargs):
            return lambda function: function

    class _DirectPytest:
        mark = _DirectMark()

    pytest = _DirectPytest()


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "benchmark_record" / "tools"
RELEASE = ROOT / "benchmark_record" / "releases" / "v1.3.0"
CONTRACT = RELEASE / "gate3_production_methods.draft.json"
MARKDOWN = RELEASE / "GATE3_PRODUCTION_METHODS_DRAFT.md"
CHECKER = TOOLS / "check_gate3_production_methods_v1_3_0.py"
CHECKER_SHA256 = "174f9ed7c84999d52dbe387853a6ba733c14d06f8809c544a409135d40712245"
CHECKER_SIZE = 28462
CHECKER_RAW = CHECKER.read_bytes()
if len(CHECKER_RAW) != CHECKER_SIZE or hashlib.sha256(CHECKER_RAW).hexdigest() != CHECKER_SHA256:
    raise RuntimeError("checker source drift")


def _load_checker(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(CHECKER)
    exec(compile(CHECKER_RAW, str(CHECKER), "exec", dont_inherit=True, optimize=0), module.__dict__)
    return module


gate3 = _load_checker("gate3_production_methods_checker")


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("ascii")


def _copies(folder: Path) -> tuple[Path, Path]:
    contract = folder / CONTRACT.name
    markdown = folder / MARKDOWN.name
    contract.write_bytes(CONTRACT.read_bytes())
    markdown.write_bytes(MARKDOWN.read_bytes())
    return contract, markdown


def _patch_outputs(monkeypatch: pytest.MonkeyPatch, contract: Path, markdown: Path) -> None:
    monkeypatch.setattr(
        gate3,
        "_checked_artifact_path",
        lambda _path, name: contract if name == gate3.CONTRACT_NAME else markdown,
    )


def _rebase_contract(monkeypatch: pytest.MonkeyPatch, raw: bytes) -> None:
    monkeypatch.setattr(gate3, "EXPECTED_CONTRACT_SIZE", len(raw))
    monkeypatch.setattr(gate3, "EXPECTED_CONTRACT_SHA256", hashlib.sha256(raw).hexdigest())


def _rebase_semantic_pins(monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    monkeypatch.setattr(gate3, "SEMANTIC_PAYLOAD_SHA256", gate3._compact_hash(payload))
    monkeypatch.setattr(gate3, "SCHEMA_TYPES_SHA256", gate3._schema_hash(payload))


def _rebase_claim_pin(monkeypatch: pytest.MonkeyPatch, payload: dict, claim_id: str) -> None:
    pins = dict(gate3.CLAIM_PAYLOAD_SHA256)
    pins[claim_id] = gate3._compact_hash(payload["claims"][claim_id])
    monkeypatch.setattr(gate3, "CLAIM_PAYLOAD_SHA256", pins)


def _rebase_source_pin(monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    monkeypatch.setattr(gate3, "SOURCE_BINDINGS_SHA256", gate3._compact_hash(payload["source_bindings"]))


def _rebase_markdown(monkeypatch: pytest.MonkeyPatch, raw: bytes) -> None:
    monkeypatch.setattr(gate3, "EXPECTED_MARKDOWN_SIZE", len(raw))
    monkeypatch.setattr(gate3, "EXPECTED_MARKDOWN_SHA256", hashlib.sha256(raw).hexdigest())


def _leaf_paths(value: object, prefix: tuple[object, ...] = ()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaf_paths(child, prefix + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _leaf_paths(child, prefix + (index,))
    else:
        yield prefix


def _mutate_leaf(value: object, path: tuple[object, ...]) -> None:
    parent = value
    for key in path[:-1]:
        parent = parent[key]
    key = path[-1]
    old = parent[key]
    if type(old) is bool:
        parent[key] = not old
    elif type(old) is int:
        parent[key] = old + 1
    elif type(old) is float:
        parent[key] = old + 0.0001
    elif isinstance(old, str):
        parent[key] = old + "_drift"
    elif old is None:
        parent[key] = "resolved_without_authority"
    else:
        raise TypeError(type(old))


def _require(condition: object) -> None:
    if not condition:
        raise RuntimeError("selftest failure")


def _assert_token(value: object, code: str) -> None:
    _require(type(value) is str)
    _require(value == code)
    _require("/" not in value and "\\" not in value and ":" not in value)


def _checker_uses_dynamic_bytecode() -> bool:
    tree = ast.parse(CHECKER_RAW.decode("ascii"))
    forbidden_imports = {"importlib", "runpy", "marshal", "zipimport"}
    forbidden_calls = {"__import__", "compile", "eval", "exec"}
    forbidden_attributes = {"exec_module", "spec_from_file_location", "CodeType"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name.split(".")[0] in forbidden_imports for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in forbidden_imports:
            return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_attributes:
                return True
    return False


def _c03_hamming(payload: dict) -> None:
    payload["claims"]["C03"]["method"]["non_survival_primary"]["missingness_distance"] = "Hamming"


def _c04_batch_purity(payload: dict) -> None:
    payload["claims"]["C04"]["eligibility"]["group_contract"] = "one row is one unit"


def _c05_family_fragmentation(payload: dict) -> None:
    payload["claims"]["C05"]["multiplicity"]["primary_family"] = "separate unadjusted component families"


def _c06_silent_clamp(payload: dict) -> None:
    payload["claims"]["C06"]["method"]["split_rule"] = "clamp or skip infeasible folds"


def _c06_technical_minimum(payload: dict) -> None:
    payload["claims"]["C06"]["eligibility"]["requirements"][3] = "two observations are calibrated support"


def _c07_target_strata(payload: dict) -> None:
    payload["claims"]["C07"]["eligibility"]["group_contract"] = "target-derived blocks are permitted"


def _c07_broad_wording(payload: dict) -> None:
    payload["claims"]["C07"]["advertised_scope"]["claim_language"] = "all leakage is detected"


def _c08_poison_reversal(payload: dict) -> None:
    payload["claims"]["C08"]["method"]["poison_contract"]["assessment_feature_poison"] = "learned_state_and_predictions_unchanged"


def _c08_missing_trace(payload: dict) -> None:
    payload["claims"]["C08"]["eligibility"]["requirements"].remove("static and runtime fit-callsite inventories reconcile exactly")


CLAIM_MUTATIONS = {
    "C03_hamming": _c03_hamming,
    "C04_missing_batch_purity": _c04_batch_purity,
    "C05_family_fragmentation": _c05_family_fragmentation,
    "C06_silent_clamp": _c06_silent_clamp,
    "C06_technical_minimum": _c06_technical_minimum,
    "C07_target_derived_strata": _c07_target_strata,
    "C07_broad_leakage_wording": _c07_broad_wording,
    "C08_poison_reversal": _c08_poison_reversal,
    "C08_missing_trace_inventory": _c08_missing_trace,
}


def test_positive_api_cli_no_override_and_optimized_fail_closed(tmp_path: Path) -> None:
    assert list(inspect.signature(gate3.validate_methods).parameters) == ["contract_path", "markdown_path"]
    assert gate3.validate_methods(CONTRACT, MARKDOWN) == gate3.SUCCESS
    command = [
        sys.executable, "-B", str(CHECKER),
        "--contract", str(CONTRACT), "--markdown", str(MARKDOWN),
    ]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert completed.returncode == 0
    assert completed.stdout == "GATE3_PRODUCTION_METHODS:PASS\n"
    assert completed.stderr == ""
    rejected = subprocess.run(command + ["--expected-sha256", "0" * 64], capture_output=True, text=True, check=False)
    assert rejected.returncode == 1
    assert rejected.stdout == "GATE3_PRODUCTION_METHODS:FAIL:INTERNAL\n"
    assert rejected.stderr == ""
    optimized = subprocess.run(
        [sys.executable, "-B", "-O", *command[2:]],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert optimized.returncode == 1
    assert optimized.stdout == "GATE3_PRODUCTION_METHODS:FAIL:INTERNAL\n"
    assert optimized.stderr == ""


def test_checker_source_is_pinned_and_has_no_dynamic_bytecode() -> None:
    assert len(CHECKER_RAW) == CHECKER_SIZE
    assert hashlib.sha256(CHECKER_RAW).hexdigest() == CHECKER_SHA256
    assert not _checker_uses_dynamic_bytecode()


def test_all_authorizations_and_claim_watchers_survive_outer_rebase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract, markdown = _copies(tmp_path)
    _patch_outputs(monkeypatch, contract, markdown)
    base = json.loads(CONTRACT.read_text(encoding="ascii"))
    for key in base["authorizations"]:
        payload = copy.deepcopy(base)
        payload["authorizations"][key] = True
        raw = _canonical(payload)
        contract.write_bytes(raw)
        _rebase_contract(monkeypatch, raw)
        _rebase_semantic_pins(monkeypatch, payload)
        assert gate3.validate_methods(contract, markdown) == "SEMANTICS"
    for mutate in CLAIM_MUTATIONS.values():
        payload = copy.deepcopy(base)
        mutate(payload)
        raw = _canonical(payload)
        contract.write_bytes(raw)
        _rebase_contract(monkeypatch, raw)
        _rebase_semantic_pins(monkeypatch, payload)
        assert gate3.validate_methods(contract, markdown) == "SEMANTICS"


def test_unresolved_authority_and_private_output_guards_survive_claim_rebase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract, markdown = _copies(tmp_path)
    _patch_outputs(monkeypatch, contract, markdown)
    base = json.loads(CONTRACT.read_text(encoding="ascii"))
    mutations = (
        lambda p: p["claims"]["C03"]["quantitative_freeze"].__setitem__("minimum_group_count", 5),
        lambda p: p["claims"]["C03"]["scenario_truth"].__setitem__("authoritative", True),
        lambda p: p["claims"]["C03"]["outputs"]["required_public_keys"].append("sample_ids"),
    )
    for mutate in mutations:
        payload = copy.deepcopy(base)
        mutate(payload)
        payload["claims"]["C03"]["outputs"]["required_public_keys"].sort()
        raw = _canonical(payload)
        contract.write_bytes(raw)
        _rebase_contract(monkeypatch, raw)
        _rebase_semantic_pins(monkeypatch, payload)
        _rebase_claim_pin(monkeypatch, payload, "C03")
        assert gate3.validate_methods(contract, markdown) == "SEMANTICS"


def test_source_pointer_claim_and_symbol_failures_survive_binding_rebase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract, markdown = _copies(tmp_path)
    _patch_outputs(monkeypatch, contract, markdown)
    base = json.loads(CONTRACT.read_text(encoding="ascii"))
    mutations = (
        (lambda p: p["source_bindings"]["diagnostic_contract"]["pointers"].__setitem__(1, "/claim_designs/C09"), "POINTER"),
        (lambda p: p["source_bindings"]["diagnostic_contract"]["claim_ids"].remove("C03"), "POINTER"),
        (lambda p: p["source_bindings"]["production_missingness"]["symbols"].append("missing_symbol"), "SOURCE"),
    )
    for mutate, code in mutations:
        payload = copy.deepcopy(base)
        mutate(payload)
        for row in payload["source_bindings"].values():
            row["claim_ids"].sort()
            row["pointers"].sort()
            row["symbols"].sort()
        raw = _canonical(payload)
        contract.write_bytes(raw)
        _rebase_contract(monkeypatch, raw)
        _rebase_semantic_pins(monkeypatch, payload)
        _rebase_source_pin(monkeypatch, payload)
        assert gate3.validate_methods(contract, markdown) == code


def test_normative_markdown_and_exact_mirror_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract, markdown = _copies(tmp_path)
    _patch_outputs(monkeypatch, contract, markdown)
    original = MARKDOWN.read_bytes()
    raw = original.replace(b'- `/activation/effective`: `false`', b'- `/activation/effective`: `true`', 1)
    assert raw != original
    markdown.write_bytes(raw)
    _rebase_markdown(monkeypatch, raw)
    assert gate3.validate_methods(contract, markdown) == "MARKDOWN"
    text = raw.decode("ascii")
    monkeypatch.setattr(gate3, "NORMATIVE_MARKDOWN_SHA256", gate3._normative_markdown_hash(text))
    assert gate3.validate_methods(contract, markdown) == "MARKDOWN"


def test_every_material_json_leaf_survives_outer_rebase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract, markdown = _copies(tmp_path)
    _patch_outputs(monkeypatch, contract, markdown)
    base = json.loads(CONTRACT.read_text(encoding="ascii"))
    paths = list(_leaf_paths(base))
    assert len(paths) > 100
    for path in paths:
        payload = copy.deepcopy(base)
        _mutate_leaf(payload, path)
        raw = _canonical(payload)
        contract.write_bytes(raw)
        _rebase_contract(monkeypatch, raw)
        assert gate3.validate_methods(contract, markdown) in {"SEMANTICS", "SOURCE", "POINTER"}


@pytest.mark.parametrize("mutation, code", [
    (lambda raw: b"\xef\xbb\xbf" + raw, "ENCODING"),
    (lambda raw: raw.replace(b"\n", b"\r\n"), "LINE_ENDING"),
    (lambda raw: raw.replace(b"{\n", b"{\n\xc3\xa9", 1), "ASCII"),
    (lambda raw: raw.replace(b'  "authorizations"', b'\t "authorizations"', 1), "CONTROL"),
    (lambda raw: raw.replace(b"{\n", b"{ \n", 1), "CANONICAL"),
    (lambda _raw: b'{"status":"x","status":"x"}\n', "DUPLICATE"),
    (lambda _raw: b'{"malformed":\n', "JSON"),
    (lambda _raw: b'{"value":NaN}\n', "JSON"),
    (lambda raw: raw.rstrip(b"\n"), "LINE_ENDING"),
])
def test_wire_attacks_return_fixed_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation, code: str) -> None:
    contract, markdown = _copies(tmp_path)
    _patch_outputs(monkeypatch, contract, markdown)
    raw = mutation(contract.read_bytes())
    contract.write_bytes(raw)
    _rebase_contract(monkeypatch, raw)
    assert gate3.validate_methods(contract, markdown) == code


def test_live_source_drift_and_one_read_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    relative, _sha256, _size = gate3.BOUND_INPUTS[-1]
    fake = tmp_path / Path(relative).name
    fake.write_bytes((ROOT / relative).read_bytes() + b"# drift\n")
    original_bound = gate3._checked_bound_path
    monkeypatch.setattr(gate3, "_checked_bound_path", lambda value: fake if value == relative else original_bound(value))
    assert gate3.validate_methods(CONTRACT, MARKDOWN) == "SOURCE"
    monkeypatch.setattr(gate3, "_checked_bound_path", original_bound)
    counts: dict[str, int] = {}
    original_read = Path.read_bytes

    def counted(path: Path) -> bytes:
        key = str(path.resolve())
        counts[key] = counts.get(key, 0) + 1
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    assert gate3.validate_methods(CONTRACT, MARKDOWN) == gate3.SUCCESS
    for path in (CONTRACT, MARKDOWN, *(ROOT / item[0] for item in gate3.BOUND_INPUTS)):
        assert counts[str(path.resolve())] == 1


def test_hostile_path_reparse_and_race_are_sanitized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hostile = tmp_path / "PRIVATE_SECRET_PATH"
    hostile.write_text("{}\n", encoding="ascii")
    assert gate3.validate_methods(hostile, MARKDOWN) == "PATH"
    completed = subprocess.run(
        [sys.executable, "-B", str(CHECKER), "--contract", str(hostile), "--markdown", str(MARKDOWN)],
        capture_output=True, text=True, check=False,
    )
    assert completed.stdout == "GATE3_PRODUCTION_METHODS:FAIL:PATH\n"
    assert completed.stderr == ""
    assert "PRIVATE_SECRET" not in completed.stdout + completed.stderr
    monkeypatch.setattr(gate3, "_identity", lambda _path, counter=iter(range(2)): (1, 1, gate3.EXPECTED_CONTRACT_SIZE, next(counter)))
    assert gate3.validate_methods(CONTRACT, MARKDOWN) == "RACE"
    link = tmp_path / "release_link"
    try:
        os.symlink(RELEASE, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        return
    assert gate3.validate_methods(link / CONTRACT.name, link / MARKDOWN.name) == "REPARSE"


def test_import_no_reads_control_exceptions_and_internal_sanitization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "read_bytes", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("read")))
    probe = _load_checker("gate3_production_methods_import_probe")
    assert probe.CONTRACT_NAME == CONTRACT.name
    for failure, expected in ((KeyboardInterrupt(), KeyboardInterrupt), (SystemExit(23), SystemExit)):
        monkeypatch.setattr(gate3, "_validation_token", lambda *_args, _failure=failure: (_ for _ in ()).throw(_failure))
        with pytest.raises(expected) as caught:
            gate3.validate_methods(CONTRACT, MARKDOWN)
        if expected is SystemExit:
            assert caught.value.code == 23
    monkeypatch.setattr(gate3, "_validation_token", lambda *_args: (_ for _ in ()).throw(ValueError("private")))
    assert gate3.validate_methods(CONTRACT, MARKDOWN) == "INTERNAL"
    output = io.StringIO()
    with redirect_stdout(output):
        assert gate3.main(["--contract", str(CONTRACT), "--markdown", str(MARKDOWN)]) == 1
    assert output.getvalue() == "GATE3_PRODUCTION_METHODS:FAIL:INTERNAL\n"


def _direct_selftest() -> None:
    _require(__debug__)
    _require(len(CHECKER_RAW) == CHECKER_SIZE)
    _require(hashlib.sha256(CHECKER_RAW).hexdigest() == CHECKER_SHA256)
    _require(not _checker_uses_dynamic_bytecode())
    _require(gate3.validate_methods(CONTRACT, MARKDOWN) == gate3.SUCCESS)
    original_path = gate3._checked_artifact_path
    original_bound = gate3._checked_bound_path
    original_identity = gate3._identity
    original_contract_hash = gate3.EXPECTED_CONTRACT_SHA256
    original_contract_size = gate3.EXPECTED_CONTRACT_SIZE
    original_markdown_hash = gate3.EXPECTED_MARKDOWN_SHA256
    original_markdown_size = gate3.EXPECTED_MARKDOWN_SIZE
    original_semantic = gate3.SEMANTIC_PAYLOAD_SHA256
    original_schema = gate3.SCHEMA_TYPES_SHA256
    try:
        with tempfile.TemporaryDirectory() as folder_name:
            folder = Path(folder_name)
            contract, markdown = _copies(folder)
            gate3._checked_artifact_path = lambda _path, name: contract if name == gate3.CONTRACT_NAME else markdown
            base = json.loads(CONTRACT.read_text(encoding="ascii"))

            def check_payload(payload: dict, code: str = "SEMANTICS") -> None:
                raw = _canonical(payload)
                contract.write_bytes(raw)
                gate3.EXPECTED_CONTRACT_SIZE = len(raw)
                gate3.EXPECTED_CONTRACT_SHA256 = hashlib.sha256(raw).hexdigest()
                gate3.SEMANTIC_PAYLOAD_SHA256 = gate3._compact_hash(payload)
                gate3.SCHEMA_TYPES_SHA256 = gate3._schema_hash(payload)
                _assert_token(gate3.validate_methods(contract, markdown), code)

            for key in base["authorizations"]:
                payload = copy.deepcopy(base)
                payload["authorizations"][key] = True
                check_payload(payload)
            for mutate in CLAIM_MUTATIONS.values():
                payload = copy.deepcopy(base)
                mutate(payload)
                check_payload(payload)

            contract.write_bytes(CONTRACT.read_bytes())
            gate3.EXPECTED_CONTRACT_SIZE = original_contract_size
            gate3.EXPECTED_CONTRACT_SHA256 = original_contract_hash
            gate3.SEMANTIC_PAYLOAD_SHA256 = original_semantic
            gate3.SCHEMA_TYPES_SHA256 = original_schema
            raw = MARKDOWN.read_bytes().replace(
                b'- `/activation/effective`: `false`',
                b'- `/activation/effective`: `true`',
                1,
            )
            markdown.write_bytes(raw)
            gate3.EXPECTED_MARKDOWN_SIZE = len(raw)
            gate3.EXPECTED_MARKDOWN_SHA256 = hashlib.sha256(raw).hexdigest()
            _assert_token(gate3.validate_methods(contract, markdown), "MARKDOWN")

            markdown.write_bytes(MARKDOWN.read_bytes())
            gate3.EXPECTED_MARKDOWN_SIZE = original_markdown_size
            gate3.EXPECTED_MARKDOWN_SHA256 = original_markdown_hash
            relative = gate3.BOUND_INPUTS[-1][0]
            fake = folder / Path(relative).name
            fake.write_bytes((ROOT / relative).read_bytes() + b"# drift\n")
            gate3._checked_bound_path = lambda value: fake if value == relative else original_bound(value)
            _assert_token(gate3.validate_methods(contract, markdown), "SOURCE")
            gate3._checked_bound_path = original_bound
            gate3._identity = lambda _path, counter=iter(range(2)): (1, 1, original_contract_size, next(counter))
            _assert_token(gate3.validate_methods(contract, markdown), "RACE")
    finally:
        gate3._checked_artifact_path = original_path
        gate3._checked_bound_path = original_bound
        gate3._identity = original_identity
        gate3.EXPECTED_CONTRACT_SHA256 = original_contract_hash
        gate3.EXPECTED_CONTRACT_SIZE = original_contract_size
        gate3.EXPECTED_MARKDOWN_SHA256 = original_markdown_hash
        gate3.EXPECTED_MARKDOWN_SIZE = original_markdown_size
        gate3.SEMANTIC_PAYLOAD_SHA256 = original_semantic
        gate3.SCHEMA_TYPES_SHA256 = original_schema
    _assert_token(gate3.validate_methods(Path(tempfile.gettempdir()) / "PRIVATE_MISSING_GATE3", MARKDOWN), "PATH")
    original_token = gate3._validation_token
    try:
        for failure, expected in ((KeyboardInterrupt(), KeyboardInterrupt), (SystemExit(23), SystemExit)):
            gate3._validation_token = lambda *_args, _failure=failure: (_ for _ in ()).throw(_failure)
            try:
                gate3.validate_methods(CONTRACT, MARKDOWN)
            except expected as caught:
                if expected is SystemExit:
                    _require(caught.code == 23)
            else:
                raise RuntimeError("control exception converted")
    finally:
        gate3._validation_token = original_token


def test_direct_selftest_wrapper() -> None:
    _direct_selftest()


def test_optimized_direct_selftest_fails_closed(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-B", "-O", str(Path(__file__).resolve())],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert completed.returncode != 0
    assert "PASS" not in completed.stdout + completed.stderr


if __name__ == "__main__":
    try:
        _direct_selftest()
    except Exception:
        raise SystemExit(1)
    print("GATE3_PRODUCTION_METHODS_SELFTEST:PASS")
