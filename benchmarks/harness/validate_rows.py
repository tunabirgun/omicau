"""Validate benchmark result rows and their execution artifacts.

The protocol does not contain a canonical aggregate row count.  The task index
is the completeness contract and is therefore required; observed result keys
can never substitute for it.  Smoke output is development evidence and
definitive output is accepted only with ``phase: definitive``.

Usage
    python validate_rows.py ../results/raw/smoke_rows.jsonl
    python validate_rows.py ../results/raw/smoke_rows.jsonl --stale-minutes 30
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:  # package import (tests/tools) and direct script execution
    from .artifacts import split_digest
except ImportError:  # pragma: no cover - direct ``python validate_rows.py`` path
    from artifacts import split_digest

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
DEVELOPMENT_ROOT = BENCH.parent / "local" / "benchmark_smoke"
ARTIFACT_ROOT = BENCH
RECORD = BENCH.parent / "benchmark_record"
SCHEMA = RECORD / "schemas" / "benchmark-result.schema.json"
PROTOCOL = RECORD / "benchmark_protocol.yaml"
METRICS = RECORD / "OUTCOME_AND_METRIC_REGISTRY.yaml"
SPLITS = BENCH / "splits"
FAILURES = BENCH / "failures"
TASK_INDEX_NAMES = ("task_index.parquet", "task_index.csv", "task_index.json",
                    "task_index.jsonl")
FAILURE_DISPOSITIONS = {
    "unsupported_task", "implementation_unsuccessful", "resource_exhausted",
    "numerical_failure", "harness_defect",
}
NON_EXECUTED_STATUSES = {
    "ineligible", "not_eligible", "excluded", "not_applicable", "not_run",
}


def configure_root(root: Path) -> Path:
    global ARTIFACT_ROOT, SPLITS, FAILURES
    ARTIFACT_ROOT = root.resolve()
    SPLITS = ARTIFACT_ROOT / "splits"
    FAILURES = ARTIFACT_ROOT / "failures"
    return ARTIFACT_ROOT


def root_for_profile(profile: str | None, root: Path | None = None) -> Path:
    return (root if root is not None else
            (DEVELOPMENT_ROOT if profile == "smoke" else BENCH)).resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    if not path.exists():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def protocol_values() -> dict[str, Any]:
    protocol = _read_yaml(PROTOCOL)
    thresholds = protocol.get("audit_thresholds", {})
    metrics = _read_yaml(METRICS).get("metrics", {})
    protocol_block = protocol.get("protocol", {})
    record_block = protocol.get("record", {})
    version = None
    if isinstance(protocol_block, dict):
        version = protocol_block.get("version")
    if version is None and isinstance(record_block, dict):
        version = record_block.get("version")
    if version is None:
        version = protocol.get("version")
    return {
        "version": None if version is None else str(version),
        "thresholds": thresholds,
        "metrics": metrics,
    }


def _jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return [], [f"cannot read {path}: {type(exc).__name__}: {exc}"]
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except Exception as exc:
            problems.append(f"line {line_number}: invalid JSON ({type(exc).__name__})")
            continue
        if not isinstance(value, dict):
            problems.append(f"line {line_number}: row is not an object")
            continue
        rows.append(value)
    return rows, problems


def _task_index_paths() -> list[Path]:
    paths: list[Path] = []
    for name in TASK_INDEX_NAMES:
        paths.extend(sorted(ARTIFACT_ROOT.rglob(name)))
    return list(dict.fromkeys(paths))


def load_task_index() -> tuple[list[dict[str, Any]], list[str], list[Path]]:
    """Load all available task indexes without making a count assumption.

    Parquet is the protocol's preferred format.  CSV/JSON/JSONL are accepted so
    the smoke harness remains usable before the final parquet writer is installed.
    """
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    paths = _task_index_paths()
    for path in paths:
        try:
            if path.suffix == ".parquet":
                import pandas as pd
                values = pd.read_parquet(path).to_dict(orient="records")
            elif path.suffix == ".csv":
                with path.open(newline="", encoding="utf-8") as handle:
                    values = list(csv.DictReader(handle))
            elif path.suffix == ".jsonl":
                values, parse_problems = _jsonl(path)
                problems.extend(f"{path}: {item}" for item in parse_problems)
            else:
                value = json.loads(path.read_text(encoding="utf-8"))
                values = value if isinstance(value, list) else value.get("tasks", [])
            if not isinstance(values, list) or any(not isinstance(v, dict) for v in values):
                raise ValueError("task index must contain a list of objects")
            records.extend(values)
        except Exception as exc:
            problems.append(f"cannot read task index {path}: {type(exc).__name__}: {exc}")
    return records, problems, paths


def _first(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return None


def task_key(record: dict[str, Any]) -> tuple[Any, ...] | None:
    """Return the logical task key shared by task-index, result and failure rows."""
    dataset = _first(record, "dataset_id", "dataset", "cohort_id", "unit_id")
    method = _first(record, "method_id", "method", "comparator_id")
    if dataset is None or method is None:
        return None
    repeat = _first(record, "repeat", "repeat_index")
    if repeat is None:
        repeat = 0
    profile = _first(record, "profile", "analysis_profile")
    split = _first(record, "split_id", "split")
    return (str(dataset), str(method), str(repeat),
            None if split is None else str(split),
            None if profile is None else str(profile))


def _loose_key(key: tuple[Any, ...] | None) -> tuple[Any, ...] | None:
    """Drop the profile but keep the split id.

    The split id is what distinguishes the safe and unsafe arms of a paired
    group-leakage unit; without it the two arms share one key and each is
    reported as a duplicate of the other.
    """
    return None if key is None else key[:4]


def _status(record: dict[str, Any]) -> str:
    value = _first(record, "status", "disposition", "outcome")
    return str(value).strip().lower() if value is not None else "expected"


def record_family(record: dict[str, Any]) -> str | None:
    unit_key = record.get("unit_key")
    if isinstance(unit_key, dict) and unit_key.get("family"):
        return str(unit_key["family"])
    value = _first(record, "family")
    return None if value is None else str(value)


def expected_tasks(index: Iterable[dict[str, Any]], profile: str | None = None,
                   families: set[str] | None = None) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]]]:
    """Return expected task keys and keys explicitly excluded from execution.

    ``families`` scopes the completeness requirement to a staged phase.  It never
    removes a task from the index; unscoped families simply stay outstanding.
    """
    expected: set[tuple[Any, ...]] = set()
    excluded: set[tuple[Any, ...]] = set()
    for record in index:
        record_profile = _first(record, "profile", "analysis_profile")
        if profile and record_profile and str(record_profile) != profile:
            continue
        key = task_key(record)
        if key is None:
            continue
        if families is not None and record_family(record) not in families:
            continue
        if _status(record) in NON_EXECUTED_STATUSES:
            excluded.add(key)
        else:
            expected.add(key)
    return expected, excluded


def _schema_type_ok(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


# Validation keywords `_schema_errors` implements.  `schema_keyword_gaps` compares
# this against the keywords the record's schemas actually use, so a schema that
# grows a keyword this checker ignores fails loudly instead of passing vacuously.
IMPLEMENTED_SCHEMA_KEYWORDS = frozenset({
    "$ref", "type", "enum", "const", "required", "properties", "patternProperties",
    "additionalProperties", "items", "minimum", "maximum", "pattern", "minLength",
    "maxLength", "minItems", "maxItems", "minProperties",
})
ANNOTATION_SCHEMA_KEYWORDS = frozenset({
    "$schema", "$id", "$defs", "title", "description", "examples", "default",
})


def schema_keyword_gaps(schema: Any) -> set[str]:
    """Return validation keywords present in ``schema`` that are not enforced."""
    found: set[str] = set()

    def walk(node: Any, in_properties: bool = False) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if not in_properties:
                    found.add(str(key))
                # Property *names* are data, not keywords, so a field called
                # "type" must not be mistaken for the type keyword.
                walk(child, in_properties=key in {"properties", "patternProperties",
                                                  "$defs"})
        elif isinstance(node, list):
            for child in node:
                walk(child, in_properties=False)

    walk(schema)
    return found - IMPLEMENTED_SCHEMA_KEYWORDS - ANNOTATION_SCHEMA_KEYWORDS


def _resolve_ref(pointer: str, root: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a local JSON pointer; remote references are not supported here."""
    if not pointer.startswith("#/"):
        return None
    node: Any = root
    for token in pointer[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or token not in node:
            return None
        node = node[token]
    return node if isinstance(node, dict) else None


def _schema_errors(value: Any, schema: dict[str, Any], path: str = "row",
                   root: dict[str, Any] | None = None) -> list[str]:
    """Check a value against the JSON Schema subset the record's schemas use.

    ``IMPLEMENTED_SCHEMA_KEYWORDS`` lists what is enforced and
    ``schema_keyword_gaps`` reports anything a schema uses beyond it.  Silently
    ignoring a keyword would make a schema pass vacuously, which is worse than
    having no schema at all.
    """
    root = schema if root is None else root
    if "$ref" in schema:
        target = _resolve_ref(str(schema["$ref"]), root)
        if target is None:
            return [f"{path} references an unresolvable schema {schema['$ref']!r}"]
        merged = {key: item for key, item in schema.items() if key != "$ref"}
        return _schema_errors(value, {**target, **merged}, path, root)

    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}={value!r} is not in enum")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}={value!r} does not equal const")
    types = schema.get("type")
    if types and not any(_schema_type_ok(value, t) for t in ([types] if isinstance(types, str) else types)):
        errors.append(f"{path} has type {type(value).__name__}, expected {types}")
        return errors

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}={value!r} is below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}={value!r} is above maximum {schema['maximum']}")
    if isinstance(value, str):
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{path}={value!r} does not match pattern {schema['pattern']!r}")
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path} has {len(value)} characters, fewer than "
                          f"{schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path} has {len(value)} characters, more than "
                          f"{schema['maxLength']}")

    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path} missing {name}")
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            errors.append(f"{path} has {len(value)} properties, fewer than "
                          f"{schema['minProperties']}")
        properties = schema.get("properties", {})
        patterns = schema.get("patternProperties", {})
        additional = schema.get("additionalProperties")
        for name, child in value.items():
            if name in properties:
                if isinstance(properties[name], dict):
                    errors.extend(_schema_errors(child, properties[name], f"{path}.{name}", root))
                continue
            matched = [sub for expression, sub in patterns.items()
                       if isinstance(sub, dict) and re.search(expression, str(name))]
            if matched:
                for sub in matched:
                    errors.extend(_schema_errors(child, sub, f"{path}.{name}", root))
            elif additional is False:
                errors.append(f"{path} has unknown field {name}")
            elif isinstance(additional, dict):
                errors.extend(_schema_errors(child, additional, f"{path}.{name}", root))

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path} has {len(value)} items, fewer than {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path} has {len(value)} items, more than {schema['maxItems']}")
        if isinstance(schema.get("items"), dict):
            for i, item in enumerate(value):
                errors.extend(_schema_errors(item, schema["items"], f"{path}[{i}]", root))
    return errors


def _failure_records() -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    if not FAILURES.exists():
        return records, problems
    for path in sorted(FAILURES.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("failure record is not an object")
            records.append(value)
            if not value.get("run_id") or not value.get("disposition"):
                problems.append(f"failure record {path.name} lacks run_id or disposition")
            elif value["disposition"] not in FAILURE_DISPOSITIONS:
                problems.append(f"failure record {path.name} has unknown disposition {value['disposition']!r}")
        except Exception as exc:
            problems.append(f"unreadable failure record {path.name}: {type(exc).__name__}")
    return records, problems


def _resolve_artifact(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [ARTIFACT_ROOT / path]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _split_artifacts() -> dict[tuple[str, str, str], tuple[str | None, str | None, Path]]:
    """Index frozen splits by (dataset, repeat, splitter arm) with a recomputed digest.

    The stored digest is never taken on trust: the third element is derived from
    the manifest's own fold payload, so a split file whose contents were changed
    after freezing disagrees with itself and with every row that cites it.
    """
    found: dict[tuple[str, str, str], tuple[str | None, str | None, Path]] = {}
    if not SPLITS.exists():
        return found
    for path in sorted(SPLITS.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            dataset = _first(value, "dataset_id", "dataset")
            repeat = _first(value, "repeat", "repeat_index")
            if dataset is None or repeat is None:
                continue
            arm = str(_first(value, "splitter_arm") or "standard")
            recomputed = split_digest(value) if isinstance(value.get("folds"), list) else None
            found[(str(dataset), str(repeat), arm)] = (
                _first(value, "sha256", "split_sha256"), recomputed, path)
        except Exception:
            continue
    return found


def _metric_ranges() -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {"AUROC": (0.0, 1.0)}
    for name, value in protocol_values()["metrics"].items():
        if isinstance(value, dict) and isinstance(value.get("range"), list) and len(value["range"]) == 2:
            ranges[name] = (float(value["range"][0]), float(value["range"][1]))
    return ranges


def _metric_range(metric: Any) -> tuple[float, float] | None:
    if metric is None:
        return None
    normalized = str(metric).replace("-", "_").replace(" ", "_").upper()
    ranges = _metric_ranges()
    if normalized in ranges:
        return ranges[normalized]
    aliases = {"AUROC": "AUROC", "AUC": "AUROC", "C_INDEX": "AUROC", "ROLE_MACRO_F1": "ROLE_MACRO_F1"}
    return ranges.get(aliases.get(normalized, normalized))


def validate_records(rows: list[dict[str, Any]], rows_path: Path | None = None,
                    profile: str | None = None, stale_minutes: float | None = 30.0,
                    check_staleness: bool = True,
                    families: set[str] | None = None) -> dict[str, Any]:
    """Run schema and execution-integrity checks over rows.

    ``rows_path`` is optional for callers such as the monitor, which validates a
    collection of files.  Problems are intentionally plain strings so the JSON
    report remains stable for shell loops and CI.
    """
    problems: list[str] = []
    warnings: list[str] = []
    notes: dict[str, Any] = {}
    protocol = protocol_values()

    if not SCHEMA.exists():
        problems.append(f"result schema missing: {SCHEMA}")
    else:
        try:
            schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
            for i, row in enumerate(rows):
                errors = _schema_errors(row, schema, f"row {i}")
                if errors:
                    problems.extend(errors[:8])
                    if len(errors) > 8:
                        problems.append(f"row {i}: {len(errors) - 8} additional schema errors")
        except Exception as exc:
            problems.append(f"cannot read result schema: {type(exc).__name__}: {exc}")

    phases = Counter(str(row.get("phase")) for row in rows)
    notes["phase_counts"] = dict(phases)
    expected_phase = {"smoke": "development", "core": "definitive", "full": "definitive"}.get(profile)
    if expected_phase and any(phase != expected_phase for phase in phases):
        problems.append(f"phase boundary violation: {dict(phases)}; expected {expected_phase!r}")
    if len(phases) > 1:
        problems.append(f"mixed development/definitive rows: {dict(phases)}")
    declared_version = protocol["version"]
    if declared_version is None:
        problems.append(f"protocol version is missing from {PROTOCOL}")
    else:
        bad_versions = {row.get("protocol_version") for row in rows
                        if row.get("protocol_version") != declared_version}
        if bad_versions:
            problems.append(
                f"rows use protocol versions other than {declared_version}: "
                f"{sorted(map(str, bad_versions))}"
            )

    predictive = [row for row in rows if row.get("method_id") != "omicau_audit"]
    audit = [row for row in rows if row.get("method_id") == "omicau_audit"]
    split_artifacts = _split_artifacts()
    split_by_pair: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    split_problems = 0
    oof_problems = 0
    metric_problems = 0
    failure_records, failure_problems = _failure_records()
    problems.extend(failure_problems)
    failure_keys = {_loose_key(task_key(item)) for item in failure_records}
    for i, row in enumerate(rows):
        run_id = row.get("run_id")
        if not run_id:
            problems.append(f"row {i}: missing run_id")
        if row.get("method_id") == "omicau_audit":
            continue
        split_id = row.get("split_id")
        split_hash = row.get("split_sha256")
        if not split_id or not split_hash:
            problems.append(f"row {i}: predictive result lacks split_id or split_sha256")
            split_problems += 1
        else:
            pair = (str(row.get("dataset_id")),
                    str(row.get("repeat", row.get("repeat_index", 0))),
                    str(row.get("splitter_arm") or "standard"))
            split_by_pair[pair].add(str(split_hash))
            artifact = split_artifacts.get(pair)
            split_path = _resolve_artifact(row.get("split_path") or row.get("split_file"))
            if split_path and not split_path.exists():
                problems.append(f"row {i}: split file missing: {split_path}")
                split_problems += 1
            if artifact:
                stored, recomputed, artifact_path = artifact
                if recomputed is None:
                    problems.append(f"row {i}: split manifest has no fold payload to hash: {artifact_path}")
                    split_problems += 1
                else:
                    if stored and str(stored) != recomputed:
                        problems.append(f"row {i}: split manifest {artifact_path} disagrees with "
                                        f"its own recomputed digest")
                        split_problems += 1
                    if str(split_hash) != recomputed:
                        problems.append(f"row {i}: split_sha256 disagrees with the digest "
                                        f"recomputed from {artifact_path}")
                        split_problems += 1
            elif not split_path and SPLITS.exists():
                problems.append(f"row {i}: no frozen split artifact for {pair}")
                split_problems += 1

        oof_path = _resolve_artifact(row.get("oof_predictions_path"))
        if oof_path is None or not oof_path.exists():
            problems.append(f"row {i}: out-of-fold file missing: {row.get('oof_predictions_path')!r}")
            oof_problems += 1
        else:
            try:
                import numpy as np
                values = np.asarray(np.load(oof_path, allow_pickle=False))
                if values.ndim != 1:
                    raise ValueError(f"OOF array has {values.ndim} dimensions")
                if not np.isfinite(values).all():
                    raise ValueError("OOF array contains non-finite values")
                if values.size and (float(values.min()) < -1e-6 or float(values.max()) > 1 + 1e-6):
                    raise ValueError("OOF probabilities are outside [0, 1]")
                if row.get("n_samples") is not None and values.size != int(row["n_samples"]):
                    raise ValueError(f"OOF length {values.size} != n_samples {row['n_samples']}")
                if values.size and float(np.std(values)) < 1e-12:
                    warnings.append(f"row {i}: OOF predictions are constant")
            except Exception as exc:
                problems.append(f"row {i}: invalid out-of-fold file {oof_path}: {exc}")
                oof_problems += 1

        value = row.get("primary_value")
        if value is None:
            if _loose_key(task_key(row)) not in failure_keys and _status(row) not in FAILURE_DISPOSITIONS:
                problems.append(f"row {i}: null primary metric has no matching failure record")
                metric_problems += 1
        elif not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            problems.append(f"row {i}: primary metric is not finite numeric")
            metric_problems += 1
        else:
            bounds = _metric_range(row.get("primary_metric"))
            if bounds and not (bounds[0] <= float(value) <= bounds[1]):
                problems.append(f"row {i}: {row.get('primary_metric')}={value} outside {bounds}")
                metric_problems += 1

    duplicate_ids = [run_id for run_id, count in Counter(row.get("run_id") for row in rows).items()
                     if run_id is not None and count > 1]
    if duplicate_ids:
        problems.append(f"duplicate run_id values: {duplicate_ids[:3]}")
    duplicate_keys = [key for key, count in Counter(_loose_key(task_key(row)) for row in rows).items()
                      if key is not None and count > 1]
    if duplicate_keys:
        problems.append(f"duplicate logical task rows: {duplicate_keys[:3]}")

    split_conflicts = {key: values for key, values in split_by_pair.items() if len(values) > 1}
    if split_conflicts:
        problems.append(f"methods saw different splits for (dataset, repeat, arm) units: "
                        f"{list(split_conflicts)[:3]}")
        split_problems += len(split_conflicts)

    null_values = [row.get("primary_value") for row in predictive
                   if (row.get("simulation") or {}).get("overlay") in {"clean_null", "no_predictive_signal"}
                   and isinstance(row.get("primary_value"), (int, float))]
    if null_values:
        reference = float(next((m.get("null_value") for name, m in protocol["metrics"].items()
                                if name == "AUROC" and isinstance(m, dict) and m.get("null_value") is not None), 0.5))
        tolerance = float(protocol["thresholds"].get("CONTROL_MARGIN", 0.12))
        mean = sum(float(v) for v in null_values) / len(null_values)
        notes["null_metric_mean"] = round(mean, 6)
        notes["null_metric_n"] = len(null_values)
        if abs(mean - reference) > tolerance:
            problems.append(f"null behavior failed: mean metric {mean:.3f} is {abs(mean - reference):.3f} from reference {reference:.3f}")

    if rows_path and check_staleness and stale_minutes is not None and rows_path.exists():
        age = max(0.0, (dt.datetime.now().timestamp() - rows_path.stat().st_mtime) / 60.0)
        notes["minutes_since_last_result"] = round(age, 1)
        if age > stale_minutes:
            problems.append(f"stale result file: {age:.1f} minutes old (threshold {stale_minutes:.1f})")

    index, index_problems, index_paths = load_task_index()
    problems.extend(index_problems)
    notes["task_index_paths"] = [str(path) for path in index_paths]
    notes["task_index_rows"] = len(index)
    expected, excluded = expected_tasks(index, profile, families)
    catalogued, catalogued_excluded = expected_tasks(index, profile)
    observed = {_loose_key(task_key(row)) for row in rows if task_key(row) is not None}
    observed_failures = {_loose_key(task_key(item)) for item in failure_records if task_key(item) is not None}
    if families is not None:
        notes["family_scope"] = sorted(families)
        notes["catalogued_tasks"] = len(catalogued)
    if not index_paths:
        problems.append("task index is required for completeness; no task index was found")
        notes["expectation_source"] = "none; task index required"
    elif expected:
        expected_loose = {_loose_key(key) for key in expected}
        # A row is unexpected only when the whole index does not contain it; a row
        # from a family outside this phase's scope is out of scope, not spurious.
        catalogued_loose = {_loose_key(key) for key in catalogued}
        missing = expected_loose - observed - observed_failures
        extra = observed - catalogued_loose - {_loose_key(key) for key in catalogued_excluded}
        notes["expected_tasks"] = len(expected_loose)
        notes["missing_tasks"] = len(missing)
        if missing:
            problems.append(f"task index has {len(missing)} missing result/failure task(s): {sorted(missing)[:3]}")
        if extra:
            problems.append(f"rows contain {len(extra)} task(s) absent from task index: {sorted(extra)[:3]}")
    elif profile:
        problems.append(f"task index has no expected tasks for profile {profile!r}")
        notes["expectation_source"] = "task index; no matching tasks"
    else:
        problems.append("task index has no expected tasks")
        notes["expectation_source"] = "task index; no expected tasks"

    notes.update({"predictive_rows": len(predictive), "audit_rows": len(audit),
                  "split_problems": split_problems, "oof_problems": oof_problems,
                  "metric_problems": metric_problems, "failure_records": len(failure_records),
                  "failure_dispositions": dict(Counter(item.get("disposition", "?") for item in failure_records))})
    for item in failure_records:
        disposition = item.get("disposition")
        if disposition in {"harness_defect", "numerical_failure", "resource_exhausted", "implementation_unsuccessful"}:
            problems.append(f"failure record {item.get('run_id', '?')}: {disposition}")
        elif disposition == "unsupported_task":
            warnings.append(f"unsupported task retained as failure: {item.get('run_id', '?')}")
    return {"problems": problems, "warnings": warnings, "notes": notes,
            "status": "PROBLEM" if problems else ("WARN" if warnings else "OK")}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = __import__("argparse").ArgumentParser(description=__doc__)
    ap.add_argument("rows", type=Path)
    ap.add_argument("--profile", choices=["smoke", "core", "full"], default=None)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--stale-minutes", type=float, default=30.0)
    ap.add_argument("--family", action="append", default=None,
                    help="scope completeness to these families, repeatable")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    configure_root(root_for_profile(args.profile, args.root))
    rows, parse_problems = _jsonl(args.rows)
    report = validate_records(rows, args.rows, args.profile, args.stale_minutes,
                              families=set(args.family) if args.family else None)
    report["problems"] = parse_problems + report["problems"]
    report["status"] = "PROBLEM" if report["problems"] else ("WARN" if report["warnings"] else "OK")
    report["rows"] = len(rows)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"{args.rows}: {report['status']} ({len(rows)} rows)")
        for problem in report["problems"]:
            print(f"  PROBLEM: {problem}")
        for warning in report["warnings"]:
            print(f"  warn   : {warning}")
        for key, value in report["notes"].items():
            print(f"  {key}: {value}")
    return 1 if report["status"] == "PROBLEM" else 0


if __name__ == "__main__":
    raise SystemExit(main())
