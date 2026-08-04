"""Between-phase verification for benchmark outputs.

Completeness is derived from the required machine-readable task index.  No
aggregate row count is canonical in the protocol.  Smoke is a separate
development validation and can never pass as definitive evidence; core and full
require ``phase: definitive`` rows and passing execution readiness.

Usage
    python phase_gate.py --phase smoke
    python phase_gate.py --phase core --json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
DEVELOPMENT_ROOT = BENCH.parent / "local" / "benchmark_smoke"
ARTIFACT_ROOT = BENCH
RESULTS = BENCH / "results" / "raw"
FAILURES = BENCH / "failures"

try:  # Works both as a package import and when run as a script.
    from .validate_rows import (_read_yaml, configure_root as configure_validation_root,
                                expected_tasks, load_task_index, protocol_values,
                                validate_records)
except ImportError:  # pragma: no cover - exercised by the CLI
    from validate_rows import (_read_yaml, configure_root as configure_validation_root,
                               expected_tasks, load_task_index, protocol_values,
                               validate_records)

RECORD = BENCH.parent / "benchmark_record"
PROTOCOL = RECORD / "benchmark_protocol.yaml"


def configure_root(phase: str, root: Path | None = None) -> Path:
    global ARTIFACT_ROOT, RESULTS, FAILURES
    ARTIFACT_ROOT = (root if root is not None else
                     (DEVELOPMENT_ROOT if phase == "smoke" else BENCH)).resolve()
    RESULTS = ARTIFACT_ROOT / "results" / "raw"
    FAILURES = ARTIFACT_ROOT / "failures"
    configure_validation_root(ARTIFACT_ROOT)
    return ARTIFACT_ROOT


def load(phase: str) -> list[dict[str, Any]]:
    """Load every result file for a profile, including resumed/sharded files."""
    rows: list[dict[str, Any]] = []
    for path in sorted(RESULTS.glob(f"{phase}*_rows.jsonl")):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
        except (OSError, json.JSONDecodeError):
            # validate_records cannot validate a row that could not be decoded;
            # retain a visible synthetic marker so the gate still fails.
            rows.append({"_load_error": str(path)})
    return rows


def _control_checks(rows: list[dict[str, Any]], notes: dict[str, Any]) -> list[str]:
    """Apply the protocol's control margin to clean audit controls."""
    protocol = _read_yaml(PROTOCOL)
    margin = float(protocol.get("audit_thresholds", {}).get("CONTROL_MARGIN", 0.12))
    null_reference = float(protocol_values()["metrics"].get("AUROC", {}).get("null_value", 0.5))
    audit = [row for row in rows if row.get("method_id") == "omicau_audit"]
    clean: list[dict[str, Any]] = []
    for row in audit:
        raw_notes = row.get("notes")
        try:
            parsed = json.loads(raw_notes) if isinstance(raw_notes, str) else (raw_notes or {})
        except (TypeError, json.JSONDecodeError):
            parsed = {}
        if not parsed.get("leakage_present"):
            clean.append(row)
    controls = [value for row in clean for value in (row.get("control_baselines") or {}).values()
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))]
    if not controls:
        return []
    mean = st.fmean(controls)
    notes["clean_control_mean"] = round(mean, 6)
    notes["control_margin"] = margin
    return ([f"clean control mean {mean:.3f} exceeds null reference + CONTROL_MARGIN ({null_reference + margin:.3f})"]
            if mean > null_reference + margin else [])


def gate(phase: str, expect: int | None = None,
         stale_minutes: float | None = 30.0,
         root: Path | None = None,
         families: set[str] | None = None) -> dict[str, Any]:
    # ``expect`` is retained for API compatibility with older callers, but is
    # intentionally ignored: the protocol has no canonical aggregate count.
    del expect
    configure_root(phase, root)
    rows = load(phase)
    if not rows:
        return {"phase": phase, "verdict": "FAIL", "failures": ["no rows found"],
                "warnings": [], "notes": {"expectation_source": "task index if present"}}

    failures: list[str] = []
    warnings: list[str] = []
    notes: dict[str, Any] = {"rows": len(rows),
                             "datasets": len({row.get("dataset_id") for row in rows if row.get("dataset_id")}),
                             "expected_evidence_phase": {"smoke": "development", "core": "definitive",
                                                          "full": "definitive"}.get(phase)}
    report = validate_records(rows, profile=phase, stale_minutes=stale_minutes,
                              check_staleness=True, families=families)
    failures.extend(report["problems"])
    warnings.extend(report["warnings"])
    notes.update(report["notes"])

    index, index_problems, index_paths = load_task_index()
    failures.extend(index_problems)
    expected, excluded = expected_tasks(index, phase, families)
    if not index_paths:
        notes["expectation_source"] = "none; task index required"
        failures.append("task index is required for completeness; no task index was found")
    elif expected:
        notes["expectation_source"] = "task index"
        notes["expected_tasks"] = len(expected)
        notes["excluded_tasks"] = len(excluded)
    else:
        notes["expectation_source"] = "task index; no matching tasks"
        failures.append(f"task index has no expected tasks for phase {phase!r}")

    if phase in {"core", "full"}:
        try:
            from benchmark_record.tools.readiness import check_readiness
            readiness = check_readiness(RECORD.parent)
        except Exception as exc:
            readiness = None
            failures.append(f"{phase} requires readiness, but readiness could not be checked: {exc}")
        if readiness is not None:
            notes["readiness_status"] = readiness.get("status")
            if readiness.get("status") != "pass":
                failures.append(f"{phase} requires passing protocol publication and execution readiness")
                failures.extend(str(item) for item in readiness.get("blockers", []))

    failures.extend(_control_checks(rows, notes))

    # The protocol's primary null metric and its reference are read from the
    # outcome registry. This is a sanity-check report, never a claim that all
    # leakage was ruled out.
    null_rows = [row for row in rows
                 if row.get("method_id") != "omicau_audit"
                 and (row.get("simulation") or {}).get("overlay") in {"clean_null", "no_predictive_signal"}
                 and isinstance(row.get("primary_value"), (int, float))]
    if null_rows:
        null_mean = st.fmean(float(row["primary_value"]) for row in null_rows)
        notes["null_control_mean"] = round(null_mean, 6)
        notes["null_control_n"] = len(null_rows)

    failures = list(dict.fromkeys(failures))
    warnings = list(dict.fromkeys(warnings))
    verdict = "FAIL" if failures else ("WARN" if warnings else "PASS")
    return {"phase": phase, "verdict": verdict, "failures": failures,
            "warnings": warnings, "notes": notes}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["smoke", "core", "full"], default="smoke")
    ap.add_argument("--stale-minutes", type=float, default=30.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--family", action="append", default=None,
                    help="scope the completeness requirement to these families, repeatable")
    args = ap.parse_args()
    report = gate(args.phase, stale_minutes=args.stale_minutes, root=args.root,
                  families=set(args.family) if args.family else None)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"phase gate: {args.phase} -> {report['verdict']}")
        for key, value in report["notes"].items():
            print(f"  {key}: {value}")
        for failure in report["failures"]:
            print(f"  FAIL: {failure}")
        for warning in report["warnings"]:
            print(f"  warn: {warning}")
    return 1 if report["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
