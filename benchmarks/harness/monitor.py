"""Read-only health check for a running benchmark.

The monitor derives progress from emitted rows and the required protocol task
index.  It never uses a fixed aggregate count or an observed-row fallback and
never treats development rows as definitive evidence.  Integrity checks are
shared with ``validate_rows`` and are run over every emitted OOF file.

Usage
    python monitor.py
    python monitor.py --json
    python monitor.py --stale-minutes 30
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
DEVELOPMENT_ROOT = BENCH.parent / "local" / "benchmark_smoke"
ARTIFACT_ROOT = BENCH
RESULTS = BENCH / "results" / "raw"
RUNS = BENCH / "runs"
FAILURES = BENCH / "failures"

try:  # Works both as a package import and when run as a script.
    from .validate_rows import (_jsonl, configure_root as configure_validation_root,
                                expected_tasks, load_task_index, validate_records)
except ImportError:  # pragma: no cover - exercised by the CLI
    from validate_rows import (_jsonl, configure_root as configure_validation_root,
                               expected_tasks, load_task_index, validate_records)

DEGENERATE_EPS = 1e-9


def configure_root(profile: str | None, root: Path | None = None) -> Path:
    global ARTIFACT_ROOT, RESULTS, RUNS, FAILURES
    ARTIFACT_ROOT = (root if root is not None else
                     (DEVELOPMENT_ROOT if profile == "smoke" else BENCH)).resolve()
    RESULTS = ARTIFACT_ROOT / "results" / "raw"
    RUNS = ARTIFACT_ROOT / "runs"
    FAILURES = ARTIFACT_ROOT / "failures"
    configure_validation_root(ARTIFACT_ROOT)
    return ARTIFACT_ROOT


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(RESULTS.glob("*_rows.jsonl")):
        parsed, problems = _jsonl(path)
        rows.extend(parsed)
        if problems:
            rows.extend({"_load_error": f"{path}: {problem}"} for problem in problems)
    return rows


def _profiles() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(RESULTS.glob("*_rows.jsonl")):
        name = path.name.removesuffix("_rows.jsonl")
        profile = name.split("_", 1)[0]
        rows, problems = _jsonl(path)
        grouped.setdefault(profile, []).extend(rows)
        if problems:
            grouped[profile].extend({"_load_error": f"{path}: {problem}"} for problem in problems)
    return grouped


def _failure_summary() -> Counter[str]:
    disposition: Counter[str] = Counter()
    if not FAILURES.exists():
        return disposition
    for path in sorted(FAILURES.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            disposition[str(value.get("disposition", "?"))] += 1
        except Exception:
            disposition["unreadable"] += 1
    return disposition


def _disk_report() -> tuple[float, dict[str, int], list[str], list[str]]:
    problems: list[str] = []
    warnings: list[str] = []
    total, used, free = shutil.disk_usage(ARTIFACT_ROOT)
    if free < 20 * 2**30:
        problems.append(f"free disk below 20 GiB ({free / 2**30:.1f} GiB)")
    elif free < 60 * 2**30:
        warnings.append(f"free disk at {free / 2**30:.1f} GiB")
    tree: dict[str, int] = {}
    for directory in (RESULTS.parent, ARTIFACT_ROOT / "splits", FAILURES, RUNS):
        if directory.exists():
            tree[directory.name] = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    leftover = tree.get("runs", 0)
    if leftover > 2 * 2**30:
        warnings.append(f"runs/ holds {leftover / 2**30:.1f} GiB of intermediates")
    return free / 2**30, tree, problems, warnings


def check(expect: int | None = None, stale_minutes: float = 30.0,
          profile: str | None = None, root: Path | None = None) -> dict[str, Any]:
    """Check outputs; ``expect`` is retained only for caller compatibility.

    The deprecated argument is deliberately ignored.  Expected work comes from
    task-index rows, never from a caller-supplied aggregate integer.
    """
    del expect
    configure_root(profile, root)
    rows = load_rows()
    problems: list[str] = []
    warnings: list[str] = []
    notes: dict[str, Any] = {}
    grouped = _profiles()
    selected = {profile: grouped.get(profile, [])} if profile else grouped
    if not selected and rows:
        selected = {"unknown": rows}
    for name, group_rows in selected.items():
        report = validate_records(group_rows, profile=name if name in {"smoke", "core", "full"} else None,
                                  stale_minutes=None, check_staleness=False)
        problems.extend(f"{name}: {item}" for item in report["problems"])
        warnings.extend(f"{name}: {item}" for item in report["warnings"])
        notes.setdefault("profile_notes", {})[name] = report["notes"]

    row_files = list(RESULTS.glob("*_rows.jsonl"))
    newest = max((path.stat().st_mtime for path in row_files), default=0.0)
    quiet_minutes = (time.time() - newest) / 60.0 if newest else math.inf
    if newest and quiet_minutes > stale_minutes:
        problems.append(f"no new results for {quiet_minutes:.1f} min (threshold {stale_minutes:.1f})")
    if not rows:
        warnings.append("no result rows found")

    index, index_problems, index_paths = load_task_index()
    problems.extend(index_problems)
    expected, excluded = expected_tasks(index, profile)
    datasets = {row.get("dataset_id") for row in rows if row.get("dataset_id") is not None}
    expected_datasets = {key[0] for key in expected} if expected else set()
    progress_denominator = len(expected_datasets)
    notes["task_index_paths"] = [str(path) for path in index_paths]
    notes["expected_tasks"] = len(expected)
    notes["excluded_tasks"] = len(excluded)
    if not index_paths:
        problems.append("task index is required for completeness; no task index was found")
        notes["expectation_source"] = "none; task index required"
    elif expected:
        notes["expectation_source"] = "task index"
    else:
        problems.append(f"task index has no expected tasks{f' for profile {profile!r}' if profile else ''}")
        notes["expectation_source"] = "task index; no expected tasks"

    disk_free, tree, disk_problems, disk_warnings = _disk_report()
    problems.extend(disk_problems)
    warnings.extend(disk_warnings)
    disposition = _failure_summary()
    phases = Counter(str(row.get("phase")) for row in rows)
    notes["phase_counts"] = dict(phases)
    problems = list(dict.fromkeys(problems))
    warnings = list(dict.fromkeys(warnings))
    return {
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rows": len(rows),
        "datasets": len(datasets),
        "methods": sorted({str(row.get("method_id")) for row in rows if row.get("method_id")}),
        "repeats": sorted({row.get("repeat", row.get("repeat_index", 0)) for row in rows}),
        "progress": f"{len(datasets)}/{progress_denominator}" if progress_denominator else str(len(datasets)),
        "progress_fraction": (len(datasets) / progress_denominator if progress_denominator else None),
        "minutes_since_last_result": None if quiet_minutes is math.inf else round(quiet_minutes, 1),
        "failures": dict(disposition),
        "disk_free_gib": round(disk_free, 1),
        "tree_bytes": tree,
        "notes": notes,
        "problems": problems,
        "warnings": warnings,
        "status": "PROBLEM" if problems else ("WARN" if warnings else "OK"),
    }


# --------------------------------------------------------------------------- #
# hourly progress report
# --------------------------------------------------------------------------- #
def _markers() -> list[dict[str, Any]]:
    directory = RUNS / "complete"
    if not directory.is_dir():
        return []
    markers = []
    for path in sorted(directory.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            markers.append(value)
    return markers


def _family_of_dataset(index: list[dict[str, Any]]) -> dict[str, str]:
    families: dict[str, str] = {}
    for record in index:
        unit = record.get("unit_key")
        if isinstance(unit, dict) and record.get("dataset_id") and unit.get("family"):
            families[str(record["dataset_id"])] = str(unit["family"])
    return families


def _throughput(markers: list[dict[str, Any]]) -> tuple[float | None, str | None]:
    """Units per hour from completion timestamps, and the most recent one."""
    stamps = sorted(str(m["completed_utc"]) for m in markers if m.get("completed_utc"))
    if len(stamps) < 2:
        return None, (stamps[-1] if stamps else None)
    try:
        first = dt.datetime.fromisoformat(stamps[0])
        last = dt.datetime.fromisoformat(stamps[-1])
    except ValueError:
        return None, stamps[-1]
    hours = (last - first).total_seconds() / 3600.0
    # n-1 intervals span the elapsed window; the first unit's own cost is outside it.
    return ((len(stamps) - 1) / hours if hours > 0 else None), stamps[-1]


def _retries() -> tuple[int, list[str]]:
    """Quarantined staging directories are the record of re-run attempts."""
    index = RUNS / "quarantine" / "index.jsonl"
    if not index.is_file():
        return 0, []
    reasons = []
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            reasons.append(str(json.loads(line).get("reason", "?")))
        except json.JSONDecodeError:
            reasons.append("unreadable quarantine record")
    return len(reasons), reasons


def _disk_growth(tree: dict[str, int], started: bool) -> dict[str, Any]:
    """Growth since the previous report, using a state file this function owns.

    Nothing is written before execution has started. ``benchmarks/runs`` must be
    empty for ``readiness._check_empty_outputs`` to pass, so a monitor run against
    the definitive root during planning must not create a file there.
    """
    state_path = RUNS / "monitor_state.json"
    total = sum(tree.values())
    previous: dict[str, Any] = {}
    if state_path.is_file():
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    now = time.time()
    growth: dict[str, Any] = {"total_bytes": total}
    if isinstance(previous.get("total_bytes"), int) and previous.get("at"):
        elapsed_hours = (now - float(previous["at"])) / 3600.0
        delta = total - int(previous["total_bytes"])
        growth["since_last_report_bytes"] = delta
        growth["mib_per_hour"] = (round(delta / 2**20 / elapsed_hours, 2)
                                  if elapsed_hours > 0 else None)
        growth["hours_since_last_report"] = round(elapsed_hours, 2)
    if started:
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps({"at": now, "total_bytes": total}), encoding="utf-8")
        except OSError:
            pass
    else:
        growth["state_not_written"] = "execution has not started; runs/ is left empty"
    return growth


def _sanity_checks(rows: list[dict[str, Any]],
                   families: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    """Report design-implied orderings. Observations only; nothing feeds back.

    A deviation here is a signal to investigate, never a reason to change a
    threshold, endpoint or cell -- benchmark_protocol.yaml forbids
    result-dependent protocol changes.
    """
    checks: dict[str, Any] = {}
    deviations: list[str] = []

    def family(row: dict[str, Any]) -> str | None:
        unit = (row.get("simulation") or {}).get("unit_key") or {}
        return unit.get("family") if isinstance(unit, dict) else families.get(
            str(row.get("dataset_id")))

    predictive = [r for r in rows if r.get("method_id") != "omicau_audit"
                  and isinstance(r.get("primary_value"), (int, float))]

    nulls = [r["primary_value"] for r in predictive
             if family(r) == "null_control_specificity"]
    if nulls:
        mean = sum(nulls) / len(nulls)
        checks["null_control_mean_auroc"] = round(mean, 4)
        checks["null_control_n"] = len(nulls)
        if abs(mean - 0.5) > 0.05:
            deviations.append(f"null-control mean AUROC {mean:.3f} is {abs(mean-0.5):.3f} "
                              "from chance; the outcome is independent of every feature "
                              "by construction")

    safe = [r["primary_value"] for r in predictive if r.get("splitter_arm") == "safe"]
    unsafe = [r["primary_value"] for r in predictive if r.get("splitter_arm") == "unsafe"]
    if safe and unsafe:
        gap = sum(unsafe) / len(unsafe) - sum(safe) / len(safe)
        checks["group_leakage_gap"] = round(gap, 4)
        checks["group_leakage_pairs"] = min(len(safe), len(unsafe))
        if gap <= 0:
            deviations.append(f"naive-minus-group-aware gap is {gap:+.3f}; ignoring "
                              "subject grouping is expected to inflate AUROC, not deflate it")

    audit = [r for r in rows if r.get("method_id") == "omicau_audit"]
    for severity_family, flag in (("batch_risk_flags", "batch_confounded"),
                                  ("missingness_risk_flags", "missingness_bias")):
        rates: dict[str, tuple[int, int]] = {}
        for row in audit:
            if family(row) != severity_family:
                continue
            condition = str((row.get("simulation") or {}).get("overlay") or "")
            severity = condition.split("__", 1)[0] or "unknown"
            fired, total = rates.get(severity, (0, 0))
            rates[severity] = (fired + bool((row.get("audit") or {}).get(flag)), total + 1)
        if rates:
            checks[f"{severity_family}_flag_rate"] = {
                k: f"{v[0]}/{v[1]}" for k, v in sorted(rates.items())}
            clean = rates.get("clean")
            severe = rates.get("severe")
            if clean and severe and clean[1] and severe[1]:
                if severe[0] / severe[1] <= clean[0] / clean[1]:
                    deviations.append(
                        f"{severity_family}: severe flag rate {severe[0]}/{severe[1]} does not "
                        f"exceed clean {clean[0]}/{clean[1]}; the severity contrast is the "
                        "registered positive/negative pair")

    by_size: dict[int, list[float]] = {}
    for row in predictive:
        if family(row) != "predictive_performance" or row.get("splitter_arm") != "standard":
            continue
        size = (row.get("simulation") or {}).get("sample_size")
        if isinstance(size, int):
            by_size.setdefault(size, []).append(row["primary_value"])
    if len(by_size) > 1:
        means = {n: sum(v) / len(v) for n, v in sorted(by_size.items())}
        checks["predictive_auroc_by_n"] = {n: round(m, 4) for n, m in means.items()}
        ordered = list(means.items())
        for (small, low), (large, high) in zip(ordered, ordered[1:]):
            if high < low - 0.02:
                deviations.append(f"mean AUROC falls from {low:.3f} at n={small} to "
                                  f"{high:.3f} at n={large}; more data is expected to help")
    return checks, deviations


def progress_report(profile: str | None = None, root: Path | None = None) -> dict[str, Any]:
    """The hourly execution report: phase, progress, throughput, cost, sanity."""
    configure_root(profile, root)
    rows = load_rows()
    index, index_problems, _ = load_task_index()
    expected, _ = expected_tasks(index, profile)
    families = _family_of_dataset(index)
    expected_datasets = {key[0] for key in expected}
    markers = _markers()
    completed = {str(m.get("dataset_id")) for m in markers if m.get("dataset_id")}

    outstanding_families = Counter(families.get(d, "unknown")
                                   for d in expected_datasets - completed)
    completed_families = Counter(families.get(d, "unknown") for d in completed)
    rate, last_completed = _throughput(markers)
    remaining = len(expected_datasets - completed)
    eta_hours = (remaining / rate) if rate else None

    costs = [r["cost"] for r in rows if isinstance(r.get("cost"), dict)]
    peak_rss = max((c["peak_rss_mib"] for c in costs
                    if isinstance(c.get("peak_rss_mib"), (int, float))), default=None)
    retries, retry_reasons = _retries()
    disk_free, tree, _, _ = _disk_report()
    checks, deviations = _sanity_checks(rows, families)

    return {
        "reported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile": profile,
        "active_phase": (outstanding_families.most_common(1)[0][0]
                         if outstanding_families else "none outstanding"),
        "families_outstanding": dict(outstanding_families),
        "families_completed": dict(completed_families),
        "units_completed": len(completed),
        "units_expected": len(expected_datasets),
        "rows": len(rows),
        "throughput_units_per_hour": None if rate is None else round(rate, 2),
        "last_unit_completed_utc": last_completed,
        "eta_hours": None if eta_hours is None else round(eta_hours, 1),
        "failures": dict(_failure_summary()),
        "retries": retries,
        "retry_reasons": retry_reasons[-3:],
        "peak_process_rss_mib": peak_rss,
        "peak_rss_is_method_exclusive": False,
        "disk_free_gib": round(disk_free, 1),
        "disk_growth": _disk_growth(tree, started=bool(markers)),
        "sanity_checks": checks,
        "expected_ordering_deviations": deviations,
        "task_index_problems": index_problems,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--profile", choices=["smoke", "core", "full"], default=None)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--stale-minutes", type=float, default=30.0)
    ap.add_argument("--progress", action="store_true",
                    help="emit the hourly execution progress report")
    args = ap.parse_args()
    if args.progress:
        report = progress_report(profile=args.profile, root=args.root)
        if args.json:
            print(json.dumps(report, indent=2, default=str))
            return 0
        print(f"[{report['reported_at']}] phase: {report['active_phase']}")
        print(f"  units      {report['units_completed']}/{report['units_expected']}  "
              f"({report['rows']} rows)")
        print(f"  throughput {report['throughput_units_per_hour']} units/h   "
              f"ETA {report['eta_hours']} h")
        print(f"  failures   {report['failures'] or '{}'}   retries {report['retries']}")
        print(f"  peak RSS   {report['peak_process_rss_mib']} MiB (process-level, not "
              "method-exclusive)")
        growth = report["disk_growth"]
        print(f"  disk       {report['disk_free_gib']} GiB free, "
              f"{growth['total_bytes'] / 2**20:.1f} MiB used"
              + (f", +{growth['mib_per_hour']} MiB/h" if growth.get("mib_per_hour") is not None else ""))
        for name, value in report["sanity_checks"].items():
            print(f"  check      {name}: {value}")
        for item in report["expected_ordering_deviations"]:
            print(f"  DEVIATION  {item}")
        for item in report["task_index_problems"]:
            print(f"  PROBLEM    {item}")
        return 0
    report = check(stale_minutes=args.stale_minutes, profile=args.profile, root=args.root)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["status"] != "PROBLEM" else 1
    print(f"[{report['checked_at']}] {report['status']}")
    print(f"  rows {report['rows']} | datasets {report['progress']} | methods {len(report['methods'])} "
          f"| repeats {report['repeats']}")
    if report["minutes_since_last_result"] is not None:
        print(f"  last result {report['minutes_since_last_result']} min ago")
    print(f"  failures {report['failures'] or '{}'} | free disk {report['disk_free_gib']} GiB")
    for problem in report["problems"]:
        print(f"  PROBLEM: {problem}")
    for warning in report["warnings"]:
        print(f"  warn   : {warning}")
    return 0 if report["status"] != "PROBLEM" else 1


if __name__ == "__main__":
    raise SystemExit(main())
