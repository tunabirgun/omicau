"""Estimate definitive campaign cost from measured per-unit timings.

Unit counts come from the protocol expansion, never from a stored total
(``benchmark_protocol.yaml#machine_summary.aggregate_counts_hardcoded: false``).
Per-unit cost comes from measured development runs, one measurement per sample
size, supplied as a JSON file of ``{"<sample_size>": seconds}``.

A sample size with no measurement is reported as unestimated rather than
interpolated: an invented cost would make an infeasible campaign look feasible.

Usage
    python estimate_campaign.py --timings local/benchmark_sizing/sizing.json
    python estimate_campaign.py --timings ... --workers 6 --json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE.parent
REPO = RECORD.parent


def load_units():
    sys.path.insert(0, str(REPO))
    import yaml
    from benchmarks.simulations.generate import protocol_units
    protocol = yaml.safe_load((RECORD / "benchmark_protocol.yaml").read_text(encoding="utf-8"))
    return protocol_units(protocol)


def load_timings(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {int(item["sample_size"]): float(item["unit_wall_seconds"])
                for item in payload if item.get("sample_size") is not None}
    return {int(key): float(value) for key, value in payload.items()}


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--exclude-family", action="append", default=["semi_synthetic_robustness"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    units = load_units()
    excluded = set(args.exclude_family)
    runnable = [u for u in units if u.family not in excluded]
    by_size = Counter(u.sample_size for u in runnable)
    timings = load_timings(args.timings)

    rows = []
    total_seconds = 0.0
    unestimated = 0
    for size, count in sorted(by_size.items(), key=lambda kv: (kv[0] is None, kv[0])):
        seconds = timings.get(size) if size is not None else None
        if seconds is None:
            unestimated += count
            rows.append({"sample_size": size, "units": count,
                         "seconds_per_unit": None, "hours": None})
            continue
        hours = count * seconds / 3600.0
        total_seconds += count * seconds
        rows.append({"sample_size": size, "units": count,
                     "seconds_per_unit": round(seconds, 1), "hours": round(hours, 1)})

    serial_hours = total_seconds / 3600.0
    report = {
        "registered_units": len(units),
        "excluded_families": sorted(excluded),
        "runnable_units": len(runnable),
        "unestimated_units": unestimated,
        "by_sample_size": rows,
        "serial_hours": round(serial_hours, 1),
        "workers": args.workers,
        "wall_hours_at_workers": round(serial_hours / max(1, args.workers), 1),
        "wall_days_at_workers": round(serial_hours / max(1, args.workers) / 24.0, 2),
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"registered units      : {report['registered_units']}")
    print(f"excluded families     : {', '.join(report['excluded_families']) or 'none'}")
    print(f"runnable units        : {report['runnable_units']}")
    print(f"{'n':>6} {'units':>7} {'s/unit':>9} {'hours':>9}")
    for row in rows:
        seconds = "unmeasured" if row["seconds_per_unit"] is None else f"{row['seconds_per_unit']:.1f}"
        hours = "-" if row["hours"] is None else f"{row['hours']:.1f}"
        print(f"{str(row['sample_size']):>6} {row['units']:>7} {seconds:>9} {hours:>9}")
    if unestimated:
        print(f"\n{unestimated} unit(s) have no measured cost and are excluded from the total")
    print(f"\nserial              : {report['serial_hours']:.1f} h")
    print(f"at {args.workers} worker(s)      : {report['wall_hours_at_workers']:.1f} h "
          f"({report['wall_days_at_workers']:.2f} days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
