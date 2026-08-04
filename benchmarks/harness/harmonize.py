"""Harmonize raw result rows into the tables the report cites.

Reads only from results/raw/, writes long-format CSV to results/harmonized/ and prints
a Markdown preview. Every table is built from the rows themselves, so a number that is
not in the results cannot appear in a table.

Interval columns are Wilson intervals for proportions and a percentile bootstrap over
datasets for means, resampling whole datasets because that is the unit of analysis in
Track A.

Usage
    python harmonize.py                     # preview + write CSVs
    python harmonize.py --min-datasets 200  # refuse to emit below a coverage floor
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
RAW = BENCH / "results" / "raw"
OUT = BENCH / "results" / "harmonized"
ARTIFACT_ROOT = BENCH


def configure_root(root: Path | None) -> Path:
    """Route every read and write through one artifact root.

    Without this the harmonizer could only read the definitive tree, which
    readiness requires to be empty before the first definitive run -- so the
    reporting step could not be exercised at all until it was too late to fix.
    """
    global ARTIFACT_ROOT, RAW, OUT, BENCH_FAILURES
    ARTIFACT_ROOT = (root or BENCH).resolve()
    RAW = ARTIFACT_ROOT / "results" / "raw"
    OUT = ARTIFACT_ROOT / "results" / "harmonized"
    BENCH_FAILURES = ARTIFACT_ROOT / "failures"
    return ARTIFACT_ROOT


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_ci(values: list[float], n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = [float(np.mean(rng.choice(arr, size=arr.size, replace=True))) for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def mcse(k: int, n: int) -> float:
    """Monte-Carlo standard error of a simulated proportion, as SAP section 5 requires."""
    if n == 0:
        return float("nan")
    p = k / n
    return math.sqrt(p * (1 - p) / n)


def load(profile: str) -> list[dict]:
    rows = []
    for p in sorted(RAW.glob(f"{profile}*_rows.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_failures() -> list[dict]:
    """Retained failures belong in every denominator, so they are read alongside rows.

    ``benchmark_protocol.yaml#statistical_analysis.analysis_population`` is
    ``all_eligible_units_including_documented_failures``. A table built only from
    successes would condition on success and reward a method that crashes on the
    hard cells.
    """
    failures = []
    directory = ARTIFACT_ROOT / "failures"
    if not directory.is_dir():
        return failures
    for path in sorted(directory.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            failures.append(value)
    return failures


def family_of(row: dict) -> str | None:
    unit = ((row.get("simulation") or {}).get("unit_key")) or {}
    return unit.get("family") if isinstance(unit, dict) else None


def role_pairs(row: dict) -> list[tuple[str, str]]:
    """The (ground truth, called) role pair for every modality in one audit row.

    An unmapped verdict stays in the list as an explicit error class. Dropping it
    would remove a wrong answer from the denominator.
    """
    truth = (row.get("simulation") or {}).get("ground_truth_roles") or {}
    called = (row.get("audit") or {}).get("modality_verdicts") or {}
    return [(str(role), str(called.get(modality) or "__unmapped__"))
            for modality, role in truth.items()]


def pooled_role_macro_f1(rows: list[dict]) -> float:
    """The registered ROLE_MACRO_F1: pooled across replicates within a cell.

    ``OUTCOME_AND_METRIC_REGISTRY.yaml#metrics.ROLE_MACRO_F1`` specifies pooling
    modality predictions across independent replicates, one F1 per ground-truth
    role represented in the cell, unweighted mean. A mean of per-dataset macro-F1
    values is a different estimator: with three modalities per dataset the
    per-dataset statistic takes only a handful of discrete values, and averaging
    them weights each replicate's roles equally regardless of how often each role
    occurs in the cell.
    """
    from sklearn.metrics import f1_score

    pairs = [pair for row in rows for pair in role_pairs(row)]
    if not pairs:
        return float("nan")
    truth = [t for t, _ in pairs]
    called = [c for _, c in pairs]
    labels = sorted(set(truth))
    return float(f1_score(truth, called, labels=labels, average="macro", zero_division=0))


def cluster_boot_ci(rows: list[dict], n_boot: int = 2000,
                    seed: int = 42) -> tuple[float, float]:
    """Cluster bootstrap over independent dataset replicates, per the registry.

    The uncertainty unit is the replicate, so whole datasets are resampled and the
    pooled statistic is recomputed on each draw -- not the per-dataset values
    averaged, which would treat modalities within a dataset as independent.
    """
    if len(rows) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sample = [rows[i] for i in rng.integers(0, len(rows), size=len(rows))]
        value = pooled_role_macro_f1(sample)
        if not math.isnan(value):
            draws.append(value)
    if len(draws) < 2:
        return (float("nan"), float("nan"))
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def dataset_cells() -> dict[str, tuple[str | None, str | None, int | None]]:
    """Map each dataset id to its (family, scenario, sample size) from the task index.

    A failure record names its dataset but not its cell. The task index is the
    only artifact that carries both, so it is what lets a retained failure be
    counted in the denominator of the condition it belongs to.
    """
    index = ARTIFACT_ROOT / "task_index.json"
    if not index.is_file():
        return {}
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    tasks = payload if isinstance(payload, list) else payload.get("tasks", [])
    cells: dict[str, tuple[str | None, str | None, int | None]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        unit = task.get("unit_key") or {}
        cells[str(task.get("dataset_id"))] = (unit.get("family"),
                                              unit.get("scenario_or_structure"),
                                              unit.get("sample_size"))
    return cells


def registered_datasets() -> int | None:
    """Count the datasets the task index requires, rather than storing a total.

    ``benchmark_protocol.yaml#machine_summary`` sets ``aggregate_counts_hardcoded:
    false`` and ``SIMULATION_DESIGN.yaml`` forbids hardcoded aggregate totals, so
    the completeness denominator is read from the execution contract on every run.
    """
    index = ARTIFACT_ROOT / "task_index.json"
    if not index.is_file():
        return None
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    tasks = payload if isinstance(payload, list) else payload.get("tasks", [])
    return len({task.get("dataset_id") for task in tasks if isinstance(task, dict)}) or None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-datasets", type=int, default=0)
    ap.add_argument("--profile", default="core", choices=["core", "full"])
    ap.add_argument("--root", type=Path, default=None,
                    help="artifact root; defaults to the definitive benchmarks/ tree")
    ap.add_argument("--no-write", action="store_true",
                    help="print the preview without writing harmonized files")
    args = ap.parse_args()
    configure_root(args.root)

    rows = load(args.profile)
    failures = load_failures()
    audit = [r for r in rows if r["method_id"] == "omicau_audit"]
    pred = [r for r in rows if r["method_id"] != "omicau_audit"]
    n_datasets = len({r["dataset_id"] for r in rows})
    if n_datasets < args.min_datasets:
        print(f"refusing: {n_datasets} datasets below the floor of {args.min_datasets}")
        return 1

    if not args.no_write:
        OUT.mkdir(parents=True, exist_ok=True)
    registered = registered_datasets()
    if registered is None:
        status = f"UNVERIFIED COVERAGE — {n_datasets} datasets, no task index found"
    elif n_datasets >= registered:
        status = f"COMPLETE — {n_datasets}/{registered} datasets"
    else:
        status = f"INTERIM — {n_datasets}/{registered} datasets"
    print(f"# Track A results ({status})\n")

    # ---- Table 4a: audit validation, by family, condition and sample size --
    # The family is part of the key: role_recovery and predictive_performance
    # units both carry overlay None, so keying on the scenario alone silently
    # merges two registered cells with different replicate counts.
    by = defaultdict(list)
    rows_by_cell = defaultdict(list)
    for r in audit:
        sim = r.get("simulation") or {}
        note = json.loads(r.get("notes") or "{}")
        metrics = r.get("secondary_metrics") or {}
        cell = (family_of(r), sim.get("overlay") or sim.get("scenario"),
                sim.get("sample_size"))
        rows_by_cell[cell].append(r)
        by[cell].append({
            # The registered role-recovery estimand is the macro-averaged role
            # classification (benchmark_protocol.yaml#scientific_scope.primary_claims),
            # so macro-F1 leads the table and accuracy is reported beside it.
            "role_f1": metrics.get("role_macro_f1"),
            "role_acc": metrics.get("role_accuracy"),
            "alarm": bool((r.get("audit") or {}).get("leakage_alarm")),
            "leak": bool(note.get("leakage_present")),
            "false_cert": bool(note.get("false_certification")),
            "false_alarm": bool(note.get("false_alarm")),
            "ctrl": [v for v in (r.get("control_baselines") or {}).values()
                     if isinstance(v, (int, float))],
        })

    audit_failures = sum(1 for f in failures if f.get("method_id") == "omicau_audit")
    print("## Table 4 — audit validation\n")
    print(f"Audit attempts that produced no row (retained failures): {audit_failures}\n")
    print("| family | condition | n | datasets | pooled role macro-F1 [95% CI] | "
          "mean per-dataset F1 | role acc | alarm rate [95% CI] (MCSE) | false cert. | "
          "false alarm | mean control |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    lines = []
    for (fam, cond, n), v in sorted(by.items(),
                                    key=lambda kv: (str(kv[0][0]), str(kv[0][1]), kv[0][2] or 0)):
        cell_rows = rows_by_cell[(fam, cond, n)]
        f1s = [x["role_f1"] for x in v if isinstance(x["role_f1"], (int, float))]
        accs = [x["role_acc"] for x in v if isinstance(x["role_acc"], (int, float))]
        f1 = pooled_role_macro_f1(cell_rows)
        per_dataset_f1 = st.fmean(f1s) if f1s else float("nan")
        acc = st.fmean(accs) if accs else float("nan")
        lo, hi = cluster_boot_ci(cell_rows)
        k_alarm = sum(x["alarm"] for x in v)
        leak_any = any(x["leak"] for x in v)
        fc = sum(x["false_cert"] for x in v)
        fa = sum(x["false_alarm"] for x in v)
        ctrl = [c for x in v for c in x["ctrl"]]
        a_lo, a_hi = wilson(k_alarm, len(v))
        cert_denominator = len(v) if leak_any else 0
        alarm_denominator = 0 if leak_any else len(v)
        row = (f"| {fam} | {cond} | {n} | {len(v)} | {f1:.3f} [{lo:.3f}, {hi:.3f}] | "
               f"{per_dataset_f1:.3f} | {acc:.3f} | "
               f"{k_alarm}/{len(v)} [{a_lo:.3f}, {a_hi:.3f}] "
               f"(±{mcse(k_alarm, len(v)):.3f}) | {fc}/{cert_denominator} | "
               f"{fa}/{alarm_denominator} | "
               f"{st.fmean(ctrl):.3f} |" if ctrl else "")
        if row:
            print(row)
            lines.append({"family": fam, "condition": cond, "n": n, "datasets": len(v),
                          "role_macro_f1_pooled": f1, "role_macro_f1_lo": lo,
                          "role_macro_f1_hi": hi,
                          "role_macro_f1_mean_per_dataset": per_dataset_f1,
                          "role_macro_f1_n": len(f1s),
                          "role_accuracy": acc, "role_accuracy_n": len(accs),
                          "audit_rows": len(v), "alarm": k_alarm,
                          "alarm_rate_lo": a_lo, "alarm_rate_hi": a_hi,
                          "alarm_rate_mcse": mcse(k_alarm, len(v)),
                          "false_certifications": fc,
                          "false_certification_denominator": cert_denominator,
                          "false_alarms": fa, "false_alarm_denominator": alarm_denominator,
                          "mean_control": st.fmean(ctrl)})

    # ---- Table 3: predictive performance, within the registered family -----
    # Pooling every non-audit row would mix the deliberately leaky unsafe arm and
    # the chance-level null controls into one per-method average.  The registered
    # estimand is the paired difference within scenario x sample size.
    # `statistical_analysis.missing_scores_from_failures` requires failure counts
    # and denominators by method and condition, so failures are joined to their
    # cell through the task index rather than reported as one total.
    cell_of = dataset_cells()
    attempted: defaultdict[tuple, int] = defaultdict(int)
    for f in failures:
        if f.get("method_id") == "omicau_audit" or f.get("splitter_arm") not in (None, "standard"):
            continue
        cell = cell_of.get(str(f.get("dataset_id")))
        if cell and cell[0] == "predictive_performance":
            attempted[(f.get("method_id"), cell[1], cell[2])] += 1
    pm = defaultdict(list)
    for r in pred:
        if family_of(r) != "predictive_performance" or r.get("splitter_arm") != "standard":
            continue
        sim = r.get("simulation") or {}
        v = r.get("primary_value")
        if isinstance(v, (int, float)) and sim.get("sample_size"):
            pm[(r["method_id"], sim.get("scenario"), sim["sample_size"])].append(v)
    print("\n## Table 3 — predictive performance, predictive_performance family, "
          "standard arm (AUROC)\n")
    print("Each cell is `mean (scored/attempted)`. `attempted` counts the scored fits "
          "plus the retained failures for that method and condition, so a method that "
          "fails cannot improve its own average by shrinking its denominator.\n")
    cells_by = sorted({(scenario, n) for _, scenario, n in pm}
                      | {(scenario, n) for _, scenario, n in attempted})
    print("| method | " + " | ".join(f"{s} n={n}" for s, n in cells_by) + " |")
    print("|---" * (len(cells_by) + 1) + "|")
    for m in sorted({m for m, _, _ in pm} | {m for m, _, _ in attempted}):
        cells = []
        for scenario, n in cells_by:
            v = pm.get((m, scenario, n), [])
            failed = attempted.get((m, scenario, n), 0)
            total = len(v) + failed
            if not total:
                cells.append("—")
            elif not v:
                cells.append(f"no scored fit (0/{total})")
            else:
                cells.append(f"{st.fmean(v):.3f} ({len(v)}/{total})")
        print(f"| {m} | " + " | ".join(cells) + " |")
    total_failed = sum(attempted.values())
    if total_failed:
        print(f"\nRetained predictive failures in this family: {total_failed}. "
              "They are counted in every denominator above and are never imputed.")

    if not args.no_write:
        (OUT / "table4_audit_validation.json").write_text(json.dumps(lines, indent=1),
                                                          encoding="utf-8")
        print(f"\nwrote {OUT / 'table4_audit_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
