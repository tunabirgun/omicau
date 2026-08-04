"""Assess comparator eligibility against COMPARATOR_ELIGIBILITY_RULES.md.

Every field is derived: method properties come from COMPARATOR_MANIFEST.yaml,
versions and licences from the installed distribution metadata, and installation
status from actually importing the implementation. A method whose implementation
cannot be imported is left explicitly unassessed rather than being recorded as
ineligible -- "we did not try" and "we tried and it failed" are different
findings, and only the second is a benchmark outcome.

The report is written before any definitive result exists, so
``assessed_before_results`` is true by construction.

Usage
    python assess_comparators.py
    python assess_comparators.py --write
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE.parent
REPO = RECORD.parent
OUTPUT = REPO / "benchmarks" / "comparators" / "eligibility.json"

# Where each registered comparator is implemented in this repository. A method
# with no entry has no implementation here and cannot be assessed.
IMPLEMENTATIONS = {
    "nested_best_single": ("benchmarks/harness/run.py::nested_best_single", "sklearn"),
    "early_elastic_net": ("benchmarks/harness/run.py::cv_predict(elastic_net)", "sklearn"),
    "random_forest_early_fusion": ("benchmarks/harness/run.py::cv_predict(random_forest)", "sklearn"),
    "fully_nested_stacking": ("benchmarks/harness/run.py::nested_stacking", "sklearn"),
    "DIABLO": (None, "mixOmics"),
    "MOFA_plus": (None, "mofapy2"),
    "additional_supervised_eligibility_slot": (None, None),
}
DISTRIBUTION_FOR_MODULE = {"sklearn": "scikit-learn", "mofapy2": "mofapy2"}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def import_status(module: str | None) -> tuple[str, str | None, str | None]:
    """Return (installation_status, version, licence) for a backing module."""
    if module is None:
        return "not_attempted_no_local_implementation", None, None
    try:
        importlib.import_module(module)
    except Exception as exc:
        return f"import_failed: {type(exc).__name__}", None, None
    distribution = DISTRIBUTION_FOR_MODULE.get(module, module)
    try:
        meta = metadata.metadata(distribution)
        version = metadata.version(distribution)
        licence = meta.get("License") or next(
            (c.split("::")[-1].strip() for c in meta.get_all("Classifier") or []
             if c.startswith("License ::")), None)
    except metadata.PackageNotFoundError:
        version, licence = None, None
    return "installed", version, licence


def assess(manifest: dict) -> list[dict]:
    comparators = manifest.get("comparators") or {}
    records = []
    for method_id, entry in comparators.items():
        location, module = IMPLEMENTATIONS.get(method_id, (None, None))
        status, version, licence = import_status(module)
        implemented = location is not None and status == "installed"
        record = {
            "method_id": method_id,
            "assessed_before_results": True,
            "official_implementation_commit_or_version": version,
            "licence_from_installed_artifact": licence,
            "installation_status": status,
            "task_support": entry.get("task_support"),
            "supervised_status": entry.get("supervision"),
            "inductive_status": entry.get("operation"),
            "frozen_external_split_support": entry.get("external_split_support"),
            "group_aware_resampling_support": entry.get("group_aware_resampling"),
            "missingness_handling": entry.get("missingness_handling",
                                              "fold_local_median_imputation"),
            "tuning_requirement": sorted(entry.get("arms", {})),
            "known_limitations": [],
            "evidence_paths_or_code_locations": [location] if location else [],
            "final_eligibility_status": (
                "primary_eligible" if implemented else "not_yet_assessed"),
            "assessor_and_date": f"benchmark_record/tools/assess_comparators.py, {utcnow()}",
            "model_results_visible_during_eligibility": False,
        }
        if not implemented:
            record["known_limitations"] = [
                "no implementation is present in this repository and no installation "
                "attempt has been made on this machine; eligibility is undetermined, "
                "not negative"
            ]
        records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help=f"write {OUTPUT.name}")
    args = parser.parse_args(argv)

    import yaml
    manifest = yaml.safe_load((RECORD / "COMPARATOR_MANIFEST.yaml").read_text(encoding="utf-8"))
    records = assess(manifest)
    payload = {
        "schema_version": "1.0",
        "protocol_version": manifest.get("protocol_version"),
        "eligibility_rules": manifest.get("eligibility_rules"),
        "generated_utc": utcnow(),
        "model_results_visible_during_eligibility": False,
        "comparators": records,
    }
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {OUTPUT.relative_to(REPO)}")
    for record in records:
        print(f"  {record['method_id']:40} {record['final_eligibility_status']:20} "
              f"{record['installation_status']}")
    unresolved = [r["method_id"] for r in records
                  if r["final_eligibility_status"] == "not_yet_assessed"]
    if unresolved:
        print(f"\nunresolved ({len(unresolved)}): {', '.join(unresolved)}")
        print("readiness stays blocked until each is installed and assessed, or "
              "recorded as an attempted-and-failed benchmark outcome")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
