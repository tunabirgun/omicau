"""Print a compact summary derived from ``benchmark_protocol.yaml``.

The script reports only values present in the machine-readable protocol and
derives products such as outer evaluations per real cohort.  It also checks the
few structural commitments that define this independent v1.0.0 design.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from record_manifest import VERSION, load_protocol, protocol_version
except ImportError:  # pragma: no cover - module import use
    from .record_manifest import VERSION, load_protocol, protocol_version

RECORD = Path(__file__).resolve().parents[1]


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first(mapping: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _int(mapping: dict[str, Any], names: Iterable[str]) -> int | None:
    value = _first(mapping, names)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _cv_values(block: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    outer = _mapping(_first(block, ("outer", "outer_cv"))) or block
    inner = _mapping(_first(block, ("inner", "inner_cv")))
    folds = _int(outer, ("n_splits", "folds", "folds_requested", "n_folds", "outer_folds"))
    repeats = _int(outer, ("n_repeats", "repeats", "outer_repeats", "partitions"))
    inner_folds = _int(inner, ("n_splits", "folds", "n_folds", "inner_folds"))
    if inner_folds is None:
        inner_folds = _int(block, ("inner_folds", "inner_n_splits"))
    return folds, repeats, inner_folds


def _count_named(value: Any) -> int | None:
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return None


def _replicate_count(block: dict[str, Any]) -> int | None:
    return _int(
        block,
        (
            "independent_replicates_per_cell",
            "paired_replicates_per_cell",
            "replicates_per_cell",
            "n_replicates",
            "replicates",
        ),
    )


def _family_units(block: dict[str, Any]) -> int | None:
    cells = block.get("cells")
    if isinstance(cells, list):
        total = 0
        for cell in cells:
            if not isinstance(cell, dict):
                return None
            replicates = _replicate_count(cell)
            if replicates is None:
                return None
            sizes = _count_named(cell.get("sample_sizes")) or 1
            total += sizes * replicates
        return total

    replicates = _replicate_count(block)
    if replicates is None:
        return None
    factor = 1
    for key, value in block.items():
        if key in {
            "paired_conditions",
            "sample_sizes",
            "scenarios",
            "cohort_structures",
            "perturbations",
        }:
            count = _count_named(value)
            if count is not None and key != "paired_conditions":
                factor *= count
    return factor * replicates


def _external_subset_units(subset: dict[str, Any]) -> int | None:
    scenarios = _count_named(subset.get("scenarios"))
    sizes = _count_named(subset.get("sample_sizes"))
    indices = _mapping(subset.get("replicate_indices"))
    start = _int(indices, ("start_inclusive", "start"))
    stop = _int(indices, ("stop_exclusive", "stop"))
    if None in (scenarios, sizes, start, stop) or stop < start:
        return None
    return scenarios * sizes * (stop - start)


def summarize(protocol: dict[str, Any]) -> dict[str, Any]:
    resampling = _mapping(protocol.get("resampling"))
    simulation_cv = _mapping(_first(
        resampling, ("simulations", "simulation", "simulated_datasets", "simulation_outer")
    ))
    real_cv = _mapping(_first(
        resampling, ("real_cohorts", "real", "cohorts", "real_outer")
    ))

    simulation = _mapping(_first(protocol, ("simulations", "simulation")))
    real_cohorts = _mapping(protocol.get("real_cohorts"))
    if not simulation_cv:
        simulation_cv = _mapping(simulation.get("outer_cv"))
    if not real_cv:
        real_cv = {
            "outer": _mapping(real_cohorts.get("outer_cv")),
            "inner": _mapping(real_cohorts.get("inner_cv")),
        }

    sim_folds, sim_repeats, _ = _cv_values(simulation_cv)
    real_folds, real_repeats, real_inner = _cv_values(real_cv)

    families = _mapping(simulation.get("families"))
    family_units = {name: _family_units(_mapping(block)) for name, block in families.items()}
    independent_units = (
        sum(value for value in family_units.values() if value is not None)
        if family_units and all(value is not None for value in family_units.values())
        else None
    )

    real_primary = _count_named(real_cohorts.get("primary_candidate_ids"))
    real_secondary = _count_named(real_cohorts.get("secondary_positive_control_ids"))

    external_subset = _mapping(simulation.get("external_comparator_subset"))
    xai = _mapping(_first(protocol, ("xai", "explainability")))
    case_studies = _int(xai, ("permitted_case_studies", "case_studies", "case_study"))

    checks = {
        "protocol_version_is_1.0.0": protocol_version(protocol) == VERSION,
        "simulation_is_one_5_fold_partition": sim_folds == 5 and sim_repeats in (None, 1),
        "real_outer_is_5_fold_x_3": real_folds == 5 and real_repeats == 3,
        "real_inner_is_3_fold": real_inner == 3,
        "external_comparisons_have_prespecified_subset": bool(external_subset),
        "xai_is_one_case_study": xai.get("default") is False and case_studies == 1,
    }
    return {
        "protocol_version": protocol_version(protocol),
        "simulation": {
            "derived_independent_units_by_family": family_units,
            "derived_independent_units_total": independent_units,
            "outer_folds": sim_folds,
            "outer_partitions": sim_repeats,
        },
        "real_cohorts": {
            "primary_count": real_primary,
            "secondary_count": real_secondary,
            "outer_folds": real_folds,
            "outer_repeats": real_repeats,
            "derived_outer_evaluations_per_cohort": (
                real_folds * real_repeats
                if real_folds is not None and real_repeats is not None
                else None
            ),
            "inner_folds": real_inner,
        },
        "external_method_simulation_subset_units": _external_subset_units(external_subset),
        "xai_case_study_count": case_studies,
        "design_checks": checks,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--strict", action="store_true", help="fail when a structural design check is false"
    )
    args = parser.parse_args()
    try:
        summary = summarize(load_protocol(RECORD))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        sim = summary["simulation"]
        real = summary["real_cohorts"]
        print(f"omicau benchmark protocol v{summary['protocol_version']}")
        print(
            "simulations: independent units; "
            f"{sim['outer_folds']}-fold CV; {sim['outer_partitions']} partition(s)"
        )
        print(
            "real cohorts: "
            f"{real['outer_folds']}-fold x {real['outer_repeats']} outer CV; "
            f"{real['inner_folds']}-fold inner CV"
        )
        if sim["derived_independent_units_total"] is not None:
            print(
                "derived independent simulation units: "
                f"{sim['derived_independent_units_total']}"
            )
            for family, count in sim["derived_independent_units_by_family"].items():
                print(f"  {family}: {count}")
        print(
            "real cohorts listed: "
            f"{real['primary_count']} primary; {real['secondary_count']} secondary"
        )
        print(
            "external-method simulation subset entries: "
            f"{summary['external_method_simulation_subset_units']}"
        )
        print(f"XAI case studies: {summary['xai_case_study_count']}")
        for name, passed in summary["design_checks"].items():
            print(f"{'OK' if passed else 'FAIL'}: {name}")

    if args.strict and not all(summary["design_checks"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
