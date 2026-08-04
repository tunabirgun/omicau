"""Validate and freeze the independent v1.0.0 prospective benchmark record.

``--check`` is read-only. ``--freeze`` runs the same gates and writes
``benchmark_record/RECORD_FREEZE.json`` only when every gate passes.  A pending
Zenodo DOI is valid for the pre-upload package and is recorded explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from record_manifest import (
        FREEZE_NAME,
        VERSION,
        collect_package_files,
        combined_manifest_sha256,
        doi_status,
        load_protocol,
        package_bytes,
        protocol_version,
        reference_errors,
    )
except ImportError:  # pragma: no cover - module import use
    from .record_manifest import (
        FREEZE_NAME,
        VERSION,
        collect_package_files,
        combined_manifest_sha256,
        doi_status,
        load_protocol,
        package_bytes,
        protocol_version,
        reference_errors,
    )

RECORD = Path(__file__).resolve().parents[1]
REPO = RECORD.parent
BENCHMARKS = REPO / "benchmarks"
FREEZE_FILE = RECORD / FREEZE_NAME
NOT_DRAFTED_RE = re.compile(r"^\s*(?:#|//|--|\*)?\s*STATUS:\s*NOT DRAFTED\s*$", re.IGNORECASE)
SEED_TOKEN_RE = re.compile(r"seed", re.IGNORECASE)

# Source/config scaffolding is deliberately absent.  Every path here is an
# execution output path and must contain no files at a prospective freeze.
ACTIVE_OUTPUT_DIRS = (
    "comparators",
    "datasets/derived_nonrestricted",
    "deviations",
    "failures",
    "figures",
    "logs",
    "manuscript_assets",
    "results",
    "runs",
    "simulations/seeds",
    "splits",
    "tables",
)


def _git(*args: str) -> str | None:
    try:
        process = subprocess.run(
            ["git", "-C", str(REPO), *args],
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return process.stdout.strip()


def _iter_scalars(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_scalars(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_scalars(child, path + (str(index),))
    else:
        yield path, value


def _integer_seed(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        return int(value)
    return None


def definitive_seeds(protocol: dict[str, Any]) -> set[int]:
    """Extract explicitly declared definitive seed values, not counts/offsets."""
    values: set[int] = set()
    ignored_tokens = {"count", "counts", "n", "number", "offset", "offsets", "replicates"}
    for path, value in _iter_scalars(protocol):
        lowered = tuple(part.lower() for part in path)
        if not any(SEED_TOKEN_RE.search(part) for part in lowered):
            continue
        if any(
            part in ignored_tokens
            or part.startswith("n_")
            or "count" in part
            or "overlap" in part
            for part in lowered
        ):
            continue
        if any(part in {"development", "pilot", "excluded", "prior"} for part in lowered):
            continue
        seed = _integer_seed(value)
        if seed is not None:
            values.add(seed)
    return values


def _seeds_from_json(value: Any, path: tuple[str, ...] = ()) -> set[int]:
    seeds: set[int] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + (str(key).lower(),)
            if key == "streams" and isinstance(child, dict):
                seeds.update(seed for seed in (_integer_seed(v) for v in child.values()) if seed is not None)
            else:
                seeds.update(_seeds_from_json(child, child_path))
    elif isinstance(value, list):
        for child in value:
            seeds.update(_seeds_from_json(child, path))
    elif path and SEED_TOKEN_RE.search(path[-1]):
        seed = _integer_seed(value)
        if seed is not None:
            seeds.add(seed)
    return seeds


def pilot_seed_sources() -> tuple[set[int], list[str], list[str]]:
    """Read all retained pilot seed registries without relying on one old layout."""
    seeds: set[int] = set()
    sources: list[str] = []
    errors: list[str] = []
    candidates = sorted(
        path for path in REPO.rglob("simulation_seeds.json")
        if RECORD not in path.parents and "delivery" not in path.parts
    )
    for path in candidates:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read pilot seed source {path.relative_to(REPO).as_posix()}: {exc}")
            continue
        extracted = _seeds_from_json(value)
        seeds.update(extracted)
        sources.append(path.relative_to(REPO).as_posix())
    return seeds, sources, errors


def nonempty_output_dirs() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for relative in ACTIVE_OUTPUT_DIRS:
        root = BENCHMARKS / relative
        if not root.exists():
            continue
        files = sorted(path.relative_to(REPO).as_posix() for path in root.rglob("*") if path.is_file())
        if files:
            found[relative] = files
    return found


def not_drafted_files(files: Iterable[Path]) -> list[str]:
    hits: list[str] = []
    for path in files:
        if path.suffix.lower() not in {".cff", ".json", ".md", ".txt", ".yaml", ".yml"}:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            hits.append(path.relative_to(RECORD).as_posix() + " (not readable as UTF-8)")
            continue
        if any(NOT_DRAFTED_RE.fullmatch(line) for line in lines):
            hits.append(path.relative_to(RECORD).as_posix())
    return hits


def threshold_check() -> tuple[int, str]:
    script = RECORD / "tools" / "check_thresholds.py"
    process = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    return process.returncode, (process.stdout + process.stderr).strip()


def run_gates(*, allow_dirty: bool) -> tuple[bool, dict[str, Any], dict[str, bytes]]:
    protocol = load_protocol(RECORD)
    files = collect_package_files(RECORD, require_freeze=False)
    entries = package_bytes(files, RECORD)
    # A re-freeze must hash only reader-facing source files, never the previous
    # freeze record itself.
    entries.pop(FREEZE_NAME, None)
    errors = reference_errors(protocol, entries)

    undrafted = not_drafted_files(files)
    if undrafted:
        errors.append("undrafted markers remain in: " + ", ".join(undrafted))

    outputs = nonempty_output_dirs()
    if outputs:
        errors.append("active benchmark output directories contain files: " + ", ".join(outputs))

    definitive = definitive_seeds(protocol)
    pilot, seed_sources, seed_errors = pilot_seed_sources()
    errors.extend(seed_errors)
    overlap = sorted(definitive & pilot)
    if not definitive:
        errors.append("no explicit definitive seed values were found in benchmark_protocol.yaml")
    if not pilot:
        errors.append("no retained pilot seed values were found; zero-overlap cannot be audited")
    if overlap:
        errors.append("definitive seeds overlap pilot seeds: " + ", ".join(map(str, overlap)))

    threshold_rc, threshold_output = threshold_check()
    if threshold_rc != 0:
        errors.append("threshold mirror check failed")

    dirty_text = _git("status", "--porcelain")
    dirty_changes = dirty_text.splitlines() if dirty_text else []
    if dirty_changes and not allow_dirty:
        errors.append("working tree is dirty (use --allow-dirty only for a declared pre-freeze review)")

    status = doi_status(protocol)
    if status["status"] == "invalid":
        errors.append(f"reserved DOI is invalid: {status['value']!r}")

    report: dict[str, Any] = {
        "protocol_version": protocol_version(protocol),
        "doi": status,
        "active_output_directories": {
            name: {"file_count": len(paths), "examples": paths[:10]}
            for name, paths in sorted(outputs.items())
        },
        "pilot_seed_audit": {
            "status": "pass" if definitive and pilot and not overlap and not seed_errors else "fail",
            "definitive_seed_count": len(definitive),
            "pilot_seed_count": len(pilot),
            "overlap_count": len(overlap),
            "overlap": overlap,
            "pilot_source_count": len(seed_sources),
        },
        "threshold_check": {"exit_code": threshold_rc, "output": threshold_output},
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(dirty_changes),
            "allow_dirty_used": allow_dirty,
        },
        "errors": errors,
    }
    return not errors, report, entries


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="run all gates and write nothing")
    mode.add_argument("--freeze", action="store_true", help="run all gates and write the freeze record")
    parser.add_argument("--allow-dirty", action="store_true", help="record and permit a dirty working tree")
    args = parser.parse_args()

    try:
        ok, report, entries = run_gates(allow_dirty=args.allow_dirty)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"record: {RECORD}")
    print(f"protocol version: {report['protocol_version']}")
    print(f"Zenodo DOI: {report['doi']['status']} ({report['doi']['value']})")
    seed = report["pilot_seed_audit"]
    print(
        "seed audit: "
        f"{seed['status']} ({seed['definitive_seed_count']} definitive; "
        f"{seed['pilot_seed_count']} pilot; {seed['overlap_count']} overlap)"
    )
    if report["active_output_directories"]:
        print("active output directories with files:")
        for name, details in report["active_output_directories"].items():
            print(f"  {name}: {details['file_count']}")
    for error in report["errors"]:
        print(f"FAIL: {error}")
    print(f"gates: {'PASS' if ok else 'FAIL'}")

    if args.check:
        return 0 if ok else 1
    if not ok:
        print("Refusing to write the freeze record.")
        return 1

    freeze = {
        "record_format": "omicau-benchmark-freeze-v1",
        "protocol_version": VERSION,
        "frozen_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "doi": report["doi"],
        "freeze_kind": "prospective_pre_upload",
        "source_commit": report["git"]["commit"],
        "git": report["git"],
        "declarations": {
            "prior_runs_are_pilot_only": True,
            "prior_pilot_outputs_excluded": True,
            "active_output_directories_empty": True,
            "definitive_pilot_seed_overlap": 0,
            "threshold_mirror_verified": True,
        },
        "pilot_seed_audit": report["pilot_seed_audit"],
        "record_content_sha256": combined_manifest_sha256(entries),
        "source_files": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(entries.items())
        },
    }
    FREEZE_FILE.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote: {FREEZE_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
