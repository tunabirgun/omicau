"""Generate and audit definitive simulation seeds without running simulations.

The protocol is the sole source for the simulation cells.  Each stream receives a
unique uint32 value derived from a canonical UTF-8 SHA-256 preimage.  In the rare
event that the first four digest bytes collide with an earlier stream, the
zero-based counter is incremented and the complete preimage is rehashed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

RECORD = Path(__file__).resolve().parents[1]
REPO = RECORD.parent
sys.path.insert(0, str(REPO))
from benchmarks.simulations.generate import (  # noqa: E402
    STREAM_LABELS,
    UnitKey,
    protocol_units,
    registry_unit_key,
    sha256_uint32,
)

PROTOCOL_PATH = RECORD / "benchmark_protocol.yaml"
REGISTRY_PATH = RECORD / "checksums" / "definitive_seed_registry.json"
AUDIT_PATH = RECORD / "checksums" / "seed_overlap_audit.json"
CHECKSUM_PATH = RECORD / "checksums" / "protocol_checksums.sha256"


def load_protocol() -> dict[str, Any]:
    value = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("benchmark_protocol.yaml must contain an object")
    return value


def expand_units(protocol: dict[str, Any]) -> list[UnitKey]:
    """Expand every independent generation unit, without hard-coded totals."""
    return protocol_units(protocol)


def build_registry(protocol: dict[str, Any]) -> dict[str, Any]:
    seed_spec = protocol["simulation"]["seed_generation"]
    namespace = str(seed_spec["namespace"])
    master_seed = int(seed_spec["master_seed"])
    used: set[int] = set()
    streams: list[dict[str, Any]] = []
    collision_count = 0
    for unit in expand_units(protocol):
        if registry_unit_key(unit) != unit:
            raise ValueError("registry expansion must emit canonical paired unit keys")
        for stream_label in STREAM_LABELS:
            counter = 0
            while True:
                seed, digest = sha256_uint32(namespace, master_seed, unit, stream_label, counter)
                if seed not in used:
                    break
                counter += 1
            used.add(seed)
            collision_count += counter
            streams.append({
                **unit.as_dict(),
                "stream_label": stream_label,
                "collision_counter": counter,
                "uint32_seed": seed,
                "sha256": digest,
            })
    return {
        "registry_format": "omicau-definitive-seed-registry-v1",
        "protocol_version": str(protocol["record"]["version"]),
        "derivation": {
            "namespace": namespace,
            "master_entropy_uint32": master_seed,
            "algorithm": str(seed_spec["algorithm"]),
            "encoding": str(seed_spec["encoding"]),
            "extraction": str(seed_spec["extraction"]),
            "collision_policy": str(seed_spec["collision_policy"]),
            "unit_key_fields": list(seed_spec["unit_key_fields"]),
            "stream_labels": list(STREAM_LABELS),
        },
        "generation_unit_count": len(expand_units(protocol)),
        "stream_seed_count": len(streams),
        "collision_rehash_count": collision_count,
        "streams": streams,
    }


def _seed_values(value: Any, path: tuple[str, ...] = ()) -> Iterable[int]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + (str(key).lower(),)
            if key == "streams" and isinstance(child, dict):
                for stream_seed in child.values():
                    if isinstance(stream_seed, int) and not isinstance(stream_seed, bool):
                        yield stream_seed
                continue
            if "seed" in str(key).lower() and isinstance(child, int) and not isinstance(child, bool):
                yield child
            else:
                yield from _seed_values(child, child_path)
    elif isinstance(value, list):
        for child in value:
            yield from _seed_values(child, path)


def pilot_seed_inventory() -> tuple[set[int], list[dict[str, Any]]]:
    inventory: set[int] = set()
    sources: list[dict[str, Any]] = []
    for path in sorted(REPO.rglob("simulation_seeds.json")):
        if RECORD in path.parents or "delivery" in path.parts:
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        seeds = sorted(set(_seed_values(value)))
        inventory.update(seeds)
        sources.append({
            "path": path.relative_to(REPO).as_posix(),
            "seed_count": len(seeds),
            "seed_values_sha256": hashlib.sha256(
                json.dumps(seeds, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        })
    return inventory, sources


def build_audit(registry: dict[str, Any]) -> dict[str, Any]:
    definitive = sorted(record["uint32_seed"] for record in registry["streams"])
    pilot, sources = pilot_seed_inventory()
    overlap = sorted(set(definitive).intersection(pilot))
    return {
        "audit_format": "omicau-definitive-pilot-seed-overlap-v1",
        "status": "pass" if pilot and not overlap else "fail",
        "definitive_seed_count": len(definitive),
        "definitive_seed_values_sha256": hashlib.sha256(
            json.dumps(definitive, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "pilot_seed_count": len(pilot),
        "pilot_seed_values_sha256": hashlib.sha256(
            json.dumps(sorted(pilot), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "pilot_sources": sources,
        "overlap_count": len(overlap),
        "overlap": overlap,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_checksum_lines() -> str:
    files = sorted(
        path for path in RECORD.rglob("*")
        if path.is_file() and "checksums" not in path.relative_to(RECORD).parts
        and "tools" not in path.relative_to(RECORD).parts and "upload" not in path.relative_to(RECORD).parts
    )
    return "".join(
        f"{sha256_file(path)}  {path.relative_to(REPO).as_posix()}\n" for path in files
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the three checksum artifacts")
    parser.add_argument("--json", action="store_true", help="print the generated audit")
    args = parser.parse_args(argv)
    protocol = load_protocol()
    registry = build_registry(protocol)
    audit = build_audit(registry)
    if args.write:
        write_json(REGISTRY_PATH, registry)
        write_json(AUDIT_PATH, audit)
        CHECKSUM_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHECKSUM_PATH.write_text(protocol_checksum_lines(), encoding="utf-8")
    report = {
        "generation_unit_count": registry["generation_unit_count"],
        "stream_seed_count": registry["stream_seed_count"],
        "collision_rehash_count": registry["collision_rehash_count"],
        "pilot_seed_count": audit["pilot_seed_count"],
        "overlap_count": audit["overlap_count"],
        "status": audit["status"],
        "wrote": bool(args.write),
    }
    print(json.dumps(report, sort_keys=True) if args.json else
          f"{report['generation_unit_count']} units; {report['stream_seed_count']} stream seeds; "
          f"{report['pilot_seed_count']} pilot seeds; {report['overlap_count']} overlap; {report['status'].upper()}")
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
