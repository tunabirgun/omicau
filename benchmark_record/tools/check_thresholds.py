"""Check the protocol's threshold mirror against omicau's source constants."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

try:
    from record_manifest import load_protocol
except ImportError:  # pragma: no cover - module import use
    from .record_manifest import load_protocol

RECORD = Path(__file__).resolve().parents[1]
SOURCE_MODULE = "omicau.interpretation.utility"
CONSTANTS = (
    "USEFUL_MARGIN",
    "GAIN_EPS",
    "GAIN_STRONG",
    "GAIN_ALPHA",
    "CKA_REDUNDANT",
    "CONTROL_MARGIN",
    "CONTROL_FAMILY_ALPHA",
    "SUBGROUP_MIN_N",
)


def _equal(actual: object, expected: object) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=0.0)
    return actual == expected


def check() -> tuple[list[str], int]:
    protocol = load_protocol(RECORD)
    mirror = protocol.get("audit_thresholds")
    if not isinstance(mirror, dict):
        return ["benchmark_protocol.yaml:audit_thresholds is missing or is not a mapping"], 0

    module_name = str(mirror.get("source_module", SOURCE_MODULE))
    if module_name != SOURCE_MODULE:
        return [f"audit_thresholds.source_module must be {SOURCE_MODULE!r}, got {module_name!r}"], 0

    module = importlib.import_module(module_name)
    errors: list[str] = []
    compared = 0
    for name in CONSTANTS:
        if name not in mirror:
            errors.append(f"audit_thresholds.{name} is missing")
            continue
        if not hasattr(module, name):
            errors.append(f"{module_name}.{name} is missing")
            continue
        compared += 1
        actual = getattr(module, name)
        expected = mirror[name]
        if not _equal(actual, expected):
            errors.append(f"{name}: protocol={expected!r}, source={actual!r}")
    return errors, compared


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        errors, compared = check()
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Compared {compared} mirrored constants with {SOURCE_MODULE}.")
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("OK: protocol thresholds match the source constants.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
