"""Capture the omicau execution environment for the definitive benchmark.

The package set is derived from the project's own declared dependencies plus the
packages the harness imports directly, so the capture cannot drift from what the
benchmark actually runs. Nothing machine-private is recorded: no hostname, user,
absolute path, credential or data location, per ``environment/README.md``.

The lock digest is a SHA-256 over the canonical JSON of the resolved package
versions and the interpreter version. It changes when, and only when, something
that can alter a numerical result changes.

Usage
    python capture_environment.py            # print the capture
    python capture_environment.py --write    # write environment/omicau-environment.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE.parent
REPO = RECORD.parent
OUTPUT = RECORD / "environment" / "omicau-environment.yaml"

# Imported directly by benchmarks/harness and benchmarks/simulations, so they
# affect definitive numbers even though the package does not declare them all.
HARNESS_IMPORTS = ("pyyaml", "psutil", "pytest")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "TOKENIZERS_PARALLELISM", "CUBLAS_WORKSPACE_CONFIG")


def declared_requirements() -> list[str]:
    """Read the distribution names omicau declares, without a hardcoded list."""
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    try:
        import tomllib
        project = tomllib.loads(text).get("project", {})
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        import tomli
        project = tomli.loads(text).get("project", {})
    specifiers = list(project.get("dependencies", []))
    for extra in (project.get("optional-dependencies") or {}).values():
        specifiers.extend(extra)
    names = set()
    for specifier in specifiers:
        name = re.split(r"[<>=!~\[;\s]", str(specifier).strip(), maxsplit=1)[0]
        if name:
            names.add(name.lower().replace("_", "-"))
    return sorted(names | set(HARNESS_IMPORTS))


def package_versions(names: list[str]) -> dict[str, str]:
    """Installed version per requested distribution; absent packages are recorded."""
    resolved: dict[str, str] = {}
    for name in names:
        try:
            resolved[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            resolved[name] = "not_installed"
    return resolved


def accelerator_state() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"torch_installed": False}
    state: dict[str, object] = {
        "torch_installed": True,
        # torch.__version__ is a str subclass that the YAML representer rejects.
        "torch_version": str(torch.__version__),
        "cuda_compiled": bool(getattr(torch.version, "cuda", None)),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_used_by_protocol": "cpu",
    }
    if torch.cuda.is_available():
        # The device name is hardware, not machine identity, and it is required
        # by the protocol's timing and determinism reporting.
        state["cuda_device_names"] = [torch.cuda.get_device_name(i)
                                      for i in range(torch.cuda.device_count())]
    return state


def cpu_topology() -> dict[str, object]:
    topology: dict[str, object] = {"logical_cores": os.cpu_count()}
    try:
        import psutil
        topology["physical_cores"] = psutil.cpu_count(logical=False)
        memory = psutil.virtual_memory()
        topology["total_memory_gib"] = round(memory.total / (1024 ** 3), 2)
    except ImportError:
        topology["physical_cores"] = None
        topology["total_memory_gib"] = None
    return topology


def capture() -> dict[str, object]:
    names = declared_requirements()
    versions = package_versions(names)
    interpreter = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
    }
    lock_payload = json.dumps({"interpreter": interpreter, "packages": versions},
                              sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": "1.0",
        "protocol_version": "1.0.0",
        "role": "system_under_test",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": cpu_topology(),
        "interpreter": interpreter,
        "packages": versions,
        "accelerator": accelerator_state(),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "environment_lock_sha256": hashlib.sha256(lock_payload).hexdigest(),
        "excluded_by_policy": ["hostname", "user", "absolute_paths", "credentials",
                               "data_locations", "subject_identifiers"],
    }


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help=f"write {OUTPUT.name}")
    args = parser.parse_args(argv)
    payload = capture()
    import yaml
    text = yaml.safe_dump(payload, sort_keys=True, allow_unicode=False)
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUTPUT.relative_to(REPO)}")
        print(f"environment_lock_sha256: {payload['environment_lock_sha256']}")
    else:
        print(text)
    missing = sorted(name for name, version in payload["packages"].items()
                     if version == "not_installed")
    if missing:
        print(f"not installed: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
