"""Shared allowlist, hashing, and metadata checks for the public record.

The Zenodo archive is intentionally narrower than the working
``benchmark_record`` directory.  Only reader-facing protocol material named in
``PACKAGE_FILES`` can enter the archive; tooling, local staging paths, captured
environments, and incidental files cannot be swept in by a recursive glob.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

VERSION = "1.0.0"
ARCHIVE_STEM = f"omicau-benchmark-v{VERSION}"
MANIFEST_NAME = "RECORD_MANIFEST.sha256"
FREEZE_NAME = "RECORD_FREEZE.json"

# Explicit, reviewed public surface.  Optional entries accommodate documents
# produced by the protocol-writing tracks without weakening the allowlist.
REQUIRED_PACKAGE_FILES = (
    "README.md",
    "BENCHMARK_PROTOCOL.md",
    "benchmark_protocol.yaml",
    "LICENSE",
    FREEZE_NAME,
)

OPTIONAL_PACKAGE_FILES = (
    "CITATION.cff",
    "CLAIM_REGISTRY.md",
    "COMPARATOR_ELIGIBILITY_RULES.md",
    "COMPARATOR_MANIFEST.yaml",
    "COMPUTE_PLAN.md",
    "DATASET_FEASIBILITY_REPORT.md",
    "DATASET_MANIFEST.yaml",
    "DEVIATION_POLICY.md",
    "DEVIATIONS.md",
    "EXPECTED_OUTPUTS.md",
    "EXCLUSION_CRITERIA.md",
    "FAILURE_REPORTING_POLICY.md",
    "HYPOTHESES.md",
    "NO_RESULTS_AT_FREEZE.md",
    "OUTCOME_AND_METRIC_REGISTRY.yaml",
    "PILOT_DISCLOSURE.md",
    "PREPROCESSING_RULES.md",
    "PUBLICATION.md",
    "REFERENCES.md",
    "RESAMPLING_AND_SPLIT_SPECIFICATION.md",
    "SIMULATION_DESIGN.yaml",
    "STATISTICAL_ANALYSIS_PLAN.md",
    "environment/README.md",
    "schemas/benchmark-result.schema.json",
    "schemas/dataset-manifest.schema.json",
    "schemas/deviation.schema.json",
    "schemas/failed-run.schema.json",
)

PACKAGE_FILES = frozenset(REQUIRED_PACKAGE_FILES + OPTIONAL_PACKAGE_FILES)
IGNORED_TOP_LEVEL = frozenset({"checksums", "local", "tools", "upload"})
ZENODO_DOI_RE = re.compile(r"\b10\.5281/zenodo\.\d+\b", re.IGNORECASE)
SEMVER_RE = re.compile(r"(?<![\w.])v?(\d+\.\d+\.\d+)(?![\w.])")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(record: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency error
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc

    path = record / "benchmark_protocol.yaml"
    if not path.is_file():
        raise ValueError(f"required protocol file is missing: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read {path} as UTF-8 YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("benchmark_protocol.yaml must contain a mapping")
    return value


def protocol_version(protocol: dict[str, Any]) -> str | None:
    for block_name in ("record", "protocol"):
        block = protocol.get(block_name)
        if isinstance(block, dict) and block.get("version") is not None:
            return str(block["version"])
    if protocol.get("version") is not None:
        return str(protocol["version"])
    return None


def _registration(protocol: dict[str, Any]) -> dict[str, Any]:
    block = protocol.get("protocol")
    if isinstance(block, dict):
        registration = block.get("registration")
        if isinstance(registration, dict):
            return registration
    registration = protocol.get("registration")
    return registration if isinstance(registration, dict) else {}


def declared_zenodo_dois(protocol: dict[str, Any]) -> tuple[str, ...]:
    found: set[str] = set()
    containers = [_registration(protocol), _mapping(protocol.get("record"))]
    for container in containers:
        for key, value in container.items():
            if "doi" not in str(key).lower() or not isinstance(value, str):
                continue
            found.update(match.lower() for match in ZENODO_DOI_RE.findall(value))
    return tuple(sorted(found))


def doi_status(protocol: dict[str, Any]) -> dict[str, str]:
    registration = _registration(protocol)
    record = _mapping(protocol.get("record"))
    value = registration.get(
        "reserved_doi", registration.get("doi", record.get("doi", "PENDING"))
    )
    text = "PENDING" if value in (None, "") else str(value).strip()
    if text.upper() == "PENDING":
        return {"status": "pending_pre_upload", "value": "PENDING"}
    if not ZENODO_DOI_RE.fullmatch(text):
        return {"status": "invalid", "value": text}
    return {"status": "reserved", "value": text.lower()}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def collect_package_files(record: Path, *, require_freeze: bool) -> list[Path]:
    required = set(REQUIRED_PACKAGE_FILES)
    if not require_freeze:
        required.remove(FREEZE_NAME)
    missing = sorted(relative for relative in required if not (record / relative).is_file())
    if missing:
        raise ValueError("required package files are missing: " + ", ".join(missing))

    return [record / relative for relative in sorted(PACKAGE_FILES)
            if (record / relative).is_file()]


def package_bytes(files: Iterable[Path], record: Path) -> dict[str, bytes]:
    return {path.relative_to(record).as_posix(): path.read_bytes() for path in files}


def manifest_bytes(entries: dict[str, bytes]) -> bytes:
    lines = [f"{sha256_bytes(data)}  {name}" for name, data in sorted(entries.items())]
    return ("\n".join(lines) + "\n").encode("utf-8")


def combined_manifest_sha256(entries: dict[str, bytes]) -> str:
    return sha256_bytes(manifest_bytes(entries))


def reference_errors(protocol: dict[str, Any], entries: dict[str, bytes]) -> list[str]:
    """Reject Zenodo identifiers and record versions not declared by this record."""
    errors: list[str] = []
    version = protocol_version(protocol)
    if version != VERSION:
        errors.append(f"protocol version is {version!r}; expected {VERSION!r}")

    allowed_dois = set(declared_zenodo_dois(protocol))
    seen_dois: dict[str, set[str]] = {}
    for name, data in entries.items():
        if Path(name).suffix.lower() not in {".cff", ".json", ".md", ".txt", ".yaml", ".yml"}:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"{name} is not valid UTF-8")
            continue
        for doi in ZENODO_DOI_RE.findall(text):
            seen_dois.setdefault(doi.lower(), set()).add(name)

        for line_number, line in enumerate(text.splitlines(), start=1):
            lower = line.lower()
            if not any(word in lower for word in ("benchmark", "protocol", "record", "zenodo")):
                continue
            for match in SEMVER_RE.finditer(line):
                found = match.group(1)
                if found != VERSION:
                    errors.append(
                        f"{name}:{line_number} names record/protocol version {found}; "
                        f"only {VERSION} is permitted"
                    )

    for doi, names in sorted(seen_dois.items()):
        if doi not in allowed_dois:
            errors.append(
                f"undeclared or superseded Zenodo DOI {doi} appears in "
                + ", ".join(sorted(names))
            )
    status = doi_status(protocol)
    if status["status"] == "invalid":
        errors.append(f"invalid reserved DOI value: {status['value']!r}")
    if status["status"] == "pending_pre_upload" and seen_dois:
        errors.append("Zenodo DOI is PENDING but a Zenodo DOI appears in package content")
    return errors
