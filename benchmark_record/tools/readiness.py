"""Read-only post-publication execution-readiness checks.

The receipt is the binding record for the published protocol.  In particular,
this module does not infer publication state from the working protocol files:
those files are the source material used to create the archive and may still
carry pre-upload values such as ``PENDING``.  The local delivery archive and
``ZENODO_RECEIPT.json`` are checked as an immutable pair.

The checks are deliberately split into two verdicts:

``protocol_publication``
    The receipt is well formed and the deposited archive can be identified and
    verified from the local delivery set.

``execution_readiness``
    The local, post-publication execution prerequisites are complete and no
    definitive output has been created yet.

This module only reads files.  It never extracts, rewrites, repackages, or
otherwise mutates the published ZIP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - handled as a report blocker
    yaml = None  # type: ignore[assignment]


VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
DOI_RE = re.compile(r"^10\.5281/zenodo\.\d+$", re.IGNORECASE)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
MD5_RE = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)
PENDING_RE = re.compile(r"(?:pending|not[_ -]?yet|unassessed|unknown)", re.IGNORECASE)

TERMINAL_ELIGIBILITY = {
    "eligible",
    "primary_eligible",
    "secondary_eligible",
    "ineligible",
    "ineligible_for_task",
    "implementation_unsuccessful",
    "excluded",
    "dataset_ineligible",
}

# These are execution products, not inputs that must be present before the
# first definitive run.  A missing directory is therefore equivalent to an
# empty directory.
#
# benchmarks/deviations is deliberately absent: DEVIATION_POLICY.md requires a
# deviation record to be written *before* the affected analysis runs, so
# demanding that the directory be empty pre-execution would make compliance
# impossible.  Its contents are validated instead, which is a stricter check.
DEFINITIVE_OUTPUT_DIRS = (
    "benchmarks/results",
    "benchmarks/runs",
    "benchmarks/logs",
    "benchmarks/figures",
    "benchmarks/tables",
    "benchmarks/failures",
    "benchmarks/manuscript_assets",
)
DEVIATION_DIR = "benchmarks/deviations"
SEMI_SYNTHETIC_TEMPLATE_DIR = "benchmarks/datasets/manifests"
SEMI_SYNTHETIC_TEMPLATE_SCHEMA = "benchmark_record/schemas/semi-synthetic-template.schema.json"

DATASET_ELIGIBILITY_DEFAULT = "benchmarks/datasets/manifests/real_cohort_eligibility.json"
COMPARATOR_ELIGIBILITY_CANDIDATES = (
    "benchmarks/comparators/eligibility.json",
    "benchmarks/comparators/comparator_eligibility.json",
    "benchmarks/comparators/eligibility.yaml",
    "benchmarks/comparators/eligibility.yml",
    "benchmarks/comparators/comparator_eligibility.yaml",
    "benchmarks/comparators/comparator_eligibility.yml",
)
SUT_ENVIRONMENT_CANDIDATES = (
    "benchmark_record/environment/omicau-environment.yaml",
    "benchmark_record/environment/omicau-environment.yml",
    "benchmark_record/environment/omicau-environment.json",
    "benchmark_record/environment/execution-environment.yaml",
    "benchmark_record/environment/execution-environment.yml",
    "benchmark_record/environment/execution-environment.json",
    "benchmark_record/environment/environment-lock.yaml",
    "benchmark_record/environment/environment-lock.yml",
    "benchmark_record/environment/environment-lock.json",
    "benchmarks/environment/omicau-environment.yaml",
    "benchmarks/environment/omicau-environment.yml",
    "benchmarks/environment/omicau-environment.json",
)
COMPARATOR_ENVIRONMENT_CANDIDATES = (
    "benchmark_record/environment/comparator-environments.yaml",
    "benchmark_record/environment/comparator-environments.yml",
    "benchmark_record/environment/comparator-environments.json",
    "benchmark_record/environment/comparator-environment.yaml",
    "benchmark_record/environment/comparator-environment.yml",
    "benchmark_record/environment/comparator-environment.json",
    "benchmarks/environment/comparator-environments.yaml",
    "benchmarks/environment/comparator-environments.yml",
    "benchmarks/environment/comparator-environments.json",
)
SCHEMA_FILES = (
    "benchmark_record/schemas/benchmark-result.schema.json",
    "benchmark_record/schemas/dataset-manifest.schema.json",
    "benchmark_record/schemas/deviation.schema.json",
    "benchmark_record/schemas/failed-run.schema.json",
    "benchmark_record/schemas/semi-synthetic-template.schema.json",
)
# Documents validated against a generated schema, as (schema, instance) pairs.
SCHEMA_INSTANCES = (
    ("benchmark_record/schemas/dataset-manifest.schema.json",
     "benchmark_record/DATASET_MANIFEST.yaml"),
)


def _relative(path: Path, repo: Path) -> str:
    try:
        return path.relative_to(repo).as_posix()
    except ValueError:
        return path.as_posix()


def _result(status: str, *, details: Any = None, blockers: Iterable[str] = ()) -> dict[str, Any]:
    value: dict[str, Any] = {"status": status, "blockers": list(blockers)}
    if details is not None:
        value["details"] = details
    return value


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _read_yaml(path: Path) -> tuple[Any | None, str | None]:
    if yaml is None:
        return None, "PyYAML is not installed"
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return None, str(exc)


def _read_structured(path: Path) -> tuple[Any | None, str | None]:
    if path.suffix.lower() == ".json":
        return _read_json(path)
    return _read_yaml(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_delivery_archive(repo: Path, name: str) -> Path | None:
    delivery = repo / "local" / "delivery"
    preferred = delivery / Path(name).stem / name
    if preferred.is_file():
        return preferred
    direct = delivery / name
    if direct.is_file():
        return direct
    matches = sorted(path for path in delivery.rglob(name) if path.is_file()) if delivery.is_dir() else []
    return matches[0] if len(matches) == 1 else None


def _doi_field(value: Any) -> bool:
    return isinstance(value, str) and DOI_RE.fullmatch(value.strip()) is not None


def _verify_zip_manifest(archive: Path) -> list[str]:
    """Verify ZIP CRCs and the reader-facing manifest without extracting it."""
    errors: list[str] = []
    try:
        with zipfile.ZipFile(archive, "r") as handle:
            bad_member = handle.testzip()
            if bad_member is not None:
                errors.append(f"deposited archive has a CRC error in {bad_member}")
            names = handle.namelist()
            manifest_names = [name for name in names if name.endswith("/RECORD_MANIFEST.sha256")]
            if len(manifest_names) != 1:
                errors.append("deposited archive does not contain exactly one RECORD_MANIFEST.sha256")
                return errors
            manifest_name = manifest_names[0]
            root = manifest_name[: -len("RECORD_MANIFEST.sha256")]
            text = handle.read(manifest_name).decode("utf-8")
            listed: set[str] = set()
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                match = re.fullmatch(r"([0-9a-fA-F]{64})  (.+)", line)
                if match is None:
                    errors.append(f"invalid archive manifest line {line_number}")
                    continue
                expected, relative = match.groups()
                if relative in listed:
                    errors.append(f"duplicate archive manifest entry {relative}")
                    continue
                listed.add(relative)
                member = root + relative
                try:
                    actual = hashlib.sha256(handle.read(member)).hexdigest()
                except KeyError:
                    errors.append(f"archive manifest names missing member {relative}")
                    continue
                if actual.lower() != expected.lower():
                    errors.append(f"archive manifest hash mismatch for {relative}")
            archived_members = {name[len(root) :] for name in names if name.startswith(root) and name != manifest_name}
            if listed != archived_members:
                missing = sorted(listed - archived_members)
                extra = sorted(archived_members - listed)
                if missing:
                    errors.append("archive manifest has missing members: " + ", ".join(missing))
                if extra:
                    errors.append("archive contains unlisted members: " + ", ".join(extra))
    except (OSError, UnicodeError, zipfile.BadZipFile) as exc:
        errors.append(f"cannot verify deposited ZIP: {exc}")
    return errors


def check_protocol_publication(repo: Path) -> dict[str, Any]:
    """Validate the receipt and its immutable local delivery archive."""
    record = repo / "benchmark_record"
    receipt_path = record / "ZENODO_RECEIPT.json"
    blockers: list[str] = []
    details: dict[str, Any] = {"receipt": _relative(receipt_path, repo)}
    receipt, error = _read_json(receipt_path)
    if error is not None:
        blockers.append(f"cannot read {_relative(receipt_path, repo)}: {error}")
        return _result("fail", details=details, blockers=blockers)
    if not isinstance(receipt, dict):
        blockers.append("ZENODO_RECEIPT.json must contain an object")
        return _result("fail", details=details, blockers=blockers)

    details["record_status"] = receipt.get("record_status")
    if receipt.get("record_status") != "published":
        blockers.append("ZENODO_RECEIPT.json record_status is not 'published'")
    for key in ("concept_doi", "version_doi"):
        if not _doi_field(receipt.get(key)):
            blockers.append(f"ZENODO_RECEIPT.json has invalid {key}")
    if not isinstance(receipt.get("publication_date"), str) or not receipt.get("publication_date"):
        blockers.append("ZENODO_RECEIPT.json publication_date is missing")
    receipt_version = receipt.get("version")
    if not isinstance(receipt_version, str) or VERSION_RE.fullmatch(receipt_version) is None:
        blockers.append("ZENODO_RECEIPT.json version is missing or not semantic version text")
    if receipt.get("archive_mutation_after_publication") is not False:
        blockers.append("ZENODO_RECEIPT.json does not assert archive_mutation_after_publication=false")

    deposited = receipt.get("deposited_file")
    if not isinstance(deposited, dict):
        blockers.append("ZENODO_RECEIPT.json deposited_file object is missing")
        return _result("fail", details=details, blockers=blockers)
    name = deposited.get("name")
    if not isinstance(name, str) or not name or Path(name).name != name or not name.lower().endswith(".zip"):
        blockers.append("ZENODO_RECEIPT.json deposited_file.name is not a ZIP filename")
        return _result("fail", details=details, blockers=blockers)
    archive = _find_delivery_archive(repo, name)
    details["local_archive"] = _relative(archive, repo) if archive else None
    if archive is None:
        blockers.append(f"local delivery archive is missing: local/delivery/**/{name}")
        return _result("fail", details=details, blockers=blockers)

    size = archive.stat().st_size
    actual_md5 = _md5(archive)
    actual_sha256 = _sha256(archive)
    details.update({"size_bytes": size, "md5": actual_md5, "sha256": actual_sha256})
    expected_size = deposited.get("size_bytes")
    if not isinstance(expected_size, int) or expected_size != size:
        blockers.append(f"deposited archive size mismatch: receipt={expected_size!r}, local={size}")
    for field, actual, pattern in (
        ("zenodo_md5", actual_md5, MD5_RE),
        ("local_md5_verified", actual_md5, MD5_RE),
        ("local_sha256", actual_sha256, SHA256_RE),
    ):
        expected = deposited.get(field)
        if not isinstance(expected, str) or pattern.fullmatch(expected) is None:
            blockers.append(f"receipt deposited_file.{field} is missing or malformed")
        elif expected.lower() != actual.lower():
            blockers.append(f"deposited archive {field} mismatch: receipt={expected}, local={actual}")
    if deposited.get("identity_check") != "pass":
        blockers.append("receipt deposited_file.identity_check is not 'pass'")

    sums_path = archive.parent / "SHA256SUMS.txt"
    if not sums_path.is_file():
        blockers.append(f"local delivery checksum file is missing: {_relative(sums_path, repo)}")
    else:
        try:
            sums = sums_path.read_text(encoding="utf-8").splitlines()
            archive_lines = [line for line in sums if line.rstrip().endswith("  " + name)]
            if not archive_lines or not archive_lines[0].lower().startswith(actual_sha256.lower() + "  "):
                blockers.append(f"SHA256SUMS.txt does not bind {name} to the local archive SHA-256")
        except (OSError, UnicodeError) as exc:
            blockers.append(f"cannot read {_relative(sums_path, repo)}: {exc}")
    blockers.extend(_verify_zip_manifest(archive))
    return _result("pass" if not blockers else "fail", details=details, blockers=blockers)


def _pending(value: Any) -> bool:
    return isinstance(value, str) and PENDING_RE.search(value) is not None


def _status(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip().lower()


def _is_terminal(value: Any) -> bool:
    status = _status(value)
    return status in TERMINAL_ELIGIBILITY or (status is not None and not _pending(status))


def _records(value: Any) -> dict[str, dict[str, Any]]:
    """Normalize common list/map eligibility report shapes by dataset/method id."""
    if isinstance(value, dict):
        for key in ("datasets", "comparators", "records", "results", "eligibility"):
            if key in value:
                return _records(value[key])
        output: dict[str, dict[str, Any]] = {}
        for key, item in value.items():
            if isinstance(item, dict):
                record = dict(item)
                identifier = record.get("dataset_id") or record.get("method_id") or record.get("id") or str(key)
                output[str(identifier)] = record
        return output
    if isinstance(value, list):
        output = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            identifier = item.get("dataset_id") or item.get("method_id") or item.get("id")
            if identifier is not None:
                output[str(identifier)] = dict(item)
        return output
    return {}


def _record_status(record: dict[str, Any]) -> Any:
    for key in ("final_eligibility_status", "eligibility_status", "feasibility_status", "status"):
        if key in record:
            return record[key]
    return None


def _load_manifest(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    value, error = _read_structured(path)
    if error is not None:
        return None, error
    return value if isinstance(value, dict) else None, None if isinstance(value, dict) else "must contain a mapping"


def _check_dataset_eligibility(repo: Path) -> tuple[dict[str, Any], set[str]]:
    record = repo / "benchmark_record"
    blockers: list[str] = []
    details: dict[str, Any] = {}
    dataset_manifest_path = record / "DATASET_MANIFEST.yaml"
    dataset_manifest, error = _load_manifest(dataset_manifest_path)
    if error is not None or dataset_manifest is None:
        blockers.append(f"cannot read {_relative(dataset_manifest_path, repo)}: {error or 'invalid'}")
        dataset_manifest = {}

    datasets = dataset_manifest.get("datasets") if isinstance(dataset_manifest.get("datasets"), dict) else {}
    required_dataset_ids = set(str(key) for key in datasets)
    workflow = dataset_manifest.get("eligibility_workflow") or {}
    output_name = workflow.get("output") if isinstance(workflow, dict) else None
    dataset_output_path = repo / (str(output_name) if isinstance(output_name, str) else DATASET_ELIGIBILITY_DEFAULT)
    details["dataset_eligibility_report"] = _relative(dataset_output_path, repo)
    dataset_report, error = _read_json(dataset_output_path)
    eligible_ids: set[str] = set()
    if error is not None:
        blockers.append(f"required dataset eligibility report is missing or unreadable: {_relative(dataset_output_path, repo)} ({error})")
    else:
        report_records = _records(dataset_report)
        details["dataset_records"] = sorted(report_records)
        missing = sorted(required_dataset_ids - set(report_records))
        if missing:
            blockers.append("dataset eligibility report omits required datasets: " + ", ".join(missing))
        for identifier, item in report_records.items():
            state = _record_status(item)
            if not _is_terminal(state):
                blockers.append(f"dataset eligibility is unresolved for {identifier}: {state!r}")
            if item.get("assessed_before_results") is False or item.get("model_results_visible_during_eligibility") is True:
                blockers.append(f"dataset eligibility for {identifier} was not outcome-blind")
            status = _status(state)
            if status and "eligible" in status and "ineligible" not in status and "not_eligible" not in status:
                eligible_ids.add(identifier)
    return _result("pass" if not blockers else "fail", details=details, blockers=blockers), eligible_ids


def _check_comparator_eligibility(repo: Path) -> dict[str, Any]:
    record = repo / "benchmark_record"
    blockers: list[str] = []
    details: dict[str, Any] = {}
    comparator_manifest_path = record / "COMPARATOR_MANIFEST.yaml"
    comparator_manifest, error = _load_manifest(comparator_manifest_path)
    if error is not None or comparator_manifest is None:
        blockers.append(f"cannot read {_relative(comparator_manifest_path, repo)}: {error or 'invalid'}")
        comparator_manifest = {}

    comparator_entries = comparator_manifest.get("comparators") if isinstance(comparator_manifest.get("comparators"), dict) else {}
    details["required_comparators"] = sorted(str(key) for key in comparator_entries)
    comparator_path = next((repo / relative for relative in COMPARATOR_ELIGIBILITY_CANDIDATES if (repo / relative).is_file()), None)
    details["comparator_eligibility_report"] = _relative(comparator_path, repo) if comparator_path else None
    if comparator_path is None:
        blockers.append("required comparator eligibility report is missing (expected benchmarks/comparators/eligibility.json or YAML equivalent)")
    else:
        comparator_report, error = _read_structured(comparator_path)
        if error is not None:
            blockers.append(f"cannot read comparator eligibility report {_relative(comparator_path, repo)}: {error}")
        else:
            comparator_records = _records(comparator_report)
            missing = sorted(set(str(key) for key in comparator_entries) - set(comparator_records))
            if missing:
                blockers.append("comparator eligibility report omits required methods: " + ", ".join(missing))
            required_fields = comparator_manifest.get("eligibility_record_required_fields") or []
            for identifier in sorted(set(str(key) for key in comparator_entries) & set(comparator_records)):
                item = comparator_records[identifier]
                state = _record_status(item)
                if not _is_terminal(state):
                    blockers.append(f"comparator eligibility is unresolved for {identifier}: {state!r}")
                if isinstance(required_fields, list):
                    missing_fields = [str(field) for field in required_fields if field not in item]
                    if missing_fields:
                        blockers.append(f"comparator eligibility record for {identifier} is missing: " + ", ".join(missing_fields))
    return _result("pass" if not blockers else "fail", details=details, blockers=blockers)


def _check_eligibility(repo: Path) -> tuple[dict[str, Any], set[str]]:
    """Both eligibility reports, retained for callers that want one verdict."""
    datasets, eligible_ids = _check_dataset_eligibility(repo)
    comparators = _check_comparator_eligibility(repo)
    blockers = datasets["blockers"] + comparators["blockers"]
    details = {**datasets.get("details", {}), **comparators.get("details", {})}
    return _result("pass" if not blockers else "fail", details=details, blockers=blockers), eligible_ids


def _iter_seed_values(value: Any) -> Iterable[int]:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in {"seed", "uint32_seed", "random_seed"} and isinstance(child, int) and not isinstance(child, bool):
                yield child
            yield from _iter_seed_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_seed_values(child)


def _configured_protocol_path(repo: Path, value: Any, default: str) -> Path:
    relative = value if isinstance(value, str) and value.strip() else default
    path = Path(relative)
    return path if path.is_absolute() else repo / path


def _check_seeds(repo: Path) -> dict[str, Any]:
    record = repo / "benchmark_record"
    blockers: list[str] = []
    protocol, protocol_error = _load_manifest(record / "benchmark_protocol.yaml")
    seed_generation = protocol.get("simulation", {}).get("seed_generation", {}) if isinstance(protocol, dict) else {}
    seed_audit = protocol.get("independence", {}).get("seed_audit", {}) if isinstance(protocol, dict) else {}
    registry_path = _configured_protocol_path(
        repo, seed_generation.get("seed_table_output"),
        "benchmark_record/checksums/definitive_seed_registry.json",
    )
    audit_path = _configured_protocol_path(
        repo, seed_audit.get("archived_output"),
        "benchmark_record/checksums/seed_overlap_audit.json",
    )
    details = {"registry": _relative(registry_path, repo), "audit": _relative(audit_path, repo)}
    # The protocol forbids storing an aggregate total (machine_summary.
    # aggregate_counts_hardcoded: false), so the expected unit and stream counts
    # are derived from the family definitions on every call.  A registry that
    # disagrees with its own protocol is the failure this check exists to catch.
    expected_keys: set[tuple[Any, ...]] = set()
    derived_units: int | None = None
    if protocol_error is None and isinstance(protocol, dict):
        try:
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from benchmarks.simulations.generate import STREAM_LABELS, protocol_units
            units = protocol_units(protocol)
            expected_keys = {
                (unit.family, unit.scenario_or_structure,
                 unit.condition_or_perturbation, unit.sample_size,
                 unit.replicate_index, stream_label)
                for unit in units for stream_label in STREAM_LABELS
            }
            derived_units = len(units)
            details["derived_generation_unit_count"] = derived_units
            details["derived_stream_key_count"] = len(expected_keys)
        except Exception as exc:
            blockers.append(f"cannot derive expected seed registry keys: {type(exc).__name__}: {exc}")
    if protocol_error is not None:
        blockers.append(f"cannot read benchmark protocol for generated seed artifacts: {protocol_error}")
    registry, error = _read_json(registry_path)
    registry_seeds: list[int] = []
    if error is not None:
        blockers.append(f"definitive seed registry is missing or unreadable: {_relative(registry_path, repo)} ({error})")
    else:
        registry_seeds = list(_iter_seed_values(registry))
        if not registry_seeds:
            blockers.append("definitive seed registry contains no explicit seed values")
        if any(seed < 0 or seed > 0xFFFFFFFF for seed in registry_seeds):
            blockers.append("definitive seed registry contains a value outside uint32 range")
        if registry_seeds and len(registry_seeds) != len(set(registry_seeds)):
            blockers.append("definitive seed registry contains duplicate seed values")
        if isinstance(registry, dict):
            streams = registry.get("streams")
            actual_keys: list[tuple[Any, ...]] = []
            if not isinstance(streams, list):
                blockers.append("definitive seed registry streams must be a list")
            else:
                actual_keys = [
                    (item.get("family"), item.get("scenario_or_structure"),
                     item.get("condition_or_perturbation"), item.get("sample_size"),
                     item.get("replicate_index"), item.get("stream_label"))
                    for item in streams if isinstance(item, dict)
                ]
                if len(actual_keys) != len(streams):
                    blockers.append("definitive seed registry contains a non-object stream record")
                if len(actual_keys) != len(set(actual_keys)):
                    blockers.append("definitive seed registry contains duplicate stream keys")
                if expected_keys and set(actual_keys) != expected_keys:
                    missing = expected_keys - set(actual_keys)
                    extra = set(actual_keys) - expected_keys
                    blockers.append(
                        "definitive seed registry keys do not exactly match protocol derivation "
                        f"({len(missing)} missing, {len(extra)} extra)"
                    )
                if derived_units is not None and registry.get("generation_unit_count") != derived_units:
                    blockers.append(
                        f"registry generation_unit_count={registry.get('generation_unit_count')!r} "
                        f"does not match the {derived_units} units derived from the protocol"
                    )
                if expected_keys and registry.get("stream_seed_count") != len(expected_keys):
                    blockers.append(
                        f"registry stream_seed_count={registry.get('stream_seed_count')!r} "
                        f"does not match the {len(expected_keys)} stream keys derived "
                        "from the protocol"
                    )
    audit, error = _read_json(audit_path)
    if error is not None:
        blockers.append(f"seed-overlap audit is missing or unreadable: {_relative(audit_path, repo)} ({error})")
    elif not isinstance(audit, dict):
        blockers.append("seed-overlap audit must contain an object")
    else:
        audit_status = str(audit.get("status", audit.get("result", ""))).lower()
        overlap_count = audit.get("overlap_count")
        overlap = audit.get("overlap")
        if audit_status not in {"pass", "passed", "ok", "success"}:
            blockers.append(f"seed-overlap audit status is not passing: {audit.get('status', audit.get('result'))!r}")
        if overlap_count != 0:
            blockers.append(f"seed-overlap audit reports overlap_count={overlap_count!r}, expected 0")
        if isinstance(overlap, list) and overlap:
            blockers.append("seed-overlap audit contains overlapping seeds: " + ", ".join(map(str, overlap[:10])))
        audit_count = audit.get("definitive_seed_count")
        if isinstance(audit_count, int) and registry_seeds and audit_count != len(registry_seeds):
            blockers.append(f"seed-overlap audit definitive count {audit_count} does not match registry count {len(registry_seeds)}")
    details["registry_seed_count"] = len(registry_seeds)
    return _result("pass" if not blockers else "fail", details=details, blockers=blockers)


def _schema_validator(repo: Path):
    """Return the harness's schema checker, which the result validator already uses."""
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from benchmarks.harness.validate_rows import _schema_errors
    return _schema_errors


def _schema_gap_reporter(repo: Path):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from benchmarks.harness.validate_rows import schema_keyword_gaps
    return schema_keyword_gaps


def _check_schemas(repo: Path) -> dict[str, Any]:
    blockers: list[str] = []
    details: dict[str, Any] = {"schemas": [], "validated_instances": []}
    loaded: dict[str, Any] = {}
    for relative in SCHEMA_FILES:
        path = repo / relative
        details["schemas"].append(relative)
        value, error = _read_json(path)
        if error is not None:
            blockers.append(f"required generated schema is missing or unreadable: {relative} ({error})")
        elif not isinstance(value, dict) or not value:
            blockers.append(f"required generated schema is empty or not an object: {relative}")
        else:
            loaded[relative] = value

    # A schema that nothing is checked against proves nothing, so each generated
    # schema with a registered instance validates that instance here.
    try:
        errors_for = _schema_validator(repo)
        gaps_in = _schema_gap_reporter(repo)
    except Exception as exc:
        blockers.append(f"cannot load the schema validator: {type(exc).__name__}: {exc}")
        return _result("fail", details=details, blockers=blockers)

    # A keyword the checker does not implement makes every instance pass
    # vacuously, so an unimplemented keyword is itself a readiness blocker.
    for relative, schema in loaded.items():
        gaps = sorted(gaps_in(schema))
        if gaps:
            blockers.append(f"{relative} uses schema keywords the validator does not "
                            f"enforce: {', '.join(gaps)}")
    for schema_relative, instance_relative in SCHEMA_INSTANCES:
        schema = loaded.get(schema_relative)
        if schema is None:
            continue
        instance, error = _read_structured(repo / instance_relative)
        if error is not None or instance is None:
            blockers.append(f"cannot read {instance_relative} for schema validation: {error or 'empty'}")
            continue
        errors = errors_for(instance, schema, instance_relative)
        details["validated_instances"].append(instance_relative)
        blockers.extend(errors[:10])
        if len(errors) > 10:
            blockers.append(f"{instance_relative}: {len(errors) - 10} further schema errors")
    return _result("pass" if not blockers else "fail", details=details, blockers=blockers)


def _check_one_environment(repo: Path, candidates: tuple[str, ...], label: str,
                           key: str) -> dict[str, Any]:
    blockers: list[str] = []
    path = next((repo / relative for relative in candidates if (repo / relative).is_file()), None)
    details: dict[str, Any] = {key: _relative(path, repo) if path else None}
    if path is None:
        blockers.append(f"{label} lock/capture is missing")
        return _result("fail", details=details, blockers=blockers)
    value, error = _read_structured(path)
    if error is not None or value in (None, {}, []):
        blockers.append(f"{label} is unreadable or empty: {_relative(path, repo)} ({error or 'empty'})")
    elif any(_pending(item) for item in _walk_scalars(value)):
        blockers.append(f"environment capture contains unresolved PENDING values: "
                        f"{_relative(path, repo)}")
    return _result("pass" if not blockers else "fail", details=details, blockers=blockers)


def _check_sut_environment(repo: Path) -> dict[str, Any]:
    return _check_one_environment(repo, SUT_ENVIRONMENT_CANDIDATES,
                                  "omicau execution environment", "sut_environment")


def _check_comparator_environments(repo: Path) -> dict[str, Any]:
    return _check_one_environment(repo, COMPARATOR_ENVIRONMENT_CANDIDATES,
                                  "comparator environment", "comparator_environments")


def _check_environments(repo: Path) -> dict[str, Any]:
    """Both environment captures, retained for callers that want one verdict."""
    sut = _check_sut_environment(repo)
    comparators = _check_comparator_environments(repo)
    blockers = sut["blockers"] + comparators["blockers"]
    details = {**sut.get("details", {}), **comparators.get("details", {})}
    return _result("pass" if not blockers else "fail", details=details, blockers=blockers)


def _walk_scalars(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_scalars(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_scalars(child)
    else:
        yield value


def _check_splits(repo: Path, eligible_ids: set[str]) -> dict[str, Any]:
    blockers: list[str] = []
    details: dict[str, Any] = {"eligible_datasets": sorted(eligible_ids), "checked": []}
    if not eligible_ids:
        blockers.append("cannot verify required real-cohort splits because no eligible dataset is resolved")
        return _result("fail", details=details, blockers=blockers)
    protocol, error = _load_manifest(repo / "benchmark_record" / "benchmark_protocol.yaml")
    if error is not None or protocol is None:
        blockers.append(f"cannot read benchmark protocol for split requirements: {error or 'invalid'}")
        return _result("fail", details=details, blockers=blockers)
    real = protocol.get("real_cohorts") or {}
    outer = real.get("outer_cv") if isinstance(real, dict) else {}
    repeats = outer.get("repeats") if isinstance(outer, dict) else None
    if not isinstance(repeats, int) or repeats < 1:
        blockers.append("benchmark protocol does not declare a positive real-cohort repeat count")
        return _result("fail", details=details, blockers=blockers)
    for dataset_id in sorted(eligible_ids):
        for repeat in range(repeats):
            path = repo / "benchmarks" / "splits" / dataset_id / f"repeat_{repeat}" / "outer_folds.json"
            if not path.is_file():
                blockers.append(f"required frozen split manifest is missing: {_relative(path, repo)}")
                continue
            value, error = _read_json(path)
            if error is not None or not isinstance(value, dict):
                blockers.append(f"frozen split manifest is unreadable or invalid: {_relative(path, repo)} ({error or 'must be an object'})")
                continue
            folds = value.get("outer_folds", value.get("folds"))
            if not isinstance(folds, list) or not folds:
                blockers.append(f"frozen split manifest has no folds: {_relative(path, repo)}")
            if not any(key in value for key in ("sha256", "SHA256")):
                blockers.append(f"frozen split manifest has no SHA-256 field: {_relative(path, repo)}")
            details["checked"].append(_relative(path, repo))
    return _result("pass" if not blockers else "fail", details=details, blockers=blockers)


def _check_empty_outputs(repo: Path) -> dict[str, Any]:
    blockers: list[str] = []
    details: dict[str, Any] = {}
    for relative in DEFINITIVE_OUTPUT_DIRS:
        path = repo / relative
        files = sorted(_relative(item, repo) for item in path.rglob("*") if item.is_file()) if path.is_dir() else []
        details[relative] = files
        if files:
            blockers.append(f"definitive output directory is not empty: {relative} ({len(files)} file(s))")
    return _result("pass" if not blockers else "fail", details=details, blockers=blockers)


def _check_deviations(repo: Path) -> dict[str, Any]:
    """Every filed deviation must be schema-valid and predate the results it affects."""
    blockers: list[str] = []
    path = repo / DEVIATION_DIR
    records = sorted(path.glob("*.json")) if path.is_dir() else []
    details: dict[str, Any] = {"records": [_relative(item, repo) for item in records]}
    if not records:
        return _result("pass", details=details, blockers=blockers)
    schema, error = _read_json(repo / "benchmark_record" / "schemas" / "deviation.schema.json")
    if error is not None or not isinstance(schema, dict):
        blockers.append(f"cannot read the deviation schema: {error or 'invalid'}")
        return _result("fail", details=details, blockers=blockers)
    try:
        errors_for = _schema_validator(repo)
    except Exception as exc:
        blockers.append(f"cannot load the schema validator: {type(exc).__name__}: {exc}")
        return _result("fail", details=details, blockers=blockers)
    for record_path in records:
        value, error = _read_json(record_path)
        name = _relative(record_path, repo)
        if error is not None or not isinstance(value, dict):
            blockers.append(f"deviation record is unreadable or not an object: {name} ({error or 'invalid'})")
            continue
        blockers.extend(errors_for(value, schema, name)[:5])
        if value.get("results_seen_before_change") == "yes":
            blockers.append(f"deviation {name} was decided after its results were seen; "
                            "it cannot be part of a pre-execution readiness pass")
    return _result("pass" if not blockers else "fail", details=details, blockers=blockers)


def _check_semi_synthetic(repo: Path) -> dict[str, Any]:
    """The semi-synthetic family needs one frozen template descriptor per structure.

    Both branches of SIMULATION_DESIGN.yaml#semi_synthetic_design, including the
    synthetic fallback, are defined against a harmonized cohort's dimension and
    missingness targets.  Without those the 180 registered units cannot be
    generated, and the family must not be able to disappear by being skipped.
    """
    blockers: list[str] = []
    details: dict[str, Any] = {"structures": [], "descriptors": {}}
    protocol, error = _load_manifest(repo / "benchmark_record" / "benchmark_protocol.yaml")
    if error is not None or protocol is None:
        blockers.append(f"cannot read benchmark protocol for semi-synthetic structures: {error or 'invalid'}")
        return _result("fail", details=details, blockers=blockers)
    family = ((protocol.get("simulation") or {}).get("families") or {}).get("semi_synthetic_robustness")
    structures = family.get("cohort_structures") if isinstance(family, dict) else None
    if not isinstance(structures, list) or not structures:
        blockers.append("benchmark protocol declares no semi-synthetic cohort structures")
        return _result("fail", details=details, blockers=blockers)
    details["structures"] = [str(item) for item in structures]

    schema, error = _read_json(repo / SEMI_SYNTHETIC_TEMPLATE_SCHEMA)
    if error is not None or not isinstance(schema, dict):
        blockers.append(f"semi-synthetic template schema is missing or unreadable: "
                        f"{SEMI_SYNTHETIC_TEMPLATE_SCHEMA} ({error or 'invalid'})")
        return _result("fail", details=details, blockers=blockers)
    try:
        errors_for = _schema_validator(repo)
    except Exception as exc:
        blockers.append(f"cannot load the schema validator: {type(exc).__name__}: {exc}")
        return _result("fail", details=details, blockers=blockers)

    filed_deviations = {
        value.get("deviation_id")
        for value in (_read_json(item)[0] for item in
                      sorted((repo / DEVIATION_DIR).glob("*.json"))
                      if (repo / DEVIATION_DIR).is_dir())
        if isinstance(value, dict)
    }
    for structure in details["structures"]:
        relative = f"{SEMI_SYNTHETIC_TEMPLATE_DIR}/semi_synthetic_template_{structure}.json"
        details["descriptors"][structure] = relative
        value, error = _read_json(repo / relative)
        if error is not None:
            blockers.append(f"semi-synthetic template descriptor is missing for "
                            f"{structure}: {relative} ({error}); see "
                            "benchmark_record/SEMI_SYNTHETIC_EXECUTION_BLOCKERS.md")
            continue
        blockers.extend(errors_for(value, schema, relative)[:5])
        magnitudes = value.get("perturbation_magnitudes") if isinstance(value, dict) else None
        if not isinstance(magnitudes, dict):
            blockers.append(f"{relative} carries no perturbation magnitudes; the three "
                            "registered operations have no magnitude in the frozen record "
                            "and cannot be executed as written")
        elif magnitudes.get("deviation_id") not in filed_deviations:
            blockers.append(f"{relative} cites deviation {magnitudes.get('deviation_id')!r}, "
                            f"which is not filed in {DEVIATION_DIR}/")
    return _result("pass" if not blockers else "fail", details=details, blockers=blockers)


# Which checks each execution phase actually depends on.  A role-recovery
# simulation does not depend on a real cohort being downloaded, so a phase-scoped
# verdict is stricter per phase than one global verdict is per phase -- it names
# exactly what that phase needs.  The global `status` remains the conjunction of
# every check and is what gates execution unless a phase is named explicitly.
PHASE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "diagnostic_simulations": (
        "definitive_seeds", "schemas", "sut_environment", "deviations",
        "empty_definitive_outputs",
    ),
    "predictive_simulations": (
        "definitive_seeds", "schemas", "sut_environment", "deviations",
        "empty_definitive_outputs",
    ),
    "external_comparators_and_semi_synthetic": (
        "definitive_seeds", "schemas", "sut_environment", "deviations",
        "empty_definitive_outputs", "comparator_eligibility",
        "comparator_environments", "semi_synthetic_templates",
    ),
    "real_cohorts": (
        "definitive_seeds", "schemas", "sut_environment", "deviations",
        "empty_definitive_outputs", "dataset_eligibility", "splits",
    ),
}
# The families each simulation phase executes, so a caller can turn a phase
# verdict into a --family selection without restating the mapping.
PHASE_FAMILIES: dict[str, tuple[str, ...]] = {
    "diagnostic_simulations": ("role_recovery", "batch_risk_flags",
                               "missingness_risk_flags", "group_leakage",
                               "null_control_specificity"),
    "predictive_simulations": ("predictive_performance", "nonlinear_secondary"),
    "external_comparators_and_semi_synthetic": ("semi_synthetic_robustness",),
    "real_cohorts": (),
}


def check_readiness(repo: Path | str | None = None) -> dict[str, Any]:
    """Return publication-validity and pre-execution-readiness verdicts.

    ``repo`` defaults to the repository containing this file.  The returned
    structure is JSON serializable and no function in this module writes to
    disk.
    """
    root = Path(repo).resolve() if repo is not None else Path(__file__).resolve().parents[2]
    publication = check_protocol_publication(root)
    dataset_eligibility, eligible_ids = _check_dataset_eligibility(root)
    execution_checks = {
        "dataset_eligibility": dataset_eligibility,
        "comparator_eligibility": _check_comparator_eligibility(root),
        "definitive_seeds": _check_seeds(root),
        "schemas": _check_schemas(root),
        "sut_environment": _check_sut_environment(root),
        "comparator_environments": _check_comparator_environments(root),
        "splits": _check_splits(root, eligible_ids),
        "semi_synthetic_templates": _check_semi_synthetic(root),
        "deviations": _check_deviations(root),
        "empty_definitive_outputs": _check_empty_outputs(root),
    }
    execution_blockers = [
        blocker
        for check in execution_checks.values()
        for blocker in check["blockers"]
    ]
    execution = {
        "status": "pass" if not execution_blockers else "fail",
        "blockers": execution_blockers,
        "checks": execution_checks,
    }
    phases = {}
    for phase, required in PHASE_REQUIREMENTS.items():
        phase_blockers = [blocker for name in required
                          for blocker in execution_checks[name]["blockers"]]
        phases[phase] = {
            "status": "pass" if not (phase_blockers or publication["blockers"]) else "fail",
            "requires": list(required),
            "families": list(PHASE_FAMILIES.get(phase, ())),
            "blockers": publication["blockers"] + phase_blockers,
        }

    blockers = publication["blockers"] + execution_blockers
    return {
        "repository": str(root),
        "read_only": True,
        "protocol_publication": publication,
        "execution_readiness": execution,
        "phases": phases,
        # Descriptive aliases make the two decision boundaries explicit to
        # callers that use the longer terminology.
        "protocol_publication_validity": publication,
        "pre_execution_readiness": execution,
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
    }


# Small API aliases for scripts that use the verb rather than the noun.
check = check_readiness
readiness = check_readiness


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=None, help="repository root (default: this repository)")
    parser.add_argument("--json", action="store_true", help="print the machine-readable report")
    args = parser.parse_args(argv)
    report = check_readiness(args.repo)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"protocol publication: {report['protocol_publication']['status'].upper()}")
        for blocker in report["protocol_publication"]["blockers"]:
            print(f"  BLOCKER: {blocker}")
        print(f"pre-execution readiness: {report['execution_readiness']['status'].upper()}")
        for name, item in report["execution_readiness"]["checks"].items():
            print(f"  {name}: {item['status'].upper()}")
            for blocker in item["blockers"]:
                print(f"    BLOCKER: {blocker}")
        print("per-phase readiness (advisory; the overall verdict is what gates execution)")
        for name, item in report["phases"].items():
            families = ", ".join(item["families"]) or "no simulation family"
            print(f"  {name}: {item['status'].upper()}  [{families}]")
            for blocker in item["blockers"]:
                print(f"    BLOCKER: {blocker}")
        print(f"overall: {report['status'].upper()}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
