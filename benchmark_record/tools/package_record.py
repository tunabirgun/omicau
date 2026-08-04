"""Build the three-file Zenodo delivery from the frozen record allowlist.

The archive is read directly from ``benchmark_record``.  It never reads a
staging ``upload/`` directory.  ``--dry-run`` validates and lists the prospective
archive without creating ``local/delivery``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

try:
    from record_manifest import (
        ARCHIVE_STEM,
        FREEZE_NAME,
        MANIFEST_NAME,
        VERSION,
        collect_package_files,
        combined_manifest_sha256,
        load_protocol,
        manifest_bytes,
        package_bytes,
        protocol_version,
        reference_errors,
        sha256_file,
    )
except ImportError:  # pragma: no cover - module import use
    from .record_manifest import (
        ARCHIVE_STEM,
        FREEZE_NAME,
        MANIFEST_NAME,
        VERSION,
        collect_package_files,
        combined_manifest_sha256,
        load_protocol,
        manifest_bytes,
        package_bytes,
        protocol_version,
        reference_errors,
        sha256_file,
    )

RECORD = Path(__file__).resolve().parents[1]
REPO = RECORD.parent
DELIVERY = REPO / "local" / "delivery" / ARCHIVE_STEM
ARCHIVE_NAME = f"{ARCHIVE_STEM}.zip"
APPLICATION_NAME = "ZENODO_APPLICATION.txt"
SUMS_NAME = "SHA256SUMS.txt"
APPLICATION_PLACEHOLDER = "{{ARCHIVE_SHA256}}"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.flag_bits |= 0x800
    return info


def validate_freeze(entries: dict[str, bytes], expected_version: str = VERSION) -> list[str]:
    errors: list[str] = []
    try:
        freeze = json.loads(entries[FREEZE_NAME].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"cannot read {FREEZE_NAME}: {exc}"]
    if freeze.get("protocol_version") != expected_version:
        errors.append(
            f"freeze is for version {freeze.get('protocol_version')!r}; expected {expected_version!r}"
        )
    source_entries = {name: data for name, data in entries.items() if name != FREEZE_NAME}
    current = combined_manifest_sha256(source_entries)
    if freeze.get("record_content_sha256") != current:
        errors.append("reader-facing record content changed after freeze; run freeze_record.py again")
    audit = freeze.get("pilot_seed_audit") or {}
    if audit.get("status") != "pass" or audit.get("overlap_count") != 0:
        errors.append("freeze does not carry a passing zero-overlap pilot seed audit")
    declarations = freeze.get("declarations") or {}
    if declarations.get("active_output_directories_empty") is not True:
        errors.append("freeze does not declare active benchmark output directories empty")
    return errors


def _published_receipt() -> bool:
    """Return whether the deposited record is marked published.

    A malformed or absent receipt is not treated as published here; the normal
    packaging validations still decide whether a new pre-publication package is
    admissible.  A published receipt is a hard immutability boundary.
    """
    try:
        receipt = json.loads((RECORD / "ZENODO_RECEIPT.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(receipt, dict) and receipt.get("record_status") == "published"


def application_bytes(archive_sha256: str) -> bytes:
    template = (RECORD / "ZENODO_DEPOSIT.md").read_text(encoding="utf-8")
    if template.count(APPLICATION_PLACEHOLDER) != 1:
        raise ValueError(
            f"ZENODO_DEPOSIT.md must contain {APPLICATION_PLACEHOLDER!r} exactly once"
        )
    return template.replace(APPLICATION_PLACEHOLDER, archive_sha256).encode("utf-8")


def write_archive(path: Path, entries: dict[str, bytes]) -> None:
    archive_entries = dict(entries)
    archive_entries[MANIFEST_NAME] = manifest_bytes(entries)
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, strict_timestamps=True
    ) as archive:
        for relative, data in sorted(archive_entries.items()):
            archive.writestr(_zip_info(f"{ARCHIVE_STEM}/{relative}"), data, compresslevel=9)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="validate and list files; write nothing")
    args = parser.parse_args()

    if not args.dry_run and _published_receipt():
        print(
            "ERROR: ZENODO_RECEIPT.json says published; refusing to overwrite "
            "the deposited ZIP or its delivery set",
            file=sys.stderr,
        )
        return 1

    try:
        protocol = load_protocol(RECORD)
        files = collect_package_files(RECORD, require_freeze=not args.dry_run)
        entries = package_bytes(files, RECORD)
        errors = reference_errors(protocol, entries)
        if not args.dry_run:
            declared_version = protocol_version(protocol)
            errors.extend(validate_freeze(entries, expected_version=declared_version or VERSION))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"source: {RECORD}")
    print(f"version: {protocol_version(protocol) or VERSION}")
    print("archive allowlist:")
    for name in sorted(entries):
        print(f"  {name}")
    print(f"  {MANIFEST_NAME} (generated)")
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    if args.dry_run:
        if FREEZE_NAME not in entries:
            print("dry run only: the freeze record is not present yet")
        print("dry run: nothing written")
        return 0

    expected = sorted((ARCHIVE_NAME, SUMS_NAME, APPLICATION_NAME))
    if DELIVERY.exists() and any(DELIVERY.iterdir()):
        existing = sorted(path.name for path in DELIVERY.iterdir() if path.is_file())
        if existing != expected or any(path.is_dir() for path in DELIVERY.iterdir()):
            print(
                f"ERROR: refusing to overwrite unexpected delivery contents: {DELIVERY}",
                file=sys.stderr,
            )
            return 1
    DELIVERY.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="omicau-zenodo-") as temporary:
        temporary_dir = Path(temporary)
        archive = temporary_dir / ARCHIVE_NAME
        write_archive(archive, entries)
        archive_sha = sha256_file(archive)
        application = temporary_dir / APPLICATION_NAME
        application.write_bytes(application_bytes(archive_sha))
        sums = temporary_dir / SUMS_NAME
        sums.write_text(
            f"{archive_sha}  {ARCHIVE_NAME}\n"
            f"{sha256_file(application)}  {APPLICATION_NAME}\n",
            encoding="utf-8",
            newline="\n",
        )
        for source in (archive, sums, application):
            os.replace(source, DELIVERY / source.name)

    actual = sorted(path.name for path in DELIVERY.iterdir() if path.is_file())
    if actual != expected:
        print(f"ERROR: delivery contents are not exact: {actual!r}", file=sys.stderr)
        return 1
    print(f"wrote: {DELIVERY}")
    print(f"archive SHA-256: {sha256_file(DELIVERY / ARCHIVE_NAME)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
