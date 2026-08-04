"""Atomic artifact promotion, completion markers and the canonical split digest.

One module owns the two conventions that the writer and the verifier must agree
on, so neither can drift from the other:

* ``split_digest`` is the single definition of a frozen split manifest's SHA-256.
  ``run.freeze_splits`` writes it and ``validate_rows`` recomputes it from the
  fold payload rather than trusting the value stored beside it.
* the completion protocol of ``benchmark_record/COMPUTE_PLAN.md`` §5: each dataset
  writes to a uniquely named staging directory, the payload is validated, the
  final files are replaced atomically, and the completion marker is written last.
  A dataset is complete only when its marker, its rows and every listed artifact
  reproduce their recorded checksums.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SPLIT_DIGEST_FIELD = "sha256"
MARKER_VERSION = 2


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# canonical digests
# --------------------------------------------------------------------------- #
def canonical_split_preimage(payload: dict[str, Any]) -> bytes:
    """Return the exact byte string a split manifest's SHA-256 is taken over.

    The digest covers the manifest without its own digest field, so a verifier
    can reproduce it from the fold payload alone.
    """
    body = {key: value for key, value in payload.items() if key != SPLIT_DIGEST_FIELD}
    return json.dumps(body, sort_keys=True).encode("utf-8")


def split_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_split_preimage(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# atomic writes
# --------------------------------------------------------------------------- #
def atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Replace ``path`` with ``data`` in one step, or leave the old file intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_text(path: Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_move(source: Path, destination: Path) -> Path:
    """Move a staged file onto its final path atomically within the same root."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(Path(source), destination)
    return destination


# --------------------------------------------------------------------------- #
# staging
# --------------------------------------------------------------------------- #
@dataclass
class DatasetStage:
    """Per-dataset staging area; nothing here is visible to a validator yet."""

    dataset_id: str
    component: str
    root: Path
    oof_dir: Path = field(init=False)
    fail_dir: Path = field(init=False)
    rows_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.oof_dir = self.root / "oof"
        self.fail_dir = self.root / "failures"
        self.rows_path = self.root / "rows.jsonl"

    def create(self) -> "DatasetStage":
        self.oof_dir.mkdir(parents=True, exist_ok=True)
        self.fail_dir.mkdir(parents=True, exist_ok=True)
        return self

    def write_rows(self, rows: Iterable[dict[str, Any]]) -> Path:
        text = "".join(json.dumps(row) + "\n" for row in rows)
        return atomic_write_text(self.rows_path, text)

    def discard(self) -> list[str]:
        """Remove the staging tree, tolerating Windows' delayed directory deletion.

        Windows can report a just-emptied directory as non-empty for a short
        while, so a single ``rmtree`` leaves empty husks behind. Silently
        ignoring that hides genuine orphans, so persistent failures are returned
        rather than swallowed.
        """
        return remove_tree(self.root)


def _clear_readonly(function, path, excinfo):  # noqa: ARG001 - shutil.rmtree callback
    try:
        os.chmod(path, stat.S_IWRITE)
        function(path)
    except OSError:
        pass


def remove_tree(path: Path, attempts: int = 6, pause: float = 0.1) -> list[str]:
    """Delete a directory tree, retrying through Windows' transient denials.

    A staging directory holds no evidence once its artifacts are promoted, so a
    residual empty husk is cosmetic. It is still returned rather than swallowed:
    an undeleted staging tree that turns out to be non-empty is a real signal.
    """
    path = Path(path)
    problems: list[str] = []
    for attempt in range(attempts):
        if not path.exists():
            return []
        try:
            shutil.rmtree(path, onexc=_clear_readonly)
            if not path.exists():
                return []
            problems = ["directory still present after removal"]
        except OSError as exc:
            problems = [f"{type(exc).__name__}: {exc}"]
        time.sleep(pause * (attempt + 1))
    if not path.exists():
        return []
    residue = sorted(item.name for item in path.rglob("*") if item.is_file())
    return problems + ([f"residual files: {residue[:5]}"] if residue else [])


def quarantine(stage_root: Path, quarantine_root: Path, reason: str) -> Path | None:
    """Retain an orphaned staging directory instead of deleting it, and record why."""
    stage_root, quarantine_root = Path(stage_root), Path(quarantine_root)
    if not stage_root.exists():
        return None
    quarantine_root.mkdir(parents=True, exist_ok=True)
    stamp = utcnow().replace(":", "").replace("-", "").replace(".", "")
    target = quarantine_root / f"{stage_root.name}__{stamp}"
    index = 1
    while target.exists():
        target = quarantine_root / f"{stage_root.name}__{stamp}__{index}"
        index += 1
    shutil.move(str(stage_root), str(target))
    with (quarantine_root / "index.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"quarantined_utc": utcnow(), "path": target.name,
                                 "reason": reason}) + "\n")
    return target


# --------------------------------------------------------------------------- #
# completion markers
# --------------------------------------------------------------------------- #
def marker_path(work_dir: Path, component: str) -> Path:
    return Path(work_dir) / "complete" / f"{component}.json"


def write_marker(path: Path, payload: dict[str, Any]) -> Path:
    """Write the completion marker last, after every artifact is already final."""
    return atomic_write_text(path, json.dumps(payload, indent=1, sort_keys=True) + "\n")


def verify_marker(marker: Path, root: Path, task_index_sha256: str,
                  expected_keys: set[tuple[Any, ...]] | None = None) -> list[str]:
    """Return the reasons a dataset must be re-run; an empty list means complete.

    Verification is deliberately exhaustive rather than existence-based: a marker
    that survives an interruption without its artifacts must not certify a dataset.
    """
    marker, root = Path(marker), Path(root)
    if not marker.is_file():
        return ["completion marker is absent"]
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"completion marker is unreadable: {type(exc).__name__}: {exc}"]
    if not isinstance(payload, dict):
        return ["completion marker is not an object"]

    problems: list[str] = []
    if payload.get("marker_version") != MARKER_VERSION:
        return [f"completion marker version {payload.get('marker_version')!r} is not "
                f"{MARKER_VERSION}; the dataset predates the current contract"]
    if payload.get("task_index_sha256") != task_index_sha256:
        return ["completion marker was written against a different task index"]

    rows = payload.get("rows")
    if not isinstance(rows, dict):
        return ["completion marker has no rows record"]
    rows_path = root / str(rows.get("path", ""))
    if not rows_path.is_file():
        problems.append(f"promoted row file is missing: {rows.get('path')!r}")
    else:
        if sha256_file(rows_path) != rows.get("sha256"):
            problems.append("promoted row file does not match its recorded SHA-256")
        observed = sum(1 for line in rows_path.read_text(encoding="utf-8").splitlines()
                       if line.strip())
        if observed != rows.get("count"):
            problems.append(f"promoted row count {observed} != recorded {rows.get('count')!r}")

    for group in ("artifacts", "failures"):
        listed = payload.get(group)
        if not isinstance(listed, dict):
            problems.append(f"completion marker has no {group} record")
            continue
        for relative, digest in listed.items():
            path = root / str(relative)
            if not path.is_file():
                problems.append(f"recorded {group[:-1]} is missing: {relative}")
            elif sha256_file(path) != digest:
                problems.append(f"recorded {group[:-1]} does not match its SHA-256: {relative}")

    if expected_keys is not None:
        covered = {tuple(key) for key in payload.get("covered_task_keys", [])}
        missing = expected_keys - covered
        if missing:
            problems.append(f"{len(missing)} expected task(s) are not covered by the marker")
    return problems
