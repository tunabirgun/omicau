from __future__ import annotations

import argparse
import base64
import copy
import csv
import gzip
import hashlib
from pathlib import Path
import tempfile
import tarfile
import zipfile


def _record_digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={encoded.decode('ascii')}"


def normalize(path: Path) -> None:
    with zipfile.ZipFile(path) as source:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise ValueError("wheel contains duplicate member names")
        wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
        record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(wheel_names) != 1 or len(record_names) != 1:
            raise ValueError("wheel must contain one WHEEL file and one RECORD file")
        wheel_name, record_name = wheel_names[0], record_names[0]
        wheel = source.read(wheel_name)
        cleaned = b"\n".join(
            line for line in wheel.splitlines() if not line.startswith(b"Generator:")
        ) + b"\n"
        if cleaned == wheel:
            return
        rows = list(csv.reader(source.read(record_name).decode("utf-8").splitlines()))
        updated_rows = []
        found_wheel = False
        for row in rows:
            if len(row) != 3:
                raise ValueError("wheel RECORD contains an invalid row")
            if row[0] == wheel_name:
                row = [wheel_name, _record_digest(cleaned), str(len(cleaned))]
                found_wheel = True
            updated_rows.append(row)
        if not found_wheel:
            raise ValueError("wheel RECORD does not bind its WHEEL metadata")
        record = "\n".join(",".join(row) for row in updated_rows) + "\n"
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with zipfile.ZipFile(temporary, "w") as destination:
                for info in source.infolist():
                    data = source.read(info.filename)
                    if info.filename == wheel_name:
                        data = cleaned
                    elif info.filename == record_name:
                        data = record.encode("utf-8")
                    destination.writestr(info, data)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def normalize_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as source:
        members = source.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("sdist contains duplicate member names")
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with temporary.open("wb") as output:
                with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as destination:
                        for member in members:
                            normalized = copy.copy(member)
                            normalized.uid = normalized.gid = 0
                            normalized.uname = normalized.gname = ""
                            normalized.mtime = 0
                            data = source.extractfile(member) if member.isfile() else None
                            destination.addfile(normalized, data)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("--sdist", action="append", default=[], type=Path)
    args = parser.parse_args()
    for wheel in args.wheels:
        normalize(wheel)
    for sdist in args.sdist:
        normalize_sdist(sdist)


if __name__ == "__main__":
    main()
