"""Shared infrastructure for the remote data-hub clients.

Network access is optional and isolated here: exponential-backoff-with-jitter
retries, a local download cache, and runtime structural gates that verify a
downloaded matrix is numeric, finite, and sample-indexed before it enters the
pipeline. ``requests`` is imported lazily so the core package never depends on
it. None of this is exercised by the offline test suite.
"""

from __future__ import annotations

import functools
import hashlib
import os
import random
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

_TRANSIENT = ("Connection", "Timeout", "Chunked", "Temporary", "Again",
              "TooManyRedirects", "IncompleteRead", "RemoteDisconnected")


def require_requests():
    """Return the ``requests`` module or raise a clear, actionable error."""
    try:
        import requests  # type: ignore
        return requests
    except ImportError as exc:  # pragma: no cover - optional dep
        from omicau._hints import extra_hint
        raise ImportError(
            "Remote data hubs need the optional 'requests' dependency: "
            f"{extra_hint('data')}. The core pipeline runs fully offline without it."
        ) from exc


def default_cache_dir() -> Path:
    """Cache root (override with the OMICAU_CACHE environment variable)."""
    root = os.environ.get("OMICAU_CACHE")
    path = Path(root) if root else Path.home() / ".omicau_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def retry_backoff(retries: int = 5, base: float = 1.0, cap: float = 30.0):
    """Decorator: retry transient network failures with jittered backoff."""
    def deco(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last = None
            for attempt in range(retries):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - classify by type name
                    last = exc
                    name = type(exc).__name__
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    transient = any(k in name for k in _TRANSIENT) or (status and status >= 500) or status == 429
                    if not transient or attempt == retries - 1:
                        raise
                    sleep = min(cap, base * (2 ** attempt)) + random.uniform(0, base)
                    time.sleep(sleep)
            raise last  # pragma: no cover
        return wrapper
    return deco


@retry_backoff()
def download_file(url: str, dest: Path, *, chunk: int = 1 << 20,
                  session=None, expected_md5: str | None = None,
                  force: bool = False) -> Path:
    """Stream ``url`` to ``dest`` with retries and an optional md5 gate (cached)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force and dest.stat().st_size > 0:
        if expected_md5 is None or _md5(dest) == expected_md5:
            return dest
    requests = require_requests()
    sess = session or requests.Session()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with sess.get(url, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in r.iter_content(chunk_size=chunk):
                if block:
                    fh.write(block)
    if expected_md5 is not None and _md5(tmp) != expected_md5:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"md5 mismatch for {url}")
    tmp.replace(dest)
    return dest


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@retry_backoff()
def get_json(url: str, *, params: dict | None = None, session=None) -> Any:
    requests = require_requests()
    sess = session or requests.Session()
    r = sess.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


@retry_backoff()
def post_json(url: str, *, json_body: Any, params: dict | None = None, session=None) -> Any:
    requests = require_requests()
    sess = session or requests.Session()
    r = sess.post(url, json=json_body, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- #
# Runtime structural gates
# --------------------------------------------------------------------------- #
def validate_matrix(frame: pd.DataFrame, name: str = "matrix") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Coerce to numeric, neutralize non-finite values, and drop dead features.

    Enforces the rigorous-validation rule: numeric-only allocations, no infinite
    variance, and string sample identifiers. Returns the cleaned frame plus a
    report of what was healed.
    """
    report: dict[str, Any] = {"name": name, "input_shape": list(frame.shape)}
    frame = frame.copy()
    frame.index = pd.Index([str(i).strip() for i in frame.index])

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    n_inf = int(np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan)).sum())
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    before_cols = numeric.shape[1]
    numeric = numeric.dropna(axis=1, how="all").dropna(axis=0, how="all")
    # infinite-variance / zero-variance tracking: drop constant columns.
    variances = numeric.var(axis=0, skipna=True)
    keep = variances[variances > 0].index
    dropped_const = before_cols - len(keep)
    numeric = numeric[keep]

    report.update({
        "output_shape": list(numeric.shape),
        "non_finite_healed": n_inf,
        "constant_features_dropped": int(dropped_const),
    })
    if numeric.shape[1] == 0:
        warnings.warn(f"validate_matrix({name}): no numeric variable features remain.", stacklevel=2)
    return numeric.astype("float64"), report


def match_samples(frames: dict[str, pd.DataFrame], target_index) -> list[str]:
    """Sample-extension match: intersect frame indices with the target labels."""
    common = set(str(i) for i in target_index)
    for f in frames.values():
        common &= set(str(i) for i in f.index)
    return sorted(common)


PROVENANCE_NOTE = (
    "Endpoints verified against public documentation as of 2026-07; open-access "
    "release identifiers can move. Re-verify before production use."
)


def write_dataset(out: Path, modalities: dict[str, pd.DataFrame], clinical: pd.DataFrame,
                  *, sample_col: str, target: str, run_name: str, source: str,
                  group: str | None = None, batch: str | None = None,
                  task: str = "auto", organism: str = "unspecified",
                  normalization_preset: str | None = None,
                  reports: dict | None = None) -> Path:
    """Emit modality CSVs, a clinical CSV, and a ready-to-run ``config.json``.

    Returns the path to the config. This is the common landing format every hub
    client produces so that ``omicau run --config <path>`` works uniformly.
    """
    import json

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mod_specs = []
    for name, frame in modalities.items():
        p = out / f"{name}.csv"
        frame.to_csv(p, index=True, index_label=sample_col, lineterminator="\n")
        mod_specs.append({"name": name, "path": p.name, "description": f"{source} {name}"})
    clin_path = out / "clinical.csv"
    clinical.to_csv(clin_path, index=False, lineterminator="\n")

    clinical_spec: dict[str, Any] = {
        "path": clin_path.name, "target": target, "sample_id": sample_col, "task": task,
    }
    if group:
        clinical_spec["group"] = group
    if batch:
        clinical_spec["batch"] = batch

    config = {
        "run_name": run_name,
        "output_dir": "run",  # relative to this config's directory
        "organism": organism,
        "modalities": mod_specs,
        "clinical": clinical_spec,
        "cv": {"n_splits": 5, "seed": 42},
        "_provenance_note": PROVENANCE_NOTE,
        "_validation": reports or {},
    }
    if normalization_preset:
        config["normalization"] = {"preset": normalization_preset}
    cfg_path = out / "config.json"
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8", newline="")
    return cfg_path
