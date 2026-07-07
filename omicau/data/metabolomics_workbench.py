"""Metabolomics Workbench client.

Pulls a study's metabolite measurement matrix and per-sample factors from the
public Metabolomics Workbench REST API (no auth) and assembles an omicau dataset
with metabolomics as the omic modality and a study factor as the target. This
adds a distinct omic layer (metabolomics) to omicau's data sources.

REST base https://www.metabolomicsworkbench.org/rest/ verified live as of
2026-07. Study ids use the ``ST######`` scheme.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from omicau.data._hub import get_json, require_requests, validate_matrix, write_dataset

BASE = "https://www.metabolomicsworkbench.org/rest"


def _records(obj) -> list[dict]:
    """The REST API returns either a dict-keyed-by-index or a bare list."""
    if isinstance(obj, dict):
        # a single record is a flat dict; many records are {"1": {...}, "2": {...}}
        if obj and all(isinstance(v, dict) for v in obj.values()) and "DATA" not in obj:
            return list(obj.values())
        return [obj]
    return list(obj)


def fetch_matrix(study_id: str, session=None) -> pd.DataFrame:
    """Build a samples x metabolites matrix from the study's measurement data."""
    session = session or require_requests().Session()
    data = _records(get_json(f"{BASE}/study/study_id/{study_id}/data", session=session))
    cols: dict[str, dict] = {}
    for entry in data:
        name = str(entry.get("metabolite_name") or entry.get("metabolite_id"))
        values = entry.get("DATA", {})
        if isinstance(values, dict) and values:
            cols[name] = {str(k): v for k, v in values.items()}
    frame = pd.DataFrame(cols)  # index = sample ids, columns = metabolites
    frame.index = frame.index.astype(str)
    return frame


def fetch_factors(study_id: str, session=None) -> pd.DataFrame:
    """Per-sample factor table (parsed from the 'factors' string), indexed by sample."""
    session = session or require_requests().Session()
    fac = _records(get_json(f"{BASE}/study/study_id/{study_id}/factors", session=session))
    rows = []
    for f in fac:
        sample = str(f.get("local_sample_id"))
        parsed = {"sample_id": sample}
        for part in str(f.get("factors", "")).split("|"):
            if ":" in part:
                k, v = part.split(":", 1)
                parsed[k.strip()] = v.strip()
        rows.append(parsed)
    df = pd.DataFrame(rows).set_index("sample_id")
    return df


def _default_target(factors: pd.DataFrame) -> str | None:
    """Pick a factor with 2-10 balanced categories as a usable classification target."""
    best, best_n = None, 0
    for c in factors.columns:
        n = factors[c].nunique(dropna=True)
        if 2 <= n <= 10 and factors[c].notna().sum() > best_n:
            best, best_n = c, factors[c].notna().sum()
    return best


def prepare(out_dir: str | Path, *, study_id: str = "ST000009", target: str | None = None,
            session=None) -> Path:
    """Assemble a Metabolomics Workbench study into an omicau dataset.

    Modality: metabolomics (samples x metabolites). Target: a study factor
    (auto-selected if not given). Returns the ``config.json`` path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    session = session or require_requests().Session()

    mat = fetch_matrix(study_id, session)
    factors = fetch_factors(study_id, session)
    if mat.empty:
        raise RuntimeError(f"No measurement data returned for study {study_id}.")

    target = target or _default_target(factors)
    if not target or target not in factors.columns:
        raise ValueError(f"No usable target factor for {study_id}. "
                         f"Available factors: {list(factors.columns)}")

    common = sorted(set(mat.index) & set(factors.index))
    if not common:
        raise RuntimeError("No overlap between metabolite samples and factor samples.")

    clean, rep = validate_matrix(mat.loc[common], name="metabolomics")
    clin = factors.loc[common, [target]].reset_index()
    clin.columns = ["sample_id", target]

    return write_dataset(out, {"metabolomics": clean}, clin, sample_col="sample_id",
                         target=target, run_name=f"metabolomics_{study_id}",
                         source=f"MetabolomicsWorkbench:{study_id}", task="classification",
                         reports={"metabolomics": rep})
