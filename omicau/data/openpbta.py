"""OpenPBTA / OpenPedCan pediatric neuro-oncology loader (public AWS S3).

Streams open-access matrices over plain anonymous HTTPS from the public S3 bucket
(no AWS credentials, no SDK). The harmonized gene-expression matrices in the
release are distributed only as ``.rds`` (R serialization), which is not reliably
readable from Python; this client therefore builds modeling-ready modalities from
the Python-friendly TSV files -- a binary putative-fusion matrix -- against the
harmonized ``histologies.tsv`` clinical table. ``list_release`` / ``download_file``
expose the full release for power users who convert the RDS matrices in R.

Bucket / release verified as of 2026-07: bucket ``d3b-openaccess-us-east-1-prd-pbta``
(``prd``, not ``prod``); OpenPedCan release ``open-targets/v15``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from omicau.data._hub import (
    default_cache_dir,
    download_file as _download_file,
    require_requests,
    validate_matrix,
    write_dataset,
)

BUCKET = "d3b-openaccess-us-east-1-prd-pbta"
BASE = f"https://{BUCKET}.s3.amazonaws.com"
RELEASE = "open-targets/v15"

# Python-friendly TSV files in the release.
HISTOLOGIES = "histologies.tsv"
FUSIONS = "fusion-putative-oncogenic.tsv"


def _cache() -> Path:
    d = default_cache_dir() / "openpbta"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_release(prefix: str = RELEASE, session=None) -> pd.DataFrame:
    """Anonymously list objects (key + size) under a release prefix via S3 REST."""
    requests = require_requests()
    sess = session or requests.Session()
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix + "/", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = sess.get(BASE + "/", params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        ns = {"s3": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
        find = (lambda el, tag: el.find(f"s3:{tag}", ns) if ns else el.find(tag))
        for c in (root.findall("s3:Contents", ns) if ns else root.findall("Contents")):
            keys.append({"key": find(c, "Key").text, "size": int(find(c, "Size").text)})
        trunc = find(root, "IsTruncated")
        if trunc is not None and trunc.text == "true":
            nt = find(root, "NextContinuationToken")
            token = nt.text if nt is not None else None
            if not token:
                break
        else:
            break
    return pd.DataFrame(keys)


def download_file(filename: str, session=None) -> Path:
    """Download a single release file to the local cache and return its path."""
    url = f"{BASE}/{RELEASE}/{filename}"
    dest = _cache() / filename
    return _download_file(url, dest, session=session or require_requests().Session())


def load_histologies(session=None) -> pd.DataFrame:
    """Load the harmonized clinical / histologies master table."""
    path = download_file(HISTOLOGIES, session)
    return pd.read_csv(path, sep="\t", low_memory=False)


def load_fusions(session=None) -> pd.DataFrame:
    """Load the recommended filtered putative-oncogenic fusion list."""
    path = download_file(FUSIONS, session)
    return pd.read_csv(path, sep="\t", low_memory=False)


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _fusion_matrix(fusions: pd.DataFrame, min_recurrence: int = 5) -> pd.DataFrame:
    """Binary samples x fused-gene matrix from the fusion list (recurrent genes)."""
    sample_col = _pick_column(fusions, ["Sample", "Kids_First_Biospecimen_ID",
                                        "BS_ID", "sample_id"])
    name_col = _pick_column(fusions, ["FusionName", "Fusion_Name", "fusion_name"])
    if sample_col is None or name_col is None:
        raise RuntimeError("Could not locate sample/fusion-name columns in the fusion file.")
    rows = []
    for _, r in fusions[[sample_col, name_col]].dropna().iterrows():
        genes = str(r[name_col]).replace("::", "--").split("--")
        for g in genes:
            g = g.strip()
            if g:
                rows.append((str(r[sample_col]), g))
    long = pd.DataFrame(rows, columns=["sample", "gene"]).drop_duplicates()
    mat = pd.crosstab(long["sample"], long["gene"]).clip(upper=1)
    recurrent = mat.columns[mat.sum(axis=0) >= min_recurrence]
    return mat[recurrent].astype(float)


def prepare(out_dir: str | Path, *, target_column: str = "broad_histology",
            min_recurrence: int = 5, session=None) -> Path:
    """Write an OpenPBTA fusion-presence dataset with a histology target.

    Builds a binary putative-fusion modality keyed by biospecimen and joins the
    harmonized histology table for the prediction target. Returns the
    ``config.json`` path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    session = session or require_requests().Session()

    hist = load_histologies(session)
    fusions = load_fusions(session)
    fusion_mat = _fusion_matrix(fusions, min_recurrence=min_recurrence)

    key = _pick_column(hist, ["Kids_First_Biospecimen_ID", "Sample", "sample_id"])
    if key is None or target_column not in hist.columns:
        raise RuntimeError(
            f"histologies.tsv missing key or target column '{target_column}'. "
            f"Available columns include: {list(hist.columns[:12])}"
        )
    hist_idx = hist.set_index(key)
    common = sorted(set(fusion_mat.index) & set(hist_idx.index.astype(str)))
    if not common:
        raise RuntimeError("No overlap between fusion samples and histology biospecimens.")

    clean, rep = validate_matrix(fusion_mat.loc[common], name="fusions")
    modalities = {"fusions": clean}
    clinical = pd.DataFrame({
        "biospecimen": common,
        target_column: [hist_idx.loc[s, target_column] if s in hist_idx.index else None
                        for s in common],
    }).dropna(subset=[target_column])

    return write_dataset(
        out, modalities, clinical, sample_col="biospecimen", target=target_column,
        run_name="openpbta_fusions", source=f"OpenPBTA/{RELEASE}",
        task="classification", reports={"fusions": rep},
    )
