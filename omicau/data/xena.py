"""UCSC Xena multi-omics hub client.

UCSC Xena serves flat, tab-delimited omic matrices and separate phenotype/clinical
matrices over public HTTPS with no authentication, across many cohorts (TCGA,
TARGET, GTEx, PCAWG, CCLE, Treehouse, ...). A modality matrix is downloaded from
``https://<hub>/download/<dataset>.gz`` (302-redirects to S3, followed
automatically), gunzipped, and read as TSV. Xena genomic matrices are stored
features-as-rows x samples-as-columns, so they are transposed to samples x
features. Phenotype matrices are already samples-as-rows and carry usable targets
(subtype, stage, survival, tissue).

Verified live against the public Xena hubs as of 2026-07; dataset ids are
hub-specific -- discover them with :func:`query_datasets`.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import pandas as pd

from omicau.data._hub import (
    default_cache_dir,
    require_requests,
    validate_matrix,
    write_dataset,
    retry_backoff,
)

HUBS = {
    "tcga": "https://tcga.xenahubs.net",
    "gdc": "https://gdc.xenahubs.net",
    "toil": "https://toil.xenahubs.net",
    "pancanatlas": "https://pancanatlas.xenahubs.net",
    "pcawg": "https://pcawg.xenahubs.net",
    "ucscpublic": "https://ucscpublic.xenahubs.net",
}

# A ready-to-run preset: TCGA breast cancer expression + PAM50 subtype target.
PRESETS = {
    "brca": {
        "hub": "tcga",
        "modalities": {"rna": "TCGA.BRCA.sampleMap/HiSeqV2"},
        "phenotype": "TCGA.BRCA.sampleMap/BRCA_clinicalMatrix",
        "target": "PAM50Call_RNAseq",
        "run_name": "xena_tcga_brca",
    },
}


def _cache() -> Path:
    d = default_cache_dir() / "xena"
    d.mkdir(parents=True, exist_ok=True)
    return d


@retry_backoff()
def query_datasets(hub: str = "tcga", keyword: str | None = None, session=None) -> list[str]:
    """List dataset names in a hub (optionally filtered by a name keyword)."""
    base = HUBS.get(hub, hub)
    requests = require_requests()
    sess = session or requests.Session()
    where = f' :where [:like :dataset.name "%{keyword}%"]' if keyword else ""
    q = f"(map :name (query {{:select [:dataset.name] :from [:dataset]{where}}}))"
    r = sess.post(base + "/data/", data=q, headers={"Content-Type": "text/plain"}, timeout=60)
    r.raise_for_status()
    return r.json()


@retry_backoff()
def _download_tsv(hub: str, dataset: str, session) -> pd.DataFrame:
    base = HUBS.get(hub, hub)
    for url in (f"{base}/download/{dataset}.gz", f"{base}/download/{dataset}"):
        r = session.get(url, timeout=300)
        if r.status_code != 200:
            continue
        raw = r.content
        if raw[:2] == b"\x1f\x8b":  # gzip magic
            raw = gzip.decompress(raw)
        return pd.read_csv(io.BytesIO(raw), sep="\t", index_col=0, low_memory=False)
    raise RuntimeError(f"Could not download Xena dataset '{dataset}' from hub '{hub}'.")


def fetch_matrix(hub: str, dataset: str, session=None) -> pd.DataFrame:
    """Fetch a genomic matrix and return it samples x features (transposed)."""
    session = session or require_requests().Session()
    df = _download_tsv(hub, dataset, session)
    return df.T  # Xena stores features x samples


def fetch_phenotype(hub: str, dataset: str, session=None) -> pd.DataFrame:
    """Fetch a phenotype/clinical matrix (already samples x fields)."""
    session = session or require_requests().Session()
    return _download_tsv(hub, dataset, session)


def prepare(out_dir: str | Path, *, preset: str | None = "brca", hub: str | None = None,
            modalities: dict[str, str] | None = None, phenotype: str | None = None,
            target: str | None = None, run_name: str | None = None, session=None) -> Path:
    """Assemble a Xena cohort into an omicau-ready dataset.

    Use a ``preset`` (default ``brca``) or pass ``hub`` + ``modalities`` (a dict of
    modality name -> Xena dataset id) + ``phenotype`` + ``target`` explicitly.
    Returns the ``config.json`` path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    session = session or require_requests().Session()

    if modalities is None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown Xena preset '{preset}'. Options: {list(PRESETS)}, "
                             "or pass hub/modalities/phenotype/target explicitly.")
        p = PRESETS[preset]
        hub, modalities, phenotype = p["hub"], p["modalities"], p["phenotype"]
        target = target or p["target"]
        run_name = run_name or p["run_name"]
    run_name = run_name or f"xena_{hub}"

    pheno = fetch_phenotype(hub, phenotype, session)
    pheno.index = pheno.index.astype(str)
    if target not in pheno.columns:
        raise ValueError(f"Target '{target}' not in phenotype (fields: {list(pheno.columns)[:15]}).")

    aligned_mods: dict[str, pd.DataFrame] = {}
    reports = {}
    for name, dataset in modalities.items():
        mat = fetch_matrix(hub, dataset, session)
        mat.index = mat.index.astype(str)
        clean, rep = validate_matrix(mat, name=name)
        aligned_mods[name] = clean
        reports[name] = rep

    common = set(pheno.index)
    for m in aligned_mods.values():
        common &= set(m.index)
    common = sorted(common)
    if not common:
        raise RuntimeError("No overlapping samples between Xena matrices and phenotype.")

    aligned_mods = {n: m.loc[m.index.intersection(common)] for n, m in aligned_mods.items()}
    clin = pheno.loc[common, [target]].reset_index()
    clin.columns = ["sampleID", target]

    return write_dataset(out, aligned_mods, clin, sample_col="sampleID", target=target,
                         run_name=run_name, source=f"UCSC-Xena:{hub}", task="auto", reports=reports)
