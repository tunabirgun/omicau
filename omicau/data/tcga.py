"""TCGA downloader via the cBioPortal public REST API.

Pulls open-access adult-cancer matrices (mRNA expression, and copy-number when
available) plus a clinical table for a public study, with no authentication. The
cBioPortal molecular-data endpoint is panel-based (it requires an explicit gene
list), so a curated cancer-gene panel is used by default. Responses are long/tidy
and pivoted to samples x features locally, then passed through the structural
gate before being written as an omicau-ready dataset.

Endpoints verified against https://www.cbioportal.org/api documentation as of
2026-07; study / profile identifiers can change -- re-verify before production.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from omicau.data._hub import (
    get_json,
    post_json,
    require_requests,
    validate_matrix,
    write_dataset,
)

API = "https://www.cbioportal.org/api"

#: A compact, well-characterized pan-cancer gene panel (HUGO symbols). The API
#: needs an explicit gene list; this keeps requests bounded and reproducible.
DEFAULT_PANEL = [
    "TP53", "EGFR", "KRAS", "NRAS", "BRAF", "PIK3CA", "PTEN", "RB1", "MYC", "CDKN2A",
    "APC", "VHL", "KIT", "FLT3", "NPM1", "IDH1", "IDH2", "DNMT3A", "TET2", "RUNX1",
    "ERBB2", "MET", "ALK", "ROS1", "CTNNB1", "SMAD4", "STK11", "KEAP1", "NF1", "ATM",
    "BRCA1", "BRCA2", "PALB2", "CDH1", "GATA3", "ESR1", "AR", "FOXA1", "MDM2", "CCND1",
    "CDK4", "CDK6", "BCL2", "JAK2", "STAT3", "NOTCH1", "FBXW7", "ARID1A", "KMT2D", "CREBBP",
    "EP300", "SMARCA4", "SETD2", "BAP1", "PBRM1", "KDM6A", "ASXL1", "EZH2", "SF3B1", "U2AF1",
]

# Candidate molecular profiles per study (mRNA, then copy-number).
_PROFILE_SUFFIXES = {
    "rna": ["_rna_seq_v2_mrna", "_rna_seq_mrna", "_mrna"],
    "cna": ["_gistic", "_cna", "_linear_CNA"],
}


def _session():
    return require_requests().Session()


def list_studies(keyword: str | None = None, session=None) -> pd.DataFrame:
    """List public studies (optionally filtered by keyword)."""
    params = {"pageSize": 100000, "direction": "ASC"}
    if keyword:
        params["keyword"] = keyword
    data = get_json(f"{API}/studies", params=params, session=session or _session())
    return pd.DataFrame(data)[["studyId", "name", "allSampleCount"]]


def _first_available_profile(study: str, kind: str, session) -> str | None:
    profiles = get_json(f"{API}/studies/{study}/molecular-profiles", session=session)
    ids = {p["molecularProfileId"] for p in profiles}
    for suffix in _PROFILE_SUFFIXES[kind]:
        cand = f"{study}{suffix}"
        if cand in ids:
            return cand
    return None


def _entrez_ids(symbols: list[str], session) -> list[int]:
    data = post_json(f"{API}/genes/fetch", params={"geneIdType": "HUGO_GENE_SYMBOL"},
                     json_body=symbols, session=session)
    return [g["entrezGeneId"] for g in data]


def _sample_ids(study: str, profile_id: str, session) -> list[str]:
    for list_id in (profile_id, f"{study}_all"):
        try:
            ids = get_json(f"{API}/sample-lists/{list_id}/sample-ids", session=session)
            if ids:
                return list(ids)
        except Exception:  # noqa: BLE001 - fall through to next candidate
            continue
    return []


def fetch_molecular(profile_id: str, entrez_ids: list[int], sample_ids: list[str],
                    session) -> pd.DataFrame:
    """Fetch one molecular profile and pivot to a samples x genes frame."""
    body = {"entrezGeneIds": entrez_ids, "sampleIds": sample_ids}
    data = post_json(f"{API}/molecular-profiles/{profile_id}/molecular-data/fetch",
                     params={"projection": "DETAILED"}, json_body=body, session=session)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame([
        {"sampleId": r["sampleId"],
         "gene": r.get("gene", {}).get("hugoGeneSymbol", str(r["entrezGeneId"])),
         "value": r.get("value")}
        for r in data
    ])
    wide = df.pivot_table(index="sampleId", columns="gene", values="value", aggfunc="mean")
    return wide


def fetch_clinical(study: str, session) -> pd.DataFrame:
    """Fetch SAMPLE-level clinical data and pivot to one row per sample."""
    params = {"clinicalDataType": "SAMPLE", "projection": "DETAILED",
              "pageSize": 10_000_000, "pageNumber": 0}
    data = get_json(f"{API}/studies/{study}/clinical-data", params=params, session=session)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame([
        {"sampleId": r["sampleId"], "attr": r["clinicalAttributeId"], "value": r["value"]}
        for r in data
    ])
    return df.pivot_table(index="sampleId", columns="attr", values="value", aggfunc="first")


def prepare(out_dir: str | Path, *, study: str = "laml_tcga", target: str | None = None,
            genes: list[str] | None = None, session=None) -> Path:
    """Download a TCGA study and write an omicau-ready dataset + config.

    Returns the path to the generated ``config.json``. ``target`` names the
    clinical column to predict; if omitted, a sensible default is chosen and the
    user is expected to confirm it.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    session = session or _session()
    genes = genes or DEFAULT_PANEL

    clinical = fetch_clinical(study, session)
    if clinical.empty:
        raise RuntimeError(f"No clinical data returned for study '{study}'.")
    sample_ids = list(clinical.index.astype(str))

    entrez = _entrez_ids(genes, session)
    modalities: dict[str, pd.DataFrame] = {}
    reports: dict[str, Any] = {}
    for kind in ("rna", "cna"):
        profile = _first_available_profile(study, kind, session)
        if not profile:
            continue
        wide = fetch_molecular(profile, entrez, sample_ids, session)
        if wide.empty:
            continue
        clean, rep = validate_matrix(wide, name=kind)
        if clean.shape[1] == 0:
            continue
        modalities[kind] = clean
        reports[kind] = rep

    if not modalities:
        raise RuntimeError(f"No molecular profiles could be fetched for '{study}'.")

    # Choose a target column if not provided.
    if target is None:
        preferred = ["OS_STATUS", "SUBTYPE", "CANCER_TYPE_DETAILED", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE"]
        target = next((c for c in preferred if c in clinical.columns), None)
        if target is None:
            target = clinical.columns[0]

    clin_out = clinical.reset_index().rename(columns={"index": "sampleId"})
    return write_dataset(out, modalities, clin_out, sample_col="sampleId", target=target,
                         run_name=f"tcga_{study}", source=f"cBioPortal:{study}", reports=reports)
