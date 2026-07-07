"""CCLE / DepMap cell-line benchmark loader.

Downloads fully public, no-auth CCLE/DepMap matrices from figshare (DepMap Public
24Q4 -- the last release with a stable figshare bundle) and maps them to a
continuous target vector (a CRISPR gene-effect / dependency score for a chosen
gene, or a drug sensitivity value). Matrices are cell-lines x features indexed by
DepMap ModelID (ACH-######). Files are cached locally; figshare ndownloader URLs
302-redirect to signed S3, which the streaming downloader follows.

File identifiers verified for DepMap Public 24Q4 (figshare article 27993248) as
of 2026-07; DepMap release cadence moves -- re-verify before production.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from omicau.data._hub import (
    default_cache_dir,
    download_file,
    require_requests,
    validate_matrix,
    write_dataset,
)

FIGSHARE = "https://ndownloader.figshare.com/files/{file_id}"

# DepMap Public 24Q4 figshare file ids (cell-lines x features, ModelID index).
FILES_24Q4 = {
    "expression": "51065489",   # OmicsExpressionProteinCodingGenesTPMLogp1.csv (log2 TPM+1)
    "crispr_effect": "51064667",  # CRISPRGeneEffect.csv (Chronos gene effect)
    "crispr_dependency": "51064631",  # CRISPRGeneDependency.csv (0-1)
    "model": "51065297",        # Model.csv (ID crosswalk / metadata)
}
RELEASE = "DepMap Public 24Q4"


def _cache(sub: str = "ccle") -> Path:
    d = default_cache_dir() / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(file_key: str, session) -> Path:
    fid = FILES_24Q4[file_key]
    dest = _cache() / f"{file_key}_{fid}.csv"
    return download_file(FIGSHARE.format(file_id=fid), dest, session=session)


def _read_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.Index([str(i).strip() for i in df.index])
    return df


def load_expression(session=None) -> pd.DataFrame:
    """Cell-lines x genes log2(TPM+1) expression (already logged; do not re-log)."""
    session = session or require_requests().Session()
    return _read_matrix(_download("expression", session))


def load_crispr_effect(session=None) -> pd.DataFrame:
    """Cell-lines x genes Chronos CRISPR gene-effect scores (target source)."""
    session = session or require_requests().Session()
    return _read_matrix(_download("crispr_effect", session))


def _clean_gene_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip the trailing '(EntrezID)' from 'SYMBOL (1234)' column labels."""
    df = df.copy()
    df.columns = [str(c).split(" (")[0].strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def prepare(out_dir: str | Path, *, target_gene: str = "SOX10", session=None) -> Path:
    """Assemble a CCLE multi-omic benchmark with a CRISPR-dependency target.

    Modalities: RNA-seq expression (and, when the file is present, additional
    layers). Target: the CRISPR gene-effect score of ``target_gene`` across cell
    lines (a continuous dependency vector). Returns the ``config.json`` path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    session = session or require_requests().Session()

    expr = _clean_gene_columns(load_expression(session))
    effect = _clean_gene_columns(load_crispr_effect(session))
    if target_gene not in effect.columns:
        raise ValueError(
            f"Gene '{target_gene}' not in CRISPR gene-effect matrix. "
            f"Example available genes: {list(effect.columns[:8])}"
        )

    common = sorted(set(expr.index) & set(effect.index))
    if not common:
        raise RuntimeError("No overlapping cell lines between expression and CRISPR matrices.")
    expr = expr.loc[common]
    target = effect.loc[common, target_gene]

    clean_expr, rep = validate_matrix(expr, name="expression")
    modalities = {"expression": clean_expr}

    clinical = pd.DataFrame({
        "ModelID": common,
        f"{target_gene}_gene_effect": target.to_numpy(dtype=float),
    }).dropna(subset=[f"{target_gene}_gene_effect"])

    return write_dataset(
        out, modalities, clinical, sample_col="ModelID",
        target=f"{target_gene}_gene_effect", run_name=f"ccle_{target_gene}",
        source=f"CCLE/{RELEASE}", task="regression", reports={"expression": rep},
    )
