"""EMBL-EBI Expression Atlas cross-organism client.

Assembles an omicau dataset from a single Expression Atlas bulk RNA-seq
experiment for ANY species in the Atlas (mouse, rat, zebrafish, fly, worm,
yeast, Arabidopsis, crops, livestock, and non-model organisms), not just human.

Per-sample expression comes from the raw-counts matrix, converted to
log2(CPM+1) per sample (within-sample library-size only, so no cross-sample
leakage; edgeR/limma-voom convention). The prediction target is an experiment
FACTOR parsed from the condensed-SDRF. Single-modality (transcriptomics) is a
valid omicau dataset -- same as the Metabolomics Workbench client.

Endpoints verified live against Atlas data (plant baseline + mouse differential)
as of 2026-07:
  https://www.ebi.ac.uk/gxa/json/experiments                          (discovery)
  https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/experiments/{ACC}/
    {ACC}-raw-counts.tsv | {ACC}-raw-counts.tsv.undecorated           (matrix)
    {ACC}.condensed-sdrf.tsv                                          (factors)
Accession + factor are the reproducible inputs; organism is provenance only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from omicau.data._hub import (
    default_cache_dir, download_file, get_json,
    require_requests, validate_matrix, write_dataset,
)

GXA_JSON = "https://www.ebi.ac.uk/gxa/json/experiments"
FTP = "https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/experiments/{acc}/{fname}"
# probe order: decorated (differential; has Gene Name) then undecorated (baseline)
_COUNT_FILES = ("{acc}-raw-counts.tsv", "{acc}-raw-counts.tsv.undecorated")
DEFAULT_ACCESSION = "E-GEOD-100100"  # mouse, 10 assays, differential, RNA-interference factor


def _cache(sub: str = "expression_atlas") -> Path:
    d = default_cache_dir() / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_experiments(session=None, *, kingdom: str | None = None,
                     species: str | None = None) -> pd.DataFrame:
    """Discovery table of Atlas experiments (optionally filtered by kingdom/species).

    Columns: accession, species, kingdom, type, n_assays, factors. Tolerates
    missing JSON fields -- json/experiments is the web-app API, not a formal REST
    contract.
    """
    session = session or require_requests().Session()
    raw = get_json(GXA_JSON, session=session).get("experiments", [])
    rows = [{
        "accession": e.get("experimentAccession"),
        "species": e.get("species"),
        "kingdom": e.get("kingdom"),
        "type": e.get("experimentType"),
        "n_assays": e.get("numberOfAssays"),
        "factors": e.get("experimentalFactors", []),
    } for e in raw]
    df = pd.DataFrame(rows)
    if kingdom:
        df = df[df["kingdom"].str.lower() == kingdom.lower()]
    if species:
        df = df[df["species"].str.contains(species, case=False, na=False)]
    return df.reset_index(drop=True)


def _experiment_meta(accession: str, session=None) -> dict:
    """species / kingdom / type / factors for one accession (provenance only)."""
    try:
        df = list_experiments(session)
        hit = df[df["accession"] == accession]
        if len(hit):
            r = hit.iloc[0]
            return {"species": r["species"], "kingdom": r["kingdom"],
                    "experiment_type": r["type"], "factors": list(r["factors"] or [])}
    except Exception:  # noqa: BLE001 - discovery is best-effort, never blocks a run
        pass
    return {"species": None, "kingdom": None, "experiment_type": None, "factors": []}


def fetch_counts(accession: str, session=None) -> tuple[pd.DataFrame, pd.Series | None, str]:
    """Download the raw-counts matrix (genes x samples). Returns (counts, gene_names, filename).

    Probes decorated then undecorated; the decorated file also yields a Gene ID ->
    Gene Name symbol map for interpretation. Raises if neither file is present.
    """
    session = session or require_requests().Session()
    last = None
    for pat in _COUNT_FILES:
        fname = pat.format(acc=accession)
        url = FTP.format(acc=accession, fname=fname)
        dest = _cache() / fname
        try:
            path = download_file(url, dest, session=session)
        except Exception as exc:  # noqa: BLE001 - try the next candidate filename
            last = exc
            continue
        df = pd.read_csv(path, sep="\t", index_col=0)
        symbols = None
        if "Gene Name" in df.columns:  # decorated: split off the symbol sidecar
            symbols = df["Gene Name"].astype(str)
            df = df.drop(columns=["Gene Name"])
        df.columns = [str(c).strip() for c in df.columns]  # SRR/ERR sample ids
        return df, symbols, fname
    raise RuntimeError(f"No raw-counts matrix for {accession} (tried {list(_COUNT_FILES)}). "
                       f"Last error: {last}")


def fetch_factors(accession: str, session=None) -> pd.DataFrame:
    """Per-sample factor table from the headerless condensed-SDRF (sample x factor)."""
    session = session or require_requests().Session()
    fname = f"{accession}.condensed-sdrf.tsv"
    path = download_file(FTP.format(acc=accession, fname=fname), _cache() / fname, session=session)
    # positional: [2]=sample, [3]=characteristic|factor, [4]=name, [5]=value, ([6]=uri)
    sdrf = pd.read_csv(path, sep="\t", header=None, dtype=str).fillna("")
    fac = sdrf[sdrf[3] == "factor"]
    if fac.empty:
        return pd.DataFrame()
    wide = (fac.pivot_table(index=2, columns=4, values=5, aggfunc="first")
               .rename_axis(index="sample_id", columns=None))
    wide.index = wide.index.astype(str)
    return wide


def _default_target(factors: pd.DataFrame) -> str | None:
    """Balanced-category heuristic (shared with metabolomics): 2-10 classes."""
    best, best_n = None, 0
    for c in factors.columns:
        n = factors[c].nunique(dropna=True)
        if 2 <= n <= 10 and factors[c].notna().sum() > best_n:
            best, best_n = c, int(factors[c].notna().sum())
    return best


def _log2_cpm(counts_samples_by_genes: pd.DataFrame) -> pd.DataFrame:
    """log2(CPM+1) per sample. Library size is a within-sample total (leakage-safe)."""
    lib = counts_samples_by_genes.sum(axis=1).replace(0, np.nan)
    cpm = counts_samples_by_genes.mul(1e6, axis=0).div(lib, axis=0)
    return np.log2(cpm + 1.0)


def prepare(out_dir: str | Path, *, accession: str = DEFAULT_ACCESSION,
            target: str | None = None, session=None) -> Path:
    """Assemble one Expression Atlas experiment into an omicau dataset.

    Modality: transcriptomics = log2(CPM+1), samples x Ensembl-gene features
    (single modality is valid, as with metabolomics). Target: an experiment
    factor (auto-selected if not given). Organism is recorded for provenance;
    Ensembl IDs run as-is with no mapping. Returns the config.json path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    session = session or require_requests().Session()

    counts_gxs, symbols, count_file = fetch_counts(accession, session)  # genes x samples
    factors = fetch_factors(accession, session)
    meta = _experiment_meta(accession, session)
    if factors.empty:
        raise RuntimeError(f"No factor rows in condensed-SDRF for {accession}.")

    target = target or _default_target(factors)
    if not target or target not in factors.columns:
        raise ValueError(f"No usable target factor for {accession}. "
                         f"Available factors: {meta.get('factors') or list(factors.columns)}")

    counts = counts_gxs.T  # -> samples x genes
    counts.index = counts.index.astype(str)
    common = sorted(set(counts.index) & set(factors.index))
    if not common:
        raise RuntimeError("No overlap between count-matrix samples and SDRF samples.")

    expr = _log2_cpm(counts.loc[common])
    clean, rep = validate_matrix(expr, name="transcriptomics")
    clin = factors.loc[common, [target]].reset_index()
    clin.columns = ["sample_id", target]

    # organism-aware, interpretation-only symbol sidecar (Ensembl-id -> symbol);
    # present only when the decorated raw-counts.tsv was used.
    if symbols is not None:
        symbols.rename_axis("gene_id").rename("gene_symbol").to_csv(
            out / "gene_symbols.csv", lineterminator="\n")

    rep.update({
        "source_file": count_file, "normalization": "log2(CPM+1) per sample",
        "species": meta.get("species"), "kingdom": meta.get("kingdom"),
        "experiment_type": meta.get("experiment_type"),
        "gene_symbols": symbols is not None,
    })
    return write_dataset(
        out, {"transcriptomics": clean}, clin, sample_col="sample_id",
        target=target, run_name=f"gxa_{accession}",
        source=f"ExpressionAtlas:{accession} ({meta.get('species') or 'organism'})",
        organism=(meta.get("species") or "unspecified"),
        task="classification", reports={"transcriptomics": rep},
    )
