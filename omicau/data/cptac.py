"""CPTAC proteogenomics loader (via the official ``cptac`` package).

Loads matched proteomics and transcriptomics for a CPTAC tumor cohort so their
cross-modality redundancy can be audited against each other. Uses the current
``cptac`` package API (cancer classes + per-datatype source tags), downloads on
first access, and collapses the ``(Name, Database_ID)`` column MultiIndex to gene
symbols. The ``cptac`` package is an optional dependency; a clear error is raised
if it is absent.

Validated against ``cptac`` 1.5.x API as of 2026-07 -- data-source tags per
cohort should be confirmed with ``list_data_sources()``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from omicau.data._hub import validate_matrix, write_dataset


def _require_cptac():
    try:
        import cptac  # type: ignore
        return cptac
    except ImportError as exc:  # pragma: no cover - optional dep
        from omicau._hints import extra_hint
        raise ImportError(
            f"CPTAC access needs the optional 'cptac' package: {extra_hint('cptac')} "
            "(or pip install cptac). It downloads cohort data on first use."
        ) from exc


def list_cancers() -> dict:
    """Return the available CPTAC cancer cohorts (abbreviation -> full name)."""
    cptac = _require_cptac()
    return cptac.get_cancer_info()


def _collapse_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce a (Name, Database_ID) column MultiIndex to unique gene symbols."""
    if isinstance(df.columns, pd.MultiIndex):
        try:
            from cptac.utils import reduce_multiindex  # type: ignore
            df = reduce_multiindex(df, levels_to_drop="Database_ID", quiet=True)
        except Exception:  # noqa: BLE001 - fall back to level-0 labels
            df = df.copy()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    return df


def load_cohort(cancer: str, *, proteomics_source: str = "umich",
                transcriptomics_source: str = "bcm") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load matched proteomics + transcriptomics frames (samples x genes)."""
    cptac = _require_cptac()
    cls = getattr(cptac, cancer.capitalize(), None)
    if cls is None:
        raise ValueError(f"Unknown CPTAC cohort '{cancer}'. Options: {list(list_cancers())}")
    cohort = cls()
    prot = _collapse_multiindex(cohort.get_proteomics(source=proteomics_source))
    rna = _collapse_multiindex(cohort.get_transcriptomics(source=transcriptomics_source))
    return prot, rna


def prepare(out_dir: str | Path, *, cancer: str = "Ucec",
            proteomics_source: str = "umich", transcriptomics_source: str = "bcm",
            target: str | None = None) -> Path:
    """Write a matched proteomics + transcriptomics dataset for a CPTAC cohort.

    The default target flags tumor vs normal samples (from the ``Sample_Tumor_Normal``
    clinical field when available), giving a classification endpoint that both
    modalities can be tested against. Returns the ``config.json`` path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cptac = _require_cptac()

    cls = getattr(cptac, cancer.capitalize())
    cohort = cls()
    prot = _collapse_multiindex(cohort.get_proteomics(source=proteomics_source))
    rna = _collapse_multiindex(cohort.get_transcriptomics(source=transcriptomics_source))

    common = sorted(set(prot.index.astype(str)) & set(rna.index.astype(str)))
    if not common:
        raise RuntimeError("No overlapping samples between proteomics and transcriptomics.")

    clean_prot, rp = validate_matrix(prot.loc[common], name="proteomics")
    clean_rna, rr = validate_matrix(rna.loc[common], name="transcriptomics")
    modalities = {"proteomics": clean_prot, "transcriptomics": clean_rna}

    # Build a clinical table with a tumor/normal endpoint when available.
    clin = pd.DataFrame(index=pd.Index(common, name="Patient_ID")).reset_index()
    label_col = target or "Sample_Tumor_Normal"
    try:
        clinical = cohort.get_clinical(source="mssm")
        if label_col in clinical.columns:
            clin[label_col] = clin["Patient_ID"].map(clinical[label_col].to_dict())
    except Exception:  # noqa: BLE001 - clinical is best-effort
        pass
    if label_col not in clin.columns or clin[label_col].isna().all():
        # Fallback endpoint: tumor samples typically lack a '.N' suffix.
        label_col = "tumor_normal"
        clin[label_col] = [("Normal" if str(s).endswith(".N") else "Tumor") for s in common]

    clin = clin.dropna(subset=[label_col])
    return write_dataset(
        out, modalities, clin, sample_col="Patient_ID", target=label_col,
        run_name=f"cptac_{cancer.lower()}", source=f"CPTAC:{cancer}",
        task="classification", reports={"proteomics": rp, "transcriptomics": rr},
    )
