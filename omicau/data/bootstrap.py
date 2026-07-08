"""Shared dataset-assembly dispatch.

Both the CLI ``omicau bootstrap`` command and the web UI's "load from a data hub"
panel call :func:`assemble`, so the two paths produce byte-identical datasets and
the same ``config.json`` -- the CLI and UI can never drift apart on hub loading.
"""

from __future__ import annotations

from pathlib import Path

# Public cohorts, in the order shown to users. `mock` is fully offline.
DATASETS = ("mock", "tcga", "ccle", "cptac", "openpbta", "xena",
            "metabolomics", "expression_atlas")

# Human-facing labels + which optional parameter each hub actually uses, so the
# UI can render only the relevant field. `param` is the wizard's free-text field.
HUB_META = {
    "mock":            {"label": "Synthetic demo (offline)",            "param": None},
    "tcga":            {"label": "TCGA / cBioPortal (human)",           "param": "study",  "hint": "e.g. laml_tcga"},
    "ccle":            {"label": "CCLE / DepMap cell lines",            "param": "target", "hint": "gene, e.g. SOX10"},
    "cptac":           {"label": "CPTAC proteogenomics (human)",       "param": "cancer", "hint": "e.g. Ucec"},
    "openpbta":        {"label": "OpenPBTA pediatric brain (human)",   "param": "target", "hint": "e.g. broad_histology"},
    "xena":            {"label": "UCSC Xena (human)",                   "param": "preset", "hint": "e.g. brca"},
    "metabolomics":    {"label": "Metabolomics Workbench",             "param": "study",  "hint": "e.g. ST000009"},
    "expression_atlas":{"label": "EMBL-EBI Expression Atlas (any organism)", "param": "study", "hint": "e.g. E-GEOD-100100"},
}


def assemble(dataset: str, out_dir: str | Path, *, study: str | None = None,
             target: str | None = None, cancer: str | None = None,
             preset: str | None = None, task: str = "classification",
             seed: int = 42, normalization: str = "log2cpm") -> Path:
    """Assemble ``dataset`` into ``out_dir`` and return the path to its config.json.

    Raises ``ValueError`` on an unknown dataset or an invalid parameter combo. This
    is the single source of truth for hub loading in both the CLI and the UI.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if normalization != "log2cpm" and dataset != "expression_atlas":
        raise ValueError(
            "normalization applies only to the expression_atlas dataset; "
            "ccle/xena/tcga ship pre-normalized (e.g. log2 TPM).")

    if dataset == "mock":
        from omicau.data.benchmark_data import write_mock_dataset
        write_mock_dataset(out_dir, task=task, seed=seed)
        return out_dir / "config.json"
    if dataset == "tcga":
        from omicau.data import tcga
        return tcga.prepare(out_dir, study=study or "laml_tcga", target=target)
    if dataset == "ccle":
        from omicau.data import ccle
        return ccle.prepare(out_dir, target_gene=target or "SOX10")
    if dataset == "cptac":
        from omicau.data import cptac
        return cptac.prepare(out_dir, cancer=cancer or "Ucec", target=target)
    if dataset == "openpbta":
        from omicau.data import openpbta
        return openpbta.prepare(out_dir, target_column=target or "broad_histology")
    if dataset == "xena":
        from omicau.data import xena
        return xena.prepare(out_dir, preset=preset or "brca", target=target)
    if dataset == "metabolomics":
        from omicau.data import metabolomics_workbench as mw
        return mw.prepare(out_dir, study_id=study or "ST000009", target=target)
    if dataset == "expression_atlas":
        from omicau.data import expression_atlas as gxa
        return gxa.prepare(out_dir, accession=study or gxa.DEFAULT_ACCESSION,
                           target=target, normalization=normalization)
    raise ValueError(f"Unknown dataset '{dataset}'.")
