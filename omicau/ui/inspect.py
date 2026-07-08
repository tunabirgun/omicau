"""Config-generation and validation core for the wizard (stack-portable).

Pure-Python, UI-agnostic logic behind the no-code flow: guess each file's omic
role, preview a matrix, describe clinical columns and the live consequence of
choosing one as target / group / batch, run the cross-file alignment (foreign-key)
check with concrete unmatched ids, and assemble the omicau config the CLI/library
consume. Everything returns JSON-native dicts so the API layer is trivial.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from omicau.config import NormalizationSpec
from omicau.data.alignment import _detect_delimiter, normalize_names

# Filename keywords -> omic role. Clinical is checked first (strongest signal).
ROLE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("clinical", ["clinical", "pheno", "phenotype", "metadata", "sample_info",
                  "samples", "label", "outcome", "target", "survival", "design", "meta"]),
    ("rna", ["rnaseq", "rna_seq", "rna", "mrna", "express", "expr", "gene", "tpm",
             "fpkm", "counts", "transcript"]),
    ("protein", ["proteom", "protein", "rppa", "prot"]),
    ("methylation", ["methylation", "methyl", "cpg", "dnam", "450k", "27k", "beta"]),
    ("mirna", ["mirna", "microrna", "mir"]),
    ("cnv", ["cnv", "copynumber", "copy_number", "gistic", "segment", "seg", "cna"]),
    ("metabolomics", ["metabolom", "metabolite", "metab"]),
    ("mutation", ["mutation", "somatic", "maf", "snv", "variant", "mut"]),
]

ROLES = ["rna", "protein", "methylation", "mirna", "cnv", "metabolomics", "mutation",
         "clinical", "other"]


def guess_role(filename: str) -> dict[str, Any]:
    """Guess a file's omic role from its name; returns role + confidence."""
    name = Path(filename).stem.lower()
    for role, kws in ROLE_KEYWORDS:
        for kw in kws:
            if kw in name:
                return {"role": role, "confidence": "high"}
    return {"role": "other", "confidence": "low"}


def _read_text(path: str | Path, limit: int | None = None) -> str:
    txt = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    return txt[:limit] if limit else txt


def _require_tabular(text: str) -> None:
    """Reject empty / header-only files with a clear message (not a pandas trace)."""
    if len([ln for ln in text.splitlines() if ln.strip()]) < 2:
        raise ValueError("File is empty or has no data rows (need a header plus at least one row).")


def inspect_matrix(path: str | Path, filename: str | None = None,
                   max_rows: int = 6, max_cols: int = 8) -> dict[str, Any]:
    """Preview a modality matrix: delimiter, shape, both axes' labels, a grid."""
    filename = filename or Path(path).name
    text = _read_text(path)
    _require_tabular(text)
    delim = _detect_delimiter(text[:8192])
    df = pd.read_csv(io.StringIO(text), sep=delim, engine="python", dtype="string",
                     index_col=0, header=0)
    df.index = pd.Index([str(i).strip() for i in df.index])
    df.columns = pd.Index([str(c).strip() for c in df.columns])
    grid = [[str(df.index[r])] + [_short(df.iat[r, c]) for c in range(min(max_cols, df.shape[1]))]
            for r in range(min(max_rows, df.shape[0]))]
    return {
        "filename": filename,
        "delimiter": {"\t": "tab", ",": "comma", ";": "semicolon", "|": "pipe"}.get(delim, delim),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "row_labels": [str(x) for x in df.index[:12]],
        "col_labels": [str(x) for x in df.columns[:12]],
        "header": [str(df.index.name or "id")] + [str(c) for c in df.columns[:max_cols]],
        "preview": grid,
        **guess_role(filename),
    }


def _short(v: Any, n: int = 12) -> str:
    s = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


# --------------------------------------------------------------------------- #
# Clinical table inspection + per-column consequences
# --------------------------------------------------------------------------- #
def read_clinical(path: str | Path) -> pd.DataFrame:
    text = _read_text(path)
    _require_tabular(text)
    delim = _detect_delimiter(text[:8192])
    return pd.read_csv(io.StringIO(text), sep=delim, engine="python", dtype="string")


def inspect_clinical(path: str | Path) -> dict[str, Any]:
    """Describe clinical columns so the UI can offer target/group/batch dropdowns."""
    df = read_clinical(path)
    cols = []
    for c in df.columns:
        s = df[c]
        numeric = pd.to_numeric(s, errors="coerce")
        n_num = int(numeric.notna().sum())
        n_unique = int(s.nunique(dropna=True))
        is_numeric = n_num >= 0.8 * max(1, s.notna().sum())
        cols.append({
            "name": str(c),
            "n_unique": n_unique,
            "n_missing": int(s.isna().sum()),
            "kind": "numeric" if is_numeric else "categorical",
            "looks_like_id": bool(n_unique == s.notna().sum() and n_unique > 1 and not is_numeric),
            "sample_values": [str(v) for v in s.dropna().unique()[:4]],
        })
    return {"n_rows": int(df.shape[0]), "columns": cols}


def target_consequence(path: str | Path, column: str, task: str = "auto") -> dict[str, Any]:
    df = read_clinical(path)
    if column not in df.columns:
        return {"ok": False, "message": f"Column '{column}' not found."}
    s = df[column]
    numeric = pd.to_numeric(s, errors="coerce")
    n_unique = int(s.nunique(dropna=True))
    present = max(1, int(s.notna().sum()))
    is_numeric = int(numeric.notna().sum()) >= 0.8 * present     # ignore missing values
    inferred = "regression" if (is_numeric and n_unique > 12) else "classification"
    if task == "auto":
        task = inferred
    out: dict[str, Any] = {"ok": True, "task": task, "n_missing": int(s.isna().sum())}
    if task == "classification":
        counts = s.value_counts(dropna=True)
        out["classes"] = {str(k): int(v) for k, v in counts.head(10).items()}
        out["n_classes"] = n_unique
        if n_unique < 2:
            out["ok"] = False
            out["message"] = ("No valid values found in the target column." if n_unique == 0
                              else "Only one class present — cannot train a classifier.")
        else:
            mn, mx = counts.min(), counts.max()
            out["balance"] = "balanced" if mn >= 0.4 * mx else "imbalanced"
            out["message"] = (f"{n_unique} classes; smallest {int(mn)}, largest {int(mx)} "
                              f"({out['balance']}).")
    else:
        vals = numeric.dropna()
        out["range"] = [float(vals.min()), float(vals.max())] if len(vals) else None
        out["message"] = (f"continuous target, {len(vals)} values, "
                          f"range {vals.min():.3g}–{vals.max():.3g}." if len(vals)
                          else "no numeric values found.")
        if len(vals) < 10:
            out["ok"] = False
            out["message"] = "Too few numeric values for a regression target."
    return out


def group_consequence(path: str | Path, column: str, sample_col: str | None = None) -> dict[str, Any]:
    df = read_clinical(path)
    if column not in df.columns:
        return {"ok": False, "message": f"Column '{column}' not found."}
    g = df[column].astype("string").fillna("NA")
    n_groups = int(g.nunique())
    sizes = g.value_counts()
    max_size = int(sizes.max()) if len(sizes) else 0
    repeated = bool((sizes > 1).any())
    return {
        "ok": n_groups >= 2,
        "n_groups": n_groups,
        "n_rows": int(len(g)),
        "repeated_measures": repeated,
        "message": (
            f"{n_groups} groups across {len(g)} rows"
            + (f"; some contribute up to {max_size} samples — group-aware splitting "
               "will keep each group entirely in train or test (prevents leakage)."
               if repeated else "; one sample per group (no repeated measures).")
        ),
    }


def batch_consequence(path: str | Path, column: str, target: str | None = None) -> dict[str, Any]:
    df = read_clinical(path)
    if column not in df.columns:
        return {"ok": False, "message": f"Column '{column}' not found."}
    b = df[column].astype("string").fillna("NA")
    n_batches = int(b.nunique())
    out: dict[str, Any] = {"ok": n_batches >= 2, "n_batches": n_batches,
                           "message": f"{n_batches} batches."}
    if target and target in df.columns and n_batches >= 2:
        ct = pd.crosstab(b, df[target].astype("string"))
        out["crosstab"] = {"rows": list(map(str, ct.index[:8])),
                           "cols": list(map(str, ct.columns[:8])),
                           "values": ct.iloc[:8, :8].astype(int).values.tolist()}
        # a batch that is nearly one target class is a confound risk
        frac = ct.div(ct.sum(axis=1), axis=0).max(axis=1)
        if (frac > 0.9).any():
            out["message"] += " Warning: at least one batch is almost entirely one outcome class (confound risk)."
    return out


# --------------------------------------------------------------------------- #
# Cross-file alignment (foreign-key) check
# --------------------------------------------------------------------------- #
def _modality_sample_ids(path: str | Path, orientation: str, norm: NormalizationSpec) -> list[str]:
    text = _read_text(path)
    delim = _detect_delimiter(text[:8192])
    df = pd.read_csv(io.StringIO(text), sep=delim, engine="python", dtype="string", index_col=0)
    labels = df.columns if orientation == "samples_as_cols" else df.index
    return normalize_names([str(x) for x in labels], norm)


def alignment_preview(modalities: list[dict], clinical_path: str | Path,
                      clinical_sample_col: str | None,
                      norm: NormalizationSpec | None = None) -> dict[str, Any]:
    """Cross-file join check: matched-across-all-layers count + unmatched examples.

    ``modalities`` is a list of {name, path, orientation}. Returns per-layer
    overlap with the clinical samples and the intersection across everything.
    """
    norm = norm or NormalizationSpec()
    clin = read_clinical(clinical_path)
    if clinical_sample_col and clinical_sample_col in clin.columns:
        # named id column: drop missing ids so '<NA>' is never treated as an id.
        clin_ids_raw = [str(x) for x in clin[clinical_sample_col] if pd.notna(x)]
    else:
        # no sample-id column -> mirror the engine, which uses the positional index.
        clin_ids_raw = [str(i) for i in range(len(clin))]
    clin_ids = set(normalize_names(clin_ids_raw, norm))

    per_layer = []
    common = set(clin_ids)
    for m in modalities:
        ids = set(_modality_sample_ids(m["path"], m.get("orientation", "samples_as_rows"), norm))
        overlap = ids & clin_ids
        common &= ids
        unmatched = sorted(ids - clin_ids)[:8]
        per_layer.append({
            "name": m["name"], "n_samples": len(ids),
            "overlap_with_clinical": len(overlap),
            "unmatched_examples": unmatched,
        })

    hint = None
    if len(common) < 0.5 * max(1, len(clin_ids)) and per_layer:
        ex = next((p["unmatched_examples"] for p in per_layer if p["unmatched_examples"]), [])
        if ex:
            hint = (f"Low overlap. Unmatched ids like {ex[:3]} — check for prefixes/suffixes "
                    "or Excel-mangled ids (e.g. 'P01' vs 'P1', or dates). Fix in the source "
                    "file and re-add; omicau does not edit your data.")
    return {
        "matched_all_layers": len(common),
        "n_clinical": len(clin_ids),
        "per_layer": per_layer,
        "hint": hint,
        "ok": len(common) >= 10,
    }


# --------------------------------------------------------------------------- #
# Config assembly
# --------------------------------------------------------------------------- #
def build_config_dict(session: dict) -> dict[str, Any]:
    """Assemble the omicau config dict from wizard state (same schema as the CLI)."""
    modalities = [
        {"name": m["name"], "path": m["path"], "orientation": m.get("orientation", "auto"),
         "description": m.get("description", "")}
        for m in session.get("modalities", [])
    ]
    clin = session.get("clinical", {})
    task = clin.get("task", "auto")
    # A confounded batch fail-closes the correction probe server-side (never trust the UI).
    confounded = bool(session.get("batch_confounded"))
    clinical_block: dict[str, Any] = {
        "path": clin.get("path"),
        "target": clin.get("target"),
        "sample_id": clin.get("sample_id"),
        "group": clin.get("group"),               # str or list[str] (composite grouping)
        "batch": clin.get("batch"),
        "task": task,
    }
    if task == "survival":                        # survival carries time + event, not a single target
        clinical_block["time"] = clin.get("time")
        clinical_block["event"] = clin.get("event")
        clinical_block["time_unit"] = clin.get("time_unit", "")
    cfg: dict[str, Any] = {
        "run_name": session.get("run_name") or "omicau_ui_run",
        "output_dir": session.get("output_dir", "run"),
        "modalities": modalities,
        "clinical": clinical_block,
        "cv": {"n_splits": session.get("n_splits", 5), "seed": session.get("seed", 42),
               "n_bootstrap": session.get("n_bootstrap", 1000),
               "batch_blocked": bool(session.get("batch_blocked", False)),
               "batch_adjust_sensitivity": bool(session.get("batch_adjust_sensitivity", False)) and not confounded},
        "neural": {"enabled": session.get("neural", True)},
        "normalization": {"preset": session.get("normalization", "none")},
        # Non-secret LLM routing only (provider/model/base_url); the API KEY is
        # NEVER part of the config -- it travels separately as an ephemeral run arg.
        "llm": _llm_block(session.get("llm")),
    }
    return cfg


def _llm_block(llm: dict | None) -> dict[str, Any]:
    """Non-secret LLM config from wizard state. Deliberately drops any key field."""
    llm = llm or {}
    if not llm.get("enabled") or (llm.get("provider", "none") == "none"):
        return {"enabled": False}
    block = {"enabled": True, "provider": llm.get("provider", "anthropic"),
             "model": llm.get("model") or "claude-sonnet-5"}
    if llm.get("base_url"):
        block["base_url"] = llm["base_url"]
    return block          # note: NO api_key / api_key_env value copied from the UI
