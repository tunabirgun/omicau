"""Missingness-bias diagnostics.

Tests whether *missingness itself* carries information about the outcome or the
batch -- the hallmark of missing-not-at-random (MNAR) data that silently biases
downstream models. Per modality it relates each sample's missingness rate to the
target (Kruskal-Wallis / Spearman) and to batch, and relates a binary
"any-missing" indicator to the target (chi-squared). p-values are FDR-corrected
across modalities (Benjamini-Hochberg). No imputation is performed here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


def _f(x) -> float | None:
    """JSON-safe float (NaN/inf -> None)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def benjamini_hochberg(pvals: list[float]) -> list[float | None]:
    """BH FDR adjustment; NaN inputs pass through as None."""
    p = np.asarray([np.nan if v is None else v for v in pvals], dtype=float)
    finite = np.isfinite(p)
    adj = np.full(p.shape, np.nan)
    if finite.sum():
        pf = p[finite]
        n = pf.size
        order = np.argsort(pf)
        ranked = pf[order] * n / (np.arange(n) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(n)
        out[order] = np.clip(ranked, 0, 1)
        adj[finite] = out
    return [_f(v) for v in adj]


def _sample_missing_rate(frame: pd.DataFrame) -> np.ndarray:
    """Per-sample fraction of missing features."""
    return frame.isna().to_numpy(dtype=float).mean(axis=1)


def missingness_diagnostics(aligned) -> dict[str, Any]:
    """Compute the full missingness-bias report for an :class:`AlignedDataset`."""
    y = aligned.y
    task = aligned.task
    batch = aligned.batch
    tests: list[dict[str, Any]] = []
    overall_modalities: dict[str, Any] = {}
    sample_missing: dict[str, list[float]] = {}
    per_feature: dict[str, list[dict[str, Any]]] = {}

    total_missing = 0.0
    total_cells = 0

    for name, mod in aligned.modalities.items():
        frame = mod.frame
        na = frame.isna()
        rate = _sample_missing_rate(frame)
        sample_missing[name] = [float(r) for r in rate]
        miss_frac = float(na.to_numpy().mean())
        total_missing += float(na.to_numpy().sum())
        total_cells += na.size

        feat_rates = na.mean(axis=0).sort_values(ascending=False)
        per_feature[name] = [
            {"feature": str(f), "missing_fraction": _f(v)}
            for f, v in feat_rates.head(25).items()
        ]
        overall_modalities[name] = {
            "missing_fraction": _f(miss_frac),
            "n_features": int(frame.shape[1]),
            "samples_with_any_missing": int((rate > 0).sum()),
            "max_feature_missing_fraction": _f(feat_rates.max() if len(feat_rates) else 0.0),
        }

        # -- missingness vs target ---------------------------------------- #
        if task == "classification":
            classes = np.unique(y.to_numpy())
            groups = [rate[y.to_numpy() == c] for c in classes]
            stat, p = _safe_kruskal(groups)
            tests.append(_mk_test(name, "kruskal_missingrate_vs_target", "target", stat, p,
                                   "Missingness rate differs across outcome classes (possible MNAR)."))
            # chi-squared: any-missing indicator vs class
            any_missing = (rate > 0).astype(int)
            chi_stat, chi_p = _safe_chi2(any_missing, y.astype("int64").to_numpy())
            tests.append(_mk_test(name, "chi2_anymissing_vs_target", "target", chi_stat, chi_p,
                                  "Presence of missing data is associated with the outcome."))
        else:  # regression
            rho, p = _safe_spearman(rate, y.to_numpy())
            tests.append(_mk_test(name, "spearman_missingrate_vs_target", "target", rho, p,
                                  "Missingness rate correlates with the continuous target."))

        # -- missingness vs batch ----------------------------------------- #
        if batch is not None:
            bvals = batch.to_numpy()
            bgroups = [rate[bvals == b] for b in np.unique(bvals)]
            stat_b, p_b = _safe_kruskal(bgroups)
            tests.append(_mk_test(name, "kruskal_missingrate_vs_batch", "batch", stat_b, p_b,
                                  "Missingness rate differs across batches (batch-linked dropout)."))

    # FDR across all tests.
    p_adj = benjamini_hochberg([t["p_value"] for t in tests])
    for t, pa in zip(tests, p_adj):
        t["p_adj"] = pa
        t["flag"] = bool(pa is not None and pa < ALPHA)

    flags = [
        f"{t['modality']}: {t['interpretation']} (p_adj={t['p_adj']:.3g})"
        for t in tests
        if t["flag"]
    ]

    return {
        "alpha": ALPHA,
        "overall": {
            "total_missing_fraction": _f(total_missing / total_cells) if total_cells else 0.0,
            "n_samples": int(aligned.n_samples),
            "modalities": overall_modalities,
        },
        "tests": tests,
        "per_feature": per_feature,
        "sample_missingness": {"sample_ids": list(aligned.sample_ids), "by_modality": sample_missing},
        "flags": flags,
    }


# --------------------------------------------------------------------------- #
# Safe statistical wrappers
# --------------------------------------------------------------------------- #
def _mk_test(modality, test, association, stat, p, interpretation) -> dict[str, Any]:
    return {
        "modality": modality,
        "test": test,
        "association": association,
        "statistic": _f(stat),
        "p_value": _f(p),
        "interpretation": interpretation,
    }


def _safe_kruskal(groups):
    groups = [np.asarray(g, float) for g in groups if len(g) > 0]
    groups = [g for g in groups if np.ptp(g) > 0 or len(g) > 1]
    if len(groups) < 2 or all(np.ptp(np.concatenate(groups)) == 0 for _ in [0]):
        return np.nan, np.nan
    try:
        if np.ptp(np.concatenate(groups)) == 0:
            return np.nan, np.nan
        return stats.kruskal(*groups)
    except (ValueError, FloatingPointError):
        return np.nan, np.nan


def _safe_chi2(a, b):
    try:
        table = pd.crosstab(pd.Series(a), pd.Series(b))
        if table.shape[0] < 2 or table.shape[1] < 2:
            return np.nan, np.nan
        chi2, p, _, _ = stats.chi2_contingency(table)
        return chi2, p
    except (ValueError, ZeroDivisionError):
        return np.nan, np.nan


def _safe_spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0 or len(a) < 3:
        return np.nan, np.nan
    try:
        res = stats.spearmanr(a, b)
        return res.statistic, res.pvalue
    except (ValueError, FloatingPointError):
        return np.nan, np.nan
