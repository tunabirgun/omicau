"""Batch-effect and confounding diagnostics.

For each modality the data are standardized (mean-imputed for the PCA projection
only -- a diagnostic view, never fed to the models) and projected onto principal
components. Batch structure is quantified by (i) the silhouette of batch labels
in PC space, (ii) one-way ANOVA and Kruskal-Wallis of the leading PCs across
batches, and (iii) the variance of PC1 explained by batch (eta-squared). A
chi-squared test between batch and a categorical target flags batch/target
confounding, the condition under which a batch effect masquerades as signal.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ALPHA = 0.05


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _mean_impute(X: np.ndarray) -> np.ndarray:
    """Column-mean imputation for the diagnostic PCA only (labelled as such)."""
    X = X.copy()
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(col_mean, idx[1])
    return X


def _encode(labels: pd.Series | None) -> np.ndarray | None:
    if labels is None:
        return None
    codes, _ = pd.factorize(labels.astype("string"), sort=True)
    return codes


def _pca_project(X: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    Xi = _mean_impute(X)
    Xs = StandardScaler().fit_transform(Xi)
    n_comp = int(min(n_components, Xs.shape[0] - 1, Xs.shape[1]))
    n_comp = max(1, n_comp)
    pca = PCA(n_components=n_comp, random_state=seed)
    coords = pca.fit_transform(Xs)
    return coords, pca.explained_variance_ratio_


def _silhouette(coords: np.ndarray, labels: np.ndarray | None) -> float | None:
    if labels is None:
        return None
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return None
    try:
        return float(silhouette_score(coords, labels))
    except ValueError:
        return None


def _anova_pc1(pc1: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """One-way ANOVA of PC1 across label groups; returns (F, p, eta_squared)."""
    groups = [pc1[labels == u] for u in np.unique(labels)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan
    try:
        F, p = stats.f_oneway(*groups)
    except (ValueError, FloatingPointError):
        F, p = np.nan, np.nan
    grand = pc1.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(((pc1 - grand) ** 2).sum())
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    return F, p, eta2


def batch_effect_diagnostics(aligned, seed: int = 42, n_components: int = 10) -> dict[str, Any]:
    """Compute batch-effect and confounding diagnostics for an aligned dataset."""
    batch = aligned.batch
    y = aligned.y
    task = aligned.task
    batch_codes = _encode(batch)
    target_codes = _encode(y.astype("string")) if task == "classification" else None

    per_modality: dict[str, Any] = {}
    pca_coords: dict[str, Any] = {}
    flags: list[str] = []

    for name, mod in aligned.modalities.items():
        coords, evr = _pca_project(mod.X, n_components, seed)
        pc1 = coords[:, 0]
        pc2 = coords[:, 1] if coords.shape[1] > 1 else np.zeros_like(pc1)

        sil_batch = _silhouette(coords, batch_codes)
        sil_target = _silhouette(coords, target_codes)

        entry: dict[str, Any] = {
            "n_components": int(coords.shape[1]),
            "explained_variance_ratio": [_f(v) for v in evr[:5]],
            "silhouette_batch": sil_batch,
            "silhouette_target": sil_target,
        }
        if batch_codes is not None:
            F, p, eta2 = _anova_pc1(pc1, batch_codes)
            try:
                kw = stats.kruskal(*[pc1[batch_codes == u] for u in np.unique(batch_codes)])
                kw_p = kw.pvalue
            except (ValueError, FloatingPointError):
                kw_p = np.nan
            entry.update({
                "pc1_anova_F": _f(F),
                "pc1_anova_p": _f(p),
                "pc1_kruskal_p": _f(kw_p),
                "batch_variance_explained_pc1": _f(eta2),
            })
            strong = (sil_batch is not None and sil_batch > 0.25) or (
                _f(eta2) is not None and eta2 > 0.15 and _f(p) is not None and p < ALPHA
            )
            entry["flag"] = bool(strong)
            if strong:
                flags.append(
                    f"{name}: batch structures the leading PCs "
                    f"(silhouette={sil_batch}, eta2_PC1={_f(eta2)})."
                )
            entry["interpretation"] = (
                "Batch strongly organizes this modality's variance."
                if strong else "No strong batch structure detected in the leading PCs."
            )
        else:
            entry["flag"] = False
            entry["interpretation"] = "No batch column supplied; batch tests skipped."

        per_modality[name] = entry
        pca_coords[name] = {
            "pc1": [float(v) for v in pc1],
            "pc2": [float(v) for v in pc2],
            "batch": (batch.astype("string").tolist() if batch is not None else None),
            "target": aligned.y_raw.astype("string").tolist() if aligned.y_raw is not None else None,
        }

    # -- batch/target confounding ----------------------------------------- #
    # The confounding-inflation risk (Nygaard et al. 2016) is a property of the
    # batch-outcome association, not the outcome type -- test both tasks.
    confounding: dict[str, Any] = {"tested": False}
    if batch is not None and task == "classification":
        try:
            table = pd.crosstab(batch.astype("string"), aligned.y_raw.astype("string"))
            chi2, p, _, _ = stats.chi2_contingency(table)
            cramers_v = _cramers_v(table)
            confounding = {
                "tested": True,
                "test": "chi2_batch_vs_target",
                "statistic": _f(chi2),
                "p_value": _f(p),
                "cramers_v": _f(cramers_v),
                "flag": bool(p is not None and p < ALPHA and cramers_v > 0.2),
            }
            if confounding["flag"]:
                flags.append(
                    f"Batch is confounded with the target "
                    f"(Cramer's V={_f(cramers_v)}, p={_f(p)}); batch effects can leak as signal."
                )
        except (ValueError, ZeroDivisionError):
            confounding = {"tested": False}
    elif batch is not None and task == "regression":
        # One-way ANOVA of the continuous outcome across batch levels + eta^2.
        try:
            y_cont = np.asarray(y, dtype=float)
            F, p, eta2 = _anova_pc1(y_cont, batch_codes)
            confounding = {
                "tested": True,
                "test": "anova_target_vs_batch",
                "statistic": _f(F),
                "p_value": _f(p),
                "eta_squared": _f(eta2),
                "flag": bool(_f(p) is not None and _f(p) < ALPHA
                             and _f(eta2) is not None and eta2 > 0.10),
            }
            if confounding["flag"]:
                flags.append(
                    f"Batch is confounded with the (continuous) outcome "
                    f"(eta2={_f(eta2)}, p={_f(p)}); batch effects can leak as signal."
                )
        except (ValueError, FloatingPointError):
            confounding = {"tested": False}

    return {
        "alpha": ALPHA,
        "batch_column": (aligned.batch.name if aligned.batch is not None else None),
        "per_modality": per_modality,
        "pca_coords": pca_coords,
        "confounding": confounding,
        "flags": flags,
    }


def _cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    denom = n * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan
