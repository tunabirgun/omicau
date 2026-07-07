"""Shared cross-validation machinery for leakage-safe fusion benchmarks.

All preprocessing (median imputation, variance filtering, standardization, and
optional univariate selection) is assembled into an sklearn ``Pipeline`` that is
fitted *inside each training fold only*, so no validation-fold statistic ever
touches training. Cross-validation is group-aware: when a group column is
present (e.g. patient id), splits keep all of a group's samples on one side to
prohibit identity leakage. Metrics are computed on pooled out-of-fold
predictions and are guarded against degenerate folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


PRIMARY_METRIC = {"classification": "auroc", "regression": "r2"}


@dataclass
class CVResult:
    """Cross-validated result for one model/feature-set combination."""

    name: str
    task: str
    metrics: dict[str, float] = field(default_factory=dict)  # pooled OOF metrics
    per_fold: list[dict[str, float]] = field(default_factory=list)
    fold_primary: list[float] = field(default_factory=list)  # per-fold primary metric
    feature_importance: dict[str, float] = field(default_factory=dict)
    n_features: int = 0
    modalities: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def primary(self) -> float:
        return self.metrics.get(PRIMARY_METRIC[self.task], float("nan"))

    @property
    def primary_std(self) -> float:
        vals = [v for v in self.fold_primary if np.isfinite(v)]
        return float(np.std(vals)) if len(vals) > 1 else 0.0


# --------------------------------------------------------------------------- #
# CV splitting
# --------------------------------------------------------------------------- #
def safe_n_splits(task: str, y: np.ndarray, groups: np.ndarray | None, requested: int) -> int:
    """Clamp fold count so every fold is populated (and both classes appear)."""
    n = max(2, requested)
    if groups is not None:
        n = min(n, len(np.unique(groups)))
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        n = min(n, int(counts.min()))
    else:
        n = min(n, len(y))
    return max(2, n)


def make_cv_splitter(
    task: str, n_splits: int, seed: int, shuffle: bool, groups: np.ndarray | None
):
    """Return the appropriate (group-aware, stratified) CV splitter."""
    if groups is not None:
        if task == "classification":
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        return GroupKFold(n_splits=n_splits)
    if task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)


# --------------------------------------------------------------------------- #
# Preprocessing (nested, leakage-safe)
# --------------------------------------------------------------------------- #
def make_pipeline(
    estimator, task: str, n_features: int, max_features: int | None, seed: int
) -> Pipeline:
    """Median-impute -> variance-filter -> standardize -> [select] -> estimator."""
    steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scale", StandardScaler()),
    ]
    if max_features and n_features > max_features:
        score_func = f_classif if task == "classification" else f_regression
        steps.append(("select", SelectKBest(score_func=score_func, k=max_features)))
    steps.append(("estimator", estimator))
    return Pipeline(steps)


# --------------------------------------------------------------------------- #
# Metrics (nan-safe)
# --------------------------------------------------------------------------- #
def _nan() -> float:
    return float("nan")


def score_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray | None,
    y_pred: np.ndarray | None,
    task: str,
) -> dict[str, float]:
    """Compute a task-appropriate metric bundle, guarding degenerate inputs."""
    y_true = np.asarray(y_true)
    out: dict[str, float] = {}
    if task == "classification":
        classes = np.unique(y_true)
        binary = len(classes) == 2
        if y_pred is None and y_score is not None:
            if y_score.ndim == 1:
                y_pred = (y_score >= 0.5).astype(int)
            else:
                y_pred = y_score.argmax(axis=1)
        # AUROC / AUPRC
        try:
            if binary and y_score is not None:
                s = y_score if y_score.ndim == 1 else y_score[:, 1]
                out["auroc"] = float(roc_auc_score(y_true, s)) if len(classes) > 1 else _nan()
                out["auprc"] = float(average_precision_score(y_true, s))
            elif y_score is not None and y_score.ndim == 2 and len(classes) > 2:
                out["auroc"] = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
                out["auprc"] = _nan()
            else:
                out["auroc"] = _nan()
                out["auprc"] = _nan()
        except ValueError:
            out["auroc"] = _nan()
            out["auprc"] = _nan()
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        out["f1"] = float(f1_score(y_true, y_pred, average="binary" if binary else "macro",
                                   zero_division=0))
        try:
            out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        except ValueError:
            out["mcc"] = _nan()
    else:
        yp = np.asarray(y_pred if y_pred is not None else y_score, dtype=float)
        out["r2"] = float(r2_score(y_true, yp)) if np.ptp(y_true) > 0 else _nan()
        out["rmse"] = float(np.sqrt(mean_squared_error(y_true, yp)))
        out["mae"] = float(mean_absolute_error(y_true, yp))
        try:
            out["spearman"] = float(stats.spearmanr(y_true, yp).statistic)
        except (ValueError, FloatingPointError):
            out["spearman"] = _nan()
        try:
            out["pearson"] = float(stats.pearsonr(y_true, yp)[0])
        except (ValueError, FloatingPointError):
            out["pearson"] = _nan()
    return out


# --------------------------------------------------------------------------- #
# Cross-validated estimator runner
# --------------------------------------------------------------------------- #
def cross_validate_estimator(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    task: str,
    estimator_factory: Callable[[], Any],
    *,
    feature_names: list[str],
    modalities: list[str],
    n_splits: int,
    seed: int,
    shuffle: bool = True,
    max_features: int | None = None,
    compute_importance: bool = False,
    importance_repeats: int = 8,
) -> CVResult:
    """Run leakage-safe CV for one estimator over feature matrix ``X``."""
    y = np.asarray(y)
    n = len(y)
    k = safe_n_splits(task, y, groups, n_splits)
    splitter = make_cv_splitter(task, k, seed, shuffle, groups)

    # Out-of-fold prediction stores.
    if task == "classification":
        n_classes = len(np.unique(y))
        oof_score = np.full((n, n_classes), np.nan)
        oof_pred = np.full(n, np.nan)
    else:
        oof_score = np.full(n, np.nan)
        oof_pred = np.full(n, np.nan)

    per_fold: list[dict[str, float]] = []
    fold_primary: list[float] = []
    importance_acc = np.zeros(X.shape[1]) if compute_importance else None
    importance_folds = 0

    for train_idx, val_idx in splitter.split(X, y, groups):
        pipe = make_pipeline(estimator_factory(), task, X.shape[1], max_features, seed)
        pipe.fit(X[train_idx], y[train_idx])

        if task == "classification":
            proba = pipe.predict_proba(X[val_idx])
            classes = pipe.named_steps["estimator"].classes_.astype(int)
            oof_score[np.ix_(val_idx, classes)] = proba
            preds = classes[proba.argmax(axis=1)]
            oof_pred[val_idx] = preds
            s_arg = oof_score if n_classes > 2 else oof_score[:, 1]
            fold_metrics = score_predictions(
                y[val_idx],
                proba if n_classes > 2 else proba[:, 1] if proba.shape[1] > 1 else proba[:, 0],
                preds,
                task,
            )
        else:
            preds = pipe.predict(X[val_idx])
            oof_pred[val_idx] = preds
            oof_score[val_idx] = preds
            fold_metrics = score_predictions(y[val_idx], preds, preds, task)

        per_fold.append(fold_metrics)
        fold_primary.append(fold_metrics.get(PRIMARY_METRIC[task], np.nan))

        if compute_importance and len(val_idx) >= 5:
            try:
                scoring = "roc_auc" if (task == "classification" and n_classes == 2) else (
                    "accuracy" if task == "classification" else "r2"
                )
                r = permutation_importance(
                    pipe, X[val_idx], y[val_idx],
                    n_repeats=importance_repeats, random_state=seed, scoring=scoring,
                )
                importance_acc += r.importances_mean
                importance_folds += 1
            except (ValueError, RuntimeError):
                pass

    # Pooled OOF metrics.
    if task == "classification":
        pooled_score = oof_score if n_classes > 2 else oof_score[:, 1]
        metrics = score_predictions(y, pooled_score, oof_pred.astype(int), task)
    else:
        metrics = score_predictions(y, oof_pred, oof_pred, task)

    importance: dict[str, float] = {}
    if compute_importance and importance_folds:
        mean_imp = importance_acc / importance_folds
        importance = {feature_names[i]: float(mean_imp[i]) for i in range(len(feature_names))}

    return CVResult(
        name=name,
        task=task,
        metrics=metrics,
        per_fold=per_fold,
        fold_primary=[float(v) for v in fold_primary],
        feature_importance=importance,
        n_features=int(X.shape[1]),
        modalities=list(modalities),
        extra={"n_splits": int(k)},
    )
