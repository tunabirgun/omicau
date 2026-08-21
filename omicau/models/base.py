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


PRIMARY_METRIC = {"classification": "auroc", "regression": "r2", "survival": "c_index"}


@dataclass
class CVResult:
    """Cross-validated result for one model/feature-set combination."""

    name: str
    task: str
    metrics: dict[str, float] = field(default_factory=dict)  # pooled OOF metrics
    per_fold: list[dict[str, float]] = field(default_factory=list)
    fold_primary: list[float] = field(default_factory=list)  # per-fold primary metric
    feature_importance: dict[str, float] = field(default_factory=dict)
    feature_importance_std: dict[str, float] = field(default_factory=dict)  # across-fold stability
    n_features: int = 0
    modalities: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    # Pooled out-of-fold predictions (kept in-memory so uncertainty, calibration,
    # and subgroup analyses can resample them; not serialized into audit.json).
    oof_true: Any = None      # true labels/values, aligned to the sample order
    oof_score: Any = None     # score feeding the primary metric (binary: P(class1); mc: proba; reg: pred)
    oof_pred: Any = None      # hard predictions
    oof_groups: Any = None    # patient/group ids for group-level resampling (or None)

    @property
    def primary(self) -> float:
        return self.metrics.get(PRIMARY_METRIC[self.task], float("nan"))

    @property
    def primary_std(self) -> float:
        vals = [v for v in self.fold_primary if np.isfinite(v)]
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-native view (finite floats only; NaN -> None)."""
        def _f(x):
            return float(x) if np.isfinite(x) else None
        return {
            "name": self.name,
            "task": self.task,
            "primary": _f(self.primary),
            "primary_std": _f(self.primary_std),
            "metrics": {k: _f(v) for k, v in self.metrics.items()},
            "per_fold": [{k: _f(v) for k, v in fm.items()} for fm in self.per_fold],
            "fold_primary": [_f(v) for v in self.fold_primary],
            "n_features": int(self.n_features),
            "modalities": list(self.modalities),
            "feature_importance": {k: _f(v) for k, v in self.feature_importance.items()},
            "feature_importance_std": {k: _f(v) for k, v in self.feature_importance_std.items()},
            "n_splits": int(self.extra.get("n_splits", 0)),
            "device": self.extra.get("device"),
            # per-fold spread of the primary metric -- fold dispersion, NOT a
            # standard error (correlated folds bias it low; use the CI instead).
            "fold_dispersion": _f(self.primary_std),
            "ci_low": self.extra.get("ci_low"),
            "ci_high": self.extra.get("ci_high"),
            "split_plan_status": self.extra.get("split_plan_status"),
            "split_plan_receipt": self.extra.get("split_plan_receipt"),
        }


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
# Preprocessing (fitted in-fold, leakage-safe)
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


def resolve_validated_cv_splits(
    validated_plan: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    task: str,
    n_splits: int,
) -> tuple[
    tuple[tuple[np.ndarray, np.ndarray], ...],
    tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...],
    dict[str, Any],
]:
    """Resolve and recheck one process-local C06 plan without exposing rows."""
    from omicau.models.split_plan import (
        ValidatedSplitPlan,
        _partition_evidence_values,
    )

    if type(validated_plan) is not ValidatedSplitPlan:
        raise ValueError("validated_split_plan_required")
    if groups is None:
        raise ValueError("validated_split_plan_requires_groups")
    y = np.asarray(y)
    groups = np.asarray(groups, dtype=object)
    n = len(y)
    if np.asarray(X).ndim != 2 or len(X) != n or groups.ndim != 1 or len(groups) != n:
        raise ValueError("validated_split_plan_wrong_row_universe")
    if task not in {"classification", "regression"}:
        raise ValueError("validated_split_plan_task_unsupported")
    validated_plan._private_validate_runtime_universe(
        groups=groups,
        task=task,
        y=y,
    )

    receipt = validated_plan.receipt()
    if receipt.get("outer_fold_count") != n_splits:
        raise ValueError("validated_split_plan_outer_fold_count_mismatch")
    outer = tuple(validated_plan._private_outer_splits())
    inner = tuple(
        tuple(validated_plan._private_inner_splits(fold))
        for fold in range(len(outer))
    )
    if len(outer) != n_splits or any(
        len(folds) != receipt.get("inner_fold_count") for folds in inner
    ):
        raise ValueError("validated_split_plan_fold_count_mismatch")

    _, expected_outer_groups, expected_inner_groups, _, _ = (
        _partition_evidence_values(validated_plan._private_partition_evidence())
    )
    universe = set(range(n))
    outer_assessment: list[int] = []
    partitions: list[tuple[np.ndarray, np.ndarray]] = []
    for fold, (train, assessment) in enumerate(outer):
        train_set = set(train.tolist())
        assessment_set = set(assessment.tolist())
        if (
            not train_set
            or not assessment_set
            or train_set & assessment_set
            or train_set | assessment_set != universe
        ):
            raise ValueError("validated_split_plan_outer_partition_invalid")
        if set(outer_assessment) & assessment_set:
            raise ValueError("validated_split_plan_outer_assessment_coverage_invalid")
        if (
            frozenset(groups[train].tolist()),
            frozenset(groups[assessment].tolist()),
        ) != expected_outer_groups[fold]:
            raise ValueError("validated_split_plan_group_universe_mismatch")
        outer_assessment.extend(assessment.tolist())
        partitions.append((train, assessment))

        inner_assessment: list[int] = []
        for child, (inner_train, inner_validation) in enumerate(inner[fold]):
            inner_train_set = set(inner_train.tolist())
            inner_validation_set = set(inner_validation.tolist())
            if (
                not inner_train_set
                or not inner_validation_set
                or inner_train_set & inner_validation_set
                or inner_train_set | inner_validation_set != train_set
            ):
                raise ValueError("validated_split_plan_inner_partition_invalid")
            if set(inner_assessment) & inner_validation_set:
                raise ValueError("validated_split_plan_inner_assessment_coverage_invalid")
            if (
                frozenset(groups[inner_train].tolist()),
                frozenset(groups[inner_validation].tolist()),
            ) != expected_inner_groups[fold][child]:
                raise ValueError("validated_split_plan_group_universe_mismatch")
            inner_assessment.extend(inner_validation.tolist())
            partitions.append((inner_train, inner_validation))
        if sorted(inner_assessment) != sorted(train.tolist()):
            raise ValueError("validated_split_plan_inner_assessment_coverage_invalid")
    if sorted(outer_assessment) != list(range(n)):
        raise ValueError("validated_split_plan_outer_assessment_coverage_invalid")

    if task == "classification":
        if "minimum_realized_training_groups_per_class" not in receipt.get(
            "support_summary", {}
        ):
            raise ValueError("validated_split_plan_task_mismatch")
        classes = set(y.tolist())
        if len(classes) < 2:
            raise ValueError("validated_split_plan_class_support_invalid")
        group_labels: dict[Any, Any] = {}
        for group, label in zip(groups.tolist(), y.tolist()):
            if group in group_labels and group_labels[group] != label:
                raise ValueError("validated_split_plan_group_outcome_mixed")
            group_labels[group] = label
        required_train = receipt["support_summary"][
            "minimum_realized_training_groups_per_class"
        ]
        required_assessment = receipt["support_summary"][
            "minimum_realized_assessment_groups_per_class"
        ]
        for train, assessment in partitions:
            if set(y[train].tolist()) != classes or set(y[assessment].tolist()) != classes:
                raise ValueError("validated_split_plan_class_support_invalid")
            train_groups = set(groups[train].tolist())
            assessment_groups = set(groups[assessment].tolist())
            train_min = min(
                sum(group_labels[group] == label for group in train_groups)
                for label in classes
            )
            assessment_min = min(
                sum(group_labels[group] == label for group in assessment_groups)
                for label in classes
            )
            if train_min < required_train or assessment_min < required_assessment:
                raise ValueError("validated_split_plan_class_support_invalid")
    elif task == "regression":
        if "minimum_realized_assessment_variance" not in receipt.get(
            "support_summary", {}
        ):
            raise ValueError("validated_split_plan_task_mismatch")
        try:
            outcome = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("validated_split_plan_regression_outcome_invalid") from error
        if not np.isfinite(outcome).all():
            raise ValueError("validated_split_plan_regression_outcome_invalid")
        group_outcomes: dict[Any, float] = {}
        for group, value in zip(groups.tolist(), outcome.tolist()):
            if group in group_outcomes and group_outcomes[group] != value:
                raise ValueError("validated_split_plan_group_outcome_mixed")
            group_outcomes[group] = value
        required_count = receipt["support_summary"][
            "minimum_realized_assessment_group_count"
        ]
        required_variance = receipt["support_summary"][
            "minimum_realized_assessment_variance"
        ]
        for _, assessment in partitions:
            values = np.asarray(
                [group_outcomes[group] for group in set(groups[assessment].tolist())],
                dtype=float,
            )
            if len(values) < required_count or np.var(values) < required_variance:
                raise ValueError("validated_split_plan_regression_support_invalid")

    return outer, inner, receipt


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
    validated_plan: Any = None,
    validated_control_contract: Any = None,
) -> CVResult:
    """Run leakage-safe CV for one estimator over feature matrix ``X``."""
    y = np.asarray(y)
    n = len(y)
    if validated_plan is None:
        k = safe_n_splits(task, y, groups, n_splits)
        splitter = make_cv_splitter(task, k, seed, shuffle, groups)
        splits = tuple(splitter.split(X, y, groups))
        split_status = "legacy_generated_non_benchmark"
        split_receipt = None
    else:
        splits, _, split_receipt = resolve_validated_cv_splits(
            validated_plan, X, y, groups, task, n_splits
        )
        k = len(splits)
        split_status = "validated_development_plan"

    fold_training_targets = None
    control_execution_receipt = None
    if validated_control_contract is not None:
        if validated_plan is None or name != "control::group_permuted_target":
            raise ValueError("validated_target_control_scope_invalid")
        from omicau.models.group_controls import _execute_fold_endpoint_permutations

        candidates, control_execution_receipt = _execute_fold_endpoint_permutations(
            contract=validated_control_contract,
            outer_splits=splits,
            groups=groups,
            task=task,
            y=y,
        )
        fold_training_targets = tuple(
            np.asarray(
                candidate["y"],
                dtype=y.dtype if task == "classification" else float,
            )
            for candidate in candidates
        )
    elif name == "control::group_permuted_target":
        raise ValueError("validated_target_control_contract_required")

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
    importance_stack: list[np.ndarray] = []   # per-fold permutation importances

    for fold, (train_idx, val_idx) in enumerate(splits):
        pipe = make_pipeline(estimator_factory(), task, X.shape[1], max_features, seed)
        training_target = (
            y[train_idx]
            if fold_training_targets is None
            else fold_training_targets[fold]
        )
        pipe.fit(X[train_idx], training_target)

        if task == "classification":
            proba = pipe.predict_proba(X[val_idx])
            classes = pipe.named_steps["estimator"].classes_.astype(int)
            oof_score[np.ix_(val_idx, classes)] = proba
            preds = classes[proba.argmax(axis=1)]
            oof_pred[val_idx] = preds
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
            # permutation_importance re-scores once per feature x repeat -- thousands of
            # tiny predicts. If the fitted estimator kept n_jobs>1, each of those predicts
            # fans out across all cores; under Windows spawn the per-call dispatch overhead
            # dominates and a high-dimensional run appears to hang. Pin n_jobs=1 for the
            # importance pass only (identical results, just no nested parallelism).
            est = pipe.named_steps.get("estimator")
            prev_njobs = getattr(est, "n_jobs", None)
            try:
                if prev_njobs not in (None, 1):
                    est.n_jobs = 1
                scoring = "roc_auc" if (task == "classification" and n_classes == 2) else (
                    "accuracy" if task == "classification" else "r2"
                )
                r = permutation_importance(
                    pipe, X[val_idx], y[val_idx],
                    n_repeats=importance_repeats, random_state=seed, scoring=scoring,
                )
                importance_stack.append(r.importances_mean)
            except (ValueError, RuntimeError):
                pass
            finally:
                if prev_njobs not in (None, 1):
                    est.n_jobs = prev_njobs

    # Pooled OOF metrics.
    if task == "classification":
        pooled_score = oof_score if n_classes > 2 else oof_score[:, 1]
        metrics = score_predictions(y, pooled_score, oof_pred.astype(int), task)
    else:
        pooled_score = oof_pred
        metrics = score_predictions(y, oof_pred, oof_pred, task)

    importance: dict[str, float] = {}
    importance_std: dict[str, float] = {}
    if compute_importance and importance_stack:
        arr = np.vstack(importance_stack)
        mean_imp = arr.mean(axis=0)
        std_imp = arr.std(axis=0) if arr.shape[0] > 1 else np.zeros_like(mean_imp)
        importance = {feature_names[i]: float(mean_imp[i]) for i in range(len(feature_names))}
        importance_std = {feature_names[i]: float(std_imp[i]) for i in range(len(feature_names))}

    return CVResult(
        name=name,
        task=task,
        metrics=metrics,
        per_fold=per_fold,
        fold_primary=[float(v) for v in fold_primary],
        feature_importance=importance,
        feature_importance_std=importance_std,
        n_features=int(X.shape[1]),
        modalities=list(modalities),
        extra={
            "n_splits": int(k),
            "split_plan_status": split_status,
            "split_plan_receipt": split_receipt,
            "control_execution_receipt": control_execution_receipt,
        },
        oof_true=y,
        oof_score=pooled_score,
        oof_pred=oof_pred,
        oof_groups=(np.asarray(groups) if groups is not None else None),
    )


def bootstrap_ci(result: CVResult, n_boot: int = 1000, seed: int = 42,
                 alpha: float = 0.05) -> tuple[float | None, float | None]:
    """Percentile CI for a result's primary metric from its pooled OOF predictions.

    Resamples whole patient/groups with replacement (falling back to samples when
    no group column) and recomputes the primary metric each time. Group-level
    resampling is the only bootstrap consistent with omicau's group-aware CV; the
    per-fold std understates the true variance because folds are correlated
    (Nadeau & Bengio, Machine Learning 52:239-281, 2003).
    """
    if result.oof_true is None or result.oof_score is None:
        return None, None
    y = np.asarray(result.oof_true)
    score = np.asarray(result.oof_score)
    pred = np.asarray(result.oof_pred)
    task = result.task
    key = PRIMARY_METRIC[task]
    n = len(y)
    n_classes = len(np.unique(y)) if task == "classification" else 0
    groups = None if result.oof_groups is None else np.asarray(result.oof_groups)
    rng = np.random.default_rng(seed)
    if groups is not None:
        uniq = np.unique(groups)
        idx_by = {g: np.where(groups == g)[0] for g in uniq}
    vals: list[float] = []
    for _ in range(n_boot):
        if groups is not None:
            chosen = rng.choice(len(uniq), size=len(uniq), replace=True)
            idx = np.concatenate([idx_by[uniq[c]] for c in chosen])
        else:
            idx = rng.integers(0, n, size=n)
        yb = y[idx]
        # require all classes present, so multiclass resamples don't silently
        # fall to the binary metric branch in score_predictions.
        if task == "classification" and len(np.unique(yb)) < n_classes:
            continue
        try:
            hard = pred[idx].astype(int) if task == "classification" else pred[idx]
            v = score_predictions(yb, score[idx], hard, task).get(key, np.nan)
            if np.isfinite(v):
                vals.append(float(v))
        except (ValueError, FloatingPointError):
            continue
    if len(vals) < 20:
        return None, None
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def attach_cis(results: list[CVResult], n_boot: int = 1000, seed: int = 42,
               alpha: float = 0.05) -> list[CVResult]:
    """Compute and cache a bootstrap CI on each result's primary metric.

    `alpha` is exposed so a caller testing a family of results can request a
    corrected interval; the control baselines use it for exactly that.
    """
    for r in results:
        lo, hi = bootstrap_ci(r, n_boot=n_boot, seed=seed, alpha=alpha)
        r.extra["ci_low"] = lo
        r.extra["ci_high"] = hi
    return results


def calibration_metrics(result: CVResult, n_bins: int = 10) -> dict[str, Any] | None:
    """Calibration of a binary classifier's OOF probabilities: Brier, ECE, curve.

    Discrimination (AUROC) says nothing about whether predicted risks match
    observed event rates (Van Calster et al., BMC Medicine 2019). Binary only.
    """
    if result.task != "classification" or result.oof_true is None or result.oof_score is None:
        return None
    y = np.asarray(result.oof_true)
    classes = np.unique(y)
    if len(classes) != 2:
        return None
    score = np.asarray(result.oof_score, dtype=float)      # P(positive class)
    y01 = (y == classes[1]).astype(int)
    keep = np.isfinite(score)
    y01, score = y01[keep], score[keep]
    if len(y01) < 10 or np.ptp(score) == 0:
        return None
    brier = float(np.mean((score - y01) ** 2))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(score, bins) - 1, 0, n_bins - 1)
    ece, curve_pred, curve_obs = 0.0, [], []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc = float(score[m].mean()), float(y01[m].mean())
        ece += (m.sum() / len(score)) * abs(acc - conf)
        curve_pred.append(conf)
        curve_obs.append(acc)
    return {
        "brier": brier, "ece": float(ece), "n": int(len(score)),
        "reliability": {"mean_predicted": curve_pred, "fraction_positive": curve_obs},
    }
