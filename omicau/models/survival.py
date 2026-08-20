"""Dependency-light survival (time-to-event) benchmarking.

scikit-survival pins an older scikit-learn and lifelines requires pandas < 3.0,
so neither fits omicau's pinned environment (sklearn 1.9 / pandas 3.0). This
module therefore implements the two standard, well-defined pieces directly:

* Harrell's concordance index (Harrell et al. 1982) -- the fraction of comparable
  (one event, later survivor) pairs a risk score orders correctly; chance = 0.5.
* A ridge-penalised Cox proportional-hazards model (Breslow partial likelihood)
  fitted with scipy.optimize, producing a per-sample risk score.

Everything else reuses the existing audit machinery: leakage-safe in-fold
preprocessing, group-aware cross-validation, negative-control baselines, and
group-level bootstrap confidence intervals. Calibration and neural survival are
out of scope (deferred), matching the plan.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from omicau.models.base import CVResult
from omicau.models.classical import resolve_cores  # noqa: F401 (parity with classical API)
from omicau.models.split_plan import ValidatedSplitPlan


_VALIDATED_SPLIT_STATUS = "validated_c06_exact_splits"
_LEGACY_SPLIT_STATUS = "legacy_nonbenchmark_dynamic_splits"


def _validated_outer_splits(
    validated_plan,
    n: int,
    groups: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
):
    if type(validated_plan) is not ValidatedSplitPlan:
        raise TypeError("c06_validated_split_plan_required")
    if groups is None or len(groups) != n:
        raise RuntimeError("c06_runtime_groups_missing_or_misaligned")
    event_identity = np.asarray(event)
    if event_identity.ndim == 1 and np.isin(event_identity, (0, 1)).all():
        event_identity = event_identity.astype(np.int64)
    validated_plan._private_validate_runtime_universe(
        groups=groups,
        task="survival",
        time=time,
        event=event_identity,
    )
    receipt = validated_plan.receipt()
    splits = tuple(validated_plan._private_outer_splits())
    if len(splits) != receipt["outer_fold_count"]:
        raise RuntimeError("c06_outer_fold_count_mismatch")
    universe = set(range(n))
    for train, assessment in splits:
        train_set = set(np.asarray(train, dtype=np.int64).tolist())
        assessment_set = set(np.asarray(assessment, dtype=np.int64).tolist())
        if not train_set or not assessment_set:
            raise RuntimeError("c06_outer_partition_empty")
        if train_set & assessment_set:
            raise RuntimeError("c06_outer_partition_overlap")
        if train_set | assessment_set != universe:
            raise RuntimeError("c06_outer_partition_universe_mismatch")
        if {groups[index] for index in train_set} & {
            groups[index] for index in assessment_set
        }:
            raise RuntimeError("c06_outer_group_overlap")
    return splits, receipt


def _has_comparable_pair(time: np.ndarray, event: np.ndarray) -> bool:
    event_rows = np.flatnonzero(event == 1)
    return any(np.any(time > time[index]) for index in event_rows)


def harrell_cindex(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C-index. A comparable pair is (i had the event, j survived past t_i);
    the score is correct when the earlier-event sample carries the higher risk."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk = np.asarray(risk, dtype=float)
    conc = disc = tied = 0.0
    ev_idx = np.where(event == 1)[0]
    for i in ev_idx:
        comparable = time > time[i]           # j outlived i's event -> orderable pair
        if not comparable.any():
            continue
        rj = risk[comparable]
        conc += np.sum(risk[i] > rj)
        disc += np.sum(risk[i] < rj)
        tied += np.sum(risk[i] == rj)
    denom = conc + disc + tied
    return 0.5 if denom == 0 else float((conc + 0.5 * tied) / denom)


def cox_fit(X: np.ndarray, time: np.ndarray, event: np.ndarray, l2: float = 10.0) -> np.ndarray:
    """Ridge-penalised Cox (Breslow partial likelihood) via L-BFGS-B; returns beta.
    The L2 penalty keeps the fit stable in the p >= n omics regime; the default was
    chosen so a signal-free null concordance sits at ~0.5 (no optimistic bias)
    while strong real signal is preserved."""
    from scipy.optimize import minimize
    X = np.asarray(X, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    order = np.argsort(time)
    Xs, ts, es = X[order], time[order], event[order]
    ev_rows = np.where(es == 1)[0]

    def negll(beta):
        eta = Xs @ beta
        ll = 0.0
        grad = np.zeros_like(beta)
        for i in ev_rows:
            rs = ts >= ts[i]                  # risk set: still at risk at t_i
            w = np.exp(eta[rs] - eta[rs].max())   # stabilised
            sw = w.sum()
            ll += eta[i] - (np.log(sw) + eta[rs].max())
            grad += Xs[i] - (w @ Xs[rs]) / sw
        return -ll + l2 * beta @ beta, -grad + 2 * l2 * beta

    res = minimize(negll, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 200})
    return res.x


class _Preproc:
    """Leakage-safe in-fold preprocessing for the p>>n survival Cox:
    median-impute -> standardize -> PCA to a low dimension. Fit on train,
    applied to val. PCA (fit in-fold) keeps the Cox in a p<<n regime so a
    signal-free null concordance stays at ~0.5 instead of overfitting spurious
    cross-feature correlations."""

    def __init__(self, max_features: int | None):
        self.max_features = max_features

    def fit(self, X):
        from sklearn.decomposition import PCA
        self.median_ = np.nanmedian(X, axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        Xf = np.where(np.isnan(X), self.median_, X)
        self.mean_ = Xf.mean(axis=0)
        self.std_ = Xf.std(axis=0)
        self.std_ = np.where(self.std_ > 0, self.std_, 1.0)
        Z = (Xf - self.mean_) / self.std_
        n, p = Z.shape
        # p<<n cap: leaves the ridge Cox well-conditioned (null concordance ~0.5).
        cap = max(2, n // 4)
        if self.max_features:
            cap = min(cap, self.max_features)
        self.n_comp_ = int(min(p, n - 1, cap))
        self.pca_ = PCA(n_components=self.n_comp_, random_state=0).fit(Z)
        return self

    def transform(self, X):
        Xf = np.where(np.isnan(X), self.median_, X)
        Z = (Xf - self.mean_) / self.std_
        return self.pca_.transform(Z)


def _splitter(event: np.ndarray, groups: np.ndarray | None, n_splits: int, seed: int):
    """Group-aware CV, stratified on the event indicator to balance censoring."""
    n_ev, n_cens = int(event.sum()), int((event == 0).sum())
    k = max(2, min(n_splits, n_ev, n_cens)) if (n_ev and n_cens) else max(2, min(n_splits, 5))
    if groups is not None:
        k = max(2, min(k, len(np.unique(groups))))
        try:
            return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed), k
        except Exception:  # noqa: BLE001
            return GroupKFold(n_splits=k), k
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed), k


def _cv_cindex(
    X, time, event, groups, n_splits, seed, max_features, l2=10.0,
    validated_plan=None,
):
    """Return (pooled_cindex, fold_cindices, oof_risk, actual_k)."""
    if validated_plan is not None:
        splits, _ = _validated_outer_splits(
            validated_plan, len(time), groups, time, event
        )
        k = len(splits)
    else:
        splitter, k = _splitter(event, groups, n_splits, seed)
        strat = event.astype(int)
        split_args = (X, strat, groups) if groups is not None else (X, strat)
        splits = tuple(splitter.split(*split_args))
    oof = np.full(len(time), np.nan)
    fold_c = []
    for tr, va in splits:
        # need events in train (to fit Cox) and at least one event in val (a c-index
        # needs a comparable event); the old unique()<1 clause could never be true.
        if validated_plan is not None:
            if event[tr].sum() < 1:
                raise RuntimeError("c06_survival_training_event_support_runtime")
            if not _has_comparable_pair(time[va], event[va]):
                raise RuntimeError("c06_survival_assessment_pair_support_runtime")
        elif event[tr].sum() < 2 or event[va].sum() < 1:
            continue
        pre = _Preproc(max_features).fit(X[tr])
        beta = cox_fit(pre.transform(X[tr]), time[tr], event[tr], l2=l2)
        risk = pre.transform(X[va]) @ beta
        oof[va] = risk
        if event[va].sum() >= 1:
            fold_c.append(harrell_cindex(time[va], event[va], risk))
    if validated_plan is not None and (np.isnan(oof).any() or len(fold_c) != k):
        raise RuntimeError("c06_survival_fold_execution_incomplete")
    mask = ~np.isnan(oof)
    pooled = harrell_cindex(time[mask], event[mask], oof[mask]) if mask.any() else float("nan")
    return pooled, fold_c, oof, k


def _boot_ci(time, event, risk, groups, n_boot, seed, alpha=0.05):
    """Group-level (or sample-level) percentile bootstrap CI of the C-index."""
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(risk)
    t, e, r = time[mask], event[mask], risk[mask]
    g = groups[mask] if groups is not None else None
    n = len(t)
    vals = []
    if g is not None:
        uniq = np.unique(g)
        idx_by = {u: np.where(g == u)[0] for u in uniq}
    for _ in range(n_boot):
        if g is not None:
            chosen = rng.choice(len(uniq), size=len(uniq), replace=True)
            idx = np.concatenate([idx_by[uniq[c]] for c in chosen])
        else:
            idx = rng.integers(0, n, size=n)
        if e[idx].sum() < 1:
            continue
        vals.append(harrell_cindex(t[idx], e[idx], r[idx]))
    if len(vals) < 20:
        return None, None
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def _result(
    name, modalities, pooled, fold_c, oof, time, event, groups, k, n_features,
    n_boot, seed, split_status, split_receipt,
):
    r = CVResult(
        name=name, task="survival",
        metrics={"c_index": pooled},
        fold_primary=[float(c) for c in fold_c],
        per_fold=[{"c_index": float(c)} for c in fold_c],
        n_features=int(n_features), modalities=list(modalities),
        oof_true=time, oof_score=oof, oof_pred=oof,
        oof_groups=(groups if groups is not None else None),
        extra={
            "n_splits": int(k),
            "split_plan_status": split_status,
            "split_plan_receipt": split_receipt,
        },
    )
    lo, hi = _boot_ci(time, event, oof, groups, n_boot, seed)
    r.extra["ci_low"], r.extra["ci_high"] = lo, hi
    return r


def run_survival_benchmark(aligned, config, validated_plan=None) -> dict[str, Any]:
    """Cox + C-index survival benchmark mirroring run_classical_benchmarks' output."""
    legacy_controls_enabled = config.controls.enabled and any(
        (
            config.controls.shuffle_target,
            config.controls.shuffle_features,
            config.controls.random_noise,
        )
    )
    if validated_plan is not None and legacy_controls_enabled:
        raise RuntimeError("validated_plan_controls_require_c07_integration")
    seed = config.seed
    n_splits = config.cv.n_splits
    n_boot = config.cv.n_bootstrap
    max_feat = config.classical.max_features
    time = np.asarray(aligned.y, dtype=float)
    event = np.asarray(aligned.event, dtype=float)
    groups = aligned.groups.to_numpy() if aligned.groups is not None else None
    mods = aligned.modality_names
    split_status = (
        _VALIDATED_SPLIT_STATUS if validated_plan is not None else _LEGACY_SPLIT_STATUS
    )
    split_receipt = validated_plan.receipt() if validated_plan is not None else None

    results: list[CVResult] = []
    for m in mods:
        Xm, fm = aligned.concat_matrix([m])
        pooled, fc, oof, k = _cv_cindex(
            Xm, time, event, groups, n_splits, seed, max_feat,
            validated_plan=validated_plan,
        )
        results.append(_result(f"cox::{m}", [m], pooled, fc, oof, time, event, groups, k, Xm.shape[1], n_boot, seed, split_status, split_receipt))

    X_all, feats_all = aligned.concat_matrix(mods)
    pooled, fc, oof, k = _cv_cindex(
        X_all, time, event, groups, n_splits, seed, max_feat,
        validated_plan=validated_plan,
    )
    results.append(_result("cox::FUSION", mods, pooled, fc, oof, time, event, groups, k, X_all.shape[1], n_boot, seed, split_status, split_receipt))
    if len(mods) > 1:
        for m in mods:
            subset = [x for x in mods if x != m]
            Xs, _ = aligned.concat_matrix(subset)
            p, f, o, kk = _cv_cindex(
                Xs, time, event, groups, n_splits, seed, max_feat,
                validated_plan=validated_plan,
            )
            results.append(_result(f"cox::FUSION-minus-{m}", subset, p, f, o, time, event, groups, kk, Xs.shape[1], n_boot, seed, split_status, split_receipt))

    # -- control baselines ------------------------------------------------- #
    controls: list[CVResult] = []
    if config.controls.enabled:
        rng = np.random.default_rng(seed)
        if config.controls.shuffle_target:
            perm = rng.permutation(len(time))          # shuffle (time, event) jointly
            p, f, o, kk = _cv_cindex(X_all, time[perm], event[perm], groups, n_splits, seed, max_feat, validated_plan=validated_plan)
            controls.append(_result("control::shuffled_target", mods, p, f, o, time[perm], event[perm], groups, kk, X_all.shape[1], n_boot, seed, split_status, split_receipt))
        if config.controls.shuffle_features:
            Xp = np.array(X_all, copy=True)
            for j in range(Xp.shape[1]):
                Xp[:, j] = rng.permutation(Xp[:, j])
            p, f, o, kk = _cv_cindex(Xp, time, event, groups, n_splits, seed, max_feat, validated_plan=validated_plan)
            controls.append(_result("control::shuffled_features", mods, p, f, o, time, event, groups, kk, Xp.shape[1], n_boot, seed, split_status, split_receipt))
        if config.controls.random_noise:
            Xn = rng.normal(size=X_all.shape)
            p, f, o, kk = _cv_cindex(Xn, time, event, groups, n_splits, seed, max_feat, validated_plan=validated_plan)
            controls.append(_result("control::random_noise", mods, p, f, o, time, event, groups, kk, Xn.shape[1], n_boot, seed, split_status, split_receipt))

    return {
        "results": results,
        "controls": controls,
        "primary_metric": "c_index",
        "reference_estimator": "cox",
        "split_execution_status": split_status,
        "split_plan_receipt": split_receipt,
    }
