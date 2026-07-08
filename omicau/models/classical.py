"""Classical multi-omic fusion benchmarks.

Early-fusion (feature-concatenation) models are cross-validated for every
modality alone, for the full fusion, and for each leave-one-modality-out subset
(so the interpretation layer can measure a modality's marginal contribution).
Three control baselines -- shuffled target, column-shuffled features, and random
noise -- are run through the identical pipeline to prove the harness does not
leak: a well-behaved pipeline scores at chance on all three.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

from omicau.models.base import CVResult, PRIMARY_METRIC, attach_cis, cross_validate_estimator


def resolve_cores(config) -> int:
    """Resolve the worker count, honoring cluster limits and leaving OS headroom."""
    requested = config.compute.cores
    if requested and requested > 0:
        return int(requested)
    total = os.cpu_count() or 2
    return max(1, min(total - 2, 16))


def _estimator_factory(key: str, task: str, seed: int, n_jobs: int) -> Callable[[], Any]:
    key = key.lower()
    if key in {"linear", "logistic", "ridge"}:
        if task == "classification":
            return lambda: LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed
            )
        return lambda: Ridge(alpha=1.0, random_state=seed)
    if key in {"random_forest", "rf", "forest"}:
        if task == "classification":
            return lambda: RandomForestClassifier(
                n_estimators=300, n_jobs=n_jobs, class_weight="balanced", random_state=seed
            )
        return lambda: RandomForestRegressor(n_estimators=300, n_jobs=n_jobs, random_state=seed)
    if key in {"gradient_boosting", "gb", "hgb"}:
        if task == "classification":
            return lambda: HistGradientBoostingClassifier(random_state=seed)
        return lambda: HistGradientBoostingRegressor(random_state=seed)
    raise ValueError(f"Unknown classical estimator key: '{key}'")


def _reference_key(keys: list[str]) -> str:
    for pref in ("random_forest", "rf"):
        if pref in [k.lower() for k in keys]:
            return "random_forest"
    return keys[0]


def _run_stacking(aligned, results, ref_key, config, groups, n_jobs, seed):
    """Late-integration stacking: cross-validate a meta-learner over the
    single-modality out-of-fold predictions. Leakage-safe because the base
    predictions are out-of-fold and the meta-learner is itself cross-validated,
    so the reported metric is honest (not in-sample on the OOF matrix)."""
    mods = aligned.modality_names
    if len(mods) < 2:
        return None
    task = aligned.task
    y = aligned.y.to_numpy()
    cl = {r.name: r for r in results}
    cols: list[np.ndarray] = []
    names: list[str] = []
    for m in mods:
        r = cl.get(f"{ref_key}::{m}")
        if r is None or r.oof_score is None:
            return None
        s = np.asarray(r.oof_score, dtype=float).reshape(len(y), -1)
        cols.append(s)
        names += [m] if s.shape[1] == 1 else [f"{m}[{j}]" for j in range(s.shape[1])]
    meta_X = np.hstack(cols)
    factory = _estimator_factory("linear", task, seed, n_jobs)
    return cross_validate_estimator(
        "stacking::FUSION", meta_X, y, groups, task, factory,
        feature_names=names, modalities=list(mods),
        n_splits=config.cv.n_splits, seed=seed, shuffle=config.cv.shuffle,
        max_features=None, compute_importance=False,
    )


def run_classical_benchmarks(aligned, config) -> dict[str, Any]:
    """Run the full classical benchmark grid over an aligned dataset."""
    task = aligned.task
    seed = config.seed
    n_jobs = resolve_cores(config)
    keys = [k for k in (config.classical.models or ["linear", "random_forest"])]
    ref_key = _reference_key(keys)
    max_feat = config.classical.max_features
    n_splits = config.cv.n_splits
    shuffle = config.cv.shuffle

    y = aligned.y.to_numpy()
    groups = aligned.groups.to_numpy() if aligned.groups is not None else None
    mods = aligned.modality_names

    def run(name, X, feats, modalities, yy=y, gg=groups, imp=False):
        factory = _estimator_factory(ref_key if imp else current_key, task, seed, n_jobs)
        return cross_validate_estimator(
            name, X, yy, gg, task, factory,
            feature_names=feats, modalities=modalities,
            n_splits=n_splits, seed=seed, shuffle=shuffle, max_features=max_feat,
            compute_importance=imp, importance_repeats=config.xai.permutation_repeats,
        )

    results: list[CVResult] = []
    X_all, feats_all = aligned.concat_matrix(mods)

    for current_key in keys:
        # single-modality baselines
        for m in mods:
            Xm, fm = aligned.concat_matrix([m])
            results.append(run(f"{current_key}::{m}", Xm, fm, [m]))
        # full fusion (compute attribution only on the reference estimator)
        want_imp = (current_key == ref_key) and config.xai.enabled
        results.append(run(f"{current_key}::FUSION", X_all, feats_all, mods, imp=want_imp))
        # leave-one-modality-out
        if len(mods) > 1:
            for m in mods:
                subset = [x for x in mods if x != m]
                Xs, fs = aligned.concat_matrix(subset)
                results.append(run(f"{current_key}::FUSION-minus-{m}", Xs, fs, subset))

    # -- control baselines (reference estimator on the full fusion) -------- #
    current_key = ref_key
    rng = np.random.default_rng(seed)
    controls: list[CVResult] = []

    if config.controls.enabled:
        if config.controls.shuffle_target:
            y_shuf = rng.permutation(y)
            controls.append(run("control::shuffled_target", X_all, feats_all, mods, yy=y_shuf))
        if config.controls.shuffle_features:
            Xp = np.array(X_all, copy=True)
            for j in range(Xp.shape[1]):
                Xp[:, j] = rng.permutation(Xp[:, j])
            controls.append(run("control::shuffled_features", Xp, feats_all, mods))
        if config.controls.random_noise:
            Xn = rng.normal(size=X_all.shape)
            controls.append(run("control::random_noise", Xn, feats_all, mods))

    # -- late-integration stacking (meta-learner over per-modality OOF) ----- #
    stack = _run_stacking(aligned, results, ref_key, config, groups, n_jobs, seed)
    if stack is not None:
        results.append(stack)

    # -- optional batch-blocked (leave-one-batch-out) generalization check --- #
    if config.cv.batch_blocked and aligned.batch is not None:
        bcodes = np.unique(aligned.batch.astype("string").to_numpy(), return_inverse=True)[1]
        n_batches = len(np.unique(bcodes))
        if n_batches >= 3:
            factory = _estimator_factory(ref_key, task, seed, n_jobs)
            bb = cross_validate_estimator(
                "stress::batch-blocked-FUSION", X_all, y, bcodes, task, factory,
                feature_names=feats_all, modalities=list(mods),
                n_splits=min(n_splits, n_batches), seed=seed, shuffle=shuffle,
                max_features=max_feat, compute_importance=False)
            results.append(bb)

    attach_cis(results + controls, n_boot=config.cv.n_bootstrap, seed=seed)

    return {
        "task": task,
        "primary_metric": PRIMARY_METRIC[task],
        "reference_estimator": ref_key,
        "estimator_keys": keys,
        "results": results,
        "controls": controls,
    }
