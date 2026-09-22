from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge

from omicau.models.base import (
    PermutationImportanceMappingError,
    _permutation_importance_original_features,
    cross_validate_estimator,
    make_cv_splitter,
    make_pipeline,
    safe_n_splits,
)


FEATURE_NAMES = [
    "rna::signal",
    "cnv::signal",
    "methylation::noise",
    "proteomics::constant",
    "rna::all_missing",
    "cnv::selection_candidate",
    "methylation::noise_2",
    "proteomics::noise_2",
]


def _fixture(task: str, dtype: type[np.floating]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(913)
    n = 48
    if task == "binary":
        y = np.arange(n) % 2
        signal = y + rng.normal(scale=0.25, size=n)
    elif task == "multiclass":
        y = np.arange(n) % 3
        signal = y + rng.normal(scale=0.25, size=n)
    else:
        y = rng.normal(size=n)
        signal = y + rng.normal(scale=0.25, size=n)
    X = np.column_stack(
        [
            signal,
            0.6 * signal + rng.normal(scale=0.4, size=n),
            rng.normal(size=n),
            np.full(n, 1.0),
            np.full(n, np.nan),
            rng.normal(size=n),
            rng.normal(size=n),
            rng.normal(size=n),
        ]
    ).astype(dtype)
    X[::7, 0] = np.nan
    X[1::9, 1] = np.nan
    return X, y


def _task_name(case: str) -> str:
    return "classification" if case in {"binary", "multiclass"} else "regression"


def _factory(task: str):
    if task == "classification":
        return lambda: LogisticRegression(max_iter=300, random_state=17)
    return lambda: Ridge()


def _scoring(case: str) -> str:
    return "roc_auc" if case == "binary" else "accuracy" if case == "multiclass" else "r2"


@pytest.mark.parametrize("case", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_transformed_estimator_importance_matches_full_pipeline_and_cv_aggregation(case, dtype):
    task = _task_name(case)
    X, y = _fixture(case, dtype)
    n_splits, seed, repeats = 3, 71, 3
    splitter = make_cv_splitter(
        task, safe_n_splits(task, y, None, n_splits), seed, True, None
    )
    raw_folds, optimized_folds = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, val_idx in splitter.split(X, y, None):
            pipe = make_pipeline(_factory(task)(), task, X.shape[1], 3, seed)
            pipe.fit(X[train_idx], y[train_idx])
            raw = permutation_importance(
                pipe,
                X[val_idx],
                y[val_idx],
                n_repeats=repeats,
                n_jobs=1,
                random_state=seed,
                scoring=_scoring(case),
            ).importances_mean
            optimized = _permutation_importance_original_features(
                pipe,
                X[val_idx],
                y[val_idx],
                feature_names=FEATURE_NAMES,
                n_repeats=repeats,
                random_state=seed,
                scoring=_scoring(case),
            )
            np.testing.assert_allclose(optimized, raw, rtol=1e-12, atol=1e-12)
            output_names = list(pipe[:-1].get_feature_names_out(FEATURE_NAMES))
            assert "rna::all_missing" not in output_names
            assert "proteomics::constant" not in output_names
            assert len(output_names) == 3
            raw_folds.append(raw)
            optimized_folds.append(optimized)

        result = cross_validate_estimator(
            "test",
            X,
            y,
            None,
            task,
            _factory(task),
            feature_names=FEATURE_NAMES,
            modalities=["rna", "cnv", "methylation", "proteomics"],
            n_splits=n_splits,
            seed=seed,
            max_features=3,
            compute_importance=True,
            importance_repeats=repeats,
        )
    expected = np.vstack(raw_folds)
    observed = np.asarray([result.feature_importance[name] for name in FEATURE_NAMES])
    observed_std = np.asarray(
        [result.feature_importance_std[name] for name in FEATURE_NAMES]
    )
    np.testing.assert_allclose(
        np.vstack(optimized_folds), expected, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        observed, expected.mean(axis=0), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        observed_std, expected.std(axis=0), rtol=1e-12, atol=1e-12
    )


def test_importance_mapping_mismatch_is_not_silently_skipped():
    X, y = _fixture("binary", np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = make_pipeline(LogisticRegression(max_iter=300), "classification", X.shape[1], 3, 7)
        pipe.fit(X[:32], y[:32])
        with pytest.raises(
            PermutationImportanceMappingError,
            match="permutation_importance_feature_name_count_mismatch",
        ):
            _permutation_importance_original_features(
                pipe,
                X[32:],
                y[32:],
                feature_names=FEATURE_NAMES[:-1],
                n_repeats=2,
                random_state=7,
                scoring="roc_auc",
            )
        with pytest.raises(
            PermutationImportanceMappingError,
            match="permutation_importance_feature_name_count_mismatch",
        ):
            cross_validate_estimator(
                "test",
                X,
                y,
                None,
                "classification",
                lambda: LogisticRegression(max_iter=300),
                feature_names=FEATURE_NAMES[:-1],
                modalities=["rna"],
                n_splits=3,
                seed=7,
                max_features=3,
                compute_importance=True,
                importance_repeats=2,
            )
