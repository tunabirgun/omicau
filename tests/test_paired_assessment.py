from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from omicau.interpretation.paired_assessment import assess_incremental, assess_standalone
from omicau.interpretation import utility
from omicau.interpretation.utility import _paired_gain
from omicau.models.base import CVResult, cross_validate_estimator
from omicau.models.classical import ClassicalBenchmarkFailure, run_classical_benchmarks
from omicau.config import ModalitySpec, OmicauConfig
from omicau.data.alignment import AlignedDataset, ModalityMatrix


def _result(
    *,
    role: str,
    full: bool = True,
    task: str = "classification",
) -> CVResult:
    y = np.array([0, 1] * 10, dtype=int)
    score = np.where(y == 1, 0.99, 0.01) if full else np.where(y == 1, 0.01, 0.99)
    fold_ids = np.arange(len(y), dtype=int) % 5
    split_identity = tuple(
        (
            tuple(np.flatnonzero(fold_ids != fold).tolist()),
            tuple(np.flatnonzero(fold_ids == fold).tolist()),
        )
        for fold in range(5)
    )
    return CVResult(
        name=role,
        task=task,
        metrics={"auroc": 1.0 if full else 0.0},
        fold_primary=[0.7, 0.8, 0.7, 0.8, 0.7],
        oof_true=y,
        oof_score=score,
        oof_pred=(score >= 0.5).astype(int),
        oof_groups=np.arange(len(y)),
        oof_row_keys=np.array(["row-" + str(i) for i in range(len(y))], dtype=object),
        oof_fold_ids=fold_ids,
        split_identity=split_identity,
        split_receipt_identity=("synthetic_receipt", 5),
        model_role=role,
        resampling_unit="group",
        score_class_order=(0, 1),
        score_positive_class=1,
        score_class_fit_identity=tuple((0, 1) for _ in range(5)),
        preprocessing_scope="fold_local_pipeline",
        preprocessing_fit_identity=tuple(pair[0] for pair in split_identity),
    )


def test_incremental_statuses_and_shared_bootstrap_vector():
    full = _result(role="full", full=True)
    reduced = _result(role="reduced:candidate", full=False)
    increased = assess_incremental(
        full, reduced, task="classification", candidate="candidate", family_size=1
    )
    assert increased["status"] == "increased_held_out_performance"
    assert increased["n_valid_resamples"] == 4000
    assert increased["ordinary_95_ci"]["lower"] == 1.0
    assert increased["decision_ci"]["upper"] == 1.0
    assert increased["ordinary_95_ci"]["quantile_method"] == "linear"

    full_bad = replace(reduced, model_role="full")
    reduced_good = replace(full, model_role="reduced:candidate")
    decreased = assess_incremental(
        full_bad, reduced_good, task="classification", candidate="candidate", family_size=1
    )
    assert decreased["status"] == "reduced_held_out_performance"

    equal = _result(role="reduced:candidate", full=True)
    unresolved = assess_incremental(
        full, equal, task="classification", candidate="candidate", family_size=2
    )
    assert unresolved["status"] == "unresolved"
    assert unresolved["n_valid_resamples"] == 8000
    assert unresolved["decision_ci"]["lower"] == 0.0
    assert unresolved["decision_ci"]["upper"] == 0.0


@pytest.mark.parametrize(
    ("alteration", "reason"),
    [
        ("reordered", "unmatched_row_keys"),
        ("duplicate", "invalid_row_key_or_fold_identity"),
        ("missing_row", "invalid_row_key_or_fold_identity"),
        ("infinite_row", "invalid_row_key_or_fold_identity"),
        ("targets", "unmatched_out_of_fold_targets"),
        ("groups", "unmatched_group_metadata"),
        ("folds", "unmatched_fold_ids"),
        ("split", "unmatched_split_identity"),
        ("nonfinite", "invalid_out_of_fold_scores"),
        ("role", "reduced_model_role_mismatch"),
        ("preprocessing", "invalid_preprocessing_scope"),
    ],
)
def test_pairing_failures_are_unavailable(alteration, reason):
    full = _result(role="full", full=True)
    reduced = _result(role="reduced:candidate", full=False)
    if alteration == "reordered":
        reduced = replace(reduced, oof_row_keys=reduced.oof_row_keys[::-1])
    elif alteration == "duplicate":
        keys = reduced.oof_row_keys.copy()
        keys[1] = keys[0]
        reduced = replace(reduced, oof_row_keys=keys)
    elif alteration == "missing_row":
        keys = reduced.oof_row_keys.copy()
        keys[0] = pd.NA
        reduced = replace(reduced, oof_row_keys=keys)
    elif alteration == "infinite_row":
        keys = reduced.oof_row_keys.copy()
        keys[0] = np.inf
        reduced = replace(reduced, oof_row_keys=keys)
    elif alteration == "targets":
        reduced = replace(reduced, oof_true=reduced.oof_true[::-1])
    elif alteration == "groups":
        reduced = replace(reduced, oof_groups=reduced.oof_groups[::-1])
    elif alteration == "folds":
        reduced = replace(reduced, oof_fold_ids=reduced.oof_fold_ids[::-1])
    elif alteration == "split":
        replacement = tuple(reversed(reduced.split_identity))
        reduced = replace(
            reduced,
            split_identity=replacement,
            preprocessing_fit_identity=tuple(pair[0] for pair in replacement),
        )
    elif alteration == "nonfinite":
        score = reduced.oof_score.copy()
        score[0] = np.nan
        reduced = replace(reduced, oof_score=score)
    elif alteration == "preprocessing":
        reduced = replace(reduced, preprocessing_scope="full_dataset")
    else:
        reduced = replace(reduced, model_role="reduced:other")
    result = assess_incremental(
        full, reduced, task="classification", candidate="candidate", family_size=1
    )
    assert result["status"] == "unavailable"
    assert result["reason"] == reason


def test_not_applicable_and_descriptive_standalone_are_separate():
    standalone = _result(role="standalone", full=True)
    assessed = assess_standalone(standalone, task="classification", n_boot=40)
    assert assessed["status"] == "descriptive"
    assert assessed["n_valid_resamples"] == 40
    assert assessed["estimate"] == 1.0

    not_applicable = assess_incremental(
        standalone, None, task="classification", candidate="candidate", family_size=0
    )
    assert not_applicable["status"] == "not_applicable"
    assert not_applicable["reason"] == "single_assay_ledger"
    assert "oof_row_keys" not in standalone.to_dict()
    assert "split_identity" not in standalone.to_dict()

    missing = assess_incremental(
        _result(role="full", full=True),
        None,
        task="classification",
        candidate="candidate",
        family_size=3,
    )
    assert missing["status"] == "unavailable"
    assert missing["reason"] == "missing_result"
    assert missing["multiplicity_family"]["family_size"] == 3


@pytest.mark.parametrize(
    ("alteration", "reason"),
    [
        ("one_group", "insufficient_independent_groups"),
        ("missing_group", "invalid_group_ids"),
        ("blank_group", "invalid_group_ids"),
        ("infinite_group", "invalid_group_ids"),
        ("no_groups", "sample_independence_not_declared"),
        ("receipt", "unmatched_split_receipt_identity"),
        ("class_order", "score_class_order_outcome_mismatch"),
        ("positive_class", "binary_score_positive_class_mismatch"),
        ("class_fit_mapping", "score_class_fit_mapping_invalid"),
        ("missing_preprocessing_evidence", "missing_fold_local_preprocessing_evidence"),
    ],
)
def test_structural_contract_failures_are_unavailable(alteration, reason):
    full = _result(role="full", full=True)
    reduced = _result(role="reduced:candidate", full=False)
    if alteration == "one_group":
        reduced = replace(reduced, oof_groups=np.repeat("same", len(reduced.oof_true)))
    elif alteration == "missing_group":
        groups = reduced.oof_groups.astype(object)
        groups[0] = pd.NA
        reduced = replace(reduced, oof_groups=groups)
    elif alteration == "blank_group":
        groups = reduced.oof_groups.astype(object)
        groups[0] = "  "
        reduced = replace(reduced, oof_groups=groups)
    elif alteration == "infinite_group":
        groups = reduced.oof_groups.astype(object)
        groups[0] = -np.inf
        reduced = replace(reduced, oof_groups=groups)
    elif alteration == "no_groups":
        reduced = replace(
            reduced,
            oof_groups=None,
            resampling_unit="sample",
            sample_independence_declared=False,
        )
    elif alteration == "receipt":
        reduced = replace(reduced, split_receipt_identity=("other_receipt", 5))
    elif alteration == "class_order":
        reduced = replace(reduced, score_class_order=(1, 0))
    elif alteration == "positive_class":
        reduced = replace(reduced, score_positive_class=0)
    elif alteration == "class_fit_mapping":
        reduced = replace(reduced, score_class_fit_identity=tuple((1, 0) for _ in range(5)))
    else:
        reduced = replace(reduced, preprocessing_fit_identity=None)
    result = assess_incremental(
        full, reduced, task="classification", candidate="candidate", family_size=1
    )
    assert result["status"] == "unavailable"
    assert result["reason"] == reason


def test_sample_resampling_requires_an_explicit_independence_contract():
    result = _result(role="standalone", full=True)
    sample_result = replace(
        result,
        oof_groups=None,
        resampling_unit="sample",
        sample_independence_declared=True,
    )
    assessment = assess_standalone(sample_result, task="classification", n_boot=40)
    assert assessment["status"] == "descriptive"
    assert assessment["resampling_unit"] == "sample"


@pytest.mark.parametrize("family_size", [True, 1.5, -1])
def test_invalid_family_size_is_unavailable(family_size):
    result = assess_incremental(
        _result(role="full", full=True),
        _result(role="reduced:candidate", full=False),
        task="classification",
        candidate="candidate",
        family_size=family_size,
    )
    assert result["status"] == "unavailable"
    assert result["reason"] == "invalid_multiplicity_family"


def _two_layer_utility_fixture():
    ids = ["u" + str(i) for i in range(20)]
    y = pd.Series([0, 1] * 10, index=ids)
    aligned = AlignedDataset(
        modalities={
            "baseline": ModalityMatrix(
                "baseline", pd.DataFrame(np.arange(40, dtype=float).reshape(20, 2), index=ids)
            ),
            "candidate": ModalityMatrix(
                "candidate", pd.DataFrame(np.arange(40, 80, dtype=float).reshape(20, 2), index=ids)
            ),
        },
        y=y,
        task="classification",
        sample_ids=ids,
    )
    full = replace(
        _result(role="full", full=True),
        name="random_forest::FUSION",
        modalities=["baseline", "candidate"],
    )
    minus_candidate = replace(
        _result(role="reduced:candidate", full=False),
        name="random_forest::FUSION-minus-candidate",
        modalities=["baseline", "candidate"],
    )
    minus_baseline = replace(
        _result(role="reduced:baseline", full=False),
        name="random_forest::FUSION-minus-baseline",
        modalities=["candidate"],
    )
    baseline = replace(
        _result(role="standalone", full=True),
        name="random_forest::baseline",
        modalities=["baseline"],
    )
    candidate = replace(
        _result(role="standalone", full=True),
        name="random_forest::candidate",
        modalities=["candidate"],
    )
    return aligned, {
        "primary_metric": "auroc",
        "reference_estimator": "random_forest",
        "results": [full, minus_candidate, minus_baseline, baseline, candidate],
        "controls": [],
        "intended_candidate_layers": ["baseline", "candidate"],
        "intended_family_size": 2,
    }


def test_utility_rejects_role_correct_reduced_model_with_candidate_still_present(monkeypatch):
    aligned, classical = _two_layer_utility_fixture()
    monkeypatch.setattr(
        utility,
        "assess_standalone",
        lambda *_args, **_kwargs: {
            "status": "descriptive", "estimate": 0.5, "ordinary_95_ci": None,
            "resampling_unit": "group", "n_valid_resamples": 0, "n_attempts": 0,
            "reason": None,
        },
    )
    monkeypatch.setattr(
        utility,
        "assess_incremental",
        lambda *_args, **_kwargs: {
            "status": "unresolved", "estimate": 0.0, "ordinary_95_ci": None,
            "decision_ci": None, "resampling_unit": "group",
            "n_valid_resamples": 0, "n_attempts": 0, "reason": None,
        },
    )
    ledger = utility.build_utility_ledger(aligned, classical, {"enabled": False}, {}, {})
    candidate = next(row for row in ledger["modality_ledger"] if row["modality"] == "candidate")
    assert candidate["incremental_assessment"]["status"] == "unavailable"
    assert candidate["incremental_assessment"]["reason"] == "reduced_model_intended_layer_mismatch"


def test_utility_rejects_corrupted_declared_family_without_rederiving_it(monkeypatch):
    aligned, classical = _two_layer_utility_fixture()
    classical["intended_family_size"] = 0
    monkeypatch.setattr(
        utility,
        "assess_standalone",
        lambda *_args, **_kwargs: {
            "status": "descriptive", "estimate": 0.5, "ordinary_95_ci": None,
            "resampling_unit": "group", "n_valid_resamples": 0, "n_attempts": 0,
            "reason": None,
        },
    )
    ledger = utility.build_utility_ledger(aligned, classical, {"enabled": False}, {}, {})
    assert ledger["intended_candidate_layers"] is None
    assert ledger["intended_family_size"] is None
    assert all(
        row["incremental_assessment"]["reason"] == "invalid_multiplicity_family"
        for row in ledger["modality_ledger"]
    )


def test_secondary_fold_context_rejects_unequal_arrays():
    with pytest.raises(ValueError, match="equal_arrays"):
        _paired_gain([0.7, 0.8], [0.7])


def test_regression_relative_improvement_can_coexist_with_negative_scores():
    y = np.arange(12, dtype=float)
    common = {
        "task": "regression",
        "metrics": {"r2": -1.0},
        "oof_true": y,
        "oof_groups": np.arange(len(y)),
        "oof_row_keys": np.array(["r" + str(i) for i in range(len(y))], dtype=object),
        "oof_fold_ids": np.arange(len(y)) % 3,
        "split_identity": tuple(
            (
                tuple(np.flatnonzero(np.arange(len(y)) % 3 != fold).tolist()),
                tuple(np.flatnonzero(np.arange(len(y)) % 3 == fold).tolist()),
            )
            for fold in range(3)
        ),
        "split_receipt_identity": ("regression_receipt", 3),
        "resampling_unit": "group",
        "preprocessing_scope": "fold_local_pipeline",
    }
    common["preprocessing_fit_identity"] = tuple(pair[0] for pair in common["split_identity"])
    full = CVResult(name="full", oof_score=np.zeros(len(y)), oof_pred=np.zeros(len(y)),
                    model_role="full", **common)
    reduced = CVResult(name="reduced", oof_score=np.full(len(y), -10.0),
                       oof_pred=np.full(len(y), -10.0), model_role="reduced:candidate", **common)
    result = assess_incremental(full, reduced, task="regression", candidate="candidate", family_size=1)
    assert result["status"] == "increased_held_out_performance"
    assert result["estimate"] > 0


def test_group_bootstrap_matches_explicit_expansion_and_linear_quantiles():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    groups = np.repeat(np.arange(4), 2)
    folds = np.arange(len(y)) % 2
    split_identity = tuple(
        (
            tuple(np.flatnonzero(folds != fold).tolist()),
            tuple(np.flatnonzero(folds == fold).tolist()),
        )
        for fold in range(2)
    )
    common = {
        "task": "classification",
        "metrics": {"auroc": 0.0},
        "oof_true": y,
        "oof_pred": (np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.6, 0.4]) >= 0.5).astype(int),
        "oof_groups": groups,
        "oof_row_keys": np.array(["t" + str(i) for i in range(len(y))], dtype=object),
        "oof_fold_ids": folds,
        "split_identity": split_identity,
        "split_receipt_identity": ("draw_receipt", 2),
        "resampling_unit": "group",
        "score_class_order": (0, 1),
        "score_positive_class": 1,
        "score_class_fit_identity": ((0, 1), (0, 1)),
        "preprocessing_scope": "fold_local_pipeline",
        "preprocessing_fit_identity": tuple(pair[0] for pair in split_identity),
    }
    full_score = np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.6, 0.4])
    reduced_score = np.array([0.4, 0.6, 0.45, 0.55, 0.5, 0.5, 0.5, 0.5])
    full = CVResult(name="full", oof_score=full_score, model_role="full", **common)
    reduced = CVResult(
        name="reduced",
        oof_score=reduced_score,
        oof_pred=(reduced_score >= 0.5).astype(int),
        model_role="reduced:candidate",
        **{key: value for key, value in common.items() if key != "oof_pred"},
    )
    assessed = assess_incremental(
        full, reduced, task="classification", candidate="candidate", family_size=1, seed=73
    )
    rng = np.random.default_rng(73)
    positions = [np.flatnonzero(groups == level) for level in np.unique(groups)]
    reference = []
    for _ in range(4000):
        selected = rng.integers(0, len(positions), size=len(positions))
        index = np.concatenate([positions[value] for value in selected])
        reference.append(
            roc_auc_score(y[index], full_score[index])
            - roc_auc_score(y[index], reduced_score[index])
        )
    reference = np.asarray(reference)
    assert assessed["n_valid_resamples"] == len(reference)
    assert assessed["ordinary_95_ci"]["lower"] == pytest.approx(
        np.quantile(reference, 0.025, method="linear")
    )
    assert assessed["ordinary_95_ci"]["upper"] == pytest.approx(
        np.quantile(reference, 0.975, method="linear")
    )


class _RecordingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, seen):
        self.seen = seen

    def fit(self, X, y):
        self.seen.append(len(X))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.4), np.full(len(X), 0.6)])


def test_cross_validation_fits_preprocessing_on_training_slices_only():
    seen: list[int] = []
    X = np.arange(48, dtype=float).reshape(12, 4)
    y = np.array([0, 1] * 6)
    result = cross_validate_estimator(
        "reference::FUSION",
        X,
        y,
        None,
        "classification",
        lambda: _RecordingClassifier(seen),
        feature_names=["f0", "f1", "f2", "f3"],
        modalities=["a", "b"],
        n_splits=3,
        seed=3,
        max_features=2,
        row_keys=["sample-" + str(i) for i in range(len(y))],
        model_role="full",
        independent_rows=True,
    )
    assert seen == [8, 8, 8]
    assert np.all(result.oof_fold_ids >= 0)
    assert result.oof_row_keys.tolist() == ["sample-" + str(i) for i in range(len(y))]


@pytest.mark.parametrize(
    "groups",
    [
        np.repeat("one", 12),
        np.array([pd.NA] + list(range(11)), dtype=object),
        np.array([""] + list(range(11)), dtype=object),
        np.array([np.inf] + list(range(11)), dtype=object),
    ],
)
def test_cross_validation_rejects_invalid_group_contract(groups):
    X = np.arange(48, dtype=float).reshape(12, 4)
    y = np.array([0, 1] * 6)
    with pytest.raises(ValueError):
        cross_validate_estimator(
            "reference::FUSION",
            X,
            y,
            groups,
            "classification",
            lambda: LogisticRegression(max_iter=1000, random_state=4),
            feature_names=["f0", "f1", "f2", "f3"],
            modalities=["a", "b"],
            n_splits=3,
            seed=4,
            row_keys=["group-" + str(i) for i in range(len(y))],
            model_role="full",
        )


def test_native_classifier_results_produce_an_available_paired_contrast():
    rng = np.random.default_rng(17)
    y = np.array([0, 1] * 20)
    candidate_signal = y.astype(float) * 4.0 + rng.normal(0.0, 0.15, size=len(y))
    baseline_noise = rng.normal(size=len(y))
    full = np.column_stack([baseline_noise, candidate_signal])
    reduced = baseline_noise.reshape(-1, 1)
    groups = np.arange(len(y))
    common = {
        "task": "classification",
        "estimator_factory": lambda: LogisticRegression(max_iter=1000, random_state=17),
        "feature_names": ["baseline", "candidate"],
        "modalities": ["baseline", "candidate"],
        "n_splits": 4,
        "seed": 17,
        "row_keys": ["native-" + str(i) for i in range(len(y))],
    }
    full_result = cross_validate_estimator(
        "reference::FUSION",
        full,
        y,
        groups,
        model_role="full",
        **common,
    )
    reduced_result = cross_validate_estimator(
        "reference::FUSION-minus-candidate",
        reduced,
        y,
        groups,
        model_role="reduced:candidate",
        **(common | {"feature_names": ["baseline"], "modalities": ["baseline"]}),
    )
    assessment = assess_incremental(
        full_result,
        reduced_result,
        task="classification",
        candidate="candidate",
        family_size=1,
    )
    assert full_result.oof_row_keys is not None
    assert full_result.oof_fold_ids is not None
    assert full_result.split_identity == reduced_result.split_identity
    assert full_result.model_role == "full"
    assert reduced_result.model_role == "reduced:candidate"
    assert assessment["status"] == "increased_held_out_performance"


def test_native_regression_results_produce_an_available_paired_contrast():
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(23)
    y = np.linspace(-2.0, 2.0, 40)
    candidate_signal = y + rng.normal(0.0, 0.05, size=len(y))
    baseline_noise = rng.normal(size=len(y))
    groups = np.arange(len(y))
    common = {
        "task": "regression",
        "estimator_factory": lambda: Ridge(alpha=1.0),
        "n_splits": 4,
        "seed": 23,
        "row_keys": ["regression-" + str(i) for i in range(len(y))],
    }
    full = cross_validate_estimator(
        "reference::FUSION",
        np.column_stack([baseline_noise, candidate_signal]),
        y,
        groups,
        feature_names=["baseline", "candidate"],
        modalities=["baseline", "candidate"],
        model_role="full",
        **common,
    )
    reduced = cross_validate_estimator(
        "reference::FUSION-minus-candidate",
        baseline_noise.reshape(-1, 1),
        y,
        groups,
        feature_names=["baseline"],
        modalities=["baseline"],
        model_role="reduced:candidate",
        **common,
    )
    assessment = assess_incremental(
        full, reduced, task="regression", candidate="candidate", family_size=1
    )
    assert full.score_class_order is None
    assert full.score_positive_class is None
    assert assessment["status"] == "increased_held_out_performance"


def test_native_multiclass_results_bind_score_class_mapping():
    rng = np.random.default_rng(31)
    y = np.repeat(np.arange(3), 12)
    candidate_signal = np.eye(3)[y] + rng.normal(0.0, 0.03, size=(len(y), 3))
    baseline_noise = rng.normal(size=(len(y), 2))
    groups = np.arange(len(y))
    common = {
        "task": "classification",
        "estimator_factory": lambda: LogisticRegression(max_iter=1000, random_state=31),
        "n_splits": 3,
        "seed": 31,
        "row_keys": ["multiclass-" + str(i) for i in range(len(y))],
    }
    full = cross_validate_estimator(
        "reference::FUSION",
        np.column_stack([baseline_noise, candidate_signal]),
        y,
        groups,
        feature_names=["n0", "n1", "c0", "c1", "c2"],
        modalities=["baseline", "candidate"],
        model_role="full",
        **common,
    )
    reduced = cross_validate_estimator(
        "reference::FUSION-minus-candidate",
        baseline_noise,
        y,
        groups,
        feature_names=["n0", "n1"],
        modalities=["baseline"],
        model_role="reduced:candidate",
        **common,
    )
    assessment = assess_incremental(
        full, reduced, task="classification", candidate="candidate", family_size=1
    )
    assert full.score_class_order == (0, 1, 2)
    assert full.score_positive_class is None
    assert full.score_class_fit_identity == ((0, 1, 2),) * 3
    assert assessment["status"] == "increased_held_out_performance"


def test_required_fit_failure_preserves_the_intended_family(monkeypatch):
    ids = ["f" + str(i) for i in range(12)]
    y = pd.Series([0, 1] * 6, index=ids)
    aligned = AlignedDataset(
        modalities={
            "baseline": ModalityMatrix(
                "baseline", pd.DataFrame(np.arange(24, dtype=float).reshape(12, 2), index=ids)
            ),
            "candidate": ModalityMatrix(
                "candidate", pd.DataFrame(np.arange(24, 48, dtype=float).reshape(12, 2), index=ids)
            ),
        },
        y=y,
        task="classification",
        sample_ids=ids,
    )
    config = OmicauConfig()
    config.classical.models = ["random_forest"]
    config.classical.max_features = None
    config.controls.enabled = False
    config.xai.enabled = False
    config.cv.n_splits = 3
    config.compute.cores = 1
    from omicau.models import classical

    original = classical.cross_validate_estimator

    def fail_only_candidate_leaveout(name, *args, **kwargs):
        if name == "random_forest::FUSION-minus-candidate":
            raise ValueError("synthetic_required_fit_failure")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(classical, "cross_validate_estimator", fail_only_candidate_leaveout)
    with pytest.raises(ClassicalBenchmarkFailure) as caught:
        run_classical_benchmarks(aligned, config)
    receipt = caught.value.failure_receipt
    assert receipt["status"] == "not_emitted_whole_benchmark_failed"
    assert receipt["fit_name"] == "random_forest::FUSION-minus-candidate"
    assert receipt["failure_reason"] == "ValueError"
    assert receipt["intended_candidate_layers"] == ["baseline", "candidate"]
    assert receipt["intended_family_size"] == 2


def test_configured_layer_absent_after_alignment_fails_before_fitting():
    ids = ["a" + str(i) for i in range(12)]
    y = pd.Series([0, 1] * 6, index=ids)
    aligned = AlignedDataset(
        modalities={
            "baseline": ModalityMatrix(
                "baseline", pd.DataFrame(np.arange(24, dtype=float).reshape(12, 2), index=ids)
            ),
            "candidate": ModalityMatrix(
                "candidate", pd.DataFrame(np.arange(24, 48, dtype=float).reshape(12, 2), index=ids)
            ),
        },
        y=y,
        task="classification",
        sample_ids=ids,
    )
    config = OmicauConfig()
    config.modalities = [
        ModalitySpec("baseline"),
        ModalitySpec("candidate"),
        ModalitySpec("absent"),
    ]
    config.classical.models = ["random_forest"]
    with pytest.raises(ClassicalBenchmarkFailure) as caught:
        run_classical_benchmarks(aligned, config)
    receipt = caught.value.failure_receipt
    assert receipt["fit_name"] == "alignment"
    assert receipt["intended_candidate_layers"] == ["baseline", "candidate", "absent"]
    assert receipt["intended_family_size"] == 3


def test_hyphenated_candidate_name_is_preserved_in_native_model_role():
    ids = ["h" + str(i) for i in range(12)]
    y = pd.Series([0, 1] * 6, index=ids)
    aligned = AlignedDataset(
        modalities={
            "baseline": ModalityMatrix(
                "baseline", pd.DataFrame(np.arange(24, dtype=float).reshape(12, 2), index=ids)
            ),
            "copy-number": ModalityMatrix(
                "copy-number", pd.DataFrame(np.arange(24, 48, dtype=float).reshape(12, 2), index=ids)
            ),
        },
        y=y,
        task="classification",
        sample_ids=ids,
        groups=pd.Series(ids, index=ids),
    )
    config = OmicauConfig()
    config.classical.models = ["random_forest"]
    config.classical.max_features = None
    config.controls.enabled = False
    config.xai.enabled = False
    config.cv.n_splits = 3
    config.compute.cores = 1
    classical = run_classical_benchmarks(aligned, config)
    reduced = next(
        result
        for result in classical["results"]
        if result.name == "random_forest::FUSION-minus-copy-number"
    )
    assert reduced.model_role == "reduced:copy-number"
