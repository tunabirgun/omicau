from __future__ import annotations

from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

import omicau.models.classical as classical
import omicau.models.base as base
from omicau.models.base import cross_validate_estimator
from omicau.models.base import CVResult
from omicau.models.split_plan import ValidatedSplitPlan, validate_split_manifest


def _manifest() -> dict[str, object]:
    return {
        "outer_folds": [
            {
                "train": [0, 1, 2, 3],
                "assessment": [4, 5, 6, 7],
                "inner_folds": [
                    {"train": [0, 1], "assessment": [2, 3]},
                    {"train": [2, 3], "assessment": [0, 1]},
                ],
            },
            {
                "train": [4, 5, 6, 7],
                "assessment": [0, 1, 2, 3],
                "inner_folds": [
                    {"train": [4, 5], "assessment": [6, 7]},
                    {"train": [6, 7], "assessment": [4, 5]},
                ],
            },
        ]
    }


def _plan(y: np.ndarray | None = None):
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1]) if y is None else y
    return validate_split_manifest(
        _manifest(),
        n_samples=8,
        groups=[f"g{index}" for index in range(8)],
        y=labels,
        task="classification",
        requested_outer_k=2,
        requested_inner_k=2,
        minimum_training_groups=2,
        minimum_assessment_groups=2,
        minimum_training_groups_per_class=1,
        minimum_assessment_groups_per_class=1,
    )


def _regression_plan():
    return validate_split_manifest(
        _manifest(),
        n_samples=8,
        groups=[f"g{index}" for index in range(8)],
        y=np.arange(8, dtype=float),
        task="regression",
        requested_outer_k=2,
        requested_inner_k=2,
        minimum_training_groups=2,
        minimum_assessment_groups=2,
        minimum_regression_assessment_groups=2,
        minimum_regression_assessment_variance=0.01,
    )


def _expanded_manifest():
    manifest = deepcopy(_manifest())
    for fold in manifest["outer_folds"]:
        fold["train"] = [row for index in fold["train"] for row in (2 * index, 2 * index + 1)]
        fold["assessment"] = [
            row for index in fold["assessment"] for row in (2 * index, 2 * index + 1)
        ]
        for inner in fold["inner_folds"]:
            inner["train"] = [
                row for index in inner["train"] for row in (2 * index, 2 * index + 1)
            ]
            inner["assessment"] = [
                row
                for index in inner["assessment"]
                for row in (2 * index, 2 * index + 1)
            ]
    return manifest


def _repeated_plan():
    groups = [f"g{index}" for index in range(8) for _ in range(2)]
    y = [label for label in [0, 1, 0, 1, 0, 1, 0, 1] for _ in range(2)]
    return validate_split_manifest(
        _expanded_manifest(),
        n_samples=16,
        groups=groups,
        y=y,
        task="classification",
        requested_outer_k=2,
        requested_inner_k=2,
        minimum_training_groups=2,
        minimum_assessment_groups=2,
        minimum_training_groups_per_class=1,
        minimum_assessment_groups_per_class=1,
    )


def _factory(task: str):
    if task == "classification":
        return lambda: LogisticRegression(max_iter=200)
    return lambda: Ridge()


def _run_base(plan, *, y=None, groups=None, X=None, n_splits=2, task="classification"):
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1]) if y is None else np.asarray(y)
    matrix = np.arange(16, dtype=float).reshape(8, 2) if X is None else np.asarray(X)
    group_values = (
        np.asarray([f"g{index}" for index in range(8)], dtype=object)
        if groups is None else np.asarray(groups, dtype=object)
    )
    return cross_validate_estimator(
        "test",
        matrix,
        labels,
        group_values,
        task,
        _factory(task),
        feature_names=["a", "b"],
        modalities=["m"],
        n_splits=n_splits,
        seed=3,
        validated_plan=plan,
    )


def _assert_runtime_mismatch_before_estimator(
    plan, *, y, groups, X=None, task="classification"
):
    constructed = 0

    def factory():
        nonlocal constructed
        constructed += 1
        return LogisticRegression(max_iter=200) if task == "classification" else Ridge()

    matrix = (
        np.arange(2 * len(y), dtype=float).reshape(len(y), 2)
        if X is None else np.asarray(X)
    )
    with pytest.raises(
        TypeError, match="^validated_split_plan_runtime_universe_mismatch$"
    ):
        cross_validate_estimator(
            "test",
            matrix,
            np.asarray(y),
            np.asarray(groups, dtype=object),
            task,
            factory,
            feature_names=["a", "b"],
            modalities=["m"],
            n_splits=2,
            seed=3,
            validated_plan=plan,
        )
    assert constructed == 0


def _config():
    return SimpleNamespace(
        seed=3,
        compute=SimpleNamespace(cores=1),
        classical=SimpleNamespace(models=["linear"], max_features=None),
        cv=SimpleNamespace(
            n_splits=2,
            shuffle=True,
            n_bootstrap=20,
            batch_blocked=False,
            batch_adjust_sensitivity=False,
            batch_adjust_min_per_batch=1,
        ),
        xai=SimpleNamespace(enabled=False, permutation_repeats=1),
        controls=SimpleNamespace(
            enabled=False,
            shuffle_target=False,
            shuffle_features=False,
            random_noise=False,
        ),
    )


def _control_contract(n_rows: int = 8) -> dict[str, object]:
    return {
        "strata": ["registered_block"] * n_rows,
        "strata_schema": {
            "name": "registered_block",
            "version": "1",
            "fields": ["registered_block"],
            "target_derived": False,
        },
        "permutation_registry": {
            "schema_version": "c07_private_registry_binding_v1",
            "purpose": "group_endpoint_permutation",
            "nonce_hex": hashlib.sha256(b"synthetic c07 registry nonce").hexdigest(),
            "registry_id": "synthetic_group_endpoint_permutation",
            "artifact": {
                "artifact_id": "synthetic_fixture",
                "sha256": hashlib.sha256(b"synthetic c07 fixture").hexdigest(),
            },
            "entries": [
                {
                    "entry_id": "synthetic_entry",
                    "artifact_sha256": hashlib.sha256(b"synthetic c07 entry").hexdigest(),
                }
            ],
        },
        "exchangeability_contract": {
            "unit": "highest_exchangeable_group",
            "scope": "outer_train_only",
        },
        "minimum_distinct_nonidentity_assignments": 1,
        "fold_seeds": [31, 37],
    }


class _Aligned:
    task = "classification"
    modality_names = ["m1", "m2"]
    batch = None
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    groups = pd.Series([f"g{index}" for index in range(8)])

    def __init__(self):
        rows = np.arange(8, dtype=float)
        self._matrices = {
            "m1": np.column_stack([rows, self.y.to_numpy()]),
            "m2": np.column_stack([rows + 100.0, 1.0 - self.y.to_numpy()]),
        }

    def concat_matrix(self, modalities):
        matrix = np.hstack([self._matrices[name] for name in modalities])
        names = [
            f"{modality}:{column}"
            for modality in modalities
            for column in range(self._matrices[modality].shape[1])
        ]
        return matrix, names


def test_supplied_plan_is_used_exactly_and_receipt_is_aggregate_only():
    result = _run_base(_plan())
    assert result.extra["n_splits"] == 2
    assert result.extra["split_plan_status"] == "validated_development_plan"
    receipt = result.extra["split_plan_receipt"]
    assert receipt["split_manifest_sha256"] is None
    assert not any("indices" in key or "groups" == key for key in receipt)
    assert np.isfinite(result.oof_pred).all()


def test_legacy_path_is_explicitly_non_benchmark():
    result = _run_base(None)
    assert result.extra["split_plan_status"] == "legacy_generated_non_benchmark"
    assert result.extra["split_plan_receipt"] is None


def test_cross_validation_derives_fold_targets_and_scores_original_truth(
    monkeypatch,
):
    fitted: list[np.ndarray] = []

    class _Estimator:
        classes_ = np.asarray([0, 1])

    class _Pipeline:
        named_steps = {"estimator": _Estimator()}

        def fit(self, X, y):
            fitted.append(np.asarray(y).copy())
            return self

        def predict_proba(self, X):
            return np.column_stack((np.full(len(X), 0.5), np.full(len(X), 0.5)))

    monkeypatch.setattr(base, "make_pipeline", lambda *args, **kwargs: _Pipeline())
    result = base.cross_validate_estimator(
        "control::group_permuted_target",
        np.arange(16, dtype=float).reshape(8, 2),
        np.asarray([0, 1, 0, 1, 0, 1, 0, 1]),
        np.asarray([f"g{index}" for index in range(8)], dtype=object),
        "classification",
        lambda: _Estimator(),
        feature_names=["a", "b"],
        modalities=["m"],
        n_splits=2,
        seed=3,
        validated_plan=_plan(),
        validated_control_contract=_control_contract(),
    )
    expected_multisets = [[0, 0, 1, 1], [0, 0, 1, 1]]
    assert [sorted(observed.tolist()) for observed in fitted] == expected_multisets
    assert all(not np.array_equal(observed, [0, 1, 0, 1]) for observed in fitted)
    assert np.array_equal(result.oof_true, np.asarray([0, 1, 0, 1, 0, 1, 0, 1]))
    assert result.extra["control_execution_receipt"]["decision"] == "development_only"


def test_raw_fold_training_target_bypass_is_not_in_the_cv_api():
    with pytest.raises(TypeError, match="fold_training_targets"):
        cross_validate_estimator(
            "test",
            np.arange(16, dtype=float).reshape(8, 2),
            np.asarray([0, 1, 0, 1, 0, 1, 0, 1]),
            np.asarray([f"g{index}" for index in range(8)], dtype=object),
            "classification",
            _factory("classification"),
            feature_names=["a", "b"],
            modalities=["m"],
            n_splits=2,
            seed=3,
            validated_plan=_plan(),
            fold_training_targets=(np.asarray([0, 1, 0, 1]),),
        )


def test_regression_uses_the_same_supplied_outer_partitions():
    result = _run_base(
        _regression_plan(), y=np.arange(8, dtype=float), task="regression"
    )
    assert result.extra["split_plan_status"] == "validated_development_plan"
    assert result.extra["n_splits"] == 2
    assert np.isfinite(result.oof_pred).all()


def test_plan_task_mismatch_fails_closed():
    with pytest.raises(TypeError, match="runtime_universe_mismatch"):
        _run_base(_plan(), y=np.arange(8, dtype=float), task="regression")


def test_same_support_class_swap_fails_before_estimator_construction():
    y = np.asarray([1, 0, 0, 1, 0, 1, 0, 1])
    _assert_runtime_mismatch_before_estimator(
        _plan(), y=y, groups=[f"g{index}" for index in range(8)]
    )


def test_regression_shift_fails_before_estimator_construction():
    _assert_runtime_mismatch_before_estimator(
        _regression_plan(),
        y=np.arange(8, dtype=float) + 10.0,
        groups=[f"g{index}" for index in range(8)],
        task="regression",
    )


@pytest.mark.parametrize(
    "groups",
    [
        ["changed", *[f"g{index}" for index in range(1, 8)]],
        ["g1", "g0", *[f"g{index}" for index in range(2, 8)]],
    ],
)
def test_group_relabel_or_reassignment_fails_before_estimator_construction(groups):
    _assert_runtime_mismatch_before_estimator(
        _plan(), y=[0, 1, 0, 1, 0, 1, 0, 1], groups=groups
    )


def test_repeated_group_row_reorder_fails_before_estimator_construction():
    groups = [f"g{index}" for index in range(8) for _ in range(2)]
    y = [label for label in [0, 1, 0, 1, 0, 1, 0, 1] for _ in range(2)]
    X = np.arange(32, dtype=float).reshape(16, 2)
    order = [0, 2, 1, *range(3, 16)]
    _assert_runtime_mismatch_before_estimator(
        _repeated_plan(),
        y=[y[index] for index in order],
        groups=[groups[index] for index in order],
        X=X[order],
    )


@pytest.mark.parametrize(
    ("change", "error_type", "message"),
    [
        (lambda kwargs: kwargs.update(X=np.ones((7, 2))), ValueError, "wrong_row_universe"),
        (lambda kwargs: kwargs.update(n_splits=3), ValueError, "outer_fold_count_mismatch"),
        (
            lambda kwargs: kwargs.update(
                groups=["changed" if index == 0 else f"g{index}" for index in range(8)]
            ),
            TypeError,
            "runtime_universe_mismatch",
        ),
        (
            lambda kwargs: kwargs.update(y=np.zeros(8, dtype=int)),
            TypeError,
            "runtime_universe_mismatch",
        ),
    ],
)
def test_supplied_plan_mismatches_fail_closed(change, error_type, message):
    kwargs = {}
    change(kwargs)
    with pytest.raises(error_type, match=message):
        _run_base(_plan(), **kwargs)


def test_assessment_reuse_or_omission_at_execution_boundary_fails(monkeypatch):
    original = ValidatedSplitPlan._private_outer_splits

    def broken(self):
        splits = list(original(self))
        yield splits[0]
        yield splits[0]

    monkeypatch.setattr(ValidatedSplitPlan, "_private_outer_splits", broken)
    with pytest.raises(ValueError, match="outer_assessment_coverage_invalid"):
        _run_base(_plan())


def test_classical_result_carries_only_aggregate_plan_status():
    output = classical.run_classical_benchmarks(
        _Aligned(), _config(), validated_plan=_plan()
    )
    assert output["split_plan_status"] == "validated_development_plan"
    assert output["split_plan_receipt"]["split_manifest_sha256"] is None
    assert all(
        result.extra["split_plan_status"] == "validated_development_plan"
        for result in output["results"]
    )


def test_plan_with_legacy_controls_fails_before_any_estimator(monkeypatch):
    config = _config()
    config.controls.enabled = True
    config.controls.shuffle_target = True
    config.controls.shuffle_features = True
    config.controls.random_noise = True
    constructed = 0
    original = classical._estimator_factory

    def watched_factory(*args, **kwargs):
        nonlocal constructed
        constructed += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(classical, "_estimator_factory", watched_factory)
    with pytest.raises(
        ValueError, match="^validated_plan_feature_controls_require_c07_integration$"
    ) as caught:
        classical.run_classical_benchmarks(
            _Aligned(), config, validated_plan=_plan()
        )
    assert constructed == 0
    assert "g0" not in str(caught.value)


def test_validated_target_control_requires_explicit_contract_before_estimator(monkeypatch):
    config = _config()
    config.controls.enabled = True
    config.controls.shuffle_target = True
    constructed = 0

    def forbidden_factory(*args, **kwargs):
        nonlocal constructed
        constructed += 1
        raise AssertionError("estimator construction must not start")

    monkeypatch.setattr(classical, "_estimator_factory", forbidden_factory)
    with pytest.raises(
        ValueError, match="^validated_plan_target_control_contract_required$"
    ):
        classical.run_classical_benchmarks(
            _Aligned(), config, validated_plan=_plan()
        )
    assert constructed == 0


def test_validated_target_control_routes_fold_targets_and_keeps_truth(monkeypatch):
    config = _config()
    config.controls.enabled = True
    config.controls.shuffle_target = True
    captured: dict[str, object] = {}

    def fake_cv(name, X, y, groups, task, factory, **kwargs):
        receipt = None
        if name == "control::group_permuted_target":
            captured["contract"] = kwargs["validated_control_contract"]
            captured["truth"] = np.asarray(y).copy()
            splits, _, _ = classical.resolve_validated_cv_splits(
                kwargs["validated_plan"], X, y, groups, task, kwargs["n_splits"]
            )
            _, receipt = classical._execute_fold_endpoint_permutations(
                contract=kwargs["validated_control_contract"],
                outer_splits=splits,
                groups=groups,
                task=task,
                y=y,
            )
        return CVResult(
            name=name,
            task=task,
            metrics={"auroc": 0.5},
            fold_primary=[0.5, 0.5],
            per_fold=[{"auroc": 0.5}, {"auroc": 0.5}],
            n_features=np.asarray(X).shape[1],
            modalities=list(kwargs["modalities"]),
            oof_true=np.asarray(y).copy(),
            oof_score=np.full(len(y), 0.5),
            oof_pred=np.zeros(len(y)),
            oof_groups=np.asarray(groups).copy(),
            extra={"n_splits": 2, "control_execution_receipt": receipt},
        )

    monkeypatch.setattr(classical, "cross_validate_estimator", fake_cv)
    monkeypatch.setattr(classical, "_run_stacking", lambda *args, **kwargs: None)
    output = classical.run_classical_benchmarks(
        _Aligned(),
        config,
        validated_plan=_plan(),
        validated_control_contract=_control_contract(),
    )
    assert captured["contract"] == _control_contract()
    assert np.array_equal(captured["truth"], _Aligned.y.to_numpy())
    assert output["controls"][0].oof_true.tolist() == _Aligned.y.tolist()
    receipt = output["control_execution_receipt"]
    assert receipt["decision"] == "development_only"
    assert receipt["fold_count"] == 2
    assert "seed" not in repr(receipt).lower()


def test_validated_target_control_runs_through_classification_pipeline():
    config = _config()
    config.controls.enabled = True
    config.controls.shuffle_target = True
    output = classical.run_classical_benchmarks(
        _Aligned(),
        config,
        validated_plan=_plan(),
        validated_control_contract=_control_contract(),
    )
    assert [control.name for control in output["controls"]] == [
        "control::group_permuted_target"
    ]
    assert output["controls"][0].oof_true.tolist() == _Aligned.y.tolist()
    assert output["control_execution_receipt"]["decision"] == "development_only"


def test_validated_target_control_runs_through_regression_pipeline():
    config = _config()
    config.controls.enabled = True
    config.controls.shuffle_target = True
    aligned = _Aligned()
    aligned.task = "regression"
    aligned.y = pd.Series(np.arange(8, dtype=float))
    aligned.modality_names = ["m1"]
    aligned._matrices = {"m1": aligned._matrices["m1"]}
    output = classical.run_classical_benchmarks(
        aligned,
        config,
        validated_plan=_regression_plan(),
        validated_control_contract=_control_contract(),
    )
    assert [control.name for control in output["controls"]] == [
        "control::group_permuted_target"
    ]
    assert np.array_equal(output["controls"][0].oof_true, aligned.y.to_numpy())
    assert output["control_execution_receipt"]["decision"] == "development_only"


def test_legacy_controls_still_run_without_validated_plan():
    config = _config()
    config.controls.enabled = True
    config.controls.shuffle_target = True
    config.controls.shuffle_features = True
    config.controls.random_noise = True
    output = classical.run_classical_benchmarks(_Aligned(), config)
    assert output["split_plan_status"] == "legacy_generated_non_benchmark"
    assert {control.name for control in output["controls"]} == {
        "control::shuffled_target",
        "control::shuffled_features",
        "control::random_noise",
    }


def test_plan_bound_batch_adjustment_is_not_silently_skipped():
    aligned = _Aligned()
    aligned.batch = pd.Series([f"b{index}" for index in range(8)])
    matrix, names = aligned.concat_matrix(aligned.modality_names)
    with pytest.raises(ValueError, match="batch_adjustment_ineligible"):
        classical._run_batch_adjusted_fusion(
            aligned,
            _config(),
            matrix,
            names,
            aligned.modality_names,
            "linear",
            aligned.y.to_numpy(),
            aligned.groups.to_numpy(),
            1,
            3,
            _plan(),
        )


def test_stacking_base_meta_features_are_nested_inner_oof(monkeypatch):
    fits: list[frozenset[int]] = []

    class SpyPipeline:
        def __init__(self):
            self.named_steps = {"estimator": self}
            self.classes_ = np.asarray([0, 1])

        def fit(self, X, y):
            if X.shape[1] == 2 and np.max(X[:, 0]) >= 1:
                fits.append(frozenset(int(value) % 100 for value in X[:, 0]))
            return self

        def predict_proba(self, X):
            probability = 0.25 + 0.5 * (np.arange(len(X)) % 2)
            return np.column_stack([1.0 - probability, probability])

    monkeypatch.setattr(classical, "make_pipeline", lambda *args, **kwargs: SpyPipeline())
    result = classical._run_nested_stacking(
        _Aligned(), "linear", _config(), _Aligned.groups.to_numpy(), 1, 3, _plan()
    )
    expected = [
        frozenset({0, 1}),
        frozenset({2, 3}),
        frozenset({0, 1, 2, 3}),
        frozenset({4, 5}),
        frozenset({6, 7}),
        frozenset({4, 5, 6, 7}),
    ]
    assert sorted(fits, key=lambda values: (min(values), len(values))) == sorted(
        expected * 2, key=lambda values: (min(values), len(values))
    )
    assert result.extra["stacking_status"] == "nested_inner_oof"


def test_invalid_manifest_cannot_reach_classical_integration():
    manifest = deepcopy(_manifest())
    manifest["outer_folds"][1]["assessment"] = [0, 1, 2, 2]
    with pytest.raises(ValueError):
        validate_split_manifest(
            manifest,
            n_samples=8,
            groups=[f"g{index}" for index in range(8)],
            y=[0, 1, 0, 1, 0, 1, 0, 1],
            task="classification",
            requested_outer_k=2,
            requested_inner_k=2,
            minimum_training_groups=2,
            minimum_assessment_groups=2,
            minimum_training_groups_per_class=1,
            minimum_assessment_groups_per_class=1,
        )
