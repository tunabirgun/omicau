from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from omicau.models import neural, survival
from omicau.models.split_plan import validate_split_manifest


def _manifest() -> dict:
    return {
        "outer_folds": [
            {
                "train": [4, 5, 6, 7],
                "assessment": [0, 1, 2, 3],
                "inner_folds": [
                    {"train": [6, 7], "assessment": [4, 5]},
                    {"train": [4, 5], "assessment": [6, 7]},
                ],
            },
            {
                "train": [0, 1, 2, 3],
                "assessment": [4, 5, 6, 7],
                "inner_folds": [
                    {"train": [2, 3], "assessment": [0, 1]},
                    {"train": [0, 1], "assessment": [2, 3]},
                ],
            },
        ]
    }


def _classification_plan():
    return validate_split_manifest(
        _manifest(),
        n_samples=8,
        groups=[f"g{index}" for index in range(8)],
        task="classification",
        requested_outer_k=2,
        requested_inner_k=2,
        minimum_training_groups=2,
        minimum_assessment_groups=2,
        y=[0, 1, 0, 1, 0, 1, 0, 1],
        minimum_training_groups_per_class=1,
        minimum_assessment_groups_per_class=1,
    )


def _survival_data():
    time = np.asarray([1.0, 4.0, 2.0, 5.0, 1.5, 4.5, 2.5, 5.5])
    event = np.asarray([1, 0, 1, 0, 1, 0, 1, 0])
    groups = np.asarray([f"g{index}" for index in range(8)], dtype=object)
    return time, event, groups


def _survival_plan():
    time, event, groups = _survival_data()
    return validate_split_manifest(
        _manifest(),
        n_samples=8,
        groups=groups,
        task="survival",
        requested_outer_k=2,
        requested_inner_k=2,
        minimum_training_groups=2,
        minimum_assessment_groups=2,
        time=time,
        event=event,
        minimum_survival_training_event_groups=1,
        minimum_survival_assessment_comparable_pairs=1,
    )


def _expanded_survival_plan():
    time, event, groups = _survival_data()
    manifest = deepcopy(_manifest())
    for outer in manifest["outer_folds"]:
        outer["train"] = [row for index in outer["train"] for row in (2 * index, 2 * index + 1)]
        outer["assessment"] = [
            row for index in outer["assessment"] for row in (2 * index, 2 * index + 1)
        ]
        for inner in outer["inner_folds"]:
            inner["train"] = [
                row for index in inner["train"] for row in (2 * index, 2 * index + 1)
            ]
            inner["assessment"] = [
                row for index in inner["assessment"] for row in (2 * index, 2 * index + 1)
            ]
    return validate_split_manifest(
        manifest,
        n_samples=16,
        groups=np.repeat(groups, 2),
        task="survival",
        requested_outer_k=2,
        requested_inner_k=2,
        minimum_training_groups=2,
        minimum_assessment_groups=2,
        time=np.repeat(time, 2),
        event=np.repeat(event, 2),
        minimum_survival_training_event_groups=1,
        minimum_survival_assessment_comparable_pairs=1,
    )


class _Aligned:
    def __init__(self, task: str):
        time, event, groups = _survival_data()
        self.task = task
        self.groups = pd.Series(groups)
        self.modality_names = ["m"]
        X = np.column_stack((time, np.arange(8, dtype=float)))
        self.modalities = {"m": SimpleNamespace(X=X, feature_names=["x0", "x1"])}
        if task == "classification":
            self.y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
            self.event = None
        else:
            self.y = pd.Series(time)
            self.event = pd.Series(event)

    def concat_matrix(self, modalities):
        arrays = [self.modalities[name].X for name in modalities]
        names = [f"{name}::{feature}" for name in modalities for feature in self.modalities[name].feature_names]
        return np.column_stack(arrays), names


def _config(task: str):
    return SimpleNamespace(
        seed=17,
        cv=SimpleNamespace(n_splits=7, shuffle=True, n_bootstrap=0),
        neural=SimpleNamespace(
            enabled=True,
            embed_dim=2,
            hidden_dim=2,
            dropout=0.0,
            pooling="mean",
            lr=0.01,
            weight_decay=0.0,
            epochs=1,
            patience=1,
            batch_size=2,
        ),
        compute=SimpleNamespace(device="cpu"),
        xai=SimpleNamespace(enabled=False),
        classical=SimpleNamespace(max_features=None),
        controls=SimpleNamespace(
            enabled=False,
            shuffle_target=False,
            shuffle_features=False,
            random_noise=False,
        ),
        task=task,
    )


def _control_contract() -> dict[str, object]:
    return {
        "strata": ["registered_block"] * 8,
        "strata_schema": {
            "name": "registered_block",
            "version": "1",
            "fields": ["registered_block"],
            "target_derived": False,
        },
        "permutation_registry": {
            "schema_version": "c07_private_registry_binding_v1",
            "purpose": "group_endpoint_permutation",
            "nonce_hex": hashlib.sha256(b"synthetic survival registry nonce").hexdigest(),
            "registry_id": "synthetic_group_endpoint_permutation",
            "artifact": {
                "artifact_id": "synthetic_fixture",
                "sha256": hashlib.sha256(b"synthetic survival fixture").hexdigest(),
            },
            "entries": [
                {
                    "entry_id": "synthetic_entry",
                    "artifact_sha256": hashlib.sha256(b"synthetic survival entry").hexdigest(),
                }
            ],
        },
        "exchangeability_contract": {
            "unit": "highest_exchangeable_group",
            "scope": "outer_train_only",
        },
        "minimum_distinct_nonidentity_assignments": 1,
        "fold_seeds": [41, 43],
    }


class _ZeroClassifier:
    def eval(self):
        return self

    def __call__(self, batch):
        size = next(iter(batch.values()))[0].shape[0]
        return torch.zeros((size, 2), dtype=torch.float32)


def test_neural_uses_exact_outer_and_plan_inner_folds(monkeypatch) -> None:
    calls = []

    def fake_train(*args, device, batch_size, **kwargs):
        calls.append((tuple(args[2]), tuple(args[3])))
        return _ZeroClassifier(), device, batch_size

    monkeypatch.setattr(neural, "_train_fold_resilient", fake_train)
    monkeypatch.setattr(neural, "attach_cis", lambda *args, **kwargs: None)
    output = neural.run_neural_benchmark(
        _Aligned("classification"), _config("classification"), _classification_plan()
    )

    assert output["split_execution_status"] == "validated_c06_exact_splits"
    assert output["split_plan_receipt"]["outer_fold_count"] == 2
    assert all(result.extra["n_splits"] == 2 for result in output["results"])
    assert all(
        result.to_dict()["split_plan_status"] == output["split_execution_status"]
        for result in output["results"]
    )
    assert calls == [((6, 7), (4, 5)), ((2, 3), (0, 1))] * 2
    public = repr(output["split_plan_receipt"])
    assert "outer_folds" not in public and "g0" not in public


def test_neural_early_stop_overlap_fails_watched() -> None:
    groups = np.asarray(["a", "b", "c"], dtype=object)
    with pytest.raises(RuntimeError, match="c06_neural_inner_partition_overlap"):
        neural._check_partition(
            np.asarray([0, 1]), np.asarray([1, 2]), np.arange(3), groups, "neural_inner"
        )


def test_neural_wrong_fold_universe_fails_watched() -> None:
    with pytest.raises(
        TypeError, match="^validated_split_plan_runtime_universe_mismatch$"
    ):
        neural._validated_partitions(
            _classification_plan(),
            7,
            np.asarray([f"g{index}" for index in range(7)]),
            "classification",
            np.asarray([0, 1, 0, 1, 0, 1, 0]),
        )


@pytest.mark.parametrize("change", ["rename", "reassign", "label_swap"])
def test_neural_runtime_universe_mismatch_precedes_training(monkeypatch, change) -> None:
    aligned = _Aligned("classification")
    if change == "rename":
        aligned.groups.iloc[0] = "renamed"
    elif change == "reassign":
        first, second = aligned.groups.iloc[0], aligned.groups.iloc[1]
        aligned.groups.iloc[0], aligned.groups.iloc[1] = second, first
    else:
        first, second = aligned.y.iloc[0], aligned.y.iloc[1]
        aligned.y.iloc[0], aligned.y.iloc[1] = second, first

    calls = 0

    def forbidden_train(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("training must not start")

    monkeypatch.setattr(neural, "_train_fold_resilient", forbidden_train)
    with pytest.raises(
        TypeError, match="^validated_split_plan_runtime_universe_mismatch$"
    ):
        neural.run_neural_benchmark(
            aligned, _config("classification"), _classification_plan()
        )
    assert calls == 0


def test_survival_executes_exact_requested_folds_and_receipt() -> None:
    time, event, groups = _survival_data()
    X = np.column_stack((time, np.arange(8, dtype=float)))
    pooled, fold_scores, oof, k = survival._cv_cindex(
        X, time, event, groups, 7, 17, None, validated_plan=_survival_plan()
    )

    assert k == 2
    assert len(fold_scores) == 2
    assert np.isfinite(pooled) and np.isfinite(oof).all()

    output = survival.run_survival_benchmark(
        _Aligned("survival"), _config("survival"), _survival_plan()
    )
    assert output["split_execution_status"] == "validated_c06_exact_splits"
    assert output["split_plan_receipt"]["outer_fold_count"] == 2
    assert all(result.extra["n_splits"] == 2 for result in output["results"])
    assert all(
        result.to_dict()["split_plan_status"] == output["split_execution_status"]
        for result in output["results"]
    )
    public = repr(output["split_plan_receipt"])
    assert "outer_folds" not in public and "g0" not in public


def test_survival_validated_plan_refuses_legacy_controls_before_private_access_or_fit(
    monkeypatch,
) -> None:
    plan = _survival_plan()
    config = _config("survival")
    config.controls.enabled = True
    config.controls.shuffle_target = True
    private_calls = fit_calls = 0

    def forbidden_private(self, **kwargs):
        nonlocal private_calls
        private_calls += 1
        raise AssertionError("private split state must not be accessed")

    def forbidden_fit(*args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        raise AssertionError("fit must not start")

    monkeypatch.setattr(
        type(plan), "_private_validate_runtime_universe", forbidden_private
    )
    monkeypatch.setattr(survival, "cox_fit", forbidden_fit)
    with pytest.raises(
        RuntimeError, match="^validated_plan_target_control_contract_required$"
    ):
        survival.run_survival_benchmark(_Aligned("survival"), config, plan)
    assert private_calls == fit_calls == 0


def test_survival_validated_target_control_uses_fold_endpoints_and_original_truth(
    monkeypatch,
) -> None:
    config = _config("survival")
    config.controls.enabled = True
    config.controls.shuffle_target = True
    captured = []

    def fake_cv(X, time, event, groups, n_splits, seed, max_features, **kwargs):
        captured.append(kwargs.get("validated_control_contract"))
        return 0.5, [0.5, 0.5], np.zeros(len(time)), 2

    monkeypatch.setattr(survival, "_cv_cindex", fake_cv)
    monkeypatch.setattr(survival, "_boot_ci", lambda *args, **kwargs: (None, None))
    aligned = _Aligned("survival")
    output = survival.run_survival_benchmark(
        aligned,
        config,
        _survival_plan(),
        validated_control_contract=_control_contract(),
    )
    control_contracts = [value for value in captured if value is not None]
    assert control_contracts == [_control_contract()]
    assert np.array_equal(output["controls"][0].oof_true, aligned.y.to_numpy())
    receipt = output["control_execution_receipt"]
    assert receipt["decision"] == "development_only"
    assert receipt["fold_count"] == 2
    assert "seed" not in repr(receipt).lower()


def test_survival_validated_target_control_runs_through_cox_pipeline() -> None:
    config = _config("survival")
    config.controls.enabled = True
    config.controls.shuffle_target = True
    aligned = _Aligned("survival")
    output = survival.run_survival_benchmark(
        aligned,
        config,
        _survival_plan(),
        validated_control_contract=_control_contract(),
    )
    assert [control.name for control in output["controls"]] == [
        "control::group_permuted_target"
    ]
    assert np.array_equal(output["controls"][0].oof_true, aligned.y.to_numpy())
    assert output["control_execution_receipt"]["decision"] == "development_only"


def test_survival_raw_fold_endpoint_bypass_is_not_in_the_cv_api() -> None:
    time, event, groups = _survival_data()
    with pytest.raises(TypeError, match="fold_training_endpoints"):
        survival._cv_cindex(
            np.column_stack((time, np.arange(8, dtype=float))),
            time,
            event,
            groups,
            2,
            17,
            None,
            validated_plan=_survival_plan(),
            fold_training_endpoints=(),
        )


def test_survival_validated_feature_controls_stay_fail_closed_before_fit(monkeypatch):
    config = _config("survival")
    config.controls.enabled = True
    config.controls.shuffle_features = True
    fit_calls = 0

    def forbidden_fit(*args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        raise AssertionError("fit must not start")

    monkeypatch.setattr(survival, "cox_fit", forbidden_fit)
    with pytest.raises(
        RuntimeError, match="^validated_plan_feature_controls_require_c07_integration$"
    ):
        survival.run_survival_benchmark(
            _Aligned("survival"), config, _survival_plan()
        )
    assert fit_calls == 0


def test_survival_zero_event_training_fold_fails_instead_of_skip(monkeypatch) -> None:
    splits = ((np.asarray([0, 1]), np.asarray([2, 3])),)
    monkeypatch.setattr(survival, "_validated_outer_splits", lambda *args: (splits, {}))
    with pytest.raises(RuntimeError, match="c06_survival_training_event_support_runtime"):
        survival._cv_cindex(
            np.ones((4, 2)), np.asarray([1.0, 2.0, 1.0, 2.0]),
            np.asarray([0, 0, 1, 0]), np.arange(4), 1, 1, None,
            validated_plan=object(),
        )


def test_survival_no_comparable_pair_fails_instead_of_chance(monkeypatch) -> None:
    splits = ((np.asarray([0, 1]), np.asarray([2, 3])),)
    monkeypatch.setattr(survival, "_validated_outer_splits", lambda *args: (splits, {}))
    with pytest.raises(RuntimeError, match="c06_survival_assessment_pair_support_runtime"):
        survival._cv_cindex(
            np.ones((4, 2)), np.ones(4), np.asarray([1, 0, 1, 0]),
            np.arange(4), 1, 1, None, validated_plan=object(),
        )


@pytest.mark.parametrize("change", ["time_shift", "event_swap"])
def test_survival_runtime_universe_mismatch_precedes_fit(monkeypatch, change) -> None:
    aligned = _Aligned("survival")
    if change == "time_shift":
        aligned.y.iloc[0] += 0.25
    else:
        first, second = aligned.event.iloc[0], aligned.event.iloc[1]
        aligned.event.iloc[0], aligned.event.iloc[1] = second, first

    calls = 0

    def forbidden_fit(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("fit must not start")

    monkeypatch.setattr(survival, "cox_fit", forbidden_fit)
    with pytest.raises(
        TypeError, match="^validated_split_plan_runtime_universe_mismatch$"
    ):
        survival.run_survival_benchmark(
            aligned, _config("survival"), _survival_plan()
        )
    assert calls == 0


def test_survival_repeated_row_reorder_precedes_fit(monkeypatch) -> None:
    time, event, groups = _survival_data()
    aligned = _Aligned("survival")
    aligned.y = pd.Series(np.repeat(time, 2))
    aligned.event = pd.Series(np.repeat(event, 2))
    aligned.groups = pd.Series(np.repeat(groups, 2))
    aligned.modalities["m"].X = np.column_stack(
        (np.repeat(time, 2), np.arange(16, dtype=float))
    )
    order = np.arange(16)
    order[[1, 2]] = order[[2, 1]]
    aligned.y = aligned.y.iloc[order].reset_index(drop=True)
    aligned.event = aligned.event.iloc[order].reset_index(drop=True)
    aligned.groups = aligned.groups.iloc[order].reset_index(drop=True)
    aligned.modalities["m"].X = aligned.modalities["m"].X[order]

    calls = 0

    def forbidden_fit(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("fit must not start")

    monkeypatch.setattr(survival, "cox_fit", forbidden_fit)
    with pytest.raises(
        TypeError, match="^validated_split_plan_runtime_universe_mismatch$"
    ):
        survival.run_survival_benchmark(
            aligned, _config("survival"), _expanded_survival_plan()
        )
    assert calls == 0


def test_legacy_paths_are_explicitly_nonbenchmark(monkeypatch) -> None:
    def fake_train(*args, device, batch_size, **kwargs):
        return _ZeroClassifier(), device, batch_size

    monkeypatch.setattr(neural, "_train_fold_resilient", fake_train)
    monkeypatch.setattr(neural, "attach_cis", lambda *args, **kwargs: None)
    output = neural.run_neural_benchmark(
        _Aligned("classification"), _config("classification")
    )
    assert output["split_execution_status"] == "legacy_nonbenchmark_dynamic_splits"
    assert output["split_plan_receipt"] is None
    assert all(
        result.to_dict()["split_plan_status"] == output["split_execution_status"]
        for result in output["results"]
    )

    survival_config = _config("survival")
    survival_config.controls.enabled = True
    survival_config.controls.shuffle_target = True
    survival_config.controls.shuffle_features = True
    survival_config.controls.random_noise = True
    survival_output = survival.run_survival_benchmark(
        _Aligned("survival"), survival_config
    )
    assert survival_output["split_execution_status"] == "legacy_nonbenchmark_dynamic_splits"
    assert survival_output["split_plan_receipt"] is None
    assert {result.name for result in survival_output["controls"]} == {
        "control::shuffled_target",
        "control::shuffled_features",
        "control::random_noise",
    }
    assert all(
        result.to_dict()["split_plan_status"]
        == survival_output["split_execution_status"]
        for result in survival_output["results"]
    )
