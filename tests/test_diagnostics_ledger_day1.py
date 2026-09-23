from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from omicau.data.alignment import AlignedDataset, ModalityMatrix
from omicau.diagnostics.batch import batch_effect_diagnostics
from omicau.diagnostics.missingness import missingness_diagnostics
from omicau.interpretation import utility
from omicau.models.base import CVResult
from omicau.reporting.reporter import build_report


def _aligned_with_distinct_batches() -> AlignedDataset:
    n = 40
    ids = [f"s{i}" for i in range(n)]
    y = pd.Series([0] * 20 + [1] * 20, index=ids, dtype=float)
    rng = np.random.default_rng(4)
    signal = pd.DataFrame(
        np.column_stack([np.r_[np.zeros(20), np.ones(20)], rng.normal(size=n)]), index=ids,
        columns=["sig", "signal_noise"],
    )
    noise = pd.DataFrame(rng.normal(size=(n, 2)), index=ids, columns=["n1", "n2"])
    signal_batch = pd.Series(["signal_a"] * 20 + ["signal_b"] * 20, index=ids, name="site_signal")
    noise_batch = pd.Series(["noise_a", "noise_b"] * 20, index=ids, name="site_noise", dtype="string")
    noise_batch.iloc[0] = pd.NA
    return AlignedDataset(
        modalities={"signal": ModalityMatrix("signal", signal), "noise": ModalityMatrix("noise", noise)},
        y=y,
        y_raw=y.astype("int64").astype("string"),
        task="classification",
        sample_ids=ids,
        batch=None,
        batch_by_modality={"signal": signal_batch, "noise": noise_batch},
    )


def _result(name: str, folds: list[float], y: np.ndarray, groups: np.ndarray, *, importance=None) -> CVResult:
    score = np.linspace(0.2, 0.8, len(y))
    return CVResult(
        name=name,
        task="classification",
        metrics={"auroc": float(np.mean(folds))},
        fold_primary=folds,
        oof_true=y,
        oof_score=score,
        oof_pred=(score >= 0.5).astype(int),
        oof_groups=groups,
        modalities=["signal", "noise"],
        feature_importance=importance or {},
        feature_importance_std={key: 0.01 for key in (importance or {})},
    )


def _classical_results(*, matched: bool) -> dict:
    y = np.array([0, 1] * 20)
    groups = np.repeat(np.arange(20), 2)
    fusion = _result(
        "linear::FUSION", [0.71, 0.76, 0.72, 0.77, 0.74], y, groups,
        importance={"signal::sig": 0.2, "noise::n1": 0.05},
    )
    loo_y = y if matched else y[::-1]
    loo = _result("linear::FUSION-minus-signal", [0.69, 0.74, 0.71, 0.74, 0.73], loo_y, groups)
    loo_noise = _result("linear::FUSION-minus-noise", [0.70, 0.75, 0.72, 0.75, 0.73], y, groups)
    standalone_signal = _result("linear::signal", [0.68] * 5, y, groups)
    standalone_noise = _result("linear::noise", [0.50] * 5, y, groups)
    return {
        "primary_metric": "auroc",
        "reference_estimator": "linear",
        "results": [fusion, loo, loo_noise, standalone_signal, standalone_noise],
        "controls": [],
    }


def test_per_modality_batch_and_missingness_use_mapped_labels_without_global_batch():
    aligned = _aligned_with_distinct_batches()
    batch = batch_effect_diagnostics(aligned)
    missing = missingness_diagnostics(aligned)

    assert batch["global_batch_available"] is False
    assert batch["per_modality"]["signal"]["batch_column"] == "site_signal"
    assert batch["per_modality"]["noise"]["batch_column"] == "site_noise"
    assert batch["per_modality"]["signal"]["target_confounding"]["flag"] is True
    assert batch["per_modality"]["noise"]["target_confounding"]["flag"] is False
    batch_tests = {row["modality"]: row for row in missing["tests"] if row["association"] == "batch"}
    assert batch_tests["signal"]["batch_column"] == "site_signal"
    assert batch_tests["noise"]["batch_column"] == "site_noise"


def test_positive_nonsignificant_gain_is_inconclusive_and_classical_only(monkeypatch):
    aligned = _aligned_with_distinct_batches()
    classical = _classical_results(matched=True)
    monkeypatch.setattr(utility, "_paired_gain", lambda *_: (0.02, 0.20))
    monkeypatch.setattr(
        utility, "_paired_gain_ci",
        lambda *_args, **_kwargs: {"low": -0.01, "high": 0.05, "resampling_unit": "group", "n_valid_resamples": 100},
    )
    result = utility.build_utility_ledger(aligned, classical, {"enabled": False}, {"per_modality": {}}, {})
    signal = next(row for row in result["modality_ledger"] if row["modality"] == "signal")

    assert signal["marginal_gain_status"] == "inconclusive_positive"
    assert "inconclusive" in signal["verdict"]
    assert result["categorical_verdict_inputs"]["neural_results_used"] is False
    assert result["categorical_verdict_inputs"]["feature_attribution_used"] is False
    assert signal["pooled_oof_marginal_gain_ci"]["resampling_unit"] == "group"


def test_foldmean_decision_estimate_is_not_paired_with_pooled_oof_interval(monkeypatch):
    """A nonlinear metric can give a pooled OOF gain unlike the fold-mean gain."""
    aligned = _aligned_with_distinct_batches()
    classical = _classical_results(matched=True)
    monkeypatch.setattr(utility, "_paired_gain", lambda *_: (0.02, 0.20))
    monkeypatch.setattr(
        utility, "_paired_gain_ci",
        lambda *_args, **_kwargs: {"low": -0.02, "high": 0.01, "resampling_unit": "group", "n_valid_resamples": 100},
    )
    result = utility.build_utility_ledger(aligned, classical, {"enabled": False}, {"per_modality": {}}, {})
    signal = next(row for row in result["modality_ledger"] if row["modality"] == "signal")

    # The fixture's OOF predictions are identical, so its pooled AUROC difference
    # is zero even though the separately supplied outer-fold estimate is positive.
    assert signal["foldmean_marginal_gain"] == 0.02
    assert signal["pooled_oof_marginal_gain"] == 0.0
    assert signal["pooled_oof_marginal_gain_ci"]["low"] == -0.02
    assert "foldmean_marginal_gain" in result["categorical_verdict_inputs"]["records"][0]
    assert "pooled_oof_marginal_gain_ci" in result["categorical_verdict_inputs"]["records"][0]


def test_unmatched_out_of_fold_records_fail_the_gain_verdict_gate(monkeypatch):
    aligned = _aligned_with_distinct_batches()
    classical = _classical_results(matched=False)
    monkeypatch.setattr(utility, "_paired_gain", lambda *_: (0.04, 0.001))
    monkeypatch.setattr(utility, "_paired_gain_ci", lambda *_args, **_kwargs: None)
    result = utility.build_utility_ledger(aligned, classical, {"enabled": False}, batch_effect_diagnostics(aligned), {})
    signal = next(row for row in result["modality_ledger"] if row["modality"] == "signal")

    assert signal["marginal_gain_eligibility"] == {"eligible": False, "reason": "unmatched_out_of_fold_targets"}
    assert signal["marginal_gain_status"] == "unavailable"
    assert not signal["verdict"].startswith("predictive")


def test_rowwise_control_receipt_is_not_used_as_global_chance_evidence(monkeypatch):
    aligned = _aligned_with_distinct_batches()
    classical = _classical_results(matched=True)
    monkeypatch.setattr(utility, "_paired_gain_ci", lambda *_args, **_kwargs: None)
    unsupported = _result("control::shuffled_target", [0.9] * 5, np.array([0, 1] * 20), np.repeat(np.arange(20), 2))
    unsupported.extra["control_execution_receipt"] = {
        "decision": "unsupported_for_group_safe_control",
        "scope": "rowwise_target_shuffle",
        "assessment_truth_status": "not_preserved",
    }
    classical["controls"] = [unsupported]
    result = utility.build_utility_ledger(aligned, classical, {"enabled": False}, {"per_modality": {}}, {})

    assert result["leakage_warning"] is False
    assert result["control_alarm_status"] == "not_evaluable_no_global_chance_eligible_control"
    assert result["controls"][0]["global_chance_evidence"] is False
    assert "not evaluated" in result["leakage_text"]


def test_conditional_control_is_descriptive_but_global_eligible_control_can_raise_alarm(monkeypatch):
    aligned = _aligned_with_distinct_batches()
    classical = _classical_results(matched=True)
    monkeypatch.setattr(utility, "_paired_gain_ci", lambda *_args, **_kwargs: None)
    control = _result("control::group_permuted_target", [0.9] * 5, np.array([0, 1] * 20), np.repeat(np.arange(20), 2))
    control.extra["control_execution_receipt"] = {
        "decision": "eligible", "expected_null_scope": "conditional", "global_chance_eligible": False,
        "assessment_truth_status": "preserved",
    }
    classical["controls"] = [control]
    conditional = utility.build_utility_ledger(aligned, classical, {"enabled": False}, {"per_modality": {}}, {})
    assert conditional["leakage_warning"] is False
    assert conditional["control_alarm_status"] == "not_evaluable_no_global_chance_eligible_control"

    control.extra["control_execution_receipt"].update({"expected_null_scope": "global", "global_chance_eligible": True})
    global_result = utility.build_utility_ledger(aligned, classical, {"enabled": False}, {"per_modality": {}}, {})
    assert global_result["leakage_warning"] is True
    assert global_result["control_alarm_status"] == "global_chance_alarm"


def test_damaged_global_control_receipt_fails_closed_for_alarm_eligibility(monkeypatch):
    """A high score cannot compensate for a missing required receipt condition."""
    aligned = _aligned_with_distinct_batches()
    classical = _classical_results(matched=True)
    monkeypatch.setattr(utility, "_paired_gain_ci", lambda *_args, **_kwargs: None)
    control = _result("control::shuffled_features", [0.9] * 5, np.array([0, 1] * 20), np.repeat(np.arange(20), 2))
    # This starts with the fixed-plan, fold-local feature-control receipt shape.
    receipt = {
        "decision": "eligible", "expected_null_scope": "global", "global_chance_eligible": True,
        "assessment_truth_status": "preserved", "scope": "outer_train_and_assessment_transformed_separately",
        "transform_role": "feature_association_stress_control",
    }
    receipt.pop("assessment_truth_status")  # representative receipt damage
    control.extra["control_execution_receipt"] = receipt
    classical["controls"] = [control]

    result = utility.build_utility_ledger(aligned, classical, {"enabled": False}, {"per_modality": {}}, {})
    assert result["controls"][0]["primary"] == 0.9
    assert result["controls"][0]["global_chance_evidence"] is False
    assert result["leakage_warning"] is False
    assert result["control_alarm_status"] == "not_evaluable_no_global_chance_eligible_control"


def test_reporting_exports_complete_attribution_and_verdict_schema(tmp_path):
    records = _classical_results(matched=True)["results"]
    models = {
        "primary_metric": "auroc", "task": "classification", "reference_estimator": "linear",
        "classical": [item.to_dict() for item in records], "controls": [], "neural": {"enabled": False, "results": []},
    }
    ledger = [{
        "modality": "signal", "n_features": 2, "standalone_primary": 0.68,
        "foldmean_marginal_gain": 0.02, "foldmean_marginal_gain_p": 0.20,
        "pooled_oof_marginal_gain": 0.00,
        "pooled_oof_marginal_gain_ci": {"low": -0.01, "high": 0.05, "resampling_unit": "group"},
        "marginal_gain_status": "inconclusive_positive", "marginal_gain_eligibility": {"eligible": True},
        "redundancy_max_cka": 0.6, "similarity_with": "noise", "batch_column": "site_signal",
        "batch_confounded": False, "missingness_biased": False,
        "verdict": "informative; incremental contribution inconclusive", "recommendation": "retain pending more evidence",
    }]
    audit = {
        "meta": {"run_name": "unit", "provenance_hash": "x", "device": "cpu", "cores": 1},
        "environment": {}, "dataset": {"n_samples": 40, "task": "classification", "modalities": []},
        "models": models, "utility": {"modality_ledger": ledger, "controls": [], "single_modality": False,
                                           "leakage_warning": False, "fusion_gain_over_best_single": 0.02,
                                           "fusion_gain_ci": None, "chance_level": 0.5,
                                           "best_model": {"primary": 0.74}, "auprc_baseline": None,
                                           "calibration": None, "subgroups": None, "batch_blocked": None,
                                           "batch_adjusted": None, "summary_flags": []},
        "diagnostics": {"missingness": {"tests": [], "sample_missingness": {"by_modality": {}}}, "batch": {}},
        "summary": {"data_hygiene_rating": "clean", "actionable_recommendations": []},
        "config": {"xai": {"top_k": 1}}, "cost_estimate": {},
    }
    assets = build_report(audit, tmp_path)
    html = assets["html"].read_text(encoding="utf-8")
    attribution = pd.read_csv(assets["feature_attribution"])
    verdict = pd.read_csv(assets["verdict_ledger"])

    assert list(attribution["feature"]) == ["signal::sig", "noise::n1"]
    assert set(("foldmean_marginal_gain", "foldmean_marginal_gain_p", "pooled_oof_marginal_gain", "pooled_oof_gain_ci_low", "pooled_oof_gain_ci_high", "gain_status", "gain_eligible", "paired_resampling_unit")) <= set(verdict.columns)
    assert verdict.loc[0, "gain_status"] == "inconclusive_positive"
    assert "Gain inconclusive" in html and "Not additive" not in html
