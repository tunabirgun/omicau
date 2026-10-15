from __future__ import annotations

import json

from omicau.interpretation.llm_summary import _sanitize_context, summarize
from omicau.reporting.reporter import (
    _answer_strip,
    _ledger_csv_rows,
    _model_card_md,
    _verdict_status,
    fig_marginal_gain,
)


def _assert_fixture_consistent(row: dict) -> None:
    assessment = row["incremental_assessment"]
    status = assessment["status"]
    ordinary, decision = assessment["ordinary_95_ci"], assessment["decision_ci"]
    if status == "increased_held_out_performance":
        assert decision["lower"] > 0
    elif status == "reduced_held_out_performance":
        assert decision["upper"] < 0
    elif status == "unresolved":
        assert decision["lower"] <= 0 <= decision["upper"]
    if ordinary is not None:
        assert decision["lower"] <= ordinary["lower"] <= ordinary["upper"] <= decision["upper"]
        assert assessment["multiplicity_family"]["family_size"] == 2
    else:
        assert status in {"unavailable", "not_applicable"}
    absolute = assessment["absolute_scores"]
    if assessment["estimate"] is None:
        assert absolute["full"] is None and absolute["reduced"] is None
    else:
        assert abs((absolute["full"] - absolute["reduced"]) - assessment["estimate"]) < 1e-12


def _row(status: str, estimate: float | None = None) -> dict:
    defaults = {
        "increased_held_out_performance": (0.012, (0.004, 0.025), (0.002, 0.030)),
        "reduced_held_out_performance": (-0.012, (-0.025, -0.004), (-0.030, -0.002)),
        "unresolved": (0.001, (-0.012, 0.019), (-0.016, 0.022)),
        "unavailable": (None, None, None),
        "not_applicable": (None, None, None),
    }
    default_estimate, ordinary_bounds, decision_bounds = defaults[status]
    estimate = default_estimate if estimate is None else estimate
    ordinary = None if ordinary_bounds is None else {
        "lower": ordinary_bounds[0], "upper": ordinary_bounds[1], "quantile_method": "linear"}
    decision = None if decision_bounds is None else {
        "lower": decision_bounds[0], "upper": decision_bounds[1], "quantile_method": "linear"}
    absolute = ({"full": None, "reduced": None, "outcome_mean_reference": 0.0, "interpretation": None}
                if estimate is None else
                {"full": -0.04, "reduced": -0.04 - estimate, "outcome_mean_reference": 0.0,
                 "interpretation": ("relative_score_increase_below_outcome_mean_reference"
                                    if estimate > 0 and -0.04 - estimate < 0 else None)})
    record = {
        "modality": "candidate", "n_features": 8, "ledger_schema_version": 2,
        "standalone_assessment": {
            "status": "descriptive", "estimate": 0.63,
            "ordinary_95_ci": {"lower": 0.55, "upper": 0.70, "quantile_method": "linear"},
            "resampling_unit": "group", "n_valid_resamples": 4000, "n_attempts": 4000,
            "reason": None,
        },
        "incremental_assessment": {
            "status": status, "estimate": estimate, "ordinary_95_ci": ordinary, "decision_ci": decision,
            "multiplicity_family": None if ordinary is None else {"family_size": 2, "family_alpha": 0.05,
                                      "decision_quantiles": [0.0125, 0.9875], "scope": "cohort_reference_ledger"},
            "resampling_unit": None if ordinary is None else "group",
            "n_valid_resamples": 0 if ordinary is None else 8000,
            "n_attempts": 0 if ordinary is None else 8000,
            "reason": "fixture_unavailable" if ordinary is None else None,
            "absolute_scores": absolute,
        },
        "diagnostic_flags": {"batch_structured": False, "batch_confounded": False,
                             "missingness_biased": False, "leakage_warning": False},
        "standalone_primary": 0.63, "foldmean_marginal_gain": estimate,
        "foldmean_marginal_gain_p": 0.02, "pooled_oof_marginal_gain": estimate,
        "pooled_oof_marginal_gain_ci": None if ordinary is None else {
            "low": ordinary["lower"], "high": ordinary["upper"]},
        "verdict": "legacy prose must not control v2 reporting", "legacy_fields_deprecated": True,
    }
    _assert_fixture_consistent(record)
    return record


def test_v2_status_drives_badge_csv_plot_and_summary() -> None:
    row = _row("increased_held_out_performance")
    assert _verdict_status(row) == "increased"
    header, rows = _ledger_csv_rows([row])
    exported = dict(zip(header, rows[0], strict=True))
    assert exported["incremental_status"] == "increased_held_out_performance"
    assert exported["legacy_verdict"] == "legacy prose must not control v2 reporting"
    assert "family decision interval" in fig_marginal_gain({"modality_ledger": [row]}, False)
    answers = _answer_strip({"single_modality": False, "modality_ledger": [row]}, "clean")
    assert answers[0]["a"] == "Increased held-out performance"
    summary = summarize({"n_samples": 80, "utility": {"single_modality": False,
                         "modality_ledger": [row], "best_model": {"name": "rf", "primary": -0.04},
                         "primary_metric": "r2", "chance_level": 0.0},
                         "missingness_flags": [], "batch_flags": []}, type("Config", (), {
                             "llm": type("LLM", (), {"enabled": False})()})())
    assert "increased held-out performance" in summary["clinical_verdict"].lower()
    assert "predictive" not in summary["clinical_verdict"].lower()


def test_v1_is_not_recategorized_from_legacy_p_values() -> None:
    row = _row("increased_held_out_performance")
    row.pop("ledger_schema_version")
    row.pop("incremental_assessment")
    assert _verdict_status(row) == "unavailable"
    header, rows = _ledger_csv_rows([row])
    exported = dict(zip(header, rows[0], strict=True))
    assert exported["incremental_status"] == "unavailable"
    assert exported["legacy_foldmean_marginal_gain_p"] == 0.02


def test_sanitized_optional_payload_contains_aggregates_only() -> None:
    row = _row("reduced_held_out_performance", -0.011)
    raw = {
        "utility": {"task": "regression", "primary_metric": "r2", "chance_level": 0.0,
                    "best_model": {"name": "rf", "primary": -0.04}, "leakage_warning": False,
                    "controls": [{"name": "control", "primary": 0.01, "oof_predictions": [1, 2]}],
                    "modality_ledger": [row]},
        "n_samples": 80, "missingness_flags": ["aggregate missingness flag"],
        "batch_flags": [], "sample_ids": ["not permitted"], "split_ids": ["not permitted"],
    }
    payload = _sanitize_context(raw)
    text = json.dumps(payload, sort_keys=True)
    assert "oof_predictions" not in text
    assert "sample_ids" not in text
    assert "split_ids" not in text
    assert payload["modality_ledger"][0]["incremental"]["status"] == "reduced_held_out_performance"


def test_model_card_labels_relative_negative_r2_correctly() -> None:
    row = _row("increased_held_out_performance")
    card = _model_card_md({"meta": {}, "dataset": {"modalities": []}, "environment": {},
                           "utility": {"best_model": {"name": "rf", "primary": -0.04},
                                       "primary_metric": "r2", "modality_ledger": [row]}})
    assert "Standalone layer scores are descriptive" in card
    assert "increased held-out performance=1" in card
