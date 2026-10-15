"""Modality-utility interpretation.

The ledger reports descriptive standalone performance, paired held-out
full-minus-reduced contrasts, and diagnostic warnings as separate outputs.
Its operational status uses only a validated paired bootstrap contrast; the
corrected fold test remains secondary stability context. Batch, missingness,
and control results are scope-limited diagnostics that do not override a valid
contrast or establish absence of leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from omicau.interpretation.paired_assessment import (
    assess_incremental,
    assess_standalone,
)

CHANCE = {"classification": 0.5, "regression": 0.0, "survival": 0.5}
# Heuristic decision thresholds. These are pragmatic cut-offs, not derived
# constants; they gate plain-language wording only and are documented as
# heuristics in the DOME limitations block. The single source of truth for the
# fusion/marginal-gain bands is here; reporting/reporter.py imports them so the
# executive strip, headline card, and summary flags cannot diverge.
USEFUL_MARGIN = 0.05   # standalone score must clear chance by this to be "useful"
GAIN_EPS = 0.01        # smallest fusion/marginal gain worth calling non-trivial
GAIN_STRONG = 0.03     # gain at/above which fusion "clearly" helps (plain language)
GAIN_ALPHA = 0.05      # significance level the paired Nadeau-Bengio test must clear
CKA_REDUNDANT = 0.5
# A control baseline trips the leakage alarm when it beats chance by more than
# this margin. Applied task-aware (chance = 0.5 AUROC for classification, 0.0 R2
# for regression), so it actually fires on regression -- a fixed 0.62 threshold
# never could, since a shuffled-target R2 sits near 0.
CONTROL_MARGIN = 0.12
# Family-wise level for the control baselines. Three controls are tested per run and
# any one of them can raise the alarm, so testing each at 5% inflates the false-alarm
# rate to roughly 15%. The interval used by the leakage gate is therefore computed at
# CONTROL_FAMILY_ALPHA / (number of controls), a Bonferroni correction over the family.
CONTROL_FAMILY_ALPHA = 0.05


def _by_name(results: list) -> dict[str, Any]:
    return {r.name: r for r in results}


def _linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    """Linear centered kernel alignment between two sample x feature matrices."""
    A = _prep_cka(A)
    B = _prep_cka(B)
    if A is None or B is None:
        return float("nan")
    if max(A.shape[1], B.shape[1]) <= A.shape[0]:
        cross = np.linalg.norm(A.T @ B) ** 2
        na = np.linalg.norm(A.T @ A)
        nb = np.linalg.norm(B.T @ B)
    else:
        gram_a = A @ A.T
        gram_b = B @ B.T
        cross = float(np.sum(gram_a * gram_b))
        na = np.linalg.norm(gram_a)
        nb = np.linalg.norm(gram_b)
    denom = na * nb
    return float(cross / denom) if denom > 0 else float("nan")


def _prep_cka(X: np.ndarray) -> np.ndarray | None:
    X = X.copy()
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(col_mean, idx[1])
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    X = X / std
    keep = np.isfinite(X).all(axis=0)
    X = X[:, keep]
    return X if X.shape[1] > 0 else None


def _paired_gain(fusion_folds: list[float], loo_folds: list[float]) -> tuple[float, float]:
    """Mean outer-fold delta and its corrected paired p-value.

    Uses the Nadeau-Bengio corrected resampled t-test rather than a plain paired
    t-test: k-fold train sets overlap, so the naive fold-difference variance is
    biased low and the uncorrected test has badly inflated Type-I error (Nadeau &
    Bengio, Machine Learning 52:239-281, 2003). The variance is inflated by
    (1/k + n_test/n_train); for k-fold, n_test/n_train = 1/(k-1).
    """
    a = np.asarray(fusion_folds, float)
    b = np.asarray(loo_folds, float)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b):
        raise ValueError("secondary_fold_test_requires_equal_arrays")
    n = len(a)
    mask = np.isfinite(a) & np.isfinite(b)
    k = int(mask.sum())
    if k < 2:
        return float("nan"), float("nan")
    d = a[mask] - b[mask]
    mean_d = float(d.mean())
    var_d = float(d.var(ddof=1))
    if var_d <= 0:
        return mean_d, float("nan")
    corr = 1.0 / k + 1.0 / (k - 1)            # Nadeau-Bengio variance inflation
    se = float(np.sqrt(corr * var_d))
    t = mean_d / se
    p = float(2 * stats.t.sf(abs(t), df=k - 1))
    return mean_d, p


def _pooled_oof_gain(best, comparator, task: str) -> float | None:
    """Pooled out-of-fold metric difference for the paired bootstrap estimand.

    This is deliberately separate from :func:`_paired_gain`: for nonlinear
    metrics such as AUROC, a mean of outer-fold differences need not equal the
    difference of pooled out-of-fold metrics.
    """
    if _gain_eligibility(best, comparator).get("eligible") is not True:
        return None
    from omicau.models.base import PRIMARY_METRIC, score_predictions

    y = np.asarray(best.oof_true)
    try:
        score_best, score_comparator = np.asarray(best.oof_score), np.asarray(comparator.oof_score)
        pred_best, pred_comparator = np.asarray(best.oof_pred), np.asarray(comparator.oof_pred)
        if task == "classification":
            pred_best, pred_comparator = pred_best.astype(int), pred_comparator.astype(int)
        key = PRIMARY_METRIC[task]
        first = score_predictions(y, score_best, pred_best, task).get(key, np.nan)
        second = score_predictions(y, score_comparator, pred_comparator, task).get(key, np.nan)
        value = float(first - second)
        return _r(value) if np.isfinite(value) else None
    except (TypeError, ValueError, FloatingPointError):
        return None


def _paired_gain_ci(best, best_single, task, n_boot: int = 1000, seed: int = 42,
                    alpha: float = 0.05) -> dict | None:
    """Percentile CI for the pooled-OOF metric difference via paired bootstrap.
    Both models are scored on the same pooled-OOF sample order, so one resample of
    the sample (or group) indices is applied to both. This interval belongs with
    :func:`_pooled_oof_gain`, not with the mean outer-fold decision statistic."""
    if best is None or best_single is None:
        return None
    y = getattr(best, "oof_true", None)
    if y is None or getattr(best_single, "oof_true", None) is None:
        return None
    from omicau.models.base import score_predictions, PRIMARY_METRIC
    key = PRIMARY_METRIC[task]
    y = np.asarray(y)
    n = len(y)
    sb, pb = np.asarray(best.oof_score), np.asarray(best.oof_pred)
    ss, ps = np.asarray(best_single.oof_score), np.asarray(best_single.oof_pred)
    if len(np.asarray(best_single.oof_true)) != n:
        return None
    if not np.array_equal(np.asarray(best_single.oof_true), y, equal_nan=True):
        return None
    n_classes = len(np.unique(y)) if task == "classification" else 0
    groups = getattr(best, "oof_groups", None)
    groups = None if groups is None else np.asarray(groups)
    other_groups = getattr(best_single, "oof_groups", None)
    other_groups = None if other_groups is None else np.asarray(other_groups)
    if (groups is None) != (other_groups is None):
        return None
    if groups is not None and (len(groups) != n or not np.array_equal(groups, other_groups)):
        return None
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
        if task == "classification" and len(np.unique(y[idx])) < n_classes:
            continue
        try:
            hb = pb[idx].astype(int) if task == "classification" else pb[idx]
            hs = ps[idx].astype(int) if task == "classification" else ps[idx]
            vb = score_predictions(y[idx], sb[idx], hb, task).get(key, np.nan)
            vs = score_predictions(y[idx], ss[idx], hs, task).get(key, np.nan)
            if np.isfinite(vb) and np.isfinite(vs):
                vals.append(float(vb - vs))
        except (ValueError, FloatingPointError):
            continue
    if len(vals) < 20:
        return None
    return {
        "low": _r(float(np.percentile(vals, 100 * alpha / 2))),
        "high": _r(float(np.percentile(vals, 100 * (1 - alpha / 2)))),
        "resampling_unit": "group" if groups is not None else "sample",
        "n_valid_resamples": int(len(vals)),
    }


def _gain_eligibility(fusion, leave_one_out) -> dict[str, Any]:
    """State whether a paired marginal-gain uncertainty calculation is valid."""
    if fusion is None or leave_one_out is None:
        return {"eligible": False, "reason": "missing_fusion_or_leave_one_out"}
    first = getattr(fusion, "oof_true", None)
    second = getattr(leave_one_out, "oof_true", None)
    if first is None or second is None:
        return {"eligible": False, "reason": "missing_out_of_fold_targets"}
    first, second = np.asarray(first), np.asarray(second)
    if len(first) != len(second) or not np.array_equal(first, second, equal_nan=True):
        return {"eligible": False, "reason": "unmatched_out_of_fold_targets"}
    groups = getattr(fusion, "oof_groups", None)
    other_groups = getattr(leave_one_out, "oof_groups", None)
    if (groups is None) != (other_groups is None):
        return {"eligible": False, "reason": "unmatched_group_metadata"}
    if groups is not None and not np.array_equal(np.asarray(groups), np.asarray(other_groups)):
        return {"eligible": False, "reason": "unmatched_group_metadata"}
    return {
        "eligible": True,
        "resampling_unit": "group" if groups is not None else "sample",
        "n_out_of_fold_rows": int(len(first)),
    }


def build_utility_ledger(
    aligned,
    classical_out: dict,
    neural_out: dict,
    batch_diag: dict | None = None,
    missing_diag: dict | None = None,
) -> dict[str, Any]:
    """Assemble the modality-utility ledger from all benchmark + diagnostic outputs."""
    task = aligned.task
    metric = classical_out["primary_metric"]
    chance = CHANCE[task]
    mods = aligned.modality_names
    single_modality = len(mods) == 1   # one layer -> no fusion/redundancy/marginal-gain to assess

    cl = _by_name(classical_out["results"])
    ref = classical_out["reference_estimator"]
    nn = _by_name(neural_out.get("results", [])) if neural_out.get("enabled") else {}

    fusion_ref = cl.get(f"{ref}::FUSION")
    nn_fusion = nn.get("neural::FUSION")

    # -- controls / leakage gate ------------------------------------------ #
    controls = [{"name": r.name, "primary": _r(r.primary),
                 "ci_low": getattr(r, "extra", {}).get("ci_low"),
                 "ci_high": getattr(r, "extra", {}).get("ci_high"),
                 "control_execution_receipt": getattr(r, "extra", {}).get("control_execution_receipt")}
                for r in classical_out.get("controls", [])]
    for control in controls:
        receipt = control.get("control_execution_receipt") or {}
        control["global_chance_evidence"] = bool(
            receipt.get("decision") == "eligible"
            and receipt.get("expected_null_scope") == "global"
            and receipt.get("global_chance_eligible") is True
            and receipt.get("assessment_truth_status") == "preserved"
        )
    alarm = chance + CONTROL_MARGIN
    present = [c for c in controls if c["primary"] is not None]
    eligible_controls = [c for c in present if c["global_chance_evidence"]]
    ineligible_controls = [c for c in present if not c["global_chance_evidence"]]

    def _sig_above_chance(c):
        # A control leaks only when BOTH conditions hold: its 95% CI lower bound
        # clears chance (the excess is distinguishable from sampling noise) AND its
        # point estimate clears the margin (the excess is large enough to matter).
        #
        # These two tests fail in opposite directions with sample size, which is why
        # requiring either one alone produced false alarms at every n. The margin
        # alone fires on small samples, where a control's score has a wide sampling
        # distribution and can clear a fixed absolute threshold by chance. The
        # interval alone fires on large samples, where a narrow interval makes a
        # trivially-above-chance control look significant. Requiring both closes each
        # gap with the other, and an effect that is neither significant nor large is
        # not evidence of leakage.
        #
        # With no interval available the margin is used alone, since that is the only
        # evidence there is.
        by_margin = c["primary"] > alarm
        if c.get("ci_low") is None:
            return by_margin
        return (c["ci_low"] > chance) and by_margin

    leaking = [c for c in eligible_controls if _sig_above_chance(c)]
    leakage = bool(leaking)
    if leaking:
        control_alarm_status = "global_chance_alarm"
        leakage_text = (
            f"A global-chance-eligible control ({leaking[0]['name'].split('::')[-1]}) is significantly above chance "
            f"(score {leaking[0]['primary']:.3f}, chance ~ {chance:.2f}); treat reported gains with "
            "caution and re-check group-aware splitting."
        )
    elif eligible_controls:
        control_alarm_status = "global_chance_no_alarm"
        leakage_text = (
            f"All global-chance-eligible controls scored near chance (~ {chance:.2f}); no alarm was raised by "
            "these evaluated target/pipeline controls. This does not establish absence of group leakage or other "
            "unmodelled sources of bias; inspect the grouping and diagnostic status."
        )
    else:
        control_alarm_status = "not_evaluable_no_global_chance_eligible_control"
        leakage_text = (
            "No global-chance-eligible control is available, so the chance-based control alarm was not evaluated. "
            "Any conditional or rowwise control result is retained descriptively only."
        )
    if ineligible_controls:
        leakage_text += (" At least one control is retained as a limited stress result but is excluded from "
                         "global-chance alarm evidence because its receipt does not establish the required "
                         "global-null, group-assignment, and assessment-truth conditions.")

    # -- redundancy (CKA) -------------------------------------------------- #
    cka = np.full((len(mods), len(mods)), np.nan)
    Xs = {m: aligned.modalities[m].X for m in mods}
    for i, mi in enumerate(mods):
        for j, mj in enumerate(mods):
            if j < i:
                cka[i, j] = cka[j, i]
            elif j == i:
                cka[i, j] = 1.0
            else:
                cka[i, j] = _linear_cka(Xs[mi], Xs[mj])

    # The reference estimator is declared before fitting by the classical runner.
    # It is never selected retrospectively by its observed OOF score.
    standalone = {
        m: (cl.get(ref + "::" + m).primary if ref + "::" + m in cl else float("nan"))
        for m in mods
    }
    intended_candidate_layers = classical_out.get("intended_candidate_layers")
    family_size = classical_out.get("intended_family_size")
    expected_family_size = (
        0
        if isinstance(intended_candidate_layers, list) and len(intended_candidate_layers) == 1
        else len(intended_candidate_layers)
        if isinstance(intended_candidate_layers, list) and len(intended_candidate_layers) > 1
        else None
    )
    if (
        not isinstance(intended_candidate_layers, list)
        or not intended_candidate_layers
        or len(set(intended_candidate_layers)) != len(intended_candidate_layers)
        or not all(isinstance(layer, str) and layer for layer in intended_candidate_layers)
        or isinstance(family_size, bool)
        or not isinstance(family_size, int)
        or family_size != expected_family_size
    ):
        intended_candidate_layers = None
        family_size = None
    single_modality = family_size == 0

    # -- per-modality ledger ---------------------------------------------- #
    ledger: list[dict[str, Any]] = []
    batch_pm = (batch_diag or {}).get("per_modality", {})
    miss_flags_by_mod = _missing_flags_by_modality(missing_diag)

    for i, m in enumerate(mods):
        loo = cl.get(ref + "::FUSION-minus-" + m)
        single = cl.get(ref + "::" + m)
        standalone_assessment = assess_standalone(single, task=task, seed=42)
        incremental = assess_incremental(
            fusion_ref,
            loo,
            task=task,
            candidate=m,
            family_size=family_size,
            seed=42,
        )
        if (
            intended_candidate_layers is not None
            and (
                fusion_ref is None
                or set(fusion_ref.modalities) != set(intended_candidate_layers)
            )
        ):
            incremental = incremental | {
                "status": "unavailable",
                "estimate": None,
                "ordinary_95_ci": None,
                "decision_ci": None,
                "reason": "full_model_intended_layer_mismatch",
            }
        if intended_candidate_layers is not None and m not in intended_candidate_layers:
            incremental = incremental | {
                "status": "unavailable",
                "estimate": None,
                "ordinary_95_ci": None,
                "decision_ci": None,
                "reason": "candidate_not_in_intended_family",
            }
        expected_reduced_layers = (
            None
            if intended_candidate_layers is None
            else set(intended_candidate_layers) - {m}
        )
        if (
            expected_reduced_layers is not None
            and loo is not None
            and set(loo.modalities) != expected_reduced_layers
        ):
            incremental = incremental | {
                "status": "unavailable",
                "estimate": None,
                "ordinary_95_ci": None,
                "decision_ci": None,
                "reason": "reduced_model_intended_layer_mismatch",
            }
        try:
            gain_c, gain_c_p = _paired_gain(fusion_ref.fold_primary, loo.fold_primary)
        except (AttributeError, TypeError, ValueError):
            gain_c, gain_c_p = float("nan"), float("nan")
        gain_eligibility = {
            "eligible": incremental["status"] not in {"unavailable", "not_applicable"},
            "reason": incremental.get("reason"),
        }
        pooled_oof_gain = incremental["estimate"]
        pooled_oof_gain_ci = incremental["ordinary_95_ci"]
        gain_n, gain_n_p = float("nan"), float("nan")
        nn_loo = nn.get("neural::FUSION-minus-" + m)
        if nn_fusion and nn_loo:
            try:
                gain_n, gain_n_p = _paired_gain(nn_fusion.fold_primary, nn_loo.fold_primary)
            except ValueError:
                gain_n, gain_n_p = float("nan"), float("nan")

        # redundancy: highest CKA with a modality that is individually stronger.
        red_partner, red_cka = None, float("nan")
        for j, mj in enumerate(mods):
            if mj == m:
                continue
            v = cka[i, j]
            if np.isfinite(v) and (not np.isfinite(red_cka) or v > red_cka):
                if standalone.get(mj, chance) >= standalone.get(m, chance):
                    red_partner, red_cka = mj, float(v)

        batch_structured = bool(batch_pm.get(m, {}).get("flag", False))
        batch_confounding = batch_pm.get(m, {}).get("target_confounding", {})
        batch_confounded = batch_structured and bool(batch_confounding.get("flag", False))
        status = incremental["status"]
        similarity_observed = (red_partner is not None and np.isfinite(red_cka)
                               and red_cka > CKA_REDUNDANT
                               and status != "increased_held_out_performance")
        verdict, rec = _ledger_text(status)
        absolute_scores = {
            "full": (
                float(fusion_ref.primary)
                if fusion_ref is not None and np.isfinite(fusion_ref.primary)
                else None
            ),
            "reduced": float(loo.primary) if loo is not None and np.isfinite(loo.primary) else None,
            "outcome_mean_reference": 0.0 if task == "regression" else None,
        }
        if (
            task == "regression"
            and incremental["estimate"] is not None
            and incremental["estimate"] > 0
            and absolute_scores["full"] is not None
            and absolute_scores["reduced"] is not None
            and absolute_scores["full"] < 0
            and absolute_scores["reduced"] < 0
        ):
            absolute_scores["interpretation"] = "relative_score_improvement_with_negative_absolute_scores"
        else:
            absolute_scores["interpretation"] = None

        top_features = _top_features(fusion_ref, m, k=8) if fusion_ref else []
        ledger.append({
            "modality": m,
            "n_features": int(aligned.modalities[m].shape[1]),
            "ledger_schema_version": 2,
            "standalone_assessment": standalone_assessment,
            "incremental_assessment": incremental | {"absolute_scores": absolute_scores},
            "diagnostic_flags": {
                "batch_structured": batch_structured,
                "batch_confounded": batch_confounded,
                "missingness_biased": bool(miss_flags_by_mod.get(m)),
                "leakage_warning": leakage,
            },
            # v1-compatible fields remain for existing readers. They are derived
            # output only and do not decide the v2 operational status.
            "legacy_fields_deprecated": True,
            "standalone_primary": standalone_assessment["estimate"],
            "foldmean_marginal_gain": _r(gain_c),
            "foldmean_marginal_gain_p": _r(gain_c_p),
            "pooled_oof_marginal_gain": pooled_oof_gain,
            "pooled_oof_marginal_gain_ci": pooled_oof_gain_ci,
            "marginal_gain_status": status,
            "marginal_gain_eligibility": gain_eligibility,
            "marginal_gain_neural": _r(gain_n),
            "marginal_gain_neural_p": _r(gain_n_p),
            "redundancy_max_cka": _r(red_cka),
            "similarity_with": red_partner if similarity_observed else None,
            "redundant_with": None,
            "batch_structured": batch_structured,
            "batch_confounded": batch_confounded,
            "batch_column": batch_pm.get(m, {}).get("batch_column"),
            "batch_target_confounding": batch_confounding,
            "missingness_biased": bool(miss_flags_by_mod.get(m)),
            "verdict": verdict,
            "recommendation": rec,
            "top_features": top_features,
        })

    # The declared reference estimator is retained; no OOF winner is selected.
    best = fusion_ref
    if single_modality:
        # the ::FUSION fit IS the single layer refit on identical data; surface the
        # honestly-named single-layer result (linear::{mod}) so no "FUSION" name leaks.
        best = cl.get(f"{ref}::{mods[0]}") or best
    best_single = None

    fusion_gain = (
        best.primary - best_single.primary if (best and best_single and np.isfinite(best.primary) and np.isfinite(best_single.primary)) else float("nan")
    )
    fusion_gain_ci = None if single_modality else _paired_gain_ci(best, best_single, task, seed=42)

    calibration = None
    auprc_baseline = None
    if task == "classification" and best is not None:
        from omicau.models.base import calibration_metrics
        calibration = calibration_metrics(best)
        yv = np.asarray(aligned.y)
        cls = np.unique(yv)
        if len(cls) == 2:                     # AUPRC is only interpretable vs prevalence
            auprc_baseline = float(np.mean(yv == cls[1]))

    subgroups = _subgroup_metrics(best, aligned, metric)

    batch_blocked = None
    bb = cl.get("stress::batch-blocked-FUSION")
    if bb is not None and fusion_ref is not None and np.isfinite(bb.primary) and np.isfinite(fusion_ref.primary):
        batch_blocked = {"primary": _r(bb.primary), "standard": _r(fusion_ref.primary),
                         "optimism_gap": _r(fusion_ref.primary - bb.primary)}

    batch_adjusted = None
    bad = cl.get("sensitivity::batch-adjusted-FUSION")
    if bad is not None and fusion_ref is not None and np.isfinite(bad.primary) and np.isfinite(fusion_ref.primary):
        batch_adjusted = {"primary": _r(bad.primary), "standard": _r(fusion_ref.primary),
                          "delta": _r(bad.primary - fusion_ref.primary),
                          "ci_low": bad.extra.get("ci_low"), "ci_high": bad.extra.get("ci_high")}

    return {
        "ledger_schema_version": 2,
        "primary_metric": metric,
        "task": task,
        "chance_level": chance,
        "best_model": _model_brief(best),
        "best_model_selection": "declared reference estimator before outcome fitting",
        "calibration": calibration,
        "auprc_baseline": auprc_baseline,
        "subgroups": subgroups,
        "batch_blocked": batch_blocked,
        "batch_adjusted": batch_adjusted,
        "single_modality": single_modality,
        "intended_candidate_layers": intended_candidate_layers,
        "intended_family_size": family_size,
        "best_single_modality": None if single_modality else _model_brief(best_single),
        "fusion_gain_over_best_single": None if single_modality else _r(fusion_gain),
        "fusion_gain_ci": fusion_gain_ci,
        "modality_ledger": ledger,
        "redundancy_matrix": {"modalities": mods, "cka": [[_r(v) for v in row] for row in cka]},
        "controls": controls,
        "leakage_warning": leakage,
        "control_alarm_status": control_alarm_status,
        "leakage_text": leakage_text,
        "categorical_verdict_inputs": {
            "model_scope": "classical_reference_estimator_only",
            "reference_estimator": ref,
            "neural_results_used": False,
            "feature_attribution_used": False,
            "decision_source": "incremental_assessment.decision_ci",
            "historical_thresholds_retired_from_categories": {
                "standalone_margin_over_chance": USEFUL_MARGIN,
                "marginal_gain": GAIN_EPS,
                "marginal_gain_alpha": GAIN_ALPHA,
                "control_margin": CONTROL_MARGIN,
            },
            "records": [{
                "modality": row["modality"],
                "ledger_schema_version": row["ledger_schema_version"],
                "standalone_assessment": row["standalone_assessment"],
                "incremental_assessment": row["incremental_assessment"],
                "diagnostic_flags": row["diagnostic_flags"],
                "legacy_fields_deprecated": row["legacy_fields_deprecated"],
                "standalone_primary": row["standalone_primary"],
                "foldmean_marginal_gain": row["foldmean_marginal_gain"],
                "foldmean_marginal_gain_p": row["foldmean_marginal_gain_p"],
                "pooled_oof_marginal_gain": row["pooled_oof_marginal_gain"],
                "pooled_oof_marginal_gain_ci": row["pooled_oof_marginal_gain_ci"],
                "marginal_gain_status": row["marginal_gain_status"],
                "marginal_gain_eligibility": row["marginal_gain_eligibility"],
                "redundancy_max_cka": row["redundancy_max_cka"],
                "similarity_with": row["similarity_with"],
                "batch_confounded": row["batch_confounded"],
                "missingness_biased": row["missingness_biased"],
                "verdict": row["verdict"],
            } for row in ledger],
        },
        "summary_flags": _summary_flags(ledger, leakage, fusion_gain, single_modality, mods),
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ledger_text(status: str) -> tuple[str, str]:
    messages = {
        "increased_held_out_performance": (
            "increased held-out performance",
            "The family decision interval is strictly above zero for this locked paired comparison.",
        ),
        "reduced_held_out_performance": (
            "reduced held-out performance",
            "The family decision interval is strictly below zero for this locked paired comparison.",
        ),
        "unresolved": (
            "unresolved incremental performance",
            "The family decision interval crosses or touches zero for this locked paired comparison.",
        ),
        "unavailable": (
            "incremental performance unavailable",
            "Required paired out-of-fold records were unavailable or failed identity validation.",
        ),
        "not_applicable": (
            "incremental performance not applicable",
            "A one-assay ledger has no leave-one-layer-out contrast.",
        ),
    }
    return messages.get(status, messages["unavailable"])


def _verdict(standalone_useful, gain_state, similarity_observed, batch_confounded, red_partner,
             miss_biased, leakage, gain_c=float("nan"), gain_c_p=float("nan")):
    """Map a modality's evidence to a verdict. gain_state is one of 'adds_sig'
    (positive leave-one-out gain that clears the paired significance test),
    'adds_ns' (positive gain not distinguishable from zero), or 'none'."""
    adds = gain_state == "adds_sig"
    if batch_confounded and not adds:
        return ("batch-confounded (no marginal gain)",
                "Batch is confounded with the outcome here and this layer adds no significant signal beyond "
                "the others; treat any apparent signal from it as untrustworthy (it may be batch leaking as signal).")
    if standalone_useful and adds:
        base = "predictive (adds marginal signal)"
        rec = "Retain: contributes information beyond the other modalities."
        if batch_confounded:
            rec += " Caution: batch is confounded with the outcome, so confirm this gain is not batch leakage."
        if miss_biased:
            rec += " Note target-associated missingness; verify the gain is not a missingness artifact."
        if leakage:
            rec += " Confirm after resolving the control-baseline leakage warning."
        return base, rec
    if standalone_useful and gain_state == "adds_ns":
        # positive leave-one-out gain, but the paired test cannot separate it from
        # zero -- do NOT badge it green; say so plainly so the reader does not read
        # a noisy gain as an established contribution.
        p_txt = f" (p={gain_c_p:.2f})" if np.isfinite(gain_c_p) else ""
        g_txt = f"+{gain_c:.3f}" if np.isfinite(gain_c) else "positive"
        return ("informative; incremental contribution inconclusive",
                f"Predictive on its own, but its leave-one-out gain ({g_txt}) is not distinguishable from zero at "
                f"this sample size{p_txt}; do not claim it improves the fusion without more data.")
    if standalone_useful:
        if similarity_observed:
            return ("informative; incremental contribution inconclusive",
                    f"Predictive on its own and similar to '{red_partner}' by CKA, but the observed "
                    "leave-one-out result does not establish redundancy or absence of complementary signal.")
        return ("informative; incremental contribution inconclusive",
                "Predictive on its own, but the available leave-one-out evidence does not establish an "
                "incremental fusion contribution or absence of one.")
    return ("no detectable signal (control-like)",
            "Performs near chance and adds nothing; a candidate to drop.")


def _verdict_single(standalone_useful, batch_confounded, miss_biased, leakage):
    """Verdict for a single-modality run: no fusion or redundancy to assess, so
    judge only whether the one layer carries real, leakage-free standalone signal."""
    if batch_confounded:
        return ("batch-confounded",
                "Batch is confounded with the outcome here, so this layer's apparent signal may be "
                "a technical artifact rather than biology; treat it as untrustworthy.")
    if not standalone_useful:
        return ("no detectable signal (near chance)",
                "Performs near chance; on its own it does not predict the outcome here.")
    rec = ("This layer carries predictive signal on its own. With one modality there is no fusion or "
           "redundancy to assess; validate externally before use.")
    if miss_biased:
        rec += " Note target-associated missingness; verify the signal is not a missingness artifact."
    if leakage:
        rec += " Confirm after resolving the control-baseline leakage warning."
    return ("predictive (standalone)", rec)


SUBGROUP_MIN_N = 20   # stratum floor; per-stratum metrics below this are too noisy to score


def _subgroup_metrics(best, aligned, metric_key, n_boot: int = 500, seed: int = 42):
    """Re-score the best model's pooled OOF within strata of the batch/site column
    — a fairness/generalization check. Pure re-aggregation, no retraining. Each
    per-stratum metric and the across-stratum gap carry a bootstrap 95% CI so a
    reader can tell a real disparity from small-sample noise (a bare max-min gap
    over tiny strata is dominated by sampling variance)."""
    if best is None or getattr(best, "oof_true", None) is None or aligned.batch is None:
        return None
    if aligned.task == "survival":
        return None                       # per-stratum C-index deferred (score_predictions is not survival-aware)
    from omicau.models.base import score_predictions
    y = np.asarray(best.oof_true)
    score = np.asarray(best.oof_score)
    pred = np.asarray(best.oof_pred)
    task = aligned.task
    n_classes = len(np.unique(y)) if task == "classification" else 0
    strata = aligned.batch.astype("string").to_numpy()
    rng = np.random.default_rng(seed)

    def _metric(idx):
        if task == "classification" and len(np.unique(y[idx])) < n_classes:
            return float("nan")
        hard = pred[idx].astype(int) if task == "classification" else pred[idx]
        v = score_predictions(y[idx], score[idx], hard, task).get(metric_key, float("nan"))
        return float(v) if v is not None else float("nan")

    rows, scored_idx = [], []
    for lvl in sorted(set(strata)):
        m = strata == lvl
        n_lvl = int(m.sum())
        idx = np.where(m)[0]
        if n_lvl < SUBGROUP_MIN_N or not np.isfinite(_metric(idx)):
            rows.append({"stratum": str(lvl), "n": n_lvl, "primary": None,
                         "ci_low": None, "ci_high": None})
            continue
        boots = []
        for _ in range(n_boot):
            bs = rng.choice(idx, size=len(idx), replace=True)
            v = _metric(bs)
            if np.isfinite(v):
                boots.append(v)
        lo = float(np.percentile(boots, 2.5)) if len(boots) >= 20 else None
        hi = float(np.percentile(boots, 97.5)) if len(boots) >= 20 else None
        rows.append({"stratum": str(lvl), "n": n_lvl, "primary": _r(_metric(idx)),
                     "ci_low": _r(lo), "ci_high": _r(hi)})
        scored_idx.append(idx)
    vals = [r["primary"] for r in rows if r["primary"] is not None]
    if len(vals) < 2:
        return None
    # Bootstrap the max-min gap jointly across the scored strata.
    gap_boot = []
    for _ in range(n_boot):
        sv = [_metric(rng.choice(idx, size=len(idx), replace=True)) for idx in scored_idx]
        if all(np.isfinite(v) for v in sv):
            gap_boot.append(max(sv) - min(sv))
    gap_lo = float(np.percentile(gap_boot, 2.5)) if len(gap_boot) >= 20 else None
    gap_hi = float(np.percentile(gap_boot, 97.5)) if len(gap_boot) >= 20 else None
    return {"by": aligned.batch.name, "metric": metric_key, "strata": rows,
            "gap": _r(max(vals) - min(vals)), "gap_ci_low": _r(gap_lo),
            "gap_ci_high": _r(gap_hi), "min_stratum": SUBGROUP_MIN_N}


def _missing_flags_by_modality(missing_diag: dict | None) -> dict[str, bool]:
    out: dict[str, bool] = {}
    if not missing_diag:
        return out
    for t in missing_diag.get("tests", []):
        if t.get("association") == "target" and t.get("flag"):
            out[t["modality"]] = True
    return out


def _top_features(fusion_result, modality: str, k: int = 8) -> list[dict[str, Any]]:
    if not fusion_result or not fusion_result.feature_importance:
        return []
    prefix = f"{modality}::"
    items = [(name, v) for name, v in fusion_result.feature_importance.items() if name.startswith(prefix)]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return [{"feature": name.split("::", 1)[1], "importance": _r(v)} for name, v in items[:k]]


def _model_brief(r) -> dict[str, Any] | None:
    if r is None:
        return None
    return {
        "name": r.name,
        "primary": _r(r.primary),
        "primary_std": _r(r.primary_std),
        "modalities": list(r.modalities),
        "metrics": {kk: _r(vv) for kk, vv in r.metrics.items()},
    }


def _summary_flags(ledger, leakage, fusion_gain, single_modality=False, mods=None) -> list[str]:
    flags = []
    if leakage:
        flags.append("Control-baseline leakage warning is active.")
    increased = [
        row for row in ledger
        if row.get("incremental_assessment", {}).get("status") == "increased_held_out_performance"
    ]
    reduced = [
        row for row in ledger
        if row.get("incremental_assessment", {}).get("status") == "reduced_held_out_performance"
    ]
    unresolved = [
        row for row in ledger
        if row.get("incremental_assessment", {}).get("status") == "unresolved"
    ]
    unavailable = [
        row for row in ledger
        if row.get("incremental_assessment", {}).get("status") == "unavailable"
    ]
    conf = [l for l in ledger if l["batch_confounded"]]
    if increased:
        flags.append(f"{len(increased)} layer(s) increased held-out performance: {', '.join(l['modality'] for l in increased)}.")
    if reduced:
        flags.append(f"{len(reduced)} layer(s) reduced held-out performance: {', '.join(l['modality'] for l in reduced)}.")
    if conf:
        flags.append(f"Batch-confounded layer(s): {', '.join(l['modality'] for l in conf)}.")
    if unresolved:
        flags.append(f"Incremental performance remains unresolved for: {', '.join(l['modality'] for l in unresolved)}.")
    if unavailable:
        flags.append(f"Incremental performance is unavailable for: {', '.join(l['modality'] for l in unavailable)}.")
    if single_modality:
        name = (mods[0] if mods else "one layer")
        flags.append(f"Single modality ({name}): fusion, redundancy, and marginal-contribution "
                     "analyses do not apply.")
    elif np.isfinite(fusion_gain):
        verb = "improves on" if fusion_gain > GAIN_EPS else "does not beat"
        flags.append(f"Fusion {verb} the best single modality by {fusion_gain:+.3f}.")
    return flags


def _r(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, 5) if np.isfinite(v) else None
