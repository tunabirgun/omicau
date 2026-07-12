"""Modality-utility interpretation.

Fuses the classical and neural benchmark results with the data-hygiene
diagnostics to answer the operational question: does each omic layer earn its
place? For every modality it reports the standalone performance, the marginal
gain from adding it to the fusion (leave-one-out delta with a paired test across
folds), representational redundancy against the other layers (linear CKA), and a
verdict that separates real predictive signal from batch artifacts and noise.
The control baselines gate the whole ledger: if a shuffled-target run scores far
above chance, a leakage warning is raised and gains are treated as untrustworthy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

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


def _by_name(results: list) -> dict[str, Any]:
    return {r.name: r for r in results}


def _linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    """Linear centered kernel alignment between two sample x feature matrices."""
    A = _prep_cka(A)
    B = _prep_cka(B)
    if A is None or B is None:
        return float("nan")
    cross = np.linalg.norm(A.T @ B) ** 2
    na = np.linalg.norm(A.T @ A)
    nb = np.linalg.norm(B.T @ B)
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
    """Mean per-fold (fusion - leave_one_out) delta and its p-value.

    Uses the Nadeau-Bengio corrected resampled t-test rather than a plain paired
    t-test: k-fold train sets overlap, so the naive fold-difference variance is
    biased low and the uncorrected test has badly inflated Type-I error (Nadeau &
    Bengio, Machine Learning 52:239-281, 2003). The variance is inflated by
    (1/k + n_test/n_train); for k-fold, n_test/n_train = 1/(k-1).
    """
    a = np.asarray(fusion_folds, float)
    b = np.asarray(loo_folds, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
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


def _paired_gain_ci(best, best_single, task, n_boot: int = 1000, seed: int = 42,
                    alpha: float = 0.05) -> dict | None:
    """Percentile CI for (best.primary - best_single.primary) via a paired bootstrap.
    Both models are scored on the same pooled-OOF sample order, so one resample of
    the sample (or group) indices is applied to both — a genuine paired interval on
    the fusion gain, so the headline number is not reported without uncertainty."""
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
    n_classes = len(np.unique(y)) if task == "classification" else 0
    groups = getattr(best, "oof_groups", None)
    groups = None if groups is None else np.asarray(groups)
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
    return {"low": _r(float(np.percentile(vals, 100 * alpha / 2))),
            "high": _r(float(np.percentile(vals, 100 * (1 - alpha / 2))))}


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
                 "ci_high": getattr(r, "extra", {}).get("ci_high")}
                for r in classical_out.get("controls", [])]
    alarm = chance + CONTROL_MARGIN
    present = [c for c in controls if c["primary"] is not None]

    def _sig_above_chance(c):
        # A control leaks if EITHER its 95% CI lower bound clears chance (a
        # significant control) OR its point estimate clears the margin. An audit
        # tool should err toward warning, so a leaking control with a wide CI is
        # not silently cleared; gating on the CI *upper* bound is avoided (it would
        # false-fire on any wide small-sample CI).
        by_ci = c.get("ci_low") is not None and c["ci_low"] > chance
        return by_ci or (c["primary"] > alarm)

    leaking = [c for c in present if _sig_above_chance(c)]
    leakage = bool(leaking)
    leakage_text = (
        f"A control baseline ({leaking[0]['name'].split('::')[-1]}) is significantly above chance "
        f"(score {leaking[0]['primary']:.3f}, chance ~ {chance:.2f}); treat reported gains with "
        "caution and re-check group-aware splitting."
        if leakage
        else (f"All control baselines scored near chance (~ {chance:.2f}); the harness shows no "
              "target or pipeline leakage. (These controls scramble the target/features; they do "
              "not by themselves rule out group leakage from a mis-set or missing group column — "
              "see the grouping status.)")
    )

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

    # standalone primaries per modality (reference estimator).
    standalone = {m: (cl.get(f"{ref}::{m}").primary if f"{ref}::{m}" in cl else float("nan")) for m in mods}

    # -- per-modality ledger ---------------------------------------------- #
    ledger: list[dict[str, Any]] = []
    batch_pm = (batch_diag or {}).get("per_modality", {})
    # A modality is "batch-confounded" only when its variance is batch-structured
    # AND batch is confounded with the outcome globally. Batch orthogonal to the
    # outcome is the harmless case (Nygaard et al. 2016) -- do not flag it.
    confounded_global = bool((batch_diag or {}).get("confounding", {}).get("flag", False))
    miss_flags_by_mod = _missing_flags_by_modality(missing_diag)

    for i, m in enumerate(mods):
        loo = cl.get(f"{ref}::FUSION-minus-{m}")
        gain_c, gain_c_p = (
            _paired_gain(fusion_ref.fold_primary, loo.fold_primary)
            if fusion_ref and loo else (float("nan"), float("nan"))
        )
        gain_n, gain_n_p = float("nan"), float("nan")
        nn_loo = nn.get(f"neural::FUSION-minus-{m}")
        if nn_fusion and nn_loo:
            gain_n, gain_n_p = _paired_gain(nn_fusion.fold_primary, nn_loo.fold_primary)

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
        batch_confounded = batch_structured and confounded_global
        standalone_useful = np.isfinite(standalone[m]) and standalone[m] > chance + USEFUL_MARGIN
        # A positive point-estimate gain is not enough to claim the layer "adds
        # signal": the leave-one-out delta must also clear the paired Nadeau-Bengio
        # test. Three states drive the verdict/badge/figure so a noisy positive
        # gain is never rendered as an established contribution.
        gain_positive = np.isfinite(gain_c) and gain_c > GAIN_EPS
        gain_significant = gain_positive and np.isfinite(gain_c_p) and gain_c_p < GAIN_ALPHA
        gain_state = "adds_sig" if gain_significant else ("adds_ns" if gain_positive else "none")
        redundant = (red_partner is not None and np.isfinite(red_cka)
                     and red_cka > CKA_REDUNDANT and gain_state != "adds_sig")

        if single_modality:
            verdict, rec = _verdict_single(
                standalone_useful, batch_confounded, bool(miss_flags_by_mod.get(m)), leakage,
            )
        else:
            verdict, rec = _verdict(
                standalone_useful, gain_state, redundant, batch_confounded, red_partner,
                bool(miss_flags_by_mod.get(m)), leakage, gain_c, gain_c_p,
            )

        top_features = _top_features(fusion_ref, m, k=8) if fusion_ref else []
        ledger.append({
            "modality": m,
            "n_features": int(aligned.modalities[m].shape[1]),
            "standalone_primary": _r(standalone[m]),
            "marginal_gain_classical": _r(gain_c),
            "marginal_gain_p": _r(gain_c_p),
            "marginal_gain_neural": _r(gain_n),
            "marginal_gain_neural_p": _r(gain_n_p),
            "redundancy_max_cka": _r(red_cka),
            "redundant_with": red_partner,
            "batch_structured": batch_structured,
            "batch_confounded": batch_confounded,
            "missingness_biased": bool(miss_flags_by_mod.get(m)),
            "verdict": verdict,
            "recommendation": rec,
            "top_features": top_features,
        })

    # -- best model across all fusion candidates -------------------------- #
    fusion_candidates = [r for r in classical_out["results"] if r.name.endswith("::FUSION")]
    fusion_candidates += [r for r in neural_out.get("results", []) if r.name == "neural::FUSION"]
    best = max(fusion_candidates, key=lambda r: (r.primary if np.isfinite(r.primary) else -1), default=None)
    if single_modality:
        # the ::FUSION fit IS the single layer refit on identical data; surface the
        # honestly-named single-layer result (linear::{mod}) so no "FUSION" name leaks.
        best = cl.get(f"{ref}::{mods[0]}") or best
    # best_single spans BOTH classical and neural single-modality models, so the
    # fusion-gain baseline is symmetric with `best` (which includes neural::FUSION).
    single_pool = [r for r in classical_out["results"] if "::" in r.name and "FUSION" not in r.name
                   and not r.name.startswith(("control::", "stress::"))]
    single_pool += [r for r in neural_out.get("results", []) if "::" in r.name and "FUSION" not in r.name]
    best_single = max(single_pool, key=lambda r: (r.primary if np.isfinite(r.primary) else -1), default=None)

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
        "primary_metric": metric,
        "task": task,
        "chance_level": chance,
        "best_model": _model_brief(best),
        "calibration": calibration,
        "auprc_baseline": auprc_baseline,
        "subgroups": subgroups,
        "batch_blocked": batch_blocked,
        "batch_adjusted": batch_adjusted,
        "single_modality": single_modality,
        "best_single_modality": None if single_modality else _model_brief(best_single),
        "fusion_gain_over_best_single": None if single_modality else _r(fusion_gain),
        "fusion_gain_ci": fusion_gain_ci,
        "modality_ledger": ledger,
        "redundancy_matrix": {"modalities": mods, "cka": [[_r(v) for v in row] for row in cka]},
        "controls": controls,
        "leakage_warning": leakage,
        "leakage_text": leakage_text,
        "summary_flags": _summary_flags(ledger, leakage, fusion_gain, single_modality, mods),
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _verdict(standalone_useful, gain_state, redundant, batch_confounded, red_partner,
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
        return ("informative alone (fusion gain not significant)",
                f"Predictive on its own, but its leave-one-out gain ({g_txt}) is not distinguishable from zero at "
                f"this sample size{p_txt}; do not claim it improves the fusion without more data.")
    if standalone_useful and redundant:
        return (f"redundant (subsumed by {red_partner})",
                f"Informative alone but its signal is largely shared with '{red_partner}'; adds little on top.")
    if standalone_useful:
        return ("informative but non-additive",
                "Predictive on its own yet not additive in fusion; likely overlapping with the retained layers.")
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
    useful = [l for l in ledger if l["verdict"].startswith("predictive")]
    dead = [l for l in ledger if l["verdict"].startswith("no detectable")]
    conf = [l for l in ledger if l["batch_confounded"]]
    if useful:
        flags.append(f"{len(useful)} modality(ies) add marginal signal: {', '.join(l['modality'] for l in useful)}.")
    if conf:
        flags.append(f"Batch-confounded layer(s): {', '.join(l['modality'] for l in conf)}.")
    if dead:
        flags.append(f"Control-like layer(s) with no signal: {', '.join(l['modality'] for l in dead)}.")
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
