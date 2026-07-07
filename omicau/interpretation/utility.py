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

CHANCE = {"classification": 0.5, "regression": 0.0}
USEFUL_MARGIN = 0.05
GAIN_EPS = 0.01
CKA_REDUNDANT = 0.5
CONTROL_ALARM = 0.62


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
    """Mean per-fold (fusion - leave_one_out) delta and paired-test p-value."""
    a = np.asarray(fusion_folds, float)
    b = np.asarray(loo_folds, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    d = a[mask] - b[mask]
    mean_d = float(d.mean())
    if np.ptp(d) == 0:
        return mean_d, float("nan")
    try:
        p = float(stats.ttest_rel(a[mask], b[mask]).pvalue)
    except (ValueError, FloatingPointError):
        p = float("nan")
    return mean_d, p


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

    cl = _by_name(classical_out["results"])
    ref = classical_out["reference_estimator"]
    nn = _by_name(neural_out.get("results", [])) if neural_out.get("enabled") else {}

    fusion_ref = cl.get(f"{ref}::FUSION")
    nn_fusion = nn.get("neural::FUSION")

    # -- controls / leakage gate ------------------------------------------ #
    controls = [{"name": r.name, "primary": _r(r.primary)} for r in classical_out.get("controls", [])]
    worst_control = max((c["primary"] for c in controls if c["primary"] is not None), default=chance)
    leakage = worst_control > CONTROL_ALARM
    leakage_text = (
        f"A control baseline scored {worst_control:.3f} (> {CONTROL_ALARM}); "
        "treat reported gains with caution and re-check group-aware splitting."
        if leakage
        else "All control baselines scored near chance; the harness shows no leakage."
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
    miss_flags_by_mod = _missing_flags_by_modality(missing_diag)

    for i, m in enumerate(mods):
        loo = cl.get(f"{ref}::FUSION-minus-{m}")
        gain_c, gain_c_p = (
            _paired_gain(fusion_ref.fold_primary, loo.fold_primary)
            if fusion_ref and loo else (float("nan"), float("nan"))
        )
        gain_n = float("nan")
        nn_loo = nn.get(f"neural::FUSION-minus-{m}")
        if nn_fusion and nn_loo:
            gain_n = nn_fusion.primary - nn_loo.primary

        # redundancy: highest CKA with a modality that is individually stronger.
        red_partner, red_cka = None, float("nan")
        for j, mj in enumerate(mods):
            if mj == m:
                continue
            v = cka[i, j]
            if np.isfinite(v) and (not np.isfinite(red_cka) or v > red_cka):
                if standalone.get(mj, chance) >= standalone.get(m, chance):
                    red_partner, red_cka = mj, float(v)

        batch_flag = bool(batch_pm.get(m, {}).get("flag", False))
        standalone_useful = np.isfinite(standalone[m]) and standalone[m] > chance + USEFUL_MARGIN
        adds = np.isfinite(gain_c) and gain_c > GAIN_EPS
        redundant = (red_partner is not None and np.isfinite(red_cka) and red_cka > CKA_REDUNDANT and not adds)

        verdict, rec = _verdict(
            standalone_useful, adds, redundant, batch_flag, red_partner,
            bool(miss_flags_by_mod.get(m)), leakage,
        )

        top_features = _top_features(fusion_ref, m, k=8) if fusion_ref else []
        ledger.append({
            "modality": m,
            "n_features": int(aligned.modalities[m].shape[1]),
            "standalone_primary": _r(standalone[m]),
            "marginal_gain_classical": _r(gain_c),
            "marginal_gain_p": _r(gain_c_p),
            "marginal_gain_neural": _r(gain_n),
            "redundancy_max_cka": _r(red_cka),
            "redundant_with": red_partner,
            "batch_confounded": batch_flag,
            "missingness_biased": bool(miss_flags_by_mod.get(m)),
            "verdict": verdict,
            "recommendation": rec,
            "top_features": top_features,
        })

    # -- best model across all fusion candidates -------------------------- #
    fusion_candidates = [r for r in classical_out["results"] if r.name.endswith("::FUSION")]
    fusion_candidates += [r for r in neural_out.get("results", []) if r.name == "neural::FUSION"]
    best = max(fusion_candidates, key=lambda r: (r.primary if np.isfinite(r.primary) else -1), default=None)
    best_single = max(
        (r for r in classical_out["results"] if "::" in r.name and not r.name.split("::")[1].startswith("FUSION")),
        key=lambda r: (r.primary if np.isfinite(r.primary) else -1), default=None,
    )

    fusion_gain = (
        best.primary - best_single.primary if (best and best_single and np.isfinite(best.primary) and np.isfinite(best_single.primary)) else float("nan")
    )

    return {
        "primary_metric": metric,
        "task": task,
        "chance_level": chance,
        "best_model": _model_brief(best),
        "best_single_modality": _model_brief(best_single),
        "fusion_gain_over_best_single": _r(fusion_gain),
        "modality_ledger": ledger,
        "redundancy_matrix": {"modalities": mods, "cka": [[_r(v) for v in row] for row in cka]},
        "controls": controls,
        "leakage_warning": leakage,
        "leakage_text": leakage_text,
        "summary_flags": _summary_flags(ledger, leakage, fusion_gain),
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _verdict(standalone_useful, adds, redundant, batch_flag, red_partner, miss_biased, leakage):
    if batch_flag and not adds:
        return ("batch-confounded (no marginal gain)",
                "Batch structures this layer's variance and it adds no signal; correct the batch effect before trusting it.")
    if standalone_useful and adds:
        base = "predictive (adds marginal signal)"
        rec = "Retain: contributes information beyond the other modalities."
        if miss_biased:
            rec += " Note target-associated missingness; verify the gain is not a missingness artifact."
        if leakage:
            rec += " Confirm after resolving the control-baseline leakage warning."
        return base, rec
    if standalone_useful and redundant:
        return (f"redundant (subsumed by {red_partner})",
                f"Informative alone but its signal is largely shared with '{red_partner}'; adds little on top.")
    if standalone_useful and not adds:
        return ("informative but non-additive",
                "Predictive on its own yet not additive in fusion; likely overlapping with the retained layers.")
    return ("no detectable signal (control-like)",
            "Performs near chance and adds nothing; a candidate to drop.")


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


def _summary_flags(ledger, leakage, fusion_gain) -> list[str]:
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
    if np.isfinite(fusion_gain):
        verb = "improves on" if fusion_gain > GAIN_EPS else "does not beat"
        flags.append(f"Fusion {verb} the best single modality by {fusion_gain:+.3f}.")
    return flags


def _r(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, 5) if np.isfinite(v) else None
