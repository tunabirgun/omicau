"""Optional LLM interpretation plugin.

This tier is strictly optional. When enabled and the ``anthropic`` package plus a
valid API key are present, a compact, PHI-free summary of the audit is sent to
the Claude API and a fixed JSON schema is parsed back. When the package, key, or
network is absent -- or the call fails or is refused -- the tool degrades to a
deterministic, rule-based summary that fills the identical schema, so the report
layout never breaks. Only aggregate statistics are ever transmitted: no sample
identifiers, feature sequences, file paths, or raw values leave the machine.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any

SCHEMA_KEYS = (
    "clinical_verdict",
    "data_hygiene_rating",
    "modality_utility_ledger",
    "actionable_recommendations",
)

_SYSTEM_PROMPT = (
    "You are a rigorous computational biology reviewer. You are given aggregate, "
    "de-identified diagnostics from a multi-omic data audit. Judge whether multi-omic "
    "fusion is justified and trustworthy for this dataset. Be concise, scientific, and "
    "cautious about batch effects, missingness bias, and leakage. Respond with STRICT "
    "JSON only (no prose, no code fences) using exactly these keys: "
    "clinical_verdict (string, 2-4 sentences for a PI/clinician), "
    "data_hygiene_rating (string: one of 'clean', 'moderate concerns', 'high concern', "
    "followed by a short rationale), "
    "modality_utility_ledger (array of objects with keys: modality, verdict, recommendation), "
    "actionable_recommendations (array of short strings)."
)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def summarize(context: dict[str, Any], config, *, api_key: str | None = None) -> dict[str, Any]:
    """Return the interpretation summary, via a user-supplied LLM if enabled else
    rule-based. The ephemeral ``api_key`` (UI-supplied, in memory only) wins over
    the named environment variable; it stays a local and is never persisted."""
    llm = config.llm
    if llm.enabled:
        key = (api_key or os.environ.get(llm.api_key_env, "")).strip()
        # Local / OpenAI-compatible servers (Ollama, LM Studio, vLLM) need no key,
        # so the presence of a key must not gate them -- only cloud providers require it.
        needs_key = (llm.provider or "anthropic").lower() not in ("local", "openai_compatible")
        if key or not needs_key:
            result = _try_llm(context, llm, key)      # key may be "" -> call_llm injects a placeholder
            if result is not None:
                result["source"] = f"llm:{llm.provider}:{llm.model}"
                return result
    fallback = _rule_based(context)
    fallback["source"] = "rule_based"
    return fallback


# --------------------------------------------------------------------------- #
# LLM path (optional, resilient)
# --------------------------------------------------------------------------- #
def _retry_backoff(fn, *, retries: int = 4, base: float = 1.0, cap: float = 20.0):
    """Call ``fn`` with exponential backoff and jitter; return None on failure."""
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - resilience: any failure -> fallback
            last = exc
            name = type(exc).__name__
            # Do not retry auth / bad-request / missing-SDK errors (no point waiting).
            if any(k in name for k in ("Authentication", "PermissionDenied", "BadRequest",
                                       "NotFound", "ImportError", "ModuleNotFound")):
                break
            sleep = min(cap, base * (2 ** attempt)) + random.uniform(0, base)
            time.sleep(sleep)
    return None


def _try_llm(context: dict[str, Any], llm, api_key: str) -> dict[str, Any] | None:
    from omicau.interpretation import llm_client

    # Fail fast (no 15s of retries) when the provider SDK is not installed: the
    # docs promise a *silent, immediate* fallback to the rule-based summary.
    # import_module consults sys.modules first, so a test-injected fake still works.
    import importlib
    sdk = "anthropic" if (llm.provider or "anthropic").lower() == "anthropic" else "openai"
    try:
        importlib.import_module(sdk)
    except ImportError:
        return None

    payload = json.dumps(_sanitize_context(context), sort_keys=True)

    def _call():
        return llm_client.call_llm(
            provider=llm.provider, model=llm.model, api_key=api_key,
            base_url=getattr(llm, "base_url", None),
            system=_SYSTEM_PROMPT, user=f"Audit diagnostics (JSON):\n{payload}",
            max_tokens=llm.max_tokens, timeout=getattr(llm, "timeout", 60.0),
            openai_api=getattr(llm, "openai_api", "chat"))

    text = _retry_backoff(_call)
    if not text:
        return None
    parsed = _parse_json(text)
    if parsed is None:
        return None
    return _coerce_schema(parsed)


def _parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _coerce_schema(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["clinical_verdict"] = str(d.get("clinical_verdict", "")).strip() or "No verdict returned."
    out["data_hygiene_rating"] = str(d.get("data_hygiene_rating", "")).strip() or "unrated"
    ledger = d.get("modality_utility_ledger", [])
    out["modality_utility_ledger"] = ledger if isinstance(ledger, list) else []
    recs = d.get("actionable_recommendations", [])
    out["actionable_recommendations"] = recs if isinstance(recs, list) else []
    return out


# --------------------------------------------------------------------------- #
# Deterministic rule-based fallback
# --------------------------------------------------------------------------- #
def _rule_based(context: dict[str, Any]) -> dict[str, Any]:
    util = context.get("utility", {})
    best = util.get("best_model") or {}
    metric = util.get("primary_metric", "score")
    chance = util.get("chance_level", 0.5)
    primary = best.get("primary")
    gain = util.get("fusion_gain_over_best_single")
    leakage = util.get("leakage_warning", False)
    ledger_in = util.get("modality_ledger", [])
    n_samples = context.get("n_samples")

    def assessment(row: dict[str, Any]) -> dict[str, Any]:
        value = row.get("incremental_assessment")
        if isinstance(value, dict):
            return value
        return {"status": "unavailable", "reason": "archived_ledger_without_family_decision_interval"}

    increased = [m["modality"] for m in ledger_in
                 if assessment(m).get("status") == "increased_held_out_performance"]
    reduced = [m["modality"] for m in ledger_in
               if assessment(m).get("status") == "reduced_held_out_performance"]
    unresolved = [m["modality"] for m in ledger_in
                  if assessment(m).get("status") == "unresolved"]
    confounded = [m["modality"] for m in ledger_in if (m.get("diagnostic_flags") or {}).get(
        "batch_confounded", m.get("batch_confounded"))]

    # data hygiene rating from the count of active flags.
    n_flags = len(context.get("missingness_flags", [])) + len(context.get("batch_flags", []))
    if leakage or n_flags >= 3:
        rating = "high concern"
    elif n_flags >= 1:
        rating = "moderate concerns"
    else:
        rating = "no raised diagnostic flags"
    rating_text = (
        f"{rating}: {n_flags} diagnostic flag(s)"
        + (", control-baseline leakage warning active" if leakage else "")
        + "."
    )

    single_modality = util.get("single_modality", False)
    perf = f"{primary:.3f}" if isinstance(primary, (int, float)) else "n/a"
    metric_label = "R²" if metric.lower() in {"r2", "r²"} else metric.upper()
    reference_text = ("the conventional outcome-mean reference R²=0"
                      if metric.lower() in {"r2", "r²"} else f"the random-ranking reference ~ {chance}")
    if single_modality:
        verdict_bits = [
            "With a single modality, the report provides descriptive standalone performance rather than a "
            f"leave-one-layer-out comparison. The declared reference model ({best.get('name', 'n/a')}) "
            f"reaches {metric_label}={perf}, interpreted against {reference_text}."
        ]
    else:
        verdict_bits = [
            f"The declared reference fusion model ({best.get('name', 'n/a')}) reaches {metric_label}={perf} "
            f"and is interpreted against {reference_text}."
        ]
    if increased:
        verdict_bits.append(
            f"The locked paired layer comparison reports increased held-out performance for: {', '.join(increased)}."
        )
    if reduced:
        verdict_bits.append(
            f"The locked paired comparison reports reduced held-out performance when retaining: {', '.join(reduced)}."
        )
    if unresolved:
        verdict_bits.append(
            f"The contribution comparison is unresolved for: {', '.join(unresolved)}."
        )
    if confounded:
        verdict_bits.append(
            f"Treat {', '.join(confounded)} with caution: batch structure dominates its variance."
        )
    if leakage:
        verdict_bits.append(
            "A control baseline scored above chance; resolve the leakage warning before trusting gains."
        )

    ledger_out = [
        {
            "modality": m["modality"],
            "verdict": assessment(m).get("status", "unavailable"),
            "recommendation": "See the report ledger for the paired interval and diagnostics.",
        }
        for m in ledger_in
    ]

    recs: list[str] = []
    if confounded:
        recs.append(f"Correct or regress out batch effects in: {', '.join(confounded)}.")
    for f in context.get("missingness_flags", [])[:3]:
        recs.append(f"Investigate missingness bias: {f}")
    if leakage:
        recs.append("Re-audit cross-validation splits for group leakage (control baseline elevated).")
    if not recs:
        recs.append(
            "No diagnostic flag was raised by the configured checks; standalone performance remains descriptive here." if single_modality
            else "No diagnostic flag was raised by the configured checks; use the paired layer categories and intervals for this locked analysis.")
    if isinstance(n_samples, int) and n_samples < 60:
        recs.append(f"Sample size is small (n={n_samples}); treat effect sizes as preliminary.")

    return {
        "clinical_verdict": " ".join(verdict_bits),
        "data_hygiene_rating": rating_text,
        "modality_utility_ledger": ledger_out,
        "actionable_recommendations": recs,
    }


# --------------------------------------------------------------------------- #
# PHI-safe context construction
# --------------------------------------------------------------------------- #
def _sanitize_context(context: dict[str, Any]) -> dict[str, Any]:
    """Whitelist aggregate fields permitted in an optional interpretation payload."""
    util = context.get("utility", {})
    def interval(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        return {key: value.get(key) for key in ("lower", "upper", "quantile_method")}

    def incremental(row: dict[str, Any]) -> dict[str, Any]:
        value = row.get("incremental_assessment")
        if not isinstance(value, dict):
            return {"status": "unavailable", "reason": "archived_ledger_without_family_decision_interval"}
        family = value.get("multiplicity_family") if isinstance(value.get("multiplicity_family"), dict) else {}
        return {
            "status": value.get("status"), "estimate": value.get("estimate"),
            "ordinary_95_ci": interval(value.get("ordinary_95_ci")),
            "decision_ci": interval(value.get("decision_ci")),
            "multiplicity_family": {key: family.get(key) for key in (
                "family_size", "family_alpha", "decision_quantiles", "scope")},
            "resampling_unit": value.get("resampling_unit"),
            "n_valid_resamples": value.get("n_valid_resamples"),
            "reason": value.get("reason"),
        }

    safe_ledger = [
        {
            "modality": m.get("modality"),
            "n_features": m.get("n_features"),
            "standalone": {
                "status": (m.get("standalone_assessment") or {}).get("status", "unavailable"),
                "estimate": (m.get("standalone_assessment") or {}).get("estimate", m.get("standalone_primary")),
                "ordinary_95_ci": interval((m.get("standalone_assessment") or {}).get("ordinary_95_ci")),
            },
            "incremental": incremental(m),
            "diagnostic_flags": {
                "batch_confounded": (m.get("diagnostic_flags") or {}).get("batch_confounded", m.get("batch_confounded")),
                "missingness_biased": (m.get("diagnostic_flags") or {}).get("missingness_biased", m.get("missingness_biased")),
                "leakage_warning": (m.get("diagnostic_flags") or {}).get("leakage_warning", False),
            },
        }
        for m in util.get("modality_ledger", [])
    ]
    return {
        "task": util.get("task"),
        "primary_metric": util.get("primary_metric"),
        "chance_level": util.get("chance_level"),
        "n_samples": context.get("n_samples"),
        "best_model": {
            "name": (util.get("best_model") or {}).get("name"),
            "primary": (util.get("best_model") or {}).get("primary"),
        },
        "fusion_gain_over_best_single": util.get("fusion_gain_over_best_single"),
        "leakage_warning": bool(util.get("leakage_warning")),
        "controls": [{key: control.get(key) for key in ("name", "primary", "ci_low", "ci_high", "global_chance_evidence")}
                     for control in util.get("controls", [])],
        "modality_ledger": safe_ledger,
        "missingness_flag_count": len(context.get("missingness_flags", [])),
        "batch_flag_count": len(context.get("batch_flags", [])),
    }


def build_context(aligned, utility_ledger, missing_diag, batch_diag) -> dict[str, Any]:
    """Assemble the compact, PHI-free context object for :func:`summarize`."""
    return {
        "n_samples": int(aligned.n_samples),
        "utility": utility_ledger,
        "missingness_flags": (missing_diag or {}).get("flags", []),
        "batch_flags": (batch_diag or {}).get("flags", []),
    }
