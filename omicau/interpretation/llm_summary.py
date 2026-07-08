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
            # Do not retry auth / bad-request style errors.
            if any(k in name for k in ("Authentication", "PermissionDenied", "BadRequest", "NotFound")):
                break
            sleep = min(cap, base * (2 ** attempt)) + random.uniform(0, base)
            time.sleep(sleep)
    return None


def _try_llm(context: dict[str, Any], llm, api_key: str) -> dict[str, Any] | None:
    from omicau.interpretation import llm_client
    payload = json.dumps(_sanitize_context(context), sort_keys=True)

    def _call():
        return llm_client.call_llm(
            provider=llm.provider, model=llm.model, api_key=api_key,
            base_url=getattr(llm, "base_url", None),
            system=_SYSTEM_PROMPT, user=f"Audit diagnostics (JSON):\n{payload}",
            max_tokens=llm.max_tokens, timeout=getattr(llm, "timeout", 60.0),
            openai_api=getattr(llm, "openai_api", "chat"))

    try:
        text = _retry_backoff(_call)
    except ImportError:            # provider SDK absent -> degrade to rule_based
        return None
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

    useful = [m["modality"] for m in ledger_in if str(m.get("verdict", "")).startswith("predictive")]
    confounded = [m["modality"] for m in ledger_in if m.get("batch_confounded")]
    dead = [m["modality"] for m in ledger_in if "no detectable" in str(m.get("verdict", ""))]

    # data hygiene rating from the count of active flags.
    n_flags = len(context.get("missingness_flags", [])) + len(context.get("batch_flags", []))
    if leakage or n_flags >= 3:
        rating = "high concern"
    elif n_flags >= 1:
        rating = "moderate concerns"
    else:
        rating = "clean"
    rating_text = (
        f"{rating}: {n_flags} diagnostic flag(s)"
        + (", control-baseline leakage warning active" if leakage else "")
        + "."
    )

    single_modality = util.get("single_modality", False)
    perf = f"{primary:.3f}" if isinstance(primary, (int, float)) else "n/a"
    if single_modality:
        verdict_bits = [
            "With a single modality, omicau runs its leakage-safe honesty check rather than a fusion "
            f"benchmark. The {best.get('name', 'n/a')} model reaches {metric.upper()}={perf} "
            f"(chance ~ {chance}) under group-aware cross-validation."
        ]
    else:
        verdict_bits = [
            f"The best fusion model ({best.get('name', 'n/a')}) reaches {metric.upper()}={perf} "
            f"(chance ~ {chance})."
        ]
    if isinstance(gain, (int, float)):
        if gain > 0.01:
            verdict_bits.append(
                f"Fusion adds {gain:+.3f} over the best single modality, so combining layers is justified."
            )
        else:
            verdict_bits.append(
                f"Fusion does not beat the best single modality ({gain:+.3f}); a single layer may suffice."
            )
    if useful:
        verdict_bits.append(f"Modalities carrying independent signal: {', '.join(useful)}.")
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
            "verdict": m.get("verdict", "n/a"),
            "recommendation": m.get("recommendation", ""),
        }
        for m in ledger_in
    ]

    recs: list[str] = []
    if confounded:
        recs.append(f"Correct or regress out batch effects in: {', '.join(confounded)}.")
    if dead:
        recs.append(f"Consider dropping control-like layers with no signal: {', '.join(dead)}.")
    for f in context.get("missingness_flags", [])[:3]:
        recs.append(f"Investigate missingness bias: {f}")
    if leakage:
        recs.append("Re-audit cross-validation splits for group leakage (control baseline elevated).")
    if not recs:
        recs.append(
            "No critical data-hygiene issues detected; the single-layer signal is the honest ceiling "
            "here — validate externally." if single_modality
            else "No critical data-hygiene issues detected; proceed with the fusion model.")
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
    """Strip anything sample-identifying; keep only aggregate statistics."""
    util = context.get("utility", {})
    safe_ledger = [
        {
            "modality": m.get("modality"),
            "n_features": m.get("n_features"),
            "standalone_primary": m.get("standalone_primary"),
            "marginal_gain_classical": m.get("marginal_gain_classical"),
            "batch_confounded": m.get("batch_confounded"),
            "missingness_biased": m.get("missingness_biased"),
            "verdict": m.get("verdict"),
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
        "leakage_warning": util.get("leakage_warning"),
        "controls": util.get("controls"),
        "modality_ledger": safe_ledger,
        "missingness_flags": context.get("missingness_flags", []),
        "batch_flags": context.get("batch_flags", []),
    }


def build_context(aligned, utility_ledger, missing_diag, batch_diag) -> dict[str, Any]:
    """Assemble the compact, PHI-free context object for :func:`summarize`."""
    return {
        "n_samples": int(aligned.n_samples),
        "utility": utility_ledger,
        "missingness_flags": (missing_diag or {}).get("flags", []),
        "batch_flags": (batch_diag or {}).get("flags", []),
    }
