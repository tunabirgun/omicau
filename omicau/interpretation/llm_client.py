"""Provider-agnostic LLM client for the optional plain-language verdict tier.

First-class providers, all reachable from the UI and the CLI config:
  anthropic  -> Claude          (anthropic SDK)
  openai     -> ChatGPT         (openai SDK, api.openai.com)
  gemini     -> Google Gemini   (openai SDK against Google's OpenAI-compatible endpoint)
  local      -> Ollama / LM Studio / vLLM and any OpenAI-compatible server (openai SDK + base_url)
  openai_compatible -> alias of ``local`` for hosted compatible gateways (Groq/Together/OpenRouter)

Key-privacy contract: the API key arrives as a function argument, is used for one
call, and is never assigned to an object attribute, returned, logged, written to
disk, or included in the provenance hash. A missing provider SDK degrades that
provider to the deterministic rule-based verdict (the caller catches ImportError).
"""

from __future__ import annotations

# Default OpenAI-compatible base URLs. The model id is always user-supplied.
GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_DEFAULT_BASE = "http://localhost:11434/v1"

# Providers that speak the OpenAI chat.completions wire format.
_OPENAI_FAMILY = {"openai", "gemini", "local", "openai_compatible"}


def call_llm(*, provider: str, model: str, api_key: str, base_url: str | None,
             system: str, user: str, max_tokens: int, timeout: float,
             openai_api: str = "chat") -> str:
    """Return the model's raw text. Raises on any failure so the caller can fall
    back to the rule-based verdict. `api_key` is a local only -- never stored."""
    provider = (provider or "anthropic").lower()
    if provider == "anthropic":
        return _call_anthropic(model, api_key, base_url, system, user, max_tokens, timeout)
    if provider in _OPENAI_FAMILY:
        url = base_url or _default_base_url(provider)
        # Gemini/local advertise chat.completions; only genuine OpenAI honours "responses".
        api = openai_api if provider == "openai" else "chat"
        # local servers (Ollama) ignore the key but the SDK still requires a non-empty string.
        key = api_key or ("ollama" if provider in ("local", "openai_compatible") else api_key)
        return _call_openai(model, key, url, system, user, max_tokens, timeout, api=api)
    raise ValueError(f"unknown LLM provider '{provider}'")


def _default_base_url(provider: str) -> str | None:
    if provider == "gemini":
        return GEMINI_OPENAI_BASE
    if provider in ("local", "openai_compatible"):
        return OLLAMA_DEFAULT_BASE     # sensible default; UI/CLI can override
    return None                        # openai -> SDK default (api.openai.com)


def _call_anthropic(model, api_key, base_url, system, user, max_tokens, timeout) -> str:
    import anthropic  # ImportError -> caller degrades to rule_based
    kw = {"api_key": api_key, "timeout": timeout}
    if base_url:                          # rare: a self-hosted Anthropic-compatible proxy
        kw["base_url"] = base_url
    client = anthropic.Anthropic(**kw)
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    if getattr(resp, "stop_reason", None) == "refusal":
        raise RuntimeError("model refused")
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def _call_openai(model, api_key, base_url, system, user, max_tokens, timeout, *, api) -> str:
    from openai import OpenAI      # ImportError -> caller degrades to rule_based
    kw = {"api_key": api_key, "timeout": timeout}
    if base_url:
        kw["base_url"] = base_url         # e.g. Gemini endpoint or http://localhost:11434/v1
    client = OpenAI(**kw)
    if api == "responses":                # opt-in only; requires openai>=1.66
        r = client.responses.create(model=model, instructions=system, input=user,
                                     max_output_tokens=max_tokens)
        return r.output_text
    r = client.chat.completions.create(    # universal path (openai>=1.0)
        model=model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return r.choices[0].message.content or ""
