"""
Thin LLM wrapper used by the EQ Decision agent (to parse free-text commands
like "make voices clearer") and the Explainer agent (to turn numeric deltas
into one plain-English sentence).

Two backends: Anthropic (ANTHROPIC_API_KEY, via the anthropic SDK) and Groq
(GROQ_API_KEY, via its OpenAI-compatible endpoint over plain HTTPS -- no
extra dependency, same approach as the Gemini path in
perception/eq_app_reader.py). Anthropic wins when both are set;
AURATUNE_LLM_PROVIDER=anthropic|groq|none forces one. See active_provider().

If neither key is set (e.g. running the validation traces offline, or in
CI), calls fall back to a deterministic rule-based implementation so the
graph still runs end-to-end -- it just loses the free-text nuance an LLM
adds. Every attempt is recorded as an LLMCall so the dashboard can show
which path actually ran.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    from config import (LLM_MODEL as _MODEL, GROQ_MODEL as _GROQ_MODEL,
                        GROQ_BASE_URL as _GROQ_BASE_URL,
                        LLM_PROVIDER as _PROVIDER_SETTING)
except Exception:  # config import shouldn't fail, but never break the fallback path
    _MODEL = os.environ.get("AURATUNE_LLM_MODEL", "claude-sonnet-5")
    _GROQ_MODEL = os.environ.get("AURATUNE_GROQ_MODEL", "openai/gpt-oss-120b")
    _GROQ_BASE_URL = os.environ.get("AURATUNE_GROQ_BASE_URL",
                                    "https://api.groq.com/openai/v1")
    _PROVIDER_SETTING = os.environ.get("AURATUNE_LLM_PROVIDER", "auto").lower()

# Anthropic keys look like sk-ant-api03-<blob>, Groq keys gsk_<blob>.
# Scrubbed from every prompt/response/error string before it can reach the
# dashboard's dev view -- a key pasted into a command box must never be
# rendered back.
_KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_\-]+|gsk_[A-Za-z0-9]+")


# What the dashboard calls each backend, so the badge never says "Claude"
# when a different model actually wrote the sentence.
PROVIDER_LABELS = {
    "anthropic": "Claude",
    "groq": "Groq",
    "none": "no LLM",
}


def redact(text: Optional[str]) -> Optional[str]:
    """Mask anything that looks like an API key. Used on everything the
    dev view in app.py displays."""
    if not text:
        return text
    return _KEY_PATTERN.sub("***REDACTED-API-KEY***", text)


@dataclass
class LLMCall:
    """One attempted LLM call, recorded so the UI can show whether the
    model actually ran or a deterministic fallback did.

    status is one of:
      "ok"          -- the model was called and returned text
      "no_api_key"  -- no provider key set (or the SDK isn't installed)
      "error"       -- the call was made but raised / returned nothing
    """
    purpose: str            # "command_parse" | "explanation"
    status: str
    provider: str = "none"  # "anthropic" | "groq" | "none"
    model: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    response: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def used_llm(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict:
        """Redacted, JSON-safe view for the dashboard's dev panel."""
        return {
            "purpose": self.purpose,
            "status": self.status,
            "used_llm": self.used_llm,
            "provider": self.provider,
            "provider_label": PROVIDER_LABELS.get(self.provider, self.provider),
            "model": self.model or None,
            "system_prompt": redact(self.system_prompt),
            "user_prompt": redact(self.user_prompt),
            "response": redact(self.response),
            "latency_ms": round(self.latency_ms, 1) if self.latency_ms is not None else None,
            "error": redact(self.error),
            **self.extra,
        }


def active_provider() -> tuple[str, Optional[str], str]:
    """Which LLM backend this run will use: (provider, api_key, model).

    Anthropic wins when both keys are set -- it's what the project was built
    against; Groq is the no-Anthropic-key path for teammates. Provider
    "none" means every call falls back to the deterministic rules.
    AURATUNE_LLM_PROVIDER forces one either way.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    if _PROVIDER_SETTING == "none":
        return "none", None, ""
    if _PROVIDER_SETTING == "groq" or (_PROVIDER_SETTING == "auto"
                                       and not anthropic_key and groq_key):
        return ("groq", groq_key, _GROQ_MODEL) if groq_key else ("none", None, "")
    if _PROVIDER_SETTING in ("auto", "anthropic") and anthropic_key:
        return "anthropic", anthropic_key, _MODEL
    return "none", None, ""


def _call_anthropic(api_key: str, model: str, system: str, user: str,
                    max_tokens: int, call: LLMCall) -> Optional[str]:
    import anthropic
    resp = anthropic.Anthropic(api_key=api_key).messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    usage = getattr(resp, "usage", None)
    if usage is not None:
        call.extra["tokens"] = {"input": getattr(usage, "input_tokens", None),
                                "output": getattr(usage, "output_tokens", None)}
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "\n".join(parts).strip() or None


def _call_groq(api_key: str, model: str, system: str, user: str,
               max_tokens: int, call: LLMCall) -> Optional[str]:
    """Groq's OpenAI-compatible chat endpoint over plain HTTPS.

    No SDK on purpose -- same approach as the Gemini path in
    perception/eq_app_reader.py, so nothing new lands in requirements.txt.
    """
    import json as _json
    import urllib.error
    import urllib.request

    # Reasoning models (gpt-oss et al.) spend part of max_completion_tokens on
    # hidden reasoning before any visible content, so a tight budget comes
    # back with an empty message. Keep reasoning short and leave headroom.
    payload = _json.dumps({
        "model": model,
        "max_completion_tokens": max_tokens + 256,
        "reasoning_effort": "low",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(
        f"{_GROQ_BASE_URL}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json",
                 # Without an explicit UA, urllib sends "Python-urllib/3.x",
                 # which Groq's edge blocks outright (Cloudflare error 1010).
                 "User-Agent": "auratune/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = _json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        # urllib's HTTPError str() is just "HTTP Error 403: Forbidden" -- the
        # useful part (bad key? unknown model? rate limited?) is in the body,
        # so surface it or the dev view can't diagnose anything.
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Groq API {exc.code}: {detail}") from None

    usage = body.get("usage") or {}
    call.extra["tokens"] = {"input": usage.get("prompt_tokens"),
                            "output": usage.get("completion_tokens")}
    choices = body.get("choices") or [{}]
    return (choices[0].get("message", {}).get("content") or "").strip() or None


def complete_with_meta(system: str, user: str, purpose: str,
                       max_tokens: int = 300) -> tuple[Optional[str], LLMCall]:
    """Same call as complete(), but also returns an LLMCall describing what
    happened -- which is what makes the "LLM vs. template" badge and the
    prompt dev view in the dashboard possible.
    """
    provider, api_key, model = active_provider()
    call = LLMCall(purpose=purpose, status="no_api_key", provider=provider,
                   model=model, system_prompt=system, user_prompt=user)
    if provider == "none" or not api_key:
        call.error = ("No ANTHROPIC_API_KEY or GROQ_API_KEY set -- using the "
                      "deterministic fallback.")
        return None, call

    started = time.perf_counter()
    try:
        if provider == "anthropic":
            text = _call_anthropic(api_key, model, system, user, max_tokens, call)
        else:
            text = _call_groq(api_key, model, system, user, max_tokens, call)
    except Exception as exc:
        call.status = "error"
        call.latency_ms = (time.perf_counter() - started) * 1000
        call.error = f"{type(exc).__name__}: {exc}"
        return None, call

    call.latency_ms = (time.perf_counter() - started) * 1000
    if text is None:
        call.status = "error"
        call.error = f"{PROVIDER_LABELS.get(provider, provider)} returned an empty response."
        return None, call

    call.status = "ok"
    call.response = text
    return text, call


def complete(system: str, user: str, max_tokens: int = 300) -> Optional[str]:
    """Return the model's text response, or None if no API key / call failed."""
    text, _ = complete_with_meta(system, user, purpose="unspecified",
                                 max_tokens=max_tokens)
    return text
