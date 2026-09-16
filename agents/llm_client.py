"""
Thin wrapper around the Anthropic API used by the EQ Decision agent (to
parse free-text commands like "make voices clearer") and the Explainer
agent (to turn numeric deltas into one plain-English sentence).

If ANTHROPIC_API_KEY isn't set (e.g. running the validation traces offline,
or in CI), calls fall back to a deterministic rule-based implementation so
the graph still runs end-to-end -- it just loses the free-text nuance an
LLM adds. Swap in a real key and nothing else about the pipeline changes.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    from config import LLM_MODEL as _MODEL
except Exception:  # config import shouldn't fail, but never break the fallback path
    _MODEL = os.environ.get("AURATUNE_LLM_MODEL", "claude-sonnet-5")

# Anthropic keys look like sk-ant-api03-<long base64ish blob>. Scrubbed from
# every prompt/response/error string before it can reach the dashboard's dev
# view -- a key pasted into a command box must never be rendered back.
_KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")


def redact(text: Optional[str]) -> Optional[str]:
    """Mask anything that looks like an API key. Used on everything the
    dev view in app.py displays."""
    if not text:
        return text
    return _KEY_PATTERN.sub("sk-ant-***REDACTED***", text)


@dataclass
class LLMCall:
    """One attempted Claude call, recorded so the UI can show whether the
    model actually ran or a deterministic fallback did.

    status is one of:
      "ok"          -- Claude was called and returned text
      "no_api_key"  -- ANTHROPIC_API_KEY unset (or the SDK isn't installed)
      "error"       -- the call was made but raised / returned nothing
    """
    purpose: str            # "command_parse" | "explanation"
    status: str
    model: str = _MODEL
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
            "model": self.model if self.used_llm else None,
            "system_prompt": redact(self.system_prompt),
            "user_prompt": redact(self.user_prompt),
            "response": redact(self.response),
            "latency_ms": round(self.latency_ms, 1) if self.latency_ms is not None else None,
            "error": redact(self.error),
            **self.extra,
        }


def _client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None


def complete_with_meta(system: str, user: str, purpose: str,
                       max_tokens: int = 300) -> tuple[Optional[str], LLMCall]:
    """Same call as complete(), but also returns an LLMCall describing what
    happened -- which is what makes the "Claude vs. template" badge and the
    prompt dev view in the dashboard possible.
    """
    call = LLMCall(purpose=purpose, status="no_api_key",
                   system_prompt=system, user_prompt=user)
    client = _client()
    if client is None:
        call.error = ("ANTHROPIC_API_KEY not set (or the anthropic package is "
                      "missing) -- using the deterministic fallback.")
        return None, call

    started = time.perf_counter()
    try:
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        text = "\n".join(parts).strip() or None
    except Exception as exc:
        call.status = "error"
        call.latency_ms = (time.perf_counter() - started) * 1000
        call.error = f"{type(exc).__name__}: {exc}"
        return None, call

    call.latency_ms = (time.perf_counter() - started) * 1000
    usage = getattr(resp, "usage", None)
    if usage is not None:
        call.extra["tokens"] = {
            "input": getattr(usage, "input_tokens", None),
            "output": getattr(usage, "output_tokens", None),
        }
    if text is None:
        call.status = "error"
        call.error = "Claude returned an empty response."
        return None, call

    call.status = "ok"
    call.response = text
    return text, call


def complete(system: str, user: str, max_tokens: int = 300) -> Optional[str]:
    """Return the model's text response, or None if no API key / call failed."""
    text, _ = complete_with_meta(system, user, purpose="unspecified",
                                 max_tokens=max_tokens)
    return text
