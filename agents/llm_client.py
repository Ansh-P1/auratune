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
from typing import Optional

try:
    from config import LLM_MODEL as _MODEL
except Exception:  # config import shouldn't fail, but never break the fallback path
    _MODEL = os.environ.get("AURATUNE_LLM_MODEL", "claude-sonnet-5")


def _client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None


def complete(system: str, user: str, max_tokens: int = 300) -> Optional[str]:
    """Return the model's text response, or None if no API key / call failed."""
    client = _client()
    if client is None:
        return None
    try:
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "\n".join(parts).strip() or None
    except Exception:
        return None
