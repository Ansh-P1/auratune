"""
Plain-English editor for an EqualizerSpec's numeric limits (dB range, step).

Not a LangGraph node -- this runs synchronously from the "Your EQ app"
editor in app.py when someone types e.g. "my range is -76.5 to 7.5 dB" or
"step is 0.5" instead of using the Min dB / Max dB / Step number inputs
directly. Same fallback philosophy as the rest of agents/: Claude parses it
when ANTHROPIC_API_KEY is set, a handful of regexes cover the common
phrasings otherwise, and worst case nothing changes rather than crashing.
"""
from __future__ import annotations

import json
import re
from typing import Dict

from agents.llm_client import complete

_FIELDS = {"gain_min_db", "gain_max_db", "step_db"}


def _rule_based_parse(text: str) -> Dict[str, float]:
    """Covers the common phrasings without needing an API key:
    "-76.5 to 7.5", "range from -76.5 to +7.5 db", "between -76.5 and 7.5",
    "min -76.5", "max is 7.5 db", "step 0.5", "0.1 db steps"."""
    out: Dict[str, float] = {}
    t = text.lower().replace("–", "-").replace("—", "-")

    m = re.search(
        r"(-?\d+\.?\d*)\s*(?:db)?\s*(?:to|and|~|through)\s*\+?(-?\d+\.?\d*)\s*(?:db)?",
        t,
    )
    if m:
        try:
            a, b = float(m.group(1)), float(m.group(2))
            out["gain_min_db"], out["gain_max_db"] = min(a, b), max(a, b)
        except ValueError:
            pass

    m = re.search(r"min(?:imum)?[^0-9\-+]{0,12}(-?\d+\.?\d*)", t)
    if m:
        out["gain_min_db"] = float(m.group(1))

    m = re.search(r"max(?:imum)?[^0-9\-+]{0,12}(-?\d+\.?\d*)", t)
    if m:
        out["gain_max_db"] = float(m.group(1))

    m = re.search(r"step[^0-9]{0,12}(\d+\.?\d*)", t) or re.search(r"(\d+\.?\d*)\s*db\s*step", t)
    if m:
        out["step_db"] = float(m.group(1))

    return out


def parse_spec_edit(text: str) -> Dict[str, float]:
    """Return whichever of gain_min_db/gain_max_db/step_db the text
    mentions. Never raises; returns {} if nothing recognizable is in there.
    """
    text = (text or "").strip()
    if not text:
        return {}

    system = (
        "The user is editing a physical EQ app's numeric limits: its "
        "minimum gain in dB, maximum gain in dB, and its slider step size "
        "in dB. Read their plain-English description and return ONLY a "
        'JSON object with any of these keys they mentioned: "gain_min_db", '
        '"gain_max_db", "step_db" (all numbers, gain_min_db < gain_max_db). '
        "No markdown fences, no other keys, no explanation. If nothing "
        "numeric is mentioned, return {}."
    )
    raw = complete(system, text, max_tokens=80)
    if raw is not None:
        try:
            cleaned = raw.strip().strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned)
            result = {k: float(v) for k, v in parsed.items() if k in _FIELDS}
            if result:
                return result
        except Exception:
            pass  # Claude answered but not with usable JSON -- fall back to regex

    return _rule_based_parse(text)
