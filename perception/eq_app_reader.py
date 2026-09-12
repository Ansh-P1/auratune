"""
Read an EqualizerSpec straight out of a screenshot of the user's EQ app.

The user uploads a photo/screenshot of their equalizer (Wavelet, Poweramp,
Spotify, a car head unit...). A vision model reads the band frequencies,
gain range, and slider step, which we validate into an `EqualizerSpec` the
rest of AuraTune already understands.

Backends, tried in this order:
  1. Google Gemini   -- if GEMINI_API_KEY / GOOGLE_API_KEY is set (free tier,
     get one in ~30s at https://aistudio.google.com/apikey). REST call, no
     extra pip dependency.
  2. Anthropic Claude -- if ANTHROPIC_API_KEY is set.
With neither, this returns a result carrying a friendly `error` string and
the Streamlit app falls back to the manual editor -- nothing crashes.
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from dsp.equalizer_spec import EqualizerSpec

try:
    from config import VISION_MODEL as _CLAUDE_MODEL
except Exception:
    _CLAUDE_MODEL = os.environ.get("AURATUNE_VISION_MODEL", "claude-sonnet-5")

try:
    from config import GEMINI_MODEL as _GEMINI_MODEL
except Exception:
    _GEMINI_MODEL = os.environ.get("AURATUNE_GEMINI_MODEL", "gemini-2.0-flash")

_SYSTEM = (
    "You read a screenshot of an audio equalizer UI and report its layout as "
    "strict JSON. You never invent bands that aren't visible. If part of the "
    "scale is covered or ambiguous, you make a sensible assumption and say so "
    "in the notes field."
)

_PROMPT = """\
This is a screenshot of an equalizer (an EQ app, media-player EQ, or car stereo).
Report its layout as a single JSON object, no prose, no markdown fences:

{
  "name": "<app / device name if visible, else 'Uploaded EQ'>",
  "band_freqs_hz": [<each slider's centre frequency in Hz, low to high, numbers only>],
  "gain_min_db": <most negative gain a single slider can reach>,
  "gain_max_db": <most positive gain a single slider can reach>,
  "step_db": <slider granularity: 1.0, 0.5, or 0.1; use 0 if it looks continuous>,
  "has_preamp": <true only if there is a SEPARATE preamp / gain control>,
  "notes": "<anything uncertain: covered scale labels, guessed range, etc.>"
}

Rules:
- "60" / "1k" / "16k" style labels -> 60, 1000, 16000.
- If the current slider VALUES are shown to one decimal (e.g. -0.1), step_db is 0.1.
- If you truly can't tell the range, use -12 and 12 and note the assumption.
- band_freqs_hz must have between 1 and 32 entries.
"""


@dataclass
class EqSpecReadResult:
    spec: Optional[EqualizerSpec] = None
    error: Optional[str] = None
    model_notes: str = ""
    backend: str = ""          # "gemini" | "claude" | ""

    @property
    def ok(self) -> bool:
        return self.spec is not None


# ---------------------------------------------------------------------------
# backends -- each returns the model's raw text (expected to contain JSON)
# ---------------------------------------------------------------------------
def _gemini_key(explicit: Optional[str]) -> Optional[str]:
    return (explicit or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY") or None)


def _read_with_gemini(image_bytes: bytes, media_type: str, api_key: str) -> str:
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{_GEMINI_MODEL}:generateContent?key={api_key}")
    body = {
        "systemInstruction": {"parts": [{"text": _SYSTEM}]},
        "contents": [{
            "parts": [
                {"text": _PROMPT},
                {"inlineData": {"mimeType": media_type,
                                "data": base64.standard_b64encode(image_bytes).decode("ascii")}},
            ],
        }],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 700},
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:200]
        if e.code in (400, 401, 403):
            raise RuntimeError("Gemini rejected the request - is the API key valid?") from e
        raise RuntimeError(f"Gemini HTTP {e.code}: {detail}") from e
    return "".join(p.get("text", "")
                   for p in data["candidates"][0]["content"]["parts"])


def _anthropic_key(explicit: Optional[str]) -> Optional[str]:
    return explicit or os.environ.get("ANTHROPIC_API_KEY") or None


def _read_with_claude(image_bytes: bytes, media_type: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=_CLAUDE_MODEL,
        max_tokens=700,
        system=_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media_type,
                    "data": base64.standard_b64encode(image_bytes).decode("ascii")}},
                {"type": "text", "text": _PROMPT},
            ],
        }],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


# ---------------------------------------------------------------------------
# parsing / validation
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model response")
    return json.loads(text[start:end + 1])


def _validate(d: dict) -> EqualizerSpec:
    freqs = d.get("band_freqs_hz") or []
    freqs = sorted(float(f) for f in freqs if float(f) > 0)
    if not (1 <= len(freqs) <= 32):
        raise ValueError(f"expected 1-32 bands, got {len(freqs)}")

    gmin = float(d.get("gain_min_db", -12.0))
    gmax = float(d.get("gain_max_db", 12.0))
    if gmin >= gmax:
        gmin, gmax = -12.0, 12.0
    gmin = max(-40.0, gmin)
    gmax = min(40.0, gmax)

    step = float(d.get("step_db", 1.0))
    if step < 0 or step > 6:
        step = 1.0

    return EqualizerSpec(
        name=str(d.get("name") or "Uploaded EQ").strip()[:60],
        band_freqs_hz=freqs,
        gain_min_db=gmin,
        gain_max_db=gmax,
        step_db=step,
        has_preamp=bool(d.get("has_preamp", False)),
        notes="Read from an uploaded screenshot. "
              + str(d.get("notes") or "").strip(),
    )


_NO_BACKEND = (
    "Screenshot reading needs a vision model. Easiest free option: get a "
    "Google Gemini key at aistudio.google.com/apikey and paste it below "
    "(or set GEMINI_API_KEY). Otherwise use the manual editor."
)


def read_equalizer_screenshot(image_bytes: bytes,
                              media_type: str = "image/png",
                              gemini_key: Optional[str] = None,
                              anthropic_key: Optional[str] = None) -> EqSpecReadResult:
    """Vision -> EqualizerSpec. Never raises; returns a result object."""
    if media_type not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
        media_type = "image/png"

    gkey = _gemini_key(gemini_key)
    akey = _anthropic_key(anthropic_key)
    if not gkey and not akey:
        return EqSpecReadResult(error=_NO_BACKEND)

    backend = "gemini" if gkey else "claude"
    try:
        if gkey:
            raw = _read_with_gemini(image_bytes, media_type, gkey)
        else:
            raw = _read_with_claude(image_bytes, media_type, akey)
        data = _extract_json(raw)
        spec = _validate(data)
        return EqSpecReadResult(spec=spec, backend=backend,
                                model_notes=str(data.get("notes") or ""))
    except Exception as exc:
        return EqSpecReadResult(
            backend=backend,
            error=f"The {backend} reader couldn't parse that screenshot ({exc}). "
                  "Try a tighter crop, or use the manual editor.")
