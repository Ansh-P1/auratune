"""
EQ Decision Agent.

Responsibility (per the architecture slide): "Blends profile + live context
+ any typed command into a target curve, volume, and bass-shelf gain."

Rule-based context blending is deterministic (auditable, no latency/cost),
matching how a real-time DSP loop should behave. A typed free-text command
("make voices clearer", "less bass, this room is boomy") is the one place
an LLM adds real value -- parsing intent into structured gain deltas -- so
that step calls Claude and falls back to keyword rules if no API key is set.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace

from dsp.parametric_eq import TargetCurve
from dsp.genre_curves import GENRE_CURVES, GENRE_BLEND_WEIGHT
from agents.llm_client import complete
from config import MAX_GAIN_DB

_NOISE_ADJUSTMENTS = {
    # noise_level -> (presence_delta, bass_delta) applied on top of baseline
    "quiet": (0.0, 0.0),
    "moderate": (1.5, -1.0),
    "noisy": (3.5, -2.5),   # boost vocal/presence band, pull bass back to avoid mud/stacking
}

_COMMAND_KEYWORDS = [
    (re.compile(r"clear(er)? voice|dialogue|speech", re.I), {"presence_gain_db": 3.0}),
    (re.compile(r"less bass|too boomy|reduce bass", re.I), {"bass_gain_db": -3.0}),
    (re.compile(r"more bass|bassier|boomier", re.I), {"bass_gain_db": 3.0}),
    (re.compile(r"brighter|more treble|crisper", re.I), {"treble_gain_db": 2.0}),
    (re.compile(r"warmer|less treble|too bright", re.I), {"treble_gain_db": -2.0}),
    (re.compile(r"louder|turn.*up", re.I), {"volume_db": 3.0}),
    (re.compile(r"quieter|turn.*down", re.I), {"volume_db": -3.0}),
]


def _rule_based_command_parse(command: str) -> dict:
    deltas: dict = {}
    for pattern, delta in _COMMAND_KEYWORDS:
        if pattern.search(command):
            deltas.update(delta)
    return deltas


def _llm_command_parse(command: str) -> dict:
    system = (
        "You convert a user's spoken/typed audio-EQ request into a JSON object "
        "with any of these optional numeric keys (dB deltas to apply on top of "
        "the current curve): volume_db, bass_gain_db, presence_gain_db, "
        "treble_gain_db. Respond with ONLY the JSON object, no prose, no "
        "markdown fences. Use modest values (typically -4 to +4 dB). If the "
        "request doesn't map to an audio adjustment, return {}."
    )
    raw = complete(system, command, max_tokens=120)
    if raw is None:
        return _rule_based_command_parse(command)
    try:
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        deltas = json.loads(cleaned)
        return {k: float(v) for k, v in deltas.items() if k in
                 {"volume_db", "bass_gain_db", "presence_gain_db", "treble_gain_db"}}
    except Exception:
        return _rule_based_command_parse(command)


def run_eq_decision_agent(state: dict) -> dict:
    baseline: TargetCurve = state["baseline_curve"]
    context = state["context"]
    command = state.get("user_command", "")

    presence_delta, bass_delta = _NOISE_ADJUSTMENTS.get(context.noise_level, (0.0, 0.0))

    decided = replace(
        baseline,
        name=f"{context.content_type}_{context.noise_level}_live",
        presence_gain_db=baseline.presence_gain_db + presence_delta,
        bass_gain_db=baseline.bass_gain_db + bass_delta,
    )

    # Genre agent runs upstream (agents/genre_agent.py) only for music
    # content; blend its bucket's tuned deltas in, scaled by both the
    # fixed GENRE_BLEND_WEIGHT and the classifier's own confidence, so a
    # shaky prediction can only nudge the curve, not dominate it.
    genre_bucket = state.get("genre_bucket")
    genre_deltas = {}
    if genre_bucket in GENRE_CURVES:
        weight = GENRE_BLEND_WEIGHT * state.get("genre_confidence", 0.0)
        bass_d, presence_d, treble_d = GENRE_CURVES[genre_bucket]
        genre_deltas = {
            "bass_gain_db": round(bass_d * weight, 2),
            "presence_gain_db": round(presence_d * weight, 2),
            "treble_gain_db": round(treble_d * weight, 2),
        }
        for key, delta in genre_deltas.items():
            setattr(decided, key, getattr(decided, key) + delta)

    command_deltas = {}
    if command:
        command_deltas = _llm_command_parse(command)
        for key, delta in command_deltas.items():
            setattr(decided, key, getattr(decided, key) + delta)

    # keep total gains within a sane, ear-safe range
    for field_name in ("volume_db", "bass_gain_db", "presence_gain_db", "treble_gain_db"):
        val = getattr(decided, field_name)
        setattr(decided, field_name, float(max(-MAX_GAIN_DB, min(MAX_GAIN_DB, val))))

    state["decided_curve"] = decided
    state["command_deltas"] = command_deltas
    state["context_deltas"] = {"presence_gain_db": presence_delta, "bass_gain_db": bass_delta}
    state["genre_deltas"] = genre_deltas
    return state
