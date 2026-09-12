"""
Profile Agent.

Responsibility (per the architecture slide): "Reads and writes the user's
audiogram, per-content-type target curves, and adjustment history in
MongoDB." No LLM call needed here -- it's a deterministic data-access step
that hands the rest of the graph a clean starting TargetCurve.
"""
from __future__ import annotations

from dsp.parametric_eq import TargetCurve
from data.db import ProfileStore


def run_profile_agent(state: dict, store: ProfileStore) -> dict:
    user_id = state["user_id"]
    content_type = state["context"].content_type
    profile = store.get_profile(user_id)

    stored = profile["target_curves"].get(content_type, {})
    baseline = TargetCurve(
        name=f"{content_type}_baseline",
        volume_db=stored.get("volume_db", 0.0),
        bass_gain_db=stored.get("bass_gain_db", 0.0),
        presence_gain_db=stored.get("presence_gain_db", 0.0),
        treble_gain_db=stored.get("treble_gain_db", 0.0),
    )

    state["profile"] = profile
    state["baseline_curve"] = baseline
    return state
