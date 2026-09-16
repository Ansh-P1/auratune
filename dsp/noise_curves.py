"""
Per-ambient-noise-bucket EQ deltas.

The ML noise classifier (perception/noise_classifier.py) predicts one of
6 broad ambient-noise buckets (ml/noise_model_def.py's NOISE_BUCKETS)
from the room's mic/ambient audio. This module maps each bucket to a
small, hand-tuned set of EQ deltas -- applied the same way the existing
noise_level deltas already are (agents/eq_decision_agent.py's
_NOISE_ADJUSTMENTS), and blended in on top of them, not instead of them:
the RMS-based quiet/moderate/noisy read is a reliable, always-on floor;
the ML bucket is a refinement layer that reacts to *what kind* of noise
it is, not just how loud.

Deliberately modest, documented judgment calls (same magnitude range as
the existing noise-level deltas and the genre deltas in genre_curves.py),
not learned values -- the ML model's job is classification (what kind of
noise is this), not regression (what exact gain is "correct").
"""
from __future__ import annotations

from typing import Dict, Tuple

# (bass_delta_db, presence_delta_db, treble_delta_db)
NOISE_CURVES: Dict[str, Tuple[float, float, float]] = {
    # Birdsong, wind, rain, water, crickets -- unobtrusive background;
    # nothing competes with the content, so barely touch the curve.
    "calm_nature": (0.0, 0.0, 0.0),
    # Footsteps, typing, door creaks, running water -- everyday household
    # sound, low-level and intermittent. A light presence nudge is enough.
    "domestic_ambient": (0.0, 0.5, 0.0),
    # Talking, laughing, clapping, a crying baby -- other voices compete
    # directly with dialogue/vocals, so lean on presence to cut through.
    "human_activity": (-0.5, 2.0, 0.0),
    # Vacuum, washing machine, engine/rotor drone, chainsaw -- a steady
    # low-frequency hum that muddies the mix; cut bass hard, add clarity.
    "mechanical_drone": (-2.5, 2.0, 1.0),
    # Door knocks, glass breaking, alarms, fireworks, thunder -- sudden,
    # sharp, hard to ignore; boost presence so dialogue stays audible
    # through the interruptions, pull bass back a little.
    "impulsive_transient": (-1.5, 2.5, 0.5),
    # Car horns, sirens, traffic engines, trains -- the classic "noisy
    # street" case; the strongest adjustment of the six.
    "traffic_urban": (-3.0, 3.5, 1.0),
}

# How much of the noise-bucket delta to actually apply on top of the
# existing noise_level + genre deltas, scaled further by the model's own
# confidence at call time (see agents/eq_decision_agent.py) -- keeps a
# low-confidence or wrong prediction from dominating the curve.
NOISE_BLEND_WEIGHT = 0.7
