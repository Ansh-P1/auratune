"""
Per-genre-bucket EQ deltas.

The ML genre classifier (perception/genre_classifier.py) predicts one of
8 broad genre buckets (ml/model_def.py's GENRE_BUCKETS) from the currently
playing music. This module maps each bucket to a small, hand-tuned set of
EQ deltas -- applied on top of the existing profile/context/command curve
by the EQ Decision agent, the same way the noise-level deltas already are
(see agents/eq_decision_agent.py's _NOISE_ADJUSTMENTS).

These are deliberately modest, documented judgment calls (same magnitude
range as the existing noise-level deltas), not learned values -- the ML
model's job is classification (which genre is this), not regression (what
exact gain is "correct"). A future iteration could learn these deltas too,
but that needs a dataset with real EQ-preference labels, which doesn't
exist publicly (see ml/README.md).
"""
from __future__ import annotations

from typing import Dict, Tuple

# (bass_delta_db, presence_delta_db, treble_delta_db)
GENRE_CURVES: Dict[str, Tuple[float, float, float]] = {
    # Sub-bass drops + bright highs are the genre's whole character.
    "electronic_dance": (3.0, 0.5, 1.5),
    # Guitars/vocals live in the presence band; keep bass from swamping them.
    "rock_metal": (1.0, 2.5, 1.0),
    # Heavy low end is the point; vocals need a presence lift to cut through.
    "hiphop_rnb": (3.5, 1.5, 0.0),
    # Balanced, radio-style curve: a little vocal presence, a little sparkle.
    "pop": (1.0, 1.5, 1.5),
    # Warm and natural -- ease off bass/treble rather than add anything.
    "acoustic_folk": (-1.0, 1.0, 0.5),
    # Wide dynamic range recordings; least interference preserves the mix.
    "classical_jazz": (0.0, 0.0, 0.5),
    # Smooth/relaxing -- gentle bass, pull back harsh treble.
    "chill_ambient": (0.5, -0.5, -1.0),
    # Percussion-forward with bright brass/strings across most traditions.
    "world_latin": (2.0, 1.5, 2.0),
}

# How much of the genre delta to actually apply on top of the existing
# context/command deltas, so a wrong or low-confidence prediction can't
# dominate the curve. Scaled further by the model's own confidence at
# call time (see agents/eq_decision_agent.py).
GENRE_BLEND_WEIGHT = 0.7
