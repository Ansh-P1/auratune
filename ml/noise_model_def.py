"""
Shared model definitions for the ambient-noise classifier -- imported by
both ml/train_noise.py (training) and perception/noise_classifier.py
(inference).

Unlike the genre classifier (ml/model_def.py), there's no proxy-feature
problem here: every feature is computed directly from a raw audio buffer
via librosa, both at training time (on ESC-50 clips) and at inference
time (on a live/synth ambient buffer) -- training and inference always
see the exact same function, not an approximation of some other system's
private features.
"""
from __future__ import annotations

# 6 EQ-relevant ambient-noise buckets -- coarser than ESC-50's 50 raw
# sound classes on purpose (see ml/train_noise.py's ESC50_TO_BUCKET for
# the full mapping and rationale). Each has its own tuned EQ deltas in
# dsp/noise_curves.py.
NOISE_BUCKETS = [
    "calm_nature",
    "domestic_ambient",
    "human_activity",
    "impulsive_transient",
    "mechanical_drone",
    "traffic_urban",
]

# The 13 numeric features used as model input, in the exact order both
# training and inference must use. All directly computed from a raw
# audio buffer via librosa -- see perception/noise_classifier.py for the
# extraction code (shared, not duplicated, since there's no proxy-feature
# split to keep decoupled like the genre classifier has).
NOISE_FEATURE_COLUMNS = [
    "rms_db", "spectral_centroid", "spectral_bandwidth", "spectral_flatness",
    "spectral_rolloff", "zero_crossing_rate", "harmonic_ratio", "onset_rate",
    "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5",
]
