"""
Noise Agent.

New pipeline node between the Profile and Genre agents. Runs on the
ambient/mic audio buffer (not the content buffer -- that's the Genre
agent's job) to classify *what kind* of ambient noise is present, using
the locally-trained model (perception/noise_classifier.py, trained by
ml/train_noise.py on the real 2,000-clip ESC-50 dataset). The EQ Decision
agent then blends that bucket's tuned deltas (dsp/noise_curves.py) in on
top of the existing RMS-based noise_level deltas -- refining *how* to
compensate, not replacing the reliable always-on loudness read.

Deterministic given a trained model -- no LLM call, like the Profile,
Genre, and Projection agents. If no ambient audio was passed in, or no
trained model exists yet (ml/models/noise_*.joblib is empty until
someone runs `python ml/train_noise.py`), this is a clean no-op:
state["noise_bucket"] stays unset and the EQ Decision agent's existing
behavior is completely unaffected -- same graceful-fallback pattern as
the rest of the pipeline.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from perception.noise_classifier import classify_noise


def run_noise_agent(state: dict, ambient_audio: Optional[np.ndarray],
                    sample_rate: int, model_name: str = "auto") -> dict:
    if ambient_audio is None:
        return state

    result = classify_noise(ambient_audio, sample_rate, model_name=model_name)
    if result.ok:
        state["noise_bucket"] = result.bucket
        state["noise_confidence"] = result.confidence
        state["noise_model_used"] = result.model_used
        state["noise_probabilities"] = result.all_probabilities
        state["noise_features"] = result.features
    else:
        state["noise_unavailable_reason"] = result.reason
    return state
