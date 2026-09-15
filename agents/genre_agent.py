"""
Genre Agent.

New pipeline node between the Profile and EQ Decision agents. Only runs
when context.content_type == "music" -- genre-based EQ tuning doesn't
make sense for a podcast or a movie's mixed dialogue+FX track. Classifies
the content audio into one of 8 EQ-relevant genre buckets using the
locally-trained model (perception/genre_classifier.py, trained by
ml/train.py on the real 114k-track Spotify Tracks Dataset). The EQ
Decision agent then blends that bucket's tuned deltas
(dsp/genre_curves.py) on top of the existing profile/context/command
deltas.

Deterministic given a trained model -- no LLM call, like the Profile and
Projection agents. If no content audio was passed in, the content isn't
music, or no trained model exists yet (ml/models/ is empty until someone
runs `python ml/train.py`), this is a clean no-op: state["genre_bucket"]
stays unset and the EQ Decision agent's existing behavior is completely
unaffected -- same graceful-fallback pattern as the rest of the pipeline.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from perception.genre_classifier import classify_genre


def run_genre_agent(state: dict, content_audio: Optional[np.ndarray],
                    sample_rate: int, model_name: str = "auto") -> dict:
    context = state["context"]
    if content_audio is None or context.content_type != "music":
        return state

    result = classify_genre(content_audio, sample_rate, model_name=model_name)
    if result.ok:
        state["genre_bucket"] = result.bucket
        state["genre_confidence"] = result.confidence
        state["genre_model_used"] = result.model_used
        state["genre_probabilities"] = result.all_probabilities
        state["genre_proxy_features"] = result.proxy_features
    else:
        state["genre_unavailable_reason"] = result.reason
    return state
