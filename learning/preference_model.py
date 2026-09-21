"""
Preference Model for AuraTune Personalization.

Learns per-user, per-context-bucket EQ preferences from explicit/implicit user
correction feedback. Computes a lightweight, explainable bias vector (gain deltas)
applied on top of the EQ decision agent's baseline output.

Key design features:
1. Exponential recency weighting: recent corrections matter more than old ones.
2. Cold-start confidence gating: smoothly scales influence as samples accumulate.
3. Ear-safe clamping: bounds learned bias to prevent excessive boosts.
4. Transparent explainability: generates human-readable reasoning strings for traces.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

from data.db import ProfileStore
from dsp.parametric_eq import TargetCurve

# Controllable EQ band fields
EQ_FIELDS = ("volume_db", "bass_gain_db", "presence_gain_db", "treble_gain_db")

# Default hyperparameters
DEFAULT_HALF_LIFE_SEC = 7 * 86400.0  # 7 days recency half-life
DEFAULT_CONFIDENCE_SCALE = 3         # 3 feedback events reach 1.0 confidence
DEFAULT_MAX_BIAS_DB = 4.0            # Max +/- 4.0 dB learned bias clamp


class PreferenceModel:
    def __init__(
        self,
        store: Optional[ProfileStore] = None,
        half_life_sec: float = DEFAULT_HALF_LIFE_SEC,
        confidence_scale: int = DEFAULT_CONFIDENCE_SCALE,
        max_bias_db: float = DEFAULT_MAX_BIAS_DB,
    ):
        self.store = store or ProfileStore()
        self.half_life_sec = max(1.0, float(half_life_sec))
        self.confidence_scale = max(1, int(confidence_scale))
        self.max_bias_db = float(max_bias_db)

    def record_feedback(
        self,
        user_id: str,
        context_bucket: str,
        agent_curve: TargetCurve | Dict[str, Any],
        user_correction_delta: Dict[str, Any],
    ) -> None:
        """Record a user adjustment event into ProfileStore."""
        curve_dict = (
            {
                "volume_db": agent_curve.volume_db,
                "bass_gain_db": agent_curve.bass_gain_db,
                "presence_gain_db": agent_curve.presence_gain_db,
                "treble_gain_db": agent_curve.treble_gain_db,
            }
            if isinstance(agent_curve, TargetCurve)
            else agent_curve
        )
        self.store.log_feedback(
            user_id=user_id,
            context_bucket=context_bucket,
            agent_curve=curve_dict,
            user_correction_delta=user_correction_delta,
        )

    def compute_bias_from_feedback(
        self,
        feedback_entries: List[Dict[str, Any]],
        current_time: Optional[float] = None,
    ) -> Tuple[Dict[str, float], float]:
        """Given a list of feedback records, compute the weighted preference bias and confidence.

        Returns:
            (bias_deltas, confidence)
            where bias_deltas is a dict {field: delta_db} and confidence is in [0.0, 1.0].
        """
        if not feedback_entries:
            return ({f: 0.0 for f in EQ_FIELDS}, 0.0)

        now = current_time if current_time is not None else time.time()
        decay_constant = math.log(2.0) / self.half_life_sec

        weighted_sums: Dict[str, float] = {f: 0.0 for f in EQ_FIELDS}
        weight_totals: Dict[str, float] = {f: 0.0 for f in EQ_FIELDS}

        for entry in feedback_entries:
            ts = float(entry.get("ts", now))
            dt = max(0.0, now - ts)
            recency_weight = math.exp(-decay_constant * dt)

            deltas = entry.get("user_correction_delta", {})
            for field in EQ_FIELDS:
                if field in deltas:
                    delta_val = float(deltas[field])
                    weighted_sums[field] += delta_val * recency_weight
                    weight_totals[field] += recency_weight

        # Cold-start confidence scaling: ramp up with number of feedback samples
        count = len(feedback_entries)
        confidence = round(min(1.0, count / self.confidence_scale), 2)

        raw_bias: Dict[str, float] = {}
        for field in EQ_FIELDS:
            if weight_totals[field] > 0:
                avg = weighted_sums[field] / weight_totals[field]
                # Scale by confidence & clamp to max_bias_db
                gated = avg * confidence
                clamped = max(-self.max_bias_db, min(self.max_bias_db, gated))
                raw_bias[field] = round(clamped, 2)
            else:
                raw_bias[field] = 0.0

        return (raw_bias, confidence)

    def get_bias(
        self,
        user_id: str,
        context_bucket: str,
        current_time: Optional[float] = None,
    ) -> Tuple[Dict[str, float], float]:
        """Fetch past feedback for user + context_bucket and return (bias_deltas, confidence)."""
        entries = self.store.get_feedback(user_id=user_id, context_bucket=context_bucket)
        return self.compute_bias_from_feedback(entries, current_time=current_time)

    def get_explanation_clause(
        self,
        bias_deltas: Dict[str, float],
        confidence: float,
        context_bucket: Optional[str] = None,
    ) -> str:
        """Format an explainable plain-English clause if preference bias was applied."""
        if confidence <= 0.0 or not bias_deltas:
            return ""

        parts = []
        eps = 0.1
        if bias_deltas.get("bass_gain_db", 0.0) > eps:
            parts.append(f"+{bias_deltas['bass_gain_db']:.1f} dB bass")
        elif bias_deltas.get("bass_gain_db", 0.0) < -eps:
            parts.append(f"{bias_deltas['bass_gain_db']:.1f} dB bass")

        if bias_deltas.get("presence_gain_db", 0.0) > eps:
            parts.append(f"+{bias_deltas['presence_gain_db']:.1f} dB presence")
        elif bias_deltas.get("presence_gain_db", 0.0) < -eps:
            parts.append(f"{bias_deltas['presence_gain_db']:.1f} dB presence")

        if bias_deltas.get("treble_gain_db", 0.0) > eps:
            parts.append(f"+{bias_deltas['treble_gain_db']:.1f} dB treble")
        elif bias_deltas.get("treble_gain_db", 0.0) < -eps:
            parts.append(f"{bias_deltas['treble_gain_db']:.1f} dB treble")

        if bias_deltas.get("volume_db", 0.0) > eps:
            parts.append(f"+{bias_deltas['volume_db']:.1f} dB volume")
        elif bias_deltas.get("volume_db", 0.0) < -eps:
            parts.append(f"{bias_deltas['volume_db']:.1f} dB volume")

        if not parts:
            return ""

        adj_str = ", ".join(parts)
        ctx_clause = f" for {context_bucket}" if context_bucket else ""
        return f"nudged {adj_str} based on your past preferences{ctx_clause}"
