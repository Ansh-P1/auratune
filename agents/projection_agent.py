"""
Projection Agent.

Sits between the EQ Decision agent and the Explainer. Takes the decided
continuous target curve and the user's EqualizerSpec (their real EQ app's
band grid) and produces the exact per-slider values they should dial in --
snapped to that app's step size and gain range, with a preamp.

Deterministic, like the Decision agent -- no LLM. If no EqualizerSpec is
attached to the run (user hasn't told us what EQ they have), this is a
no-op and downstream code just shows the smooth curve.
"""
from __future__ import annotations

from typing import Optional

from dsp.parametric_eq import ParametricEQ
from dsp.equalizer_spec import EqualizerSpec
from dsp.eq_projection import project_curve


def run_projection_agent(state: dict, eq: ParametricEQ,
                         spec: Optional[EqualizerSpec] = None) -> dict:
    spec = spec or state.get("equalizer_spec")
    if spec is None:
        state["projected_eq"] = None
        return state
    state["projected_eq"] = project_curve(state["decided_curve"], eq, spec)
    return state
