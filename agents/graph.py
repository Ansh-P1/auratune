"""
LangGraph pipeline: Profile Agent -> EQ Decision Agent -> Explainer Agent.

State flows through as a plain dict (langgraph's TypedDict pattern), which
keeps this file readable and makes it trivial to add a 4th node later
(e.g. a Safety agent that hard-caps gain deltas) without touching the
agents themselves.
"""
from __future__ import annotations

from typing import TypedDict, Any, Optional

from langgraph.graph import StateGraph, END

from data.db import ProfileStore
from dsp.parametric_eq import ParametricEQ, TargetCurve
from dsp.equalizer_spec import EqualizerSpec
from dsp.eq_projection import ProjectedEQ
from perception.context_classifier import Context
from agents.profile_agent import run_profile_agent
from agents.eq_decision_agent import run_eq_decision_agent
from agents.projection_agent import run_projection_agent
from agents.explainer_agent import run_explainer_agent


class PipelineState(TypedDict, total=False):
    user_id: str
    context: Context
    user_command: str
    equalizer_spec: Optional[EqualizerSpec]
    profile: dict
    baseline_curve: TargetCurve
    decided_curve: TargetCurve
    command_deltas: dict
    context_deltas: dict
    projected_eq: Optional[ProjectedEQ]
    eq: ParametricEQ
    explanation: str


def build_graph(store: ProfileStore, eq: ParametricEQ,
                equalizer_spec: Optional[EqualizerSpec] = None):
    graph = StateGraph(PipelineState)

    graph.add_node("profile_agent", lambda s: run_profile_agent(s, store))
    graph.add_node("eq_decision_agent", run_eq_decision_agent)
    graph.add_node("projection_agent", lambda s: run_projection_agent(s, eq, equalizer_spec))
    graph.add_node("explainer_agent", lambda s: run_explainer_agent(s, eq))

    graph.set_entry_point("profile_agent")
    graph.add_edge("profile_agent", "eq_decision_agent")
    graph.add_edge("eq_decision_agent", "projection_agent")
    graph.add_edge("projection_agent", "explainer_agent")
    graph.add_edge("explainer_agent", END)

    return graph.compile()


def run_pipeline(
    store: ProfileStore,
    eq: ParametricEQ,
    user_id: str,
    context: Context,
    user_command: str = "",
    equalizer_spec: Optional[EqualizerSpec] = None,
) -> PipelineState:
    """Convenience one-shot call used by the Streamlit app and validation script."""
    app = build_graph(store, eq, equalizer_spec)
    result = app.invoke({
        "user_id": user_id,
        "context": context,
        "user_command": user_command,
        "equalizer_spec": equalizer_spec,
    })
    # persist the decided curve + a history entry
    content_type = context.content_type
    profile_update: dict = {
        "target_curves": {
            **store.get_profile(user_id)["target_curves"],
            content_type: {
                "volume_db": result["decided_curve"].volume_db,
                "bass_gain_db": result["decided_curve"].bass_gain_db,
                "presence_gain_db": result["decided_curve"].presence_gain_db,
                "treble_gain_db": result["decided_curve"].treble_gain_db,
            },
        },
    }
    if equalizer_spec is not None:
        # remember the user's EQ so the next run defaults to it
        profile_update["equalizer_spec"] = equalizer_spec.to_dict()
    store.save_profile(user_id, profile_update)

    projected = result.get("projected_eq")
    store.log_adjustment(user_id, {
        "content_type": content_type,
        "noise_level": context.noise_level,
        "command": user_command,
        "explanation": result["explanation"],
        "context_deltas": result["context_deltas"],
        "command_deltas": result["command_deltas"],
        "projected_eq": projected.to_dict() if projected is not None else None,
    })
    return result
