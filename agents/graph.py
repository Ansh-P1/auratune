"""
LangGraph pipeline: Profile Agent -> EQ Decision Agent -> Explainer Agent.

State flows through as a plain dict (langgraph's TypedDict pattern), which
keeps this file readable and makes it trivial to add a 4th node later
(e.g. a Safety agent that hard-caps gain deltas) without touching the
agents themselves.
"""
from __future__ import annotations

from typing import TypedDict, Any, Dict, Optional

import numpy as np
from langgraph.graph import StateGraph, END

from data.db import ProfileStore
from dsp.parametric_eq import ParametricEQ, TargetCurve
from dsp.equalizer_spec import EqualizerSpec
from dsp.eq_projection import ProjectedEQ
from perception.context_classifier import Context
from agents.profile_agent import run_profile_agent
from agents.noise_agent import run_noise_agent
from agents.genre_agent import run_genre_agent
from agents.eq_decision_agent import run_eq_decision_agent
from agents.projection_agent import run_projection_agent
from agents.explainer_agent import run_explainer_agent
from agents.trace import traced


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
    genre_bucket: Optional[str]
    genre_confidence: float
    genre_model_used: str
    genre_probabilities: Dict[str, float]
    genre_proxy_features: Dict[str, float]
    genre_deltas: dict
    genre_unavailable_reason: str
    noise_bucket: Optional[str]
    noise_confidence: float
    noise_model_used: str
    noise_probabilities: Dict[str, float]
    noise_features: Dict[str, float]
    noise_deltas: dict
    noise_unavailable_reason: str
    projected_eq: Optional[ProjectedEQ]
    eq: ParametricEQ
    explanation: str
    # Observability (agents/trace.py) -- recorded per run so the dashboard
    # can show one row per node and say whether Claude or a fallback ran.
    agent_trace: list
    llm_calls: list
    command_parse_source: str    # "llm" | "rules" | "none"
    explanation_source: str      # "llm" | "template"
    explanation_provider: str    # "anthropic" | "groq"
    explanation_model: str


def build_graph(store: ProfileStore, eq: ParametricEQ,
                equalizer_spec: Optional[EqualizerSpec] = None,
                content_audio: Optional[np.ndarray] = None,
                ambient_audio: Optional[np.ndarray] = None,
                sample_rate: int = 44100,
                genre_model: str = "auto",
                noise_model: str = "auto"):
    graph = StateGraph(PipelineState)

    # Every node goes through traced() (agents/trace.py), which times it and
    # records what it put into the state -- pure recording, the node's own
    # return value is passed straight through.
    graph.add_node("profile_agent", traced("profile_agent", lambda s: run_profile_agent(s, store)))
    graph.add_node("noise_agent", traced("noise_agent", lambda s: run_noise_agent(s, ambient_audio, sample_rate, noise_model)))
    graph.add_node("genre_agent", traced("genre_agent", lambda s: run_genre_agent(s, content_audio, sample_rate, genre_model)))
    graph.add_node("eq_decision_agent", traced("eq_decision_agent", run_eq_decision_agent))
    graph.add_node("projection_agent", traced("projection_agent", lambda s: run_projection_agent(s, eq, equalizer_spec)))
    graph.add_node("explainer_agent", traced("explainer_agent", lambda s: run_explainer_agent(s, eq)))

    graph.set_entry_point("profile_agent")
    graph.add_edge("profile_agent", "noise_agent")
    graph.add_edge("noise_agent", "genre_agent")
    graph.add_edge("genre_agent", "eq_decision_agent")
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
    content_audio: Optional[np.ndarray] = None,
    ambient_audio: Optional[np.ndarray] = None,
    sample_rate: int = 44100,
    genre_model: str = "auto",
    noise_model: str = "auto",
) -> PipelineState:
    """Convenience one-shot call used by the Streamlit app and validation script.

    content_audio: the currently-playing content buffer, used only by the
    Genre agent (see agents/genre_agent.py) to run the local genre/mood
    classifier when context.content_type == "music".
    ambient_audio: the room/mic buffer, used only by the Noise agent (see
    agents/noise_agent.py) to run the local noise-type classifier. Both
    optional -- omit either and the pipeline behaves exactly as before
    that classifier existed.
    """
    app = build_graph(store, eq, equalizer_spec, content_audio, ambient_audio,
                      sample_rate, genre_model, noise_model)
    result = app.invoke({
        "user_id": user_id,
        "context": context,
        "user_command": user_command,
        "equalizer_spec": equalizer_spec,
        "agent_trace": [],
        "llm_calls": [],
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
        "explanation_source": result.get("explanation_source"),
        "command_parse_source": result.get("command_parse_source"),
        "context_deltas": result["context_deltas"],
        "command_deltas": result["command_deltas"],
        "genre_bucket": result.get("genre_bucket"),
        "genre_confidence": result.get("genre_confidence"),
        "genre_deltas": result.get("genre_deltas"),
        "noise_bucket": result.get("noise_bucket"),
        "noise_confidence": result.get("noise_confidence"),
        "noise_deltas": result.get("noise_deltas"),
        "projected_eq": projected.to_dict() if projected is not None else None,
    })
    return result
