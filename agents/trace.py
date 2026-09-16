"""
Agent trace.

The pipeline is six LangGraph nodes deep (profile -> noise -> genre ->
eq_decision -> projection -> explainer), but until now the dashboard only
ever showed the final sentence plus one flattened blob of deltas. Nobody
looking at a run could tell which node did what, which ones no-opped, or
whether Claude actually ran.

This module is the recording seam: graph.py wraps every node with
`traced()`, which times it and captures a one-line summary of what that
node put into the state. Agents that call Claude additionally record an
LLMCall (agents/llm_client.py) via `record_llm_call()`. The result is
state["agent_trace"] -- an ordered list of steps app.py renders as the
"Agent trace" view, one row per node, in execution order.

Recording only. Nothing here changes a curve, so a trace failure can
never break an adaptation.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from agents.llm_client import LLMCall

# Short human-readable labels + what each node is responsible for, shown in
# the dashboard's trace view so the node names aren't bare function names.
NODE_LABELS: dict[str, tuple[str, str]] = {
    "profile_agent": ("Profile", "Loads the stored hearing profile and baseline curve"),
    "noise_agent": ("Noise", "Classifies the ambient buffer's noise type (local ML model)"),
    "genre_agent": ("Genre", "Classifies music content into an EQ-relevant genre bucket"),
    "eq_decision_agent": ("EQ decision", "Blends profile + context + ML + command into a curve"),
    "projection_agent": ("Projection", "Snaps the curve onto your EQ app's real sliders"),
    "explainer_agent": ("Explainer", "Writes the one-sentence plain-English explanation"),
}

# Which nodes can call Claude at all -- the other four are deterministic.
# Used by the trace view to distinguish "this node ran rules" from "this
# node has no LLM in it to begin with".
LLM_CAPABLE_NODES = {"eq_decision_agent", "explainer_agent"}


def record_llm_call(state: dict, call: LLMCall) -> None:
    """Append one attempted Claude call to the state's LLM log."""
    state.setdefault("llm_calls", []).append(call)


def _summarize(node: str, state: dict) -> tuple[str, dict]:
    """One-line status plus the interesting detail for a finished node."""
    if node == "profile_agent":
        baseline = state.get("baseline_curve")
        name = getattr(baseline, "name", "-")
        return f"Loaded baseline curve `{name}`", {
            "baseline_curve": baseline.to_dict() if baseline is not None else None,
        }

    if node == "noise_agent":
        if state.get("noise_bucket"):
            conf = state.get("noise_confidence", 0.0)
            return (f"Heard {state['noise_bucket'].replace('_', ' ')} "
                    f"({conf * 100:.0f}% confidence)"), {
                "model_used": state.get("noise_model_used"),
                "noise_probabilities": state.get("noise_probabilities"),
            }
        reason = state.get("noise_unavailable_reason")
        return (f"Skipped — {reason}" if reason
                else "Skipped — no ambient audio passed in"), {}

    if node == "genre_agent":
        if state.get("genre_bucket"):
            conf = state.get("genre_confidence", 0.0)
            return (f"Heard {state['genre_bucket'].replace('_', '/')} "
                    f"({conf * 100:.0f}% confidence)"), {
                "model_used": state.get("genre_model_used"),
                "genre_probabilities": state.get("genre_probabilities"),
            }
        content_type = getattr(state.get("context"), "content_type", "?")
        if content_type != "music":
            return f"Skipped — content is {content_type}, not music", {}
        reason = state.get("genre_unavailable_reason")
        return (f"Skipped — {reason}" if reason
                else "Skipped — no content audio passed in"), {}

    if node == "eq_decision_agent":
        source = state.get("command_parse_source", "none")
        if source == "none":
            summary = "Applied context deltas (no typed command)"
        elif source == "llm":
            summary = "Applied context deltas + parsed your command with the LLM"
        else:
            summary = "Applied context deltas + parsed your command with keyword rules"
        return summary, {
            "context_deltas": state.get("context_deltas"),
            "noise_deltas": state.get("noise_deltas"),
            "genre_deltas": state.get("genre_deltas"),
            "command_deltas": state.get("command_deltas"),
            "decided_curve": (state["decided_curve"].to_dict()
                              if state.get("decided_curve") is not None else None),
        }

    if node == "projection_agent":
        proj = state.get("projected_eq")
        if proj is None:
            return "Skipped — no EQ app selected", {}
        return f"Snapped the curve onto {proj.spec_name}", {
            "fit_error_db": proj.fit_error_db,
            "preamp_db": proj.preamp_db,
            "clipped_freqs": proj.clipped_freqs,
        }

    if node == "explainer_agent":
        source = state.get("explanation_source", "template")
        if source == "llm":
            model = state.get("explanation_model") or "the LLM"
            wrote = f"`{model}` wrote"
        else:
            wrote = "Template wrote"
        return f"{wrote} the explanation", {"explanation": state.get("explanation")}

    return "Ran", {}


def traced(node: str, fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """Wrap a graph node so it records an ordered, timed trace entry.

    The wrapped node's own return value is passed through untouched, so
    adding or removing tracing can never change what the pipeline decides.
    """
    def wrapper(state: dict) -> dict:
        started = time.perf_counter()
        before = len(state.get("llm_calls") or [])
        new_state = fn(state)
        elapsed_ms = (time.perf_counter() - started) * 1000

        label, description = NODE_LABELS.get(node, (node, ""))
        summary, detail = _summarize(node, new_state)
        calls = (new_state.get("llm_calls") or [])[before:]

        trace: list = new_state.setdefault("agent_trace", [])
        trace.append({
            "step": len(trace) + 1,
            "node": node,
            "label": label,
            "description": description,
            "summary": summary,
            "skipped": summary.startswith("Skipped"),
            "duration_ms": round(elapsed_ms, 1),
            "llm_capable": node in LLM_CAPABLE_NODES,
            "llm_calls": [c.to_dict() for c in calls],
            "detail": detail,
        })
        return new_state

    return wrapper


def explanation_source(state: dict) -> Optional[str]:
    """"llm" or "template" -- what actually wrote the sentence on screen."""
    return state.get("explanation_source")
