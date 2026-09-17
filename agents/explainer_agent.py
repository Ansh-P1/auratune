"""
Explainer Agent.

Responsibility (per the architecture slide): "Turns the deltas and their
trigger -- noise, content type, or command -- into one plain-English
sentence." Calls Claude for natural phrasing; falls back to a templated
sentence if no API key is set, so the UI always has something to show.
"""
from __future__ import annotations

from dsp.parametric_eq import ParametricEQ
from agents.llm_client import complete_with_meta
from agents.trace import record_llm_call


def _template_sentence(state: dict) -> str:
    ctx = state["context"]
    deltas = state["eq"].delta(state["baseline_curve"], state["decided_curve"])
    # Small threshold, not a strict 0 check -- a delta that's nonzero but
    # rounds to "0.0" at 1 decimal (e.g. -0.02) would otherwise still get
    # announced ("pulled bass back 0.0 dB"), which reads like a bug.
    eps = 0.05
    bits = []
    if deltas["presence_gain_db"] > eps:
        bits.append(f"boosted vocal clarity by {deltas['presence_gain_db']:.1f} dB")
    elif deltas["presence_gain_db"] < -eps:
        bits.append(f"eased vocal presence by {abs(deltas['presence_gain_db']):.1f} dB")
    if deltas["bass_gain_db"] < -eps:
        bits.append(f"pulled bass back {abs(deltas['bass_gain_db']):.1f} dB")
    elif deltas["bass_gain_db"] > eps:
        bits.append(f"added {deltas['bass_gain_db']:.1f} dB of bass")
    if abs(deltas["treble_gain_db"]) > eps:
        direction = "brightened" if deltas["treble_gain_db"] > 0 else "warmed"
        bits.append(f"{direction} treble by {abs(deltas['treble_gain_db']):.1f} dB")
    if abs(deltas["volume_db"]) > eps:
        direction = "raised" if deltas["volume_db"] > 0 else "lowered"
        bits.append(f"{direction} volume {abs(deltas['volume_db']):.1f} dB")

    if not bits:
        return f"No change needed -- your {ctx.content_type} curve already fits a {ctx.noise_level} room."

    # Only credit the command when it actually moved something. A command we
    # couldn't parse (empty command_deltas) leaves changes that came from the
    # room/ML rules -- attributing those to the user reads as if the app
    # understood a request it silently dropped.
    command_worked = state.get("user_command") and state.get("command_deltas")
    trigger = "your command" if command_worked else f"a {ctx.noise_level} environment"
    return f"Because of {trigger} during {ctx.content_type}, I " + ", ".join(bits) + "."


def _llm_sentence(state: dict) -> str:
    ctx = state["context"]
    deltas = state["eq"].delta(state["baseline_curve"], state["decided_curve"])
    # Same rule as the template above: an unparsed command isn't the trigger.
    trigger = (state["user_command"] if state.get("command_deltas")
               else f"{ctx.noise_level} ambient noise")
    system = (
        "You explain an automatic audio-EQ adjustment to a non-technical user "
        "in exactly ONE short, friendly sentence. Mention the trigger and the "
        "audible effect, not raw jargon. No preamble, no quotes. Describe ONLY "
        "the adjustments in the deltas you are given: a delta of 0 means that "
        "band did not change, so never mention it. Do not invent changes."
    )
    user = (
        f"Content type: {ctx.content_type}. Trigger: {trigger}. "
        f"Deltas (dB): volume={deltas['volume_db']}, bass={deltas['bass_gain_db']}, "
        f"presence/vocal={deltas['presence_gain_db']}, treble={deltas['treble_gain_db']}."
    )
    sentence, call = complete_with_meta(system, user, purpose="explanation",
                                        max_tokens=100)
    record_llm_call(state, call)
    return sentence


def _genre_clause(state: dict) -> str:
    """One extra clause naming the ML-detected genre, when the Genre agent
    (agents/genre_agent.py) ran and actually changed anything."""
    bucket = state.get("genre_bucket")
    genre_deltas = state.get("genre_deltas") or {}
    if not bucket or not any(abs(v) >= 0.05 for v in genre_deltas.values()):
        return ""
    label = bucket.replace("_", "/")
    confidence = state.get("genre_confidence", 0.0)
    model = state.get("genre_model_used", "")
    return (f" Sounds like {label} ({confidence * 100:.0f}% confidence, "
            f"local {model} model), so I leaned the curve that way.")


def _noise_clause(state: dict) -> str:
    """One extra clause naming the ML-detected ambient noise type, when
    the Noise agent (agents/noise_agent.py) ran and actually changed
    anything."""
    bucket = state.get("noise_bucket")
    noise_deltas = state.get("noise_deltas") or {}
    if not bucket or not any(abs(v) >= 0.05 for v in noise_deltas.values()):
        return ""
    label = bucket.replace("_", " ")
    confidence = state.get("noise_confidence", 0.0)
    model = state.get("noise_model_used", "")
    return (f" The room sounds like {label} ({confidence * 100:.0f}% confidence, "
            f"local {model} model).")


def _projection_clause(state: dict) -> str:
    """One extra sentence naming the actual slider moves for the user's EQ app."""
    proj = state.get("projected_eq")
    if proj is None:
        return ""
    from dsp.eq_projection import _fmt_hz

    moves = [b for b in proj.bands if abs(b.set_gain_db) >= 0.05]
    moves.sort(key=lambda b: abs(b.set_gain_db), reverse=True)
    app = proj.spec_name.split(" - ")[0].split(" (")[0]

    if not moves:
        base = f" On your {app} EQ, leave every band flat"
    else:
        listed = ", ".join(f"{_fmt_hz(b.freq_hz)} {b.set_gain_db:+.1f}" for b in moves[:4])
        base = f" On your {app} EQ, set {listed}"
        if len(moves) > 4:
            base += f" (+{len(moves) - 4} smaller)"

    tail = ""
    if abs(proj.preamp_db) >= 0.05:
        word = "preamp" if proj.has_preamp else "and drop the volume"
        tail = f", {word} {proj.preamp_db:+.1f} dB"
    clip = ""
    if proj.clipped_freqs:
        spec = state.get("equalizer_spec")
        limit = f"±{abs(spec.gain_max_db):.0f} dB " if spec is not None else ""
        clip = (f" Your app's {limit}limit capped "
                f"{', '.join(_fmt_hz(f) for f in proj.clipped_freqs)}.")
    return base + tail + "." + clip


def run_explainer_agent(state: dict, eq: ParametricEQ) -> dict:
    state["eq"] = eq
    sentence = _llm_sentence(state)
    state["explanation_source"] = "llm" if sentence else "template"
    if sentence:
        # Which backend actually wrote it -- the dashboard badge names the
        # model rather than assuming Claude (see agents/llm_client.py).
        last_call = (state.get("llm_calls") or [])[-1]
        state["explanation_provider"] = last_call.provider
        state["explanation_model"] = last_call.model
    if not sentence:
        sentence = _template_sentence(state)
    state["explanation"] = sentence + _noise_clause(state) + _genre_clause(state) + _projection_clause(state)
    return state
