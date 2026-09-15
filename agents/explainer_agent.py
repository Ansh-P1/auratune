"""
Explainer Agent.

Responsibility (per the architecture slide): "Turns the deltas and their
trigger -- noise, content type, or command -- into one plain-English
sentence." Calls Claude for natural phrasing; falls back to a templated
sentence if no API key is set, so the UI always has something to show.
"""
from __future__ import annotations

from dsp.parametric_eq import ParametricEQ
from agents.llm_client import complete


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

    trigger = "your command" if state.get("user_command") else f"a {ctx.noise_level} environment"
    return f"Because of {trigger} during {ctx.content_type}, I " + ", ".join(bits) + "."


def _llm_sentence(state: dict) -> str:
    ctx = state["context"]
    deltas = state["eq"].delta(state["baseline_curve"], state["decided_curve"])
    trigger = state.get("user_command") or f"{ctx.noise_level} ambient noise"
    system = (
        "You explain an automatic audio-EQ adjustment to a non-technical user "
        "in exactly ONE short, friendly sentence. Mention the trigger and the "
        "audible effect, not raw jargon. No preamble, no quotes."
    )
    user = (
        f"Content type: {ctx.content_type}. Trigger: {trigger}. "
        f"Deltas (dB): volume={deltas['volume_db']}, bass={deltas['bass_gain_db']}, "
        f"presence/vocal={deltas['presence_gain_db']}, treble={deltas['treble_gain_db']}."
    )
    return complete(system, user, max_tokens=100)


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
    if not sentence:
        sentence = _template_sentence(state)
    state["explanation"] = sentence + _genre_clause(state) + _projection_clause(state)
    return state
