"""
AuraTune -- Live Adaptive Mode.

The one-shot "Run adaptation" flow on the main page samples the room once.
This page keeps listening, keeps deciding, and keeps the EQ correct without
anyone touching anything -- the "Real-Time Adaptive Listening Mode" brief's
Part 5 deliverable (dashboard, evaluation, final integration).

Parts 1 (continuous ambient sensing, perception/live_monitor.py) and 2
(smooth EQ transitions, dsp/curve_smoothing.py) are both merged now, so
this page runs the real things: perception.live_monitor.LiveMonitor
listens to the real system microphone (via perception/live_capture.py's
sounddevice capture) when one is available, falling back to cycling the
three synth_scenario demo clips when it isn't (e.g. a headless server with
no audio hardware). dsp.curve_smoothing.CurveRamper computes the actual
glide steps between curves. Everything else (context classification, the
agent pipeline, personalization) is unchanged: perception.context_classifier,
agents.graph.run_pipeline, and learning.preference_model all run unmodified.

Threading note: LiveMonitor samples on its own background thread and fires
on_change() callbacks from that thread. Streamlit's session_state isn't
thread-safe to touch from there (see live_monitor.py's own docstring), so
the callback below never reads or writes st.session_state directly -- it
only runs the real pipeline and puts the result on a plain thread-safe
queue.Queue, which the main script thread drains on every rerun.

The callback also reimplements agents.live_loop.LiveLoop's cooldown gate
(same algorithm, same 8s default) inline rather than importing LiveLoop
itself, because LiveLoop's on_curve_step(curve, is_final) callback doesn't
surface the full PipelineState (baseline curve, agent trace) this page's
event log and feedback panel need. The actual smoothing math is still the
real dsp.curve_smoothing.CurveRamper -- the same class LiveLoop uses.

Run with: streamlit run app.py   (this page appears in the sidebar nav)
"""
from __future__ import annotations

import queue
import threading
import time
from datetime import datetime

import streamlit as st

from dsp.curve_smoothing import CurveRamper
from dsp.equalizer_spec import all_specs
from dsp.parametric_eq import ParametricEQ, TargetCurve
from perception.context_classifier import Context
from perception.live_capture import is_available as mic_is_available
from perception.live_monitor import LiveMonitor
from perception.synth_scenarios import synth_scenario, SCENARIO_CONTENT_TYPE
from data.db import ProfileStore
from agents.graph import run_pipeline
from learning.preference_model import PreferenceModel

from ui import theme
from ui import components as c
from ui import charts

SR = 44100
USER_ID = "demo_user"
N_RAMP_STEPS = 5
COOLDOWN_S = 8.0

FIELD_LABELS = {
    "volume_db": "volume",
    "bass_gain_db": "bass",
    "presence_gain_db": "presence",
    "treble_gain_db": "treble",
}

_RAMPER = CurveRamper(steps=N_RAMP_STEPS)

st.set_page_config(page_title="AuraTune -- Live Mode", layout="wide",
                   initial_sidebar_state="collapsed")

st.session_state.setdefault("dark_mode_enabled", st.session_state.get("dark_mode", True))
st.session_state.setdefault("live_mode_enabled", False)
st.session_state["live_mode_page"] = st.session_state.live_mode_enabled
DARK = st.session_state.dark_mode_enabled
theme.inject(DARK)


@st.cache_resource
def get_store():
    return ProfileStore()


@st.cache_resource
def get_eq():
    return ParametricEQ(SR)


# ---------------------------------------------------------------------------
# Session state for the live loop
# ---------------------------------------------------------------------------
st.session_state.setdefault("live_on", False)
st.session_state.setdefault("live_monitor", None)
st.session_state.setdefault("live_queue", None)
st.session_state.setdefault("live_using_real_mic", False)
st.session_state.setdefault("live_baseline_curve", None)
st.session_state.setdefault("live_current_curve", None)
st.session_state.setdefault("live_pending_steps", [])
st.session_state.setdefault("live_log", [])
st.session_state.setdefault("live_last_ctx", None)
st.session_state.setdefault("live_last_result", None)
st.session_state.setdefault("live_command_state", {"text": ""})
st.session_state.setdefault("live_last_error", None)
st.session_state.setdefault("live_apply_state", {"spec": None, "apply_live": False})


def _dominant_field_delta(deltas: dict):
    field, value = max(deltas.items(), key=lambda kv: abs(kv[1]))
    return field, value


def _make_demo_clip_source(cycle=("quiet_podcast", "noisy_music", "home_movie"), repeats=4):
    """No real mic available -- cycle the 3 demo scenarios instead, holding
    each for `repeats` samples in a row (like a room that actually stays in
    one state for a while), matching LiveMonitor.set_clip_source()'s
    fn() -> (ambient, content, content_type_hint) contract."""
    state = {"idx": 0, "count": 0}

    def _next():
        kind = cycle[state["idx"]]
        state["count"] += 1
        if state["count"] >= repeats:
            state["count"] = 0
            state["idx"] = (state["idx"] + 1) % len(cycle)
        ambient, content = synth_scenario(kind, SR)
        return ambient, content, SCENARIO_CONTENT_TYPE[kind]

    return _next


def _make_live_reaction(store, eq, events_queue: "queue.Queue", command_state: dict,
                        apply_state: dict, cooldown_s: float = COOLDOWN_S):
    """Returns a LiveMonitor.on_change callback: real cooldown gate + real
    pipeline run, result handed to the main thread via the queue. Runs on
    LiveMonitor's background thread -- must never touch st.session_state.

    command_state and apply_state are plain dicts the main thread updates in
    place every rerun (as the user types a command, or changes the EQ app /
    apply-live toggle) -- reading a single dict key like this from another
    thread is safe in CPython (the GIL makes the read/write atomic), so no
    extra locking is needed just for this, unlike the cooldown timestamp
    below which involves a check-then-set.

    apply_state["apply_live"] + ["spec"] control whether every confirmed
    adaptation is also pushed live to a real EQ target (currently only
    Equalizer APO -- see dsp/eq_projection.py's apply_live, which was wired
    all the way through run_pipeline/build_graph/run_projection_agent for
    this). When on, this actually rewrites Equalizer APO's config.txt on
    the user's machine on every confirmed change -- a real, audible,
    system-wide side effect, not just a chart update.

    Any exception from run_pipeline (a slow/failed LLM call, a missing
    model file, etc.) is caught and reported as an error event instead of
    propagating -- an uncaught exception here would otherwise escape
    LiveMonitor's sampling loop and silently kill the whole background
    thread, which looks exactly like "the mic stopped being detected" with
    zero indication of why.
    """
    cooldown_state = {"last_run": None}
    lock = threading.Lock()

    def on_context_change(ctx: Context) -> None:
        now = time.monotonic()
        with lock:
            if cooldown_state["last_run"] is not None and now - cooldown_state["last_run"] < cooldown_s:
                return
            cooldown_state["last_run"] = now
        command = command_state.get("text", "").strip()
        spec = apply_state.get("spec")
        apply_live = bool(apply_state.get("apply_live")) and spec is not None
        try:
            result = run_pipeline(store, eq, USER_ID, ctx, command, sample_rate=SR,
                                  equalizer_spec=spec, apply_live=apply_live)
        except Exception as exc:
            events_queue.put({"error": f"{type(exc).__name__}: {exc}"})
            return
        events_queue.put({"ctx": ctx, "result": result})

    return on_context_change


def _describe_change(old_ctx, new_ctx, deltas: dict) -> str:
    field, value = _dominant_field_delta(deltas)
    label = FIELD_LABELS[field]
    verb = "boosted" if value >= 0 else "cut"
    headline = f"{verb} {label} {value:+.1f}dB"
    if old_ctx is None:
        return f"started on {new_ctx.noise_level}/{new_ctx.content_type} -- {headline}"
    bits = []
    if old_ctx.noise_level != new_ctx.noise_level:
        bits.append(f"noise went {old_ctx.noise_level} to {new_ctx.noise_level}")
    if old_ctx.content_type != new_ctx.content_type:
        bits.append(f"content switched {old_ctx.content_type} to {new_ctx.content_type}")
    if not bits:
        bits.append(f"{new_ctx.noise_level}/{new_ctx.content_type} confirmed")
    return f"{', '.join(bits)}, {headline}"


def _reset_live_state(interval_sec: float, eq: ParametricEQ):
    """Auto mode just turned on: build a real LiveMonitor, wire it to a
    fresh queue, and start it sampling (real mic if one exists, else the
    synthetic scenario cycle)."""
    q: "queue.Queue" = queue.Queue()
    monitor = LiveMonitor(debounce_samples=2)
    use_real_mic = mic_is_available()
    if not use_real_mic:
        monitor.set_clip_source(_make_demo_clip_source())
    monitor.on_change(_make_live_reaction(get_store(), eq, q, st.session_state.live_command_state,
                                          st.session_state.live_apply_state))
    monitor.start(sample_rate=SR, interval_sec=interval_sec)

    st.session_state.live_monitor = monitor
    st.session_state.live_queue = q
    st.session_state.live_using_real_mic = use_real_mic
    st.session_state.live_baseline_curve = None
    st.session_state.live_current_curve = None
    st.session_state.live_pending_steps = []
    st.session_state.live_log = []
    st.session_state.live_last_ctx = None
    st.session_state.live_last_result = None


def _stop_live_state():
    if st.session_state.live_monitor is not None:
        st.session_state.live_monitor.stop()
    st.session_state.live_monitor = None
    st.session_state.live_queue = None


def _drain_queue(eq: ParametricEQ):
    """Main-thread only: pull every pipeline result the background thread
    has produced since the last rerun and update the UI state from it. If
    more than one arrived, only the latest one's glide is kept -- a rapid
    second change should snap the target forward, not queue up two glides."""
    q = st.session_state.live_queue
    if q is None:
        return
    drained = False
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
        drained = True
        if "error" in item:
            st.session_state.live_last_error = item["error"]
            st.session_state.live_log.insert(0, {"ts": datetime.now().strftime("%H:%M:%S"),
                                                  "text": f"⚠ pipeline error: {item['error']}"})
            st.session_state.live_log = st.session_state.live_log[:20]
            continue
        st.session_state.live_last_error = None
        ctx = item["ctx"]
        result = item["result"]
        new_curve = result["decided_curve"]
        old_curve = st.session_state.live_current_curve or result["baseline_curve"]
        if st.session_state.live_baseline_curve is None:
            st.session_state.live_baseline_curve = result["baseline_curve"]

        deltas = eq.delta(old_curve, new_curve)
        text = _describe_change(st.session_state.live_last_ctx, ctx, deltas)
        projected = result.get("projected_eq")
        if projected is not None and projected.live_detail:
            live_icon = "✅" if projected.live_applied else "⚠"
            text += f" -- {live_icon} {projected.live_detail}"
        st.session_state.live_log.insert(0, {"ts": datetime.now().strftime("%H:%M:%S"), "text": text})
        st.session_state.live_log = st.session_state.live_log[:20]

        st.session_state.live_pending_steps = _RAMPER.ramp(old_curve, new_curve)
        st.session_state.live_last_ctx = ctx
        st.session_state.live_last_result = result
    return drained


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def _return_to_main() -> None:
    st.session_state.live_mode_enabled = st.session_state.live_mode_page


title_col, mode_col = st.columns([7, 1])
with title_col:
    c.eyebrow("Real-time adaptive listening")
    st.markdown('<h1>Live <span class="at-acc">Mode</span></h1>', unsafe_allow_html=True)
with mode_col:
    st.write("")
    st.toggle("Live mode", key="live_mode_page", on_change=_return_to_main)
st.caption("Keeps listening, keeps deciding, keeps the EQ correct -- no button to click. "
          "Uses the real microphone when one's available, and the real smooth-glide EQ "
          "transition; everything downstream (classification, agents, personalization) "
          "is the same real pipeline the main page uses.")

# Same no-op-rerun-in-a-callback issue as app.py's toggle -- the navigation
# has to happen here, in the main script body, not inside on_change.
if not st.session_state.live_mode_enabled:
    st.switch_page("app.py")

_specs = all_specs()
_eq_options = ["(none, just show the curve)"] + list(_specs.keys())
_default_key = "equalizer_apo" if "equalizer_apo" in _specs else _eq_options[0]
_default_idx = _eq_options.index(_default_key) if _default_key in _eq_options else 0

eq_l, eq_r = st.columns([2, 1])
with eq_l:
    _selected_key = st.selectbox(
        "EQ app for live apply",
        _eq_options,
        index=_default_idx,
        format_func=lambda k: _specs[k].name if k in _specs else k,
        help="Pick your real EQ app so Live Mode can show the exact slider values -- "
             "and, for apps that support it, push them live automatically.",
    )
selected_spec = _specs.get(_selected_key)
with eq_r:
    st.write("")
    apply_live_toggle = False
    if selected_spec is not None and selected_spec.live_target:
        apply_live_toggle = st.checkbox(
            f"Apply live to {selected_spec.name}",
            value=bool(st.session_state.live_apply_state.get("apply_live", False)),
            help="Rewrites Equalizer APO's config.txt on every confirmed adaptation -- "
                 "a real, audible, system-wide change to whatever's currently playing, "
                 "not just an update to the chart below.",
        )
    elif selected_spec is not None:
        st.caption(f"{selected_spec.name} has no live target -- slider values shown, not auto-applied.")
st.session_state.live_apply_state["spec"] = selected_spec
st.session_state.live_apply_state["apply_live"] = apply_live_toggle

top_l, top_r = st.columns([2, 1])
with top_r:
    interval_sec = st.slider("Sensing interval (sec)", 0.5, 4.0, 1.5, 0.5,
                             help="How often the room is re-sampled. Also used as this "
                                  "page's own refresh rate. A real mic recording takes "
                                  "~2s on top of this gap.")
with top_l:
    live_on = st.toggle("Auto mode", value=st.session_state.live_on,
                        help="When on, this listens continuously (real mic if you have "
                             "one, otherwise a cycling demo scenario) and re-adapts the "
                             "EQ on its own, gliding smoothly between curves.")
    if live_on and not st.session_state.live_on:
        _reset_live_state(interval_sec, get_eq())
    elif not live_on and st.session_state.live_on:
        _stop_live_state()
    st.session_state.live_on = live_on
    if live_on:
        st.caption("\U0001F3A4 Listening on the real microphone" if st.session_state.live_using_real_mic
                   else "\U0001F50C No microphone detected -- cycling the 3 demo scenarios instead")

command_text = st.text_input(
    "Live command (optional)",
    value=st.session_state.live_command_state.get("text", ""),
    placeholder="e.g. more bass, brighter, louder, less bass, make voices clearer",
    help="Applied to every future automatic adaptation while Auto mode is on -- takes "
         "effect from the next confirmed room reading, not instantly (same debounce + "
         "8s cooldown as everything else here). Parsed by the LLM if one's configured, "
         "otherwise matched against keyword rules (see agents/eq_decision_agent.py).",
)
st.session_state.live_command_state["text"] = command_text

store = get_store()
eq = get_eq()

if st.session_state.live_on:
    _drain_queue(eq)
    if st.session_state.live_pending_steps:
        st.session_state.live_current_curve = st.session_state.live_pending_steps.pop(0)

if st.session_state.live_last_error:
    st.error(f"Last live adaptation failed: {st.session_state.live_last_error}")

result_container = st.container(border=True)
with result_container:
    st.subheader("Live EQ curve")
    baseline = st.session_state.live_baseline_curve
    current = st.session_state.live_current_curve
    if baseline is None or current is None:
        c.empty_state(title="Auto mode is off" if not st.session_state.live_on else "Listening...",
                     body="Turn on Auto mode above to start the live loop.")
    else:
        freqs_before, mag_before = eq.frequency_response(baseline)
        freqs_after, mag_after = eq.frequency_response(current)
        charts.eq_curve(freqs_before, mag_before, freqs_after, mag_after)

log_col, feedback_col = st.columns([2, 1])
with log_col:
    with st.container(border=True):
        c.eyebrow("Event log")
        if not st.session_state.live_log:
            st.caption("No automatic changes yet.")
        else:
            for entry in st.session_state.live_log:
                st.markdown(f"`{entry['ts']}` -- {entry['text']}")

with feedback_col:
    with st.container(border=True):
        c.eyebrow("How's this curve?")
        last_ctx = st.session_state.live_last_ctx
        last_result = st.session_state.live_last_result
        if last_ctx is None or last_result is None:
            st.caption("Feedback opens up after the first automatic adaptation.")
        else:
            context_bucket = f"{last_ctx.noise_level}_{last_ctx.content_type}"
            decided = last_result["decided_curve"]
            pref_model = PreferenceModel(store=store)
            up, down = st.columns(2)
            if up.button("Good \U0001F44D", use_container_width=True):
                pref_model.record_feedback(USER_ID, context_bucket, decided, {})
                st.toast("Thanks -- noted as correct for this context.")
            if down.button("Too much \U0001F44E", use_container_width=True):
                # Nudge: ask for half of whatever the agent just changed to be
                # undone, on the field that moved most -- the only signal we
                # can infer from a plain thumbs-down with no slider attached.
                field, value = _dominant_field_delta(
                    eq.delta(st.session_state.live_baseline_curve, decided))
                pref_model.record_feedback(USER_ID, context_bucket, decided, {field: -value / 2})
                st.toast(f"Thanks -- nudging {FIELD_LABELS[field]} back for {context_bucket}.")

if st.session_state.live_on:
    time.sleep(interval_sec)
    st.rerun()
