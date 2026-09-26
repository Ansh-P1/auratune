"""
AuraTune -- Live Adaptive Mode.

The one-shot "Run adaptation" flow on the main page samples the room once.
This page keeps listening, keeps deciding, and keeps the EQ correct without
anyone touching anything -- the "Real-Time Adaptive Listening Mode" brief's
Part 5 deliverable (dashboard, evaluation, final integration).

Parts 1 (continuous ambient sensing) and 2 (real-time decision loop + smooth
EQ transitions) don't exist in this repo yet, so this page runs against the
stand-ins in validation/live_mode_fakes.py -- see that file's docstring for
the exact swap points. Everything else (context classification, the agent
pipeline, personalization) is the real thing: perception.context_classifier,
agents.graph.run_pipeline, and whatever bias learning.preference_model has
already learned from past feedback all run unmodified.

Run with: streamlit run app.py   (this page appears in the sidebar nav)
"""
from __future__ import annotations

import time
from datetime import datetime

import streamlit as st

from dsp.parametric_eq import ParametricEQ, TargetCurve
from perception.context_classifier import classify, Context
from perception.synth_scenarios import synth_scenario, SCENARIO_CONTENT_TYPE
from data.db import ProfileStore
from agents.graph import run_pipeline
from learning.preference_model import PreferenceModel

from ui import theme
from ui import components as c
from ui import charts

from validation.live_mode_fakes import FakeLiveMonitor, ramp_steps, dominant_field_delta, FIELD_LABELS

SR = 44100
USER_ID = "demo_user"
N_RAMP_STEPS = 5

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
st.session_state.setdefault("live_baseline_curve", None)
st.session_state.setdefault("live_current_curve", None)
st.session_state.setdefault("live_pending_steps", [])
st.session_state.setdefault("live_log", [])
st.session_state.setdefault("live_last_ctx", None)
st.session_state.setdefault("live_last_result", None)
st.session_state.setdefault("live_ticks", 0)


def _reset_live_state():
    st.session_state.live_monitor = FakeLiveMonitor()
    st.session_state.live_baseline_curve = None
    st.session_state.live_current_curve = None
    st.session_state.live_pending_steps = []
    st.session_state.live_log = []
    st.session_state.live_last_ctx = None
    st.session_state.live_last_result = None
    st.session_state.live_ticks = 0


def _describe_change(old_ctx, new_ctx, deltas: dict) -> str:
    field, value = dominant_field_delta(deltas)
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


def _run_tick(store, eq):
    """Advance the fake live loop by exactly one tick. Either plays the next
    queued smoothing step, or samples the room and -- if that's a debounced,
    confirmed change -- re-runs the real pipeline for a fresh target curve
    and queues a fresh glide toward it."""
    if st.session_state.live_pending_steps:
        st.session_state.live_current_curve = st.session_state.live_pending_steps.pop(0)
        return

    monitor: FakeLiveMonitor = st.session_state.live_monitor
    kind = monitor.next_scenario_kind()
    ambient, content = synth_scenario(kind, SR)
    ctx: Context = classify(ambient, content, SR, content_type_hint=SCENARIO_CONTENT_TYPE[kind])

    if not monitor.observe(ctx):
        return

    result = run_pipeline(store, eq, USER_ID, ctx, "", content_audio=content,
                          ambient_audio=ambient, sample_rate=SR)
    new_curve = result["decided_curve"]
    old_curve = st.session_state.live_current_curve or result["baseline_curve"]
    if st.session_state.live_baseline_curve is None:
        st.session_state.live_baseline_curve = result["baseline_curve"]

    deltas = eq.delta(old_curve, new_curve)
    text = _describe_change(st.session_state.live_last_ctx, ctx, deltas)
    st.session_state.live_log.insert(0, {"ts": datetime.now().strftime("%H:%M:%S"), "text": text})
    st.session_state.live_log = st.session_state.live_log[:20]

    st.session_state.live_pending_steps = ramp_steps(old_curve, new_curve, N_RAMP_STEPS)
    st.session_state.live_last_ctx = ctx
    st.session_state.live_last_result = result


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def _return_to_main() -> None:
    st.session_state.live_mode_enabled = st.session_state.live_mode_page
    if not st.session_state.live_mode_enabled:
        st.switch_page("app.py")


title_col, mode_col = st.columns([7, 1])
with title_col:
    c.eyebrow("Real-time adaptive listening")
    st.markdown('<h1>Live <span class="at-acc">Mode</span></h1>', unsafe_allow_html=True)
with mode_col:
    st.write("")
    st.toggle("Live mode", key="live_mode_page", on_change=_return_to_main)
st.caption("Keeps listening, keeps deciding, keeps the EQ correct -- no button to click. "
          "Sensing and smoothing here stand in for Parts 1 & 2 until those land; "
          "everything downstream (classification, agents, personalization) is the real pipeline.")

top_l, top_r = st.columns([2, 1])
with top_l:
    live_on = st.toggle("Auto mode", value=st.session_state.live_on,
                        help="When on, this page simulates a room that keeps changing and "
                             "keeps re-adapting, the same way a real LiveMonitor would.")
    if live_on and not st.session_state.live_on:
        _reset_live_state()
    st.session_state.live_on = live_on
with top_r:
    interval_sec = st.slider("Tick interval (sec)", 0.5, 4.0, 1.5, 0.5,
                             help="How often the fake sensor samples the room / advances a glide step.")

store = get_store()
eq = get_eq()

if st.session_state.live_on:
    _run_tick(store, eq)

result_container = st.container(border=True)
with result_container:
    st.subheader("Live EQ curve")
    baseline = st.session_state.live_baseline_curve
    current = st.session_state.live_current_curve
    if baseline is None or current is None:
        c.empty_state(title="Auto mode is off" if not st.session_state.live_on else "Listening...",
                     body="Turn on Auto mode above to start the simulated live loop.")
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
                field, value = dominant_field_delta(
                    eq.delta(st.session_state.live_baseline_curve, decided))
                pref_model.record_feedback(USER_ID, context_bucket, decided, {field: -value / 2})
                st.toast(f"Thanks -- nudging {FIELD_LABELS[field]} back for {context_bucket}.")

if st.session_state.live_on:
    time.sleep(interval_sec)
    st.rerun()
