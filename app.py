"""
AuraTune -- Streamlit dashboard.

Simulates the live loop: pick/generate an ambient + content scenario,
run it through the perception -> LangGraph -> DSP pipeline, and show the
live EQ curve plus the plain-English explanation. A chat box lets the user
type live commands ("make voices clearer") that the EQ Decision agent
folds into the curve.

"Your EQ app" lets the user pick the real EQ they have (or a spec Claude
wrote from a screenshot); the pipeline then snaps the adapted curve onto
that app's exact sliders, step size, and gain range.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dsp.parametric_eq import ParametricEQ
from dsp.equalizer_spec import EqualizerSpec, all_specs, save_spec
from perception.eq_app_reader import read_equalizer_screenshot
from perception.context_classifier import classify, Context
from perception.synth_scenarios import synth_scenario, SCENARIOS as SCENARIO_LABELS, SCENARIO_CONTENT_TYPE
from perception import genre_classifier
from data.db import ProfileStore
from agents.graph import run_pipeline

st.set_page_config(page_title="AuraTune", page_icon="🎧", layout="wide")

SR = 44100
USER_ID = "demo_user"


@st.cache_resource
def get_store():
    return ProfileStore()


@st.cache_resource
def get_eq():
    return ParametricEQ(SR)


def _slugify(name: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in name]
    return "".join(keep).strip("_") or "custom_eq"


@st.cache_data(show_spinner="Reading your EQ screenshot…")
def _read_screenshot_cached(image_bytes: bytes, media_type: str, gemini_key: str):
    """Cache by image bytes (+ key) so we don't re-call the vision model on every rerun."""
    res = read_equalizer_screenshot(image_bytes, media_type,
                                    gemini_key=gemini_key or None)
    return ((res.spec.to_dict() if res.spec else None),
            res.error, res.model_notes, res.backend)


def _spec_editor(prefill: EqualizerSpec | None, key: str) -> EqualizerSpec | None:
    """Editable band/step/range fields. Prefilled from `prefill` when given."""
    p = prefill
    name = st.text_input("Name", p.name if p else "My EQ", key=f"{key}_name")
    freq_default = (", ".join(_num(f) for f in p.band_freqs_hz) if p
                    else "62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000")
    freq_str = st.text_input("Band frequencies (Hz, comma-separated)",
                             freq_default, key=f"{key}_freqs")
    c1, c2, c3 = st.columns(3)
    gmin = c1.number_input("Min dB", value=float(p.gain_min_db) if p else -12.0,
                           step=1.0, key=f"{key}_gmin")
    gmax = c2.number_input("Max dB", value=float(p.gain_max_db) if p else 12.0,
                           step=1.0, key=f"{key}_gmax")
    step = c3.number_input("Step dB (0 = continuous)",
                           value=float(p.step_db) if p else 1.0,
                           step=0.1, min_value=0.0, key=f"{key}_step")
    has_preamp = st.checkbox("Has a separate preamp / gain slider",
                             value=bool(p.has_preamp) if p else False, key=f"{key}_pre")
    try:
        freqs = sorted(float(x) for x in freq_str.replace(" ", "").split(",") if x)
    except ValueError:
        st.error("Couldn't parse the frequency list.")
        return None
    if not freqs:
        st.error("Add at least one band frequency.")
        return None

    spec = EqualizerSpec(
        name=name, band_freqs_hz=freqs,
        gain_min_db=float(gmin), gain_max_db=float(gmax), step_db=float(step),
        has_preamp=bool(has_preamp),
        notes=(p.notes if p else "Built in the AuraTune manual editor."),
    )
    if st.button("Save to eq_specs/ (reuse it later)", key=f"{key}_save"):
        path = save_spec(_slugify(name), spec)
        st.success(f"Saved {path.name} — it'll be in the dropdown next time.")
    return spec


def _num(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def eq_spec_picker() -> EqualizerSpec | None:
    """Pick a built-in / saved spec, upload a screenshot, or build one by hand."""
    st.subheader("Your EQ app")
    specs = all_specs()
    saved = list(specs.keys())
    options = ["📷 Upload a screenshot…"] + saved + ["Manual…", "(none — just show the curve)"]
    choice = st.selectbox(
        "Which EQ are you dialing in?",
        options,
        index=1 if saved else 0,
        format_func=lambda k: specs[k].name if k in specs else k,
    )

    if choice == "(none — just show the curve)":
        return None

    if choice in specs:
        spec = specs[choice]
        st.caption(
            f"{len(spec.band_freqs_hz)} bands · "
            f"{spec.gain_min_db:+.0f}…{spec.gain_max_db:+.0f} dB · "
            f"{spec.step_db or 'continuous'} dB step"
            + ("" if not spec.has_preamp else " · has preamp")
        )
        if spec.notes:
            st.caption(f"_{spec.notes}_")
        return spec

    if choice == "Manual…":
        return _spec_editor(None, key="manual")

    # --- 📷 upload a screenshot -------------------------------------------
    has_env_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                       or os.environ.get("ANTHROPIC_API_KEY"))
    gemini_key = ""
    if not has_env_key:
        gemini_key = st.text_input(
            "Free Google Gemini API key",
            type="password",
            help="Get one in ~30s at aistudio.google.com/apikey. Used only for "
                 "this session, never saved. Or set GEMINI_API_KEY in your env.",
            placeholder="AIza…",
        )

    upload = st.file_uploader("Photo or screenshot of your EQ",
                              type=["png", "jpg", "jpeg", "webp"])
    if upload is None:
        st.caption("Snap your EQ app's screen. AuraTune reads the bands, step "
                   "size and range, then you confirm and it's set.")
        return None

    st.image(upload, caption="Your upload", width=220)
    media_type = upload.type or "image/png"
    spec_dict, error, model_notes, backend = _read_screenshot_cached(
        upload.getvalue(), media_type, gemini_key)

    if error:
        st.warning(error)
        st.caption("…or enter it by hand:")
        return _spec_editor(None, key="upload_fallback")

    detected = EqualizerSpec.from_dict(spec_dict)
    st.success(f"Read **{detected.name}** — {len(detected.band_freqs_hz)} bands, "
               f"{detected.step_db or 'continuous'} dB step (via {backend}). "
               f"Check it below, then use it.")
    if model_notes:
        st.caption(f"Note from the reader: _{model_notes}_")
    return _spec_editor(detected, key="upload")


st.title("AuraTune")
st.caption("Adaptive Audio Personalization Engine — perception → 3-agent LangGraph → your EQ")

col_left, col_right = st.columns([1, 1.4])

with col_left:
    st.subheader("Scenario")
    scenario_key = st.selectbox(
        "Simulated context",
        list(SCENARIO_LABELS.keys()),
        format_func=lambda k: SCENARIO_LABELS[k],
    )
    user_command = st.text_input(
        "Live command (optional)",
        placeholder="e.g. make voices clearer, less bass",
    )

    if genre_classifier.available():
        model_options = ["auto (best)"] + genre_classifier.list_models()
        genre_model_choice = st.selectbox(
            "Genre model (local ML)", model_options,
            help="For music content, locally classifies the genre/mood into "
                 "one of 8 buckets and leans the EQ curve accordingly. "
                 "Trained on the 114k-track Spotify dataset -- see ml/README.md.",
        )
        genre_model_name = "auto" if genre_model_choice.startswith("auto") else genre_model_choice
    else:
        genre_model_name = "auto"
        st.caption("No trained genre model found — run `python ml/train.py` "
                   "once to enable local ML genre-aware EQ tuning.")

    st.divider()
    eq_spec = eq_spec_picker()

    st.divider()
    run_clicked = st.button("Run adaptation", type="primary")

    st.divider()
    st.subheader("History")
    store = get_store()
    history = store.get_history(USER_ID, limit=8)
    if not history:
        st.caption("No adjustments logged yet.")
    for h in reversed(history):
        st.markdown(f"- **{h['content_type']}** / {h['noise_level']} — {h['explanation']}")

with col_right:
    if run_clicked:
        ambient, content = synth_scenario(scenario_key, SR)
        ctx: Context = classify(ambient, content, SR,
                                 content_type_hint=SCENARIO_CONTENT_TYPE[scenario_key])
        eq = get_eq()

        with st.spinner("Running perception → agents → DSP…"):
            result = run_pipeline(store, eq, USER_ID, ctx, user_command,
                                  equalizer_spec=eq_spec, content_audio=content,
                                  sample_rate=SR, genre_model=genre_model_name)

        st.subheader("Detected context")
        c1, c2, c3 = st.columns(3)
        c1.metric("Noise level", ctx.noise_level)
        c2.metric("Content type", ctx.content_type)
        c3.metric("Ambient level", f"{ctx.ambient_rms_db} dB")

        proj = result.get("projected_eq")

        if result.get("genre_bucket"):
            st.subheader("Detected genre (local ML model)")
            g1, g2, g3 = st.columns(3)
            g1.metric("Genre bucket", result["genre_bucket"].replace("_", " / "))
            g2.metric("Confidence", f"{result.get('genre_confidence', 0) * 100:.0f}%")
            g3.metric("Model used", result.get("genre_model_used", "-"))
            probs = result.get("genre_probabilities") or {}
            if probs:
                st.bar_chart(dict(sorted(probs.items(), key=lambda kv: -kv[1])))
        elif result.get("genre_unavailable_reason"):
            st.caption(f"Genre model unavailable: {result['genre_unavailable_reason']}")

        st.subheader("Live EQ curve")
        freqs_before, mag_before = eq.frequency_response(result["baseline_curve"])
        freqs_after, mag_after = eq.frequency_response(result["decided_curve"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freqs_before, y=mag_before, name="Stored baseline",
                                  line=dict(dash="dash", color="gray")))
        fig.add_trace(go.Scatter(x=freqs_after, y=mag_after, name="Live adapted curve",
                                  line=dict(color="#2E86AB", width=3)))
        if proj is not None:
            fig.add_trace(go.Scatter(
                x=[b.freq_hz for b in proj.bands],
                y=[b.set_gain_db for b in proj.bands],
                name=f"{proj.spec_name} sliders",
                mode="markers+lines",
                line=dict(color="#E8871E", width=1, dash="dot"),
                marker=dict(size=10, color="#E8871E"),
            ))
        fig.update_xaxes(type="log", title="Frequency (Hz)")
        fig.update_yaxes(title="Gain (dB)")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        if proj is not None:
            st.subheader(f"Set these on {proj.spec_name}")
            st.table(proj.as_table_rows())
            note = f"Curve fit within ±{proj.fit_error_db:.1f} dB of the ideal across bands."
            if proj.clipped_freqs:
                note += (" Some bands hit the app's range limit — that's the closest "
                         "it can get.")
            if not proj.has_preamp:
                note += ("  *This app has no preamp; the value is how much to lower the "
                         "media/app volume so the boosts don't clip.")
            st.caption(note)

            txt = "\n".join(f"{r['Frequency']}\t{r['Set to (dB)']}" for r in proj.as_table_rows())
            st.download_button("Download these settings (.txt)", txt,
                               file_name=f"{_slugify(proj.spec_name)}_settings.txt")

        st.subheader("Explanation")
        st.info(result["explanation"])

        with st.expander("Raw deltas (debug)"):
            dbg = {
                "context_deltas": result["context_deltas"],
                "command_deltas": result["command_deltas"],
                "genre_deltas": result.get("genre_deltas"),
                "genre_proxy_features": result.get("genre_proxy_features"),
                "decided_curve": result["decided_curve"].to_dict(),
            }
            if proj is not None:
                dbg["projected_eq"] = proj.to_dict()
            st.json(dbg)
    else:
        st.caption("Pick a scenario, choose your EQ app, and click **Run adaptation**.")
