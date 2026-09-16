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

Frontend layer lives in ui/ (theme, components, charts) -- this file is
layout + orchestration only. The agents/dsp/perception/data backend is
untouched from the original build.

Run with: streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

from dsp.parametric_eq import ParametricEQ
from dsp.equalizer_spec import EqualizerSpec, all_specs, save_spec
from perception.eq_app_reader import read_equalizer_screenshot
from perception.context_classifier import classify, Context
from perception.synth_scenarios import synth_scenario, SCENARIOS as SCENARIO_LABELS, SCENARIO_CONTENT_TYPE
from perception.live_capture import record_ambient, is_available as mic_is_available, MicUnavailableError
from perception import genre_classifier
from perception import noise_classifier
from data.db import ProfileStore
from agents.graph import run_pipeline

from ui import theme
from ui import components as c
from ui import charts

st.set_page_config(page_title="AuraTune", page_icon="🎧", layout="wide",
                   initial_sidebar_state="collapsed")

SR = 44100
USER_ID = "demo_user"
REALTIME_KEY = "__realtime__"

st.session_state.setdefault("dark_mode", True)
DARK = st.session_state.dark_mode
theme.inject(DARK)

# ---------------------------------------------------------------------------
# Cached resources / lookups
# ---------------------------------------------------------------------------
@st.cache_resource
def get_store():
    return ProfileStore()


@st.cache_resource
def get_eq():
    return ParametricEQ(SR)


def _slugify(name: str) -> str:
    keep = [ch.lower() if ch.isalnum() else "_" for ch in name]
    return "".join(keep).strip("_") or "custom_eq"


def _num(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


_MODEL_DISPLAY_NAMES = {
    "gradient_boosting": "Gradient Boost",
    "neural_net": "Neural Net",
    "logistic_regression": "Logistic Reg.",
}

_GENRE_DISPLAY_NAMES = {
    "electronic_dance": "Electronic",
    "rock_metal": "Rock / Metal",
    "hiphop_rnb": "Hip-Hop / R&B",
    "pop": "Pop",
    "acoustic_folk": "Acoustic",
    "classical_jazz": "Classical / Jazz",
    "chill_ambient": "Chill",
    "world_latin": "World / Latin",
}

_GENRE_CHART_NAMES = {
    "electronic_dance": "Electronic", "rock_metal": "Rock", "hiphop_rnb": "Hip-Hop",
    "pop": "Pop", "acoustic_folk": "Acoustic", "classical_jazz": "Classical",
    "chill_ambient": "Chill", "world_latin": "World",
}

_DEVICE_TYPES = {
    "apple_music_10band": "earbuds",
    "bose_music_app": "overear",
    "google_pixel_buds_pro_5band": "earbuds",
    "nothing_x_advanced_8band": "earbuds",
    "oneplus_heymelody_5band": "earbuds",
    "samsung_soundalive_9band": "earbuds",
    "sennheiser_smart_control": "overear",
    "sony_headphones_app": "overear",
    "soundcore_app_8band": "overear",
}


def _model_label(name: str) -> str:
    return _MODEL_DISPLAY_NAMES.get(name, name)


def _genre_label(bucket: str) -> str:
    return _GENRE_DISPLAY_NAMES.get(bucket, bucket.replace("_", " / "))


@st.cache_data(show_spinner="Reading your EQ screenshot…")
def _read_screenshot_cached(image_bytes: bytes, media_type: str, gemini_key: str):
    res = read_equalizer_screenshot(image_bytes, media_type, gemini_key=gemini_key or None)
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
    if st.button("Save to eq_specs/ (reuse it later)", key=f"{key}_save", type="primary"):
        path = save_spec(_slugify(name), spec)
        st.success(f"Saved {path.name} — it'll be in the dropdown next time.")
    return spec


def eq_spec_picker() -> EqualizerSpec | None:
    """Pick a built-in / saved spec, upload a screenshot, or build one by hand."""
    specs = all_specs()
    saved = list(specs.keys())
    options = ["📷 Upload a screenshot…"] + saved + ["Manual…", "(none — just show the curve)"]
    choice = st.selectbox(
        "Which EQ are you dialing in?",
        options,
        index=1 if saved else 0,
        format_func=lambda k: specs[k].name if k in specs else k,
    )

    device_type = _DEVICE_TYPES.get(choice, "generic")
    icon_l, icon_r = st.columns([1, 6])
    with icon_l:
        st.markdown(c.device_icon(device_type), unsafe_allow_html=True)
    with icon_r:
        label = {"earbuds": "Earbuds", "overear": "Over-ear headphones"}.get(
            device_type, "Generic EQ app")
        st.caption(f"_{label} (generic icon, not the actual product)_"
                  if device_type != "generic" else f"_{label}_")

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

    # --- upload a screenshot ------------------------------------------
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


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
hl, hr = st.columns([7, 1])
with hl:
    c.hero()
with hr:
    st.write("")
    st.toggle("🌙 Dark", key="dark_mode")

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1.4], gap="medium")

with col_left:
    with st.container(border=True):
        c.eyebrow("Scenario")
        scenario_options = list(SCENARIO_LABELS.keys()) + [REALTIME_KEY]
        scenario_labels = {**SCENARIO_LABELS, REALTIME_KEY: "Real-time (10s mic capture)"}
        scenario_key = st.selectbox(
            "Simulated context",
            scenario_options,
            format_func=lambda k: scenario_labels[k],
        )

        realtime_content_hint = "music"
        if scenario_key == REALTIME_KEY:
            realtime_content_hint = st.selectbox(
                "What's playing? (real-time mode has no separate content feed,"
                " so tell it what to expect)",
                ["podcast", "music", "movie"], index=1,
            )
            if not mic_is_available():
                st.caption("⚠️ No microphone detected on this machine/browser session — "
                          "Run adaptation will show a friendly error instead of capturing audio.")
            else:
                st.caption("Clicking Run adaptation will record 10 seconds from your "
                          "default microphone — speak, play music, or just let the "
                          "room's ambient noise through.")

        youtube_url = st.text_input(
            "▶ Test with a YouTube video (optional)",
            placeholder="https://www.youtube.com/watch?v=…",
            help="Plays the video through your speakers so the 10-second mic "
                 "capture above can pick it up — a convenience for testing, "
                 "not a direct audio feed.",
        )
        if youtube_url.strip():
            st.video(youtube_url.strip())

        user_command = st.text_input(
            "Live command (optional)",
            placeholder="e.g. make voices clearer, less bass",
        )

    with st.container(border=True):
        t_eq, t_models = st.tabs(["Your EQ app", "ML models"])

        with t_eq:
            eq_spec = eq_spec_picker()

        with t_models:
            if genre_classifier.available():
                model_options = ["auto (best)"] + genre_classifier.list_models()
                genre_model_choice = st.selectbox(
                    "Genre model (local ML)", model_options,
                    format_func=lambda k: k if k.startswith("auto") else _model_label(k),
                    help="For music content, locally classifies the genre/mood into "
                         "one of 8 buckets and leans the EQ curve accordingly. "
                         "Trained on the 114k-track Spotify dataset — see ml/README.md.",
                )
                genre_model_name = "auto" if genre_model_choice.startswith("auto") else genre_model_choice
            else:
                genre_model_name = "auto"
                st.caption("No trained genre model found — run `python ml/train.py` "
                          "once to enable local ML genre-aware EQ tuning.")

            if noise_classifier.available():
                noise_model_options = ["auto (best)"] + noise_classifier.list_models()
                noise_model_choice = st.selectbox(
                    "Noise model (local ML)", noise_model_options,
                    format_func=lambda k: k if k.startswith("auto") else _model_label(k),
                    help="Classifies the ambient/room noise into one of 6 buckets "
                         "(calm nature, domestic, human activity, mechanical drone, "
                         "impulsive/transient, traffic/urban) and leans the EQ curve "
                         "accordingly. Trained on the real ESC-50 dataset — see ml/README.md.",
                )
                noise_model_name = "auto" if noise_model_choice.startswith("auto") else noise_model_choice
            else:
                noise_model_name = "auto"
                st.caption("No trained noise model found — run `python ml/train_noise.py` "
                          "once to enable local ML noise-type-aware EQ tuning.")

        run_clicked = st.button("▶  Run adaptation", type="primary", use_container_width=True)

    with st.container(border=True):
        c.eyebrow("History")
        store = get_store()
        c.timeline(store.get_history(USER_ID, limit=8))

with col_right:
    if run_clicked:
        mic_error = None
        content = None

        if scenario_key == REALTIME_KEY:
            try:
                with st.spinner("🎙️ Listening for 10 seconds…"):
                    ambient = record_ambient(10.0, SR)
                content_type_hint = realtime_content_hint
            except MicUnavailableError as exc:
                mic_error = str(exc)
                ambient = None
        else:
            ambient, content = synth_scenario(scenario_key, SR)
            content_type_hint = SCENARIO_CONTENT_TYPE[scenario_key]

        if mic_error:
            st.error(f"Couldn't capture from the microphone: {mic_error}")
        else:
            ctx: Context = classify(ambient, content if content is not None else ambient,
                                    SR, content_type_hint=content_type_hint)
            eq = get_eq()
            with st.spinner("Running perception → agents → DSP…"):
                result = run_pipeline(store, eq, USER_ID, ctx, user_command,
                                      equalizer_spec=eq_spec, content_audio=content,
                                      ambient_audio=ambient, sample_rate=SR,
                                      genre_model=genre_model_name, noise_model=noise_model_name)

            with st.container(border=True):
                c.eyebrow("Detected context")
                c.stats(
                    [("Noise level", ctx.noise_level),
                     ("Content type", ctx.content_type),
                     ("Ambient", f"{ctx.ambient_rms_db} dB")],
                    hot=1, mono={2},
                )

            proj = result.get("projected_eq")

            if result.get("noise_bucket"):
                with st.container(border=True):
                    c.eyebrow("Noise type · local ML")
                    n1, n2 = st.columns([1.4, 1])
                    with n1:
                        c.stats([
                            ("Bucket", result["noise_bucket"].replace("_", " ").title()),
                            ("Model", _model_label(result.get("noise_model_used", "—"))),
                        ])
                    with n2:
                        c.meter("Confidence", result.get("noise_confidence", 0))
                    nprobs = result.get("noise_probabilities") or {}
                    if nprobs:
                        charts.probability_bars(
                            {k.replace("_", " ").title(): v for k, v in nprobs.items()},
                            accent="warm",
                        )
            elif result.get("noise_unavailable_reason"):
                st.caption(f"Noise model unavailable: {result['noise_unavailable_reason']}")

            if result.get("genre_bucket"):
                with st.container(border=True):
                    c.eyebrow("Genre · local ML")
                    g1, g2 = st.columns([1.4, 1])
                    with g1:
                        c.stats([
                            ("Bucket", _genre_label(result["genre_bucket"])),
                            ("Model", _model_label(result.get("genre_model_used", "—"))),
                        ])
                    with g2:
                        c.meter("Confidence", result.get("genre_confidence", 0))
                    probs = result.get("genre_probabilities") or {}
                    if probs:
                        charts.probability_bars(
                            {_GENRE_CHART_NAMES.get(k, k): v for k, v in probs.items()},
                            accent="accent",
                        )
            elif result.get("genre_unavailable_reason"):
                st.caption(f"Genre model unavailable: {result['genre_unavailable_reason']}")

            with st.container(border=True):
                c.eyebrow("Live EQ curve")
                freqs_before, mag_before = eq.frequency_response(result["baseline_curve"])
                freqs_after, mag_after = eq.frequency_response(result["decided_curve"])
                charts.eq_curve(freqs_before, mag_before, freqs_after, mag_after, proj=proj)

            if proj is not None:
                with st.container(border=True):
                    c.eyebrow(f"Set these on {proj.spec_name}")
                    gmin = eq_spec.gain_min_db if eq_spec is not None else -12.0
                    gmax = eq_spec.gain_max_db if eq_spec is not None else 12.0
                    c.fader_rack(proj.bands, gain_min=gmin, gain_max=gmax,
                               clipped=set(proj.clipped_freqs or []))

                    note = f"Curve fit within ±{proj.fit_error_db:.1f} dB of the ideal across bands."
                    if proj.clipped_freqs:
                        note += (" Bands outlined in red hit the app's range limit — "
                                "that's the closest it can get.")
                    if not proj.has_preamp:
                        note += (" *This app has no preamp; the value is how much to lower the "
                                "media/app volume so the boosts don't clip.")
                    st.caption(note)

                    with st.expander("Values as a table"):
                        st.table(proj.as_table_rows())

                    txt = "\n".join(f"{r['Frequency']}\t{r['Set to (dB)']}"
                                    for r in proj.as_table_rows())
                    st.download_button("⬇ Download these settings (.txt)", txt,
                                       file_name=f"{_slugify(proj.spec_name)}_settings.txt")

            with st.container(border=True):
                c.eyebrow("Explanation")
                st.info(result["explanation"])

            with st.expander("Raw deltas (debug)"):
                dbg = {
                    "context_deltas": result["context_deltas"],
                    "command_deltas": result["command_deltas"],
                    "genre_deltas": result.get("genre_deltas"),
                    "genre_proxy_features": result.get("genre_proxy_features"),
                    "noise_deltas": result.get("noise_deltas"),
                    "noise_features": result.get("noise_features"),
                    "decided_curve": result["decided_curve"].to_dict(),
                }
                if proj is not None:
                    dbg["projected_eq"] = proj.to_dict()
                st.json(dbg)
    else:
        with st.container(border=True):
            c.empty_state()
