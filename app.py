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
from perception.live_capture import decode_browser_audio, MicUnavailableError
from perception import genre_classifier
from perception import noise_classifier
from data.db import ProfileStore
from agents.graph import run_pipeline
from agents.llm_client import PROVIDER_LABELS
from agents.spec_edit_parser import parse_spec_edit

st.set_page_config(page_title="AuraTune", page_icon="🎧", layout="wide")

SR = 44100
USER_ID = "demo_user"

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
DARK = st.session_state.dark_mode

# ---------------------------------------------------------------------------
# Visual theme -- soft, minimalist, with a light/dark toggle (see the
# toggle widget near the title). .streamlit/config.toml sets the base
# Streamlit theme (fixed at server start, can't change at runtime); this
# CSS layer is what actually switches on the fly, driven by
# st.session_state.dark_mode. __VARS__/__SHADOW__/__PRIMARY_TEXT__/__CHIP_COLOR__ are
# plain string placeholders (not an f-string) so the CSS itself never
# needs its braces escaped.
# ---------------------------------------------------------------------------
def _build_css(dark: bool) -> str:
    if dark:
        vars_css = """
    --at-primary: #D5B893;
    --at-primary-soft: #3D3320;
    --at-accent: #FFB37E;
    --at-accent-soft: #3A2A1E;
    --at-ink: #E7E9F5;
    --at-muted: #C5CAE9;
    --at-card: #1B1E30;
    --at-border: #2D3150;
    --at-bg: #12141F;
"""
        shadow = "0 2px 14px rgba(0, 0, 0, 0.35)"
        primary_text = "#25344F"
        chip_color = "#FFB37E"
    else:
        vars_css = """
    --at-primary: #743014;
    --at-primary-soft: #F2DFDA;
    --at-accent: #E8925A;
    --at-accent-soft: #F3E4D3;
    --at-ink: #3A3428;
    --at-muted: #8C8370;
    --at-card: #FFFCF6;
    --at-border: #E8DFC9;
    --at-bg: #F5EFE1;
"""
        shadow = "0 2px 10px rgba(45, 49, 66, 0.04)"
        primary_text = "#ffffff"
        chip_color = "#A6551F"

    template = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root __VARS__
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
[data-testid="stAppViewContainer"], .stApp { background: var(--at-bg) !important; }
[data-testid="stHeader"] { background: var(--at-bg) !important; }
.block-container { padding-top: 3.5rem; padding-bottom: 3rem; max-width: 1200px; }
h1 { font-weight: 700 !important; color: var(--at-ink) !important; letter-spacing: -0.02em; }
h2, h3, h4 { font-weight: 600 !important; color: var(--at-ink) !important; }
p, span, label, .stMarkdown, small, div, li { color: var(--at-ink); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * ,
small, [data-testid="stMarkdownContainer"] em, .stCaption {
    color: var(--at-muted) !important;
}
[data-testid="stWidgetLabel"] p { color: var(--at-ink) !important; }
[data-testid="stTooltipIcon"] { color: var(--at-muted) !important; }
[data-testid="stMarkdownContainer"] { color: var(--at-ink); }
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--at-border) !important;
    border-radius: 18px !important;
    background: var(--at-card);
    box-shadow: __SHADOW__;
    padding: 0.25rem 0.25rem;
    margin-bottom: 1rem;
}
[data-testid="stMetric"] {
    background: var(--at-primary-soft);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    border: 1px solid var(--at-border);
}
[data-testid="stMetricLabel"] { color: var(--at-muted) !important; }
[data-testid="stMetricValue"] {
    color: var(--at-ink) !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    line-height: 1.3 !important;
    word-break: break-word !important;
}
div.stButton > button, div.stDownloadButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
    border: none !important;
    background: var(--at-primary-soft) !important;
    color: var(--at-ink) !important;
}
div.stButton > button[kind="primary"] {
    background: var(--at-primary) !important;
    box-shadow: none !important;
    color: __PRIMARY_TEXT__ !important;
}
/* The button's label sits in a nested <p>, which the global
   "p, span, label... { color: var(--at-ink) }" rule further down also
   matches -- being the more specific (innermost) element, it was winning
   over the button's own color and making the text hard to read. */
div.stButton > button[kind="primary"] p { color: __PRIMARY_TEXT__ !important; }
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: none !important;
}
[data-testid="stToggle"] label p { color: var(--at-ink) !important; }
/* Streamlit's selectbox moved to a react-aria ComboBox with no [data-baseweb]
   attributes at all -- the old BaseWeb selectors above matched nothing.
   Targeting by role instead, since that's the stable part across versions. */
[data-testid="stSelectbox"] input[role="combobox"] { color: var(--at-ink) !important; }
[data-testid="stSelectbox"] [role="group"] {
    background: var(--at-card) !important;
    border-color: var(--at-border) !important;
}
.stTextInput input, .stNumberInput input, .stFileUploader section {
    border-radius: 12px !important;
    border-color: var(--at-border) !important;
    background: var(--at-card) !important;
    color: var(--at-ink) !important;
}
/* Browsers dim placeholder text by default (often ~50% opacity), which on
   top of an already-muted colour made it too faint to read against the
   dark card background. Forcing full opacity here. */
.stTextInput input::placeholder, .stNumberInput input::placeholder {
    color: var(--at-muted) !important;
    opacity: 1 !important;
}
/* The open dropdown list renders in a portal appended to <body>, not inside
   the app tree -- selecting it structurally by its listbox child since its
   own wrapper has no stable class or role. */
body div:has(> [role="listbox"]) { background: var(--at-card) !important; border: 1px solid var(--at-border) !important; }
[role="option"] { background: transparent !important; color: var(--at-ink) !important; }
[role="option"]:hover, [role="option"][data-hovered="true"] { background: var(--at-primary-soft) !important; }
[data-testid="stAlert"] { border-radius: 14px !important; background: var(--at-primary-soft) !important; color: var(--at-ink) !important; }
[data-testid="stExpander"] { border-radius: 14px !important; border-color: var(--at-border) !important; overflow: hidden; background: var(--at-card) !important; }
[data-testid="stExpander"] summary { color: var(--at-ink) !important; }
[data-testid="stTable"] { border-radius: 12px; overflow: hidden; }
[data-testid="stTable"] table, [data-testid="stTable"] th, [data-testid="stTable"] td {
    background: var(--at-card) !important; color: var(--at-ink) !important; border-color: var(--at-border) !important;
}
[data-testid="stArrowVegaLiteChart"] { background: #FFFCF6 !important; border-radius: 10px; padding: 8px; }
.at-chip {
    display: inline-block;
    background: var(--at-accent-soft);
    color: __CHIP_COLOR__;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 0.3rem 0.6rem;
    border-radius: 999px;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
    line-height: 1.4;
}
.llm-badge {
    display: inline-block;
    background: var(--at-primary-soft);
    color: var(--at-primary);
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    border: 1px solid var(--at-border);
    margin-top: 0.55rem;
    float: right;
}
.trace-meta { color: var(--at-muted); font-size: 0.78rem; }
.trace-summary { color: var(--at-ink); font-size: 0.9rem; }
</style>
"""
    return (template.replace("__VARS__", "{" + vars_css + "}")
                    .replace("__SHADOW__", shadow)
                    .replace("__PRIMARY_TEXT__", primary_text)
                    .replace("__CHIP_COLOR__", chip_color))


st.markdown(_build_css(DARK), unsafe_allow_html=True)


@st.cache_resource
def get_store():
    return ProfileStore()


@st.cache_resource
def get_eq():
    return ParametricEQ(SR)


def _slugify(name: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in name]
    return "".join(keep).strip("_") or "custom_eq"


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
# Shorter single-word labels for the probability bar chart, where multi-word
# labels wrap/clip on the narrow rotated category axis.
_GENRE_CHART_NAMES = {
    "electronic_dance": "Electronic", "rock_metal": "Rock", "hiphop_rnb": "Hip-Hop",
    "pop": "Pop", "acoustic_folk": "Acoustic", "classical_jazz": "Classical",
    "chill_ambient": "Chill", "world_latin": "World",
}


def _model_label(name: str) -> str:
    return _MODEL_DISPLAY_NAMES.get(name, name)


def _genre_label(bucket: str) -> str:
    return _GENRE_DISPLAY_NAMES.get(bucket, bucket.replace("_", " / "))


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

    _NL_FIELD_TO_WIDGET = {"gain_min_db": "gmin", "gain_max_db": "gmax", "step_db": "step"}
    nl_col, nl_btn_col = st.columns([5, 1])
    nl_text = nl_col.text_input(
        "Or just describe it in plain English",
        placeholder='e.g. "my range is -76.5 to 7.5 dB" or "step is 0.5 dB"',
        key=f"{key}_nl",
    )
    nl_btn_col.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    if nl_btn_col.button("Apply", key=f"{key}_nl_apply"):
        if nl_text.strip():
            changes = parse_spec_edit(nl_text)
            if changes:
                for field, value in changes.items():
                    st.session_state[f"{key}_{_NL_FIELD_TO_WIDGET[field]}"] = value
                st.success("Updated " + ", ".join(changes) + " below.")
            else:
                st.caption('Couldn\'t find a number in that -- try e.g. "range -76.5 to 7.5 dB".')

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
        st.success(f"Saved {path.name}. It'll be in the dropdown next time.")
    return spec


def _num(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Generic device-type icons for "Your EQ app". Never a real product photo or
# logo (copyright/trademark risk for a public repo + deployed site) -- just
# hand-drawn line art keyed to the *kind* of device a preset represents, so
# picking "OnePlus Buds" still shows something earbud-shaped without
# claiming to depict the actual product.
# ---------------------------------------------------------------------------
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

_DEVICE_ICON_PATHS = {
    "earbuds": (
        '<circle cx="8" cy="9" r="3"/><path d="M8 12v6"/>'
        '<circle cx="16" cy="9" r="3"/><path d="M16 12v6"/>'
    ),
    "overear": (
        '<path d="M4 13v-1a8 8 0 0 1 16 0v1"/>'
        '<rect x="2.5" y="13" width="4" height="7" rx="1.5"/>'
        '<rect x="17.5" y="13" width="4" height="7" rx="1.5"/>'
    ),
    "generic": (
        '<line x1="5" y1="4" x2="5" y2="20"/><circle cx="5" cy="9" r="1.6" fill="currentColor"/>'
        '<line x1="12" y1="4" x2="12" y2="20"/><circle cx="12" cy="15" r="1.6" fill="currentColor"/>'
        '<line x1="19" y1="4" x2="19" y2="20"/><circle cx="19" cy="11" r="1.6" fill="currentColor"/>'
    ),
}


def _device_icon_svg(spec_key: str) -> str:
    device_type = _DEVICE_TYPES.get(spec_key, "generic")
    color = "#E8D1A7" if DARK else "#743014"
    inner = _DEVICE_ICON_PATHS[device_type]
    return (
        f'<svg width="44" height="44" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="1.6" stroke-linecap="round" '
        f'stroke-linejoin="round">{inner}</svg>'
    )


def eq_spec_picker() -> EqualizerSpec | None:
    """Pick a built-in / saved spec, upload a screenshot, or build one by hand."""
    st.subheader("Your EQ app")
    specs = all_specs()
    saved = list(specs.keys())
    options = ["Upload a screenshot…"] + saved + ["Manual…", "(none, just show the curve)"]
    choice = st.selectbox(
        "Which EQ are you dialing in?",
        options,
        index=len(options) - 1,  # neutral by default -- don't assume a device the user may not own
        format_func=lambda k: specs[k].name if k in specs else k,
    )
    icon_l, icon_r = st.columns([1, 6])
    with icon_l:
        st.markdown(_device_icon_svg(choice), unsafe_allow_html=True)
    with icon_r:
        device_type = _DEVICE_TYPES.get(choice, "generic")
        label = {"earbuds": "Earbuds", "overear": "Over-ear headphones"}.get(
            device_type, "Generic EQ app")
        st.caption(f"_{label} (generic icon, not the actual product)_"
                   if device_type != "generic" else f"_{label}_")

    if choice == "(none, just show the curve)":
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
        with st.expander("✏️ Edit values (e.g. match your app's actual dB range)"):
            st.caption("Presets are a starting point — your actual app may differ "
                       "(e.g. Wavelet's preamp can go down to −76.5 dB, not the "
                       "±12 dB shown above). Adjust the range, step, or bands to "
                       "match what your app really shows, then optionally save it "
                       "as its own preset below.")
            edited = _spec_editor(spec, key=f"edit_{_slugify(choice)}")
            if edited is not None:
                spec = edited
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
    st.success(f"Read **{detected.name}**: {len(detected.band_freqs_hz)} bands, "
               f"{detected.step_db or 'continuous'} dB step (via {backend}). "
               f"Check it below, then use it.")
    if model_notes:
        st.caption(f"Note from the reader: _{model_notes}_")
    return _spec_editor(detected, key="upload")


_, top_r = st.columns([6, 1])
with top_r:
    st.toggle("Dark", key="dark_mode")
st.title("AuraTune")
st.caption("Perception → 3-agent LangGraph → your EQ, explained in plain English")

REALTIME_KEY = "__realtime__"
store = get_store()

row1_left, row1_right = st.columns([1, 1], gap="medium")

with row1_left:
    with st.container(border=True):
        st.subheader("Scenario")
        scenario_options = list(SCENARIO_LABELS.keys()) + [REALTIME_KEY]
        scenario_labels = {**SCENARIO_LABELS, REALTIME_KEY: "Real-time (10s mic capture)"}
        scenario_key = st.selectbox(
            "Simulated context",
            scenario_options,
            format_func=lambda k: scenario_labels[k],
        )

        realtime_content_hint = "music"
        realtime_audio = None
        if scenario_key == REALTIME_KEY:
            realtime_content_hint = st.selectbox(
                "What's playing? (real-time mode has no separate content feed,"
                " so tell it what to expect)",
                ["podcast", "music", "movie"], index=1,
            )
            st.caption("Record a clip below -- speak, play music, or just let the "
                       "room's ambient noise through -- then click Run adaptation. "
                       "Recording happens in your browser, so this works even on a "
                       "deployed server with no microphone of its own.")
            realtime_audio = st.audio_input("🎙️ Record ambient audio", key="realtime_mic_input")

            youtube_url = st.text_input(
                "▶ Test with a YouTube video (optional)",
                placeholder="https://www.youtube.com/watch?v=…",
                help="Plays the video through your speakers so the 10-second mic "
                     "capture above can pick it up -- a convenience for testing, "
                     "not a direct audio feed. A browser can't read a YouTube "
                     "iframe's audio from the page around it (cross-origin "
                     "security), so real-time mode always \"hears\" it the same "
                     "way your actual microphone would.",
            )
            if youtube_url.strip():
                st.video(youtube_url.strip())

        user_command = st.text_input(
            "Live command (optional)",
            placeholder="e.g. make voices clearer, less bass",
        )

        if genre_classifier.available():
            model_options = ["auto (best)"] + genre_classifier.list_models()
            genre_model_choice = st.selectbox(
                "Genre model (local ML)", model_options,
                format_func=lambda k: k if k.startswith("auto") else _model_label(k),
                help="For music content, locally classifies the genre/mood into "
                     "one of 8 buckets and leans the EQ curve accordingly. "
                     "Trained on the 114k-track Spotify dataset -- see ml/README.md.",
            )
            genre_model_name = "auto" if genre_model_choice.startswith("auto") else genre_model_choice
        else:
            genre_model_name = "auto"
            st.caption("Genre-aware tuning isn't available in this deployment. "
                       "Your EQ still adapts based on room noise and content type.")

        if noise_classifier.available():
            noise_model_options = ["auto (best)"] + noise_classifier.list_models()
            noise_model_choice = st.selectbox(
                "Noise model (local ML)", noise_model_options,
                format_func=lambda k: k if k.startswith("auto") else _model_label(k),
                help="Classifies the ambient/room noise into one of 6 buckets "
                     "(calm nature, domestic, human activity, mechanical drone, "
                     "impulsive/transient, traffic/urban) and leans the EQ curve "
                     "accordingly. Trained on the real ESC-50 dataset -- see ml/README.md.",
            )
            noise_model_name = "auto" if noise_model_choice.startswith("auto") else noise_model_choice
        else:
            noise_model_name = "auto"
            st.caption("Noise-type tuning isn't available in this deployment. "
                       "Your EQ still adapts based on overall noise level and content type.")

with row1_right:
    with st.container(border=True):
        eq_spec = eq_spec_picker()

run_clicked = st.button("▶  Run adaptation", type="primary", use_container_width=True)

if run_clicked:
    mic_error = None
    content = None
    if scenario_key == REALTIME_KEY:
        if realtime_audio is None:
            mic_error = ("No audio recorded yet -- click the microphone icon "
                         "above, record a clip, then click Run adaptation again.")
            ambient = None
        else:
            try:
                with st.spinner("Decoding your recording…"):
                    ambient = decode_browser_audio(realtime_audio.getvalue(), SR)
                content_type_hint = realtime_content_hint
            except MicUnavailableError as exc:
                mic_error = str(exc)
                ambient = None
    else:
        ambient, content = synth_scenario(scenario_key, SR)
        content_type_hint = SCENARIO_CONTENT_TYPE[scenario_key]

    if mic_error:
        st.session_state["last_error"] = mic_error
        st.session_state.pop("last_result", None)
    else:
        ctx: Context = classify(ambient, content if content is not None else ambient,
                                SR, content_type_hint=content_type_hint)
        eq = get_eq()

        with st.spinner("Running perception → agents → DSP…"):
            result = run_pipeline(store, eq, USER_ID, ctx, user_command,
                                  equalizer_spec=eq_spec, content_audio=content,
                                  ambient_audio=ambient, sample_rate=SR,
                                  genre_model=genre_model_name, noise_model=noise_model_name)

        # Stash rather than render here: run_pipeline() above already wrote
        # this run's entry to the store, but the History panel (further down
        # this same script pass) already rendered from the store *before*
        # that write happened. Rerunning is what lets History pick up the
        # fresh entry immediately instead of only on the next unrelated click.
        st.session_state["last_result"] = {"ctx": ctx, "result": result}
        st.session_state.pop("last_error", None)
    st.rerun()

if st.session_state.get("last_error"):
    st.error(f"Couldn't capture from the microphone: {st.session_state['last_error']}")
elif st.session_state.get("last_result"):
    ctx = st.session_state["last_result"]["ctx"]
    result = st.session_state["last_result"]["result"]
    eq = get_eq()
    proj = result.get("projected_eq")

    # Summary-first: the headline result (why, curve, exact slider moves) is
    # its own tab so it's the first thing visible -- not buried under
    # detection cards, the agent trace, and debug JSON in one long scroll.
    tab_result, tab_detected, tab_trace, tab_debug = st.tabs(
        ["Result", "Detected", "Agent trace", "Debug"])

    with tab_result:
        with st.container(border=True):
            exp_source = result.get("explanation_source", "template")
            if exp_source == "llm":
                model = result.get("explanation_model", "an LLM")
                provider = PROVIDER_LABELS.get(result.get("explanation_provider"), "")
                badge = f"🤖 Written by {model}" + (f" ({provider})" if provider else "")
            else:
                badge = "📋 Written by the built-in template"
            head, tag = st.columns([3, 2])
            head.subheader("💬 Explanation")
            tag.markdown(f"<div class='llm-badge'>{badge}</div>",
                         unsafe_allow_html=True)
            st.info(result["explanation"])
            cmd_source = result.get("command_parse_source", "none")
            if cmd_source != "none":
                st.caption("Your typed command was parsed by "
                           + ("**the LLM**." if cmd_source == "llm"
                              else "**keyword rules** (no API key, or the call failed)."))

        with st.container(border=True):
            st.subheader("Live EQ curve")
            freqs_before, mag_before = eq.frequency_response(result["baseline_curve"])
            freqs_after, mag_after = eq.frequency_response(result["decided_curve"])

            curve_color = "#E8D1A7" if DARK else "#743014"
            baseline_color = "#454A6E" if DARK else "#D6CBAE"
            grid_color = "#2D3150" if DARK else "#EFE7D4"
            plot_bg = "#1B1E30" if DARK else "#FFFCF6"
            text_color = "#E7E9F5" if DARK else "#3A3428"

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=freqs_before, y=mag_before, name="Stored baseline",
                                      line=dict(dash="dash", color=baseline_color)))
            fig.add_trace(go.Scatter(x=freqs_after, y=mag_after, name="Live adapted curve",
                                      line=dict(color=curve_color, width=3)))
            if proj is not None:
                fig.add_trace(go.Scatter(
                    x=[b.freq_hz for b in proj.bands],
                    y=[b.set_gain_db for b in proj.bands],
                    name=f"{proj.spec_name} sliders",
                    mode="markers+lines",
                    line=dict(color="#F5A56B", width=1, dash="dot"),
                    marker=dict(size=10, color="#F5A56B"),
                ))
            fig.update_xaxes(type="log", title="Frequency (Hz)", gridcolor=grid_color)
            fig.update_yaxes(title="Gain (dB)", gridcolor=grid_color)
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                               legend=dict(orientation="h", y=1.1),
                               plot_bgcolor=plot_bg, paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="Inter, sans-serif", color=text_color))
            st.plotly_chart(fig, use_container_width=True)

        if proj is not None:
            with st.container(border=True):
                st.subheader(f"Set these on {proj.spec_name}")
                st.table(proj.as_table_rows())
                note = f"Curve fit within ±{proj.fit_error_db:.1f} dB of the ideal across bands."
                if proj.clipped_freqs:
                    note += (" Some bands hit the app's range limit. That's the closest "
                             "it can get.")
                if not proj.has_preamp:
                    note += ("  *This app has no preamp; the value is how much to lower the "
                             "media/app volume so the boosts don't clip.")
                st.caption(note)

                txt = "\n".join(f"{r['Frequency']}\t{r['Set to (dB)']}" for r in proj.as_table_rows())
                st.download_button("⬇  Download these settings (.txt)", txt,
                                   file_name=f"{_slugify(proj.spec_name)}_settings.txt")

    with tab_detected:
        with st.container(border=True):
            st.subheader("Detected context")
            c1, c2, c3 = st.columns(3)
            c1.metric("Noise level", ctx.noise_level)
            c2.metric("Content type", ctx.content_type)
            c3.metric("Ambient level", f"{ctx.ambient_rms_db} dB")

        if result.get("noise_bucket"):
            with st.container(border=True):
                st.subheader("Detected noise type (local ML model)")
                n1, n2, n3 = st.columns(3)
                n1.metric("Noise bucket", result["noise_bucket"].replace("_", " ").title())
                n2.metric("Confidence", f"{result.get('noise_confidence', 0) * 100:.0f}%")
                n3.metric("Model used", _model_label(result.get("noise_model_used", "-")))
                nprobs = result.get("noise_probabilities") or {}
                if nprobs:
                    nlabeled = {k.replace("_", " ").title(): v for k, v in
                               sorted(nprobs.items(), key=lambda kv: -kv[1])}
                    st.bar_chart(nlabeled, color="#E8925A")
        elif result.get("noise_unavailable_reason"):
            # The raw reason (result['noise_unavailable_reason']) is a
            # developer-facing string -- local file paths, errno text --
            # set in agents/noise_agent.py. Deliberately not shown to the
            # user here; see agents/noise_agent.py if it needs to change.
            st.caption("Noise-type tuning isn't available in this deployment. "
                       "Your EQ still adapts based on overall noise level and content type.")

        if result.get("genre_bucket"):
            with st.container(border=True):
                st.subheader("Detected genre (local ML model)")
                g1, g2, g3 = st.columns(3)
                g1.metric("Genre bucket", _genre_label(result["genre_bucket"]))
                g2.metric("Confidence", f"{result.get('genre_confidence', 0) * 100:.0f}%")
                g3.metric("Model used", _model_label(result.get("genre_model_used", "-")))
                probs = result.get("genre_probabilities") or {}
                if probs:
                    labeled = {_GENRE_CHART_NAMES.get(k, k): v for k, v in
                              sorted(probs.items(), key=lambda kv: -kv[1])}
                    st.bar_chart(labeled, color="#E8D1A7" if DARK else "#743014")
        elif result.get("genre_unavailable_reason"):
            # Same deliberate choice as the noise reason above -- see
            # agents/genre_agent.py for the raw (developer-facing) string.
            st.caption("Genre-aware tuning isn't available in this deployment. "
                       "Your EQ still adapts based on room noise and content type.")

    with tab_trace:
        with st.container(border=True):
            st.subheader("🧭 Agent trace")
            st.caption("One row per LangGraph node, in the order it ran.")
            for step in result.get("agent_trace", []):
                icon = "⏭️" if step["skipped"] else "✅"
                llm_tag = ""
                for call in step["llm_calls"]:
                    llm_tag = (f" · 🤖 {call['provider_label']}" if call["used_llm"]
                               else " · 📋 fallback")
                st.markdown(
                    f"**{icon} {step['step']}. {step['label']}** "
                    f"<span class='trace-meta'>{step['duration_ms']:.0f} ms{llm_tag}</span><br>"
                    f"<span class='trace-summary'>{step['summary']}</span>",
                    unsafe_allow_html=True)
                with st.expander(f"Details — {step['description']}"):
                    st.json(step["detail"])

        llm_calls = [c for s in result.get("agent_trace", []) for c in s["llm_calls"]]
        with st.expander("🔎 LLM prompts (dev view)"):
            if not llm_calls:
                st.caption("No LLM call was attempted this run — the explainer "
                           "always tries one, so this is unexpected.")
            for call in llm_calls:
                status = {"ok": "✅ The model answered",
                          "no_api_key": "📋 No API key — deterministic fallback used",
                          "error": "⚠️ Call failed — deterministic fallback used"}.get(
                              call["status"], call["status"])
                st.markdown(f"**{call['purpose']}** — {status}"
                            + (f" · `{call['model']}` ({call['provider_label']})" if call["model"] else "")
                            + (f" · {call['latency_ms']:.0f} ms"
                               if call["latency_ms"] else ""))
                if call["error"]:
                    st.caption(call["error"])
                st.caption("System prompt")
                st.code(call["system_prompt"], language="text")
                st.caption("User prompt")
                st.code(call["user_prompt"], language="text")
                if call["response"]:
                    st.caption("Response")
                    st.code(call["response"], language="text")
                st.divider()
            st.caption("API keys are stripped from everything shown here "
                       "(agents/llm_client.py `redact()`).")

    with tab_debug:
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
        st.markdown("#### Ready when you are")
        st.caption("Pick a scenario and your EQ app above, then click "
                   "**Run adaptation** to see the live curve and explanation appear here.")

with st.container(border=True):
    st.subheader("History")
    history = store.get_history(USER_ID, limit=8)
    if not history:
        st.caption("No adjustments logged yet.")
    for h in reversed(history):
        st.markdown(f"- **{h['content_type']}** / {h['noise_level']}: {h['explanation']}")
