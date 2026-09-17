# 🎧 AuraTune — Adaptive Audio Personalization Engine

<p align="center">

  [![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-auratune--eq.streamlit.app-FF4B4B?style=for-the-badge)](https://auratune-eq.streamlit.app)
  [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
  [![LangGraph](https://img.shields.io/badge/LangGraph-Agents-1C9E7C?style=flat-square)](https://langchain-ai.github.io/langgraph/)
  [![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](#)
</p>

<p align="center">
  <b>Real-time, explainable EQ that adapts to your room, your content, and your ears.</b><br>
  <i>Perception → 3-Agent LangGraph → Parametric EQ → One plain-English sentence why.</i>
</p>

---

### ✨ What it does

| | |
|---|---|
| 🎙️ **Hears the room** | Detects ambient noise level + *type* (6 classes via local ML) |
| 🎬 **Knows the content** | Podcast / Music / Movie + genre/mood (8 buckets, 114k-track model) |
| 🧠 **Decides** | LangGraph pipeline blends profile + context + your live command (`"less bass"`) into a target curve |
| 🎚️ **Maps to YOUR EQ** | Snaps the ideal curve onto your actual app's sliders — upload a screenshot and it reads it |
| 💬 **Explains** | One sentence, no jargon — plus a full agent trace |

```
🎵 Audio + 🎙️ Ambient Mic  →  👁 Perception (Demucs / HPSS + Classifiers)  →  🤖 LangGraph (Profile → Decision → Explainer)  →  🎛️ Parametric EQ  →  📊 Streamlit
```

---

### ⚡ Quick Start

```bash
git clone https://github.com/Ansh-P1/auratune.git && cd auratune
python3 -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

> **Zero-config.** No keys, no DB, no GPU needed — everything falls back gracefully. Add keys to unlock the full pipeline ↓

---

### 🎛️ Try it

1. Pick a **Scenario** — `Quiet + Podcast` / `Noisy + Music` / `Home + Movie` or **🎙️ Real-time (10s mic capture)** for your actual room
2. (optional) Type a command — *“make voices clearer”*, *“less bass, room is boomy”*
3. Tell it **Your EQ app** — screenshot / preset / manual
4. Hit **▶ Run adaptation** → live vs stored curve + exact slider values + one-sentence *why*

History panel on the left keeps every run.

---

### 🎚️ Your Real EQ — 3 ways

Every EQ is a fixed grid (N bands, range, step). AuraTune samples the ideal curve at your frequencies, snaps to your step, and computes a safe preamp.

| Method | How |
|---|---|
| **📷 Screenshot** | Upload your EQ screen → Gemini (free tier) or Claude reads bands/step/range → you confirm |
| **📦 Preset** | `wavelet_9band` · `iso_10band` · `spotify_5band` · `car_3band` + any `eq_specs/*.json` |
| **✏️ Manual** | Type bands + range + step → *Save to eq_specs/* for next time |

> Tip: Presets are starting points — edit range/step to match what your app *actually* shows (e.g. Wavelet preamp → −76.5 dB).

---

### 🧠 Local ML — On-device, no API

| Classifier | Dataset | Models trained | Best | Accuracy* |
|---|---|---|---|---|
| **Genre / Mood** (8 buckets) | 114k Spotify tracks | Logistic Reg · HistGradientBoosting · PyTorch NN | **~53%** | random = 12.5% |
| **Noise Type** (6 buckets) | 2k ESC-50 clips | Logistic Reg · Random Forest (bagging) · HGB (boosting) | **~64%** | random = ~17% |

Blended into the curve weighted by confidence. Fully optional — pipeline works without it.

<details>
<summary><b>▶ Train locally</b></summary>

```bash
# Genre
mkdir -p ml/data
curl -L "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv" -o ml/data/spotify_tracks_raw.csv
python ml/train.py

# Noise type
curl -L "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip" -o ml/data/esc50_master.zip
unzip ml/data/esc50_master.zip -d ml/data/
python ml/train_noise.py
```
Artifacts go to `ml/models/` (gitignored). Full write-up → [`ml/README.md`](ml/README.md) · review notebooks: `ml/model_review.ipynb`

</details>

---

### 🔧 Configuration

All optional — app runs without any of them:

| Variable | Purpose | Fallback |
|---|---|---|
| `ANTHROPIC_API_KEY` | LLM decisions + explanations + screenshot | Keyword rules + template |
| `GROQ_API_KEY` | Same as above via `openai/gpt-oss-120b` | — |
| `GEMINI_API_KEY` | Screenshot reading (free tier, [get key](https://aistudio.google.com/apikey)) | Claude or manual entry |
| `MONGO_URI` | Persistent profile storage | `data/profiles.local.json` |

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or GROQ_API_KEY=gsk_...
export GEMINI_API_KEY=AIza...         # for screenshot reading
export MONGO_URI=mongodb+srv://...    # for shared profiles
# or put them in a local .env (gitignored) — see config.py
```

---

### ✅ Validate & Test

```bash
python3 validation/generate_traces.py   # 3 traces → validation/output/ (PNG curves + report.md)
python3 tests/test_dsp.py
python3 tests/test_eq_projection.py
python3 tests/test_classifier.py        # needs librosa
```

| Scenario | Expected | Status |
|---|---|---|
| Quiet + Podcast | Minimal compensation, near stored curve | ✅ |
| Noisy + Music | Presence boost +3.5 dB, bass −2.5 dB | ✅ |
| Home + Movie | Switches to movie target curve | ✅ |

---

### 🚀 Deploy free (Streamlit Cloud)

1. [share.streamlit.io](https://share.streamlit.io) → **Sign in with GitHub** → **New app** → repo `auratune` · branch `main` · file `app.py` → **Deploy**
2. (optional) **Settings → Secrets** → paste `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` as TOML
3. Copy your `https://*.streamlit.app` link to the top of this README

---

<details>
<summary><b>📁 Project Structure</b></summary>

```
app.py                    → Streamlit dashboard (entry point)
config.py                 → env-driven settings
dsp/                      → parametric_eq.py · equalizer_spec.py · eq_projection.py
perception/               → context/genre/noise classifiers · stem separation (Demucs→HPSS) · live_capture · eq_app_reader
agents/                   → profile · noise · genre · eq_decision · projection · explainer + graph.py (LangGraph)
data/db.py                → MongoDB with local-JSON fallback
eq_specs/                 → one JSON per real EQ app
ml/                       → train.py · train_noise.py · model_def.py · models/ (gitignored)
validation/               → generate_traces.py
tests/                    → test_*.py
```

</details>

<details>
<summary><b>🔍 What's real vs. simulated?</b></summary>

Built sandbox-first, so every integration is real with a documented fallback — not a stub:

- **Audio** — `synth_scenarios.py` for demos; `live_capture.py` via `sounddevice` for real mic (browser `audio_input` on deployed site)
- **Demucs** — real weights when reachable, else `librosa.effects.hpss`
- **LLM** — real Anthropic/Groq/Gemini calls when keys set, else deterministic rules
- **MongoDB** — real `pymongo`, else transparent local JSON
- **Genre ML** — 114k real tracks; 11/13 features estimated from signal via `librosa`, 2 pinned (see `ml/README.md`)
- **Noise ML** — 2k ESC-50 clips; zero proxy gap — same `extract_noise_features()` in train & inference

DSP, LangGraph wiring, and UI run for real, no mocking.

</details>

---

<p align="center">
  <sub>Built for the architecture in <code>Grp_186__PPT.pptx</code> · Feedback & issues → <a href="https://github.com/anomalyco/opencode">opencode</a></sub>
</p>
