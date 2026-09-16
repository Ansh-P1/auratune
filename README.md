# AuraTune — Adaptive Audio Personalization Engine

**Live app:** [auratune-eq.streamlit.app](https://auratune-eq.streamlit.app)

Real-time, explainable EQ personalization. A perception layer reads the
room (ambient noise) and the content (podcast / music / movie), a 3-agent
LangGraph pipeline turns that plus your stored hearing profile into a
target EQ curve, and a plain-English sentence tells you why it changed.
It also reads a photo of *your actual EQ app* and tells you the exact
slider values to set.

Built to match the architecture in `Grp_186__PPT.pptx`:

```
Audio + ambient mic  ──▶  Perception module   ──▶  LangGraph pipeline   ──▶  Parametric EQ engine  ──▶  Streamlit dashboard
(playing content +        (Demucs stems +          (Profile → EQ            (applies the                (live curve +
 room noise)                context classifier)      decision → Explainer)   target curve)               explanations)
```

## Project layout

```
auratune/
├── app.py                        # Streamlit dashboard (entry point)
├── config.py                     # env-driven settings
├── dsp/
│   ├── parametric_eq.py          # biquad EQ engine + frequency response
│   ├── equalizer_spec.py         # describes a real EQ (bands, step, range) + presets
│   └── eq_projection.py          # snap the ideal curve onto a real EQ's sliders
├── perception/
│   ├── context_classifier.py     # noise level + content type
│   ├── eq_app_reader.py          # screenshot of an EQ app -> EqualizerSpec (vision)
│   ├── stem_separation.py        # Demucs wrapper (falls back to HPSS)
│   ├── synth_scenarios.py        # synthetic audio for demos/validation
│   ├── live_capture.py           # real mic capture (sounddevice) for real-time mode
│   ├── genre_classifier.py       # loads ml/models/, predicts genre from real audio
│   └── noise_classifier.py       # loads ml/models/, predicts ambient noise type
├── agents/
│   ├── profile_agent.py          # reads/writes MongoDB profile
│   ├── noise_agent.py            # ambient audio -> local ML noise-type prediction
│   ├── genre_agent.py            # music content -> local ML genre/mood prediction
│   ├── eq_decision_agent.py      # context + noise + genre + command -> target curve
│   ├── projection_agent.py       # target curve -> your EQ app's exact slider values
│   ├── explainer_agent.py        # deltas -> one plain-English sentence
│   ├── llm_client.py             # thin Anthropic API wrapper w/ fallback
│   └── graph.py                  # LangGraph wiring of the agents
├── data/
│   └── db.py                     # MongoDB w/ local-JSON fallback
├── eq_specs/                      # one JSON per real EQ app (e.g. from a screenshot)
├── ml/                            # local ML models -- see ml/README.md
│   ├── train.py                  # genre/mood: 3 models on the 114k-track Spotify dataset
│   ├── model_def.py              # shared neural-net architecture + genre buckets
│   ├── model_review.ipynb        # pre-run notebook reviewing all 3 genre models
│   ├── train_noise.py            # noise-type: 3 models (incl. bagging + boosting) on ESC-50
│   ├── noise_model_def.py        # noise buckets + feature columns
│   └── models/                   # trained artifacts (gitignored -- run train*.py)
├── validation/
│   └── generate_traces.py        # generates the 3 required validation traces
└── tests/
    ├── test_dsp.py
    ├── test_eq_projection.py
    ├── test_eq_app_reader.py
    ├── test_classifier.py
    ├── test_genre_classifier.py
    └── test_noise_classifier.py
```

## How to use the app

1. **Try it live** — open the deployed link at the top of this README, no
   install needed.
2. **Or run it yourself** (see [Setup](#setup) + [Run the dashboard](#run-the-dashboard)
   below), then:
   - Pick a **scenario** (quiet room + podcast / noisy environment + music /
     home + movie), or **🎙️ Real-time (10s mic capture)** to record 10
     seconds from your actual microphone and adapt to the real room instead
     of a simulated one (see [Real-time mic mode](#real-time-mic-mode)).
   - Optionally type a **live command** in plain English ("make voices
     clearer", "less bass, this room is boomy").
   - Under **Your EQ app**, tell it which EQ you actually have — upload a
     screenshot, pick a preset, or enter it by hand (see
     [Adapting to your actual EQ app](#adapting-to-your-actual-eq-app)).
   - Click **Run adaptation**. You'll get: the detected room/content
     context, the live vs. stored EQ curve, the exact slider values for
     your EQ app, and a one-sentence explanation of what changed and why.
3. Repeat with a different scenario or command — the **History** panel on
   the left keeps a running log of every adaptation.

## Setup

```bash
git clone https://github.com/Ansh-P1/auratune.git
cd auratune
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Everything runs with **zero external services** out of the box:
- No `ANTHROPIC_API_KEY`? Set `GROQ_API_KEY` instead and those two steps run
  on Groq's OpenAI-compatible API (`openai/gpt-oss-120b` by default);
  Anthropic wins when both are set. With neither, the EQ Decision agent uses
  keyword-rule command parsing and the Explainer agent uses a templated
  sentence. Either key can go in a local `.env` (gitignored) instead of your
  shell — see `config.py`. Which one actually ran is shown per run in the
  dashboard's explanation badge and Agent trace (`docs/agent_pipeline.md`).
- No `MONGO_URI` (or Mongo unreachable)? The profile store falls back to a
  local JSON file (`data/profiles.local.json`).
- No network access to Meta's model hub for Demucs weights? Stem separation
  falls back to `librosa`'s harmonic-percussive split.

Set these to unlock the full pipeline:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # free-text commands + natural explanations (+ EQ-screenshot reading)
export GROQ_API_KEY=gsk_...           # same two features, if you don't have an Anthropic key
export GEMINI_API_KEY=AIza...         # EQ-screenshot reading, free tier (aistudio.google.com/apikey)
export MONGO_URI=mongodb+srv://...    # persistent, shared profile storage
```

## Run the dashboard

```bash
streamlit run app.py
```

Pick a scenario (quiet+podcast / noisy+music / home+movie), optionally type
a live command ("make voices clearer", "less bass"), and click **Run
adaptation** to see the live vs. stored EQ curve and the agent's explanation.

## Adapting to your actual EQ app

The agents reason about a smooth target curve, but every real EQ is a fixed
grid — N sliders at fixed frequencies, a limited gain range, a fixed step
(1 dB, 0.5 dB, or Wavelet's 0.1 dB). Under **Your EQ app** in the dashboard,
tell AuraTune which EQ you have. It then samples the ideal curve at your
slider frequencies, snaps each to what your app can dial in, and works out a
preamp so the boosts don't clip — the output is a table of exact values plus
a matching sentence:

> "…On your Wavelet EQ, set 4 kHz +4.4, 62.5 Hz −4.3, 2 kHz +3.5… and drop
> the volume −4.4 dB."

Three ways to set your EQ, in the **Your EQ app** dropdown:

1. **📷 Upload a screenshot** — snap your EQ app's screen; a vision model
   (`perception/eq_app_reader.py`) reads the band frequencies, step, and
   range, you eyeball the parsed values, and it's set. Uses **Google Gemini**
   (free tier — get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
   and paste it into the field, or set `GEMINI_API_KEY`) or `ANTHROPIC_API_KEY`
   if that's what you have. With neither, it falls back to manual entry.
2. **Presets** — `wavelet_9band`, `iso_10band`, `spotify_5band`, `car_3band`
   (in `dsp/equalizer_spec.py`), plus any JSON in `eq_specs/`.
3. **Manual** — type the band list, step, and range. "Save to eq_specs/"
   keeps it in the dropdown for next time.

Claude can also write a spec for you outside the app: send it a screenshot
and it drops a `<name>.json` in `eq_specs/`. See `eq_specs/README.md`.

## Local ML genre/mood classifier

For music content, AuraTune can classify the genre/mood into one of 8
EQ-relevant buckets (electronic/dance, rock/metal, hip-hop/R&B, pop,
acoustic/folk, classical/jazz, chill/ambient, world/latin) using a model
**trained locally on real data** -- no API calls, no network access needed
at inference time. Three models are trained on the same split (Logistic
Regression, a tuned `HistGradientBoostingClassifier`, and a PyTorch
neural network, 200 epochs) on the
[Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
(114,000 real tracks) -- **~53% test accuracy** on the best model (8-way
classification; random baseline is 12.5%). The best-performing one is
used by default, and the predicted bucket's hand-tuned EQ deltas
(`dsp/genre_curves.py`) get blended into the curve, weighted by the
model's own confidence. `ml/model_review.ipynb` is a pre-run notebook
reviewing all 3 models -- comparison charts, confusion matrix, training
curve, feature importance, a live classification demo.

```bash
mkdir -p ml/data
curl -L "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv" \
  -o ml/data/spotify_tracks_raw.csv
python ml/train.py
```

Then just run the app as usual -- if `ml/models/` exists, the "Genre
model (local ML)" picker appears in the sidebar and the Genre agent
(`agents/genre_agent.py`) runs automatically for music scenarios. Skip
this step and everything still works exactly as before -- it's a clean,
optional add-on. Full write-up (dataset, genre-bucket mapping rationale,
model comparison, and how live audio is turned into the model's input
features): **[`ml/README.md`](ml/README.md)**.

## Local ML ambient-noise-type classifier

A second, independent local ML component: instead of just *how loud* the
room is (the always-on RMS-based quiet/moderate/noisy read), a model
classifies *what kind* of ambient noise is present -- one of 6 buckets
(calm nature, domestic ambient, human activity, mechanical drone, impulsive
transient, traffic/urban) -- and refines the EQ curve on top of the
loudness-based adjustment. Trained on **2,000 real labeled clips** from the
[ESC-50 dataset](https://github.com/karolpiczak/ESC-50). Same 3-model
setup as the genre classifier -- Logistic Regression baseline, a **bagging**
Random Forest, and a **boosting** `HistGradientBoostingClassifier` -- and
the best one is picked automatically:

| Model | Type | Test accuracy | Test F1 (macro) |
|---|---|---|---|
| Logistic Regression | linear baseline | ~48% | ~0.42 |
| Random Forest | bagging ensemble | ~63% | ~0.61 |
| HistGradientBoostingClassifier | boosting ensemble | **~64%** | ~0.62 |

(Random baseline for 6 balanced-ish classes is ~17%; see
`ml/models/noise_metrics.json` after training for the exact run.) Unlike
the genre classifier, there's no proxy-feature gap here -- training and
inference call the exact same `extract_noise_features()` function.

```bash
mkdir -p ml/data
curl -L "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip" \
  -o ml/data/esc50_master.zip
unzip ml/data/esc50_master.zip -d ml/data/
python ml/train_noise.py
```

Full write-up: **[`ml/README.md`](ml/README.md)**.

### Real-time mic mode

Selecting **🎙️ Real-time (10s mic capture)** in the Scenario picker and
clicking **Run adaptation** records 10 real seconds from your default
microphone (`perception/live_capture.py`) and runs it through the noise
classifier above -- genuinely live, not simulated. There's no separate
"what's playing" feed in this mode (a single mic can't isolate content
from room noise the way two synthetic buffers can), so you tell it what
to expect (podcast/music/movie) from a dropdown; genre classification is
skipped for that run since it needs a real content signal, but noise-type
classification runs at full fidelity. If no microphone is available
(`perception/live_capture.py`'s `is_available()` returns `False`), the
option still appears but shows a friendly error instead of crashing.

Real-time mode also has a **"Test with a YouTube video"** field — paste a
link and `st.video()` embeds the player right there, so you can play
something through your speakers for the mic to pick up without leaving
the tab. It's convenience playback only, not a direct audio feed: a
browser can't read a YouTube iframe's audio from the surrounding page
(cross-origin security sandboxing applies to embedded video the same way
it would to any other site), so AuraTune "hears" it exactly the way a
real microphone would hear anything else playing in the room.

## Run the validation traces

Reproduces the 3 traces from the project's own validation plan:

```bash
python3 validation/generate_traces.py
```

Outputs to `validation/output/`: one PNG per scenario (before/after curve)
plus `report.md` with the detected context, numeric deltas, and the
Explainer agent's sentence for each.

| Scenario | Expected behavior | Verified |
|---|---|---|
| Quiet room + podcast | Minimal compensation; curve near stored podcast target | ✅ "No change needed" |
| Noisy environment + music | Vocal/presence boost, bass pulled back to avoid noise stacking | ✅ +3.5 dB presence, −2.5 dB bass |
| Home + movie | Stored movie target curve, isolating content-type switching | ✅ correctly switches curve on content type |

## Run unit tests

```bash
python3 tests/test_dsp.py
python3 tests/test_eq_projection.py
python3 tests/test_eq_app_reader.py
python3 tests/test_classifier.py           # needs librosa installed
python3 tests/test_genre_classifier.py     # needs librosa installed
python3 tests/test_noise_classifier.py     # needs librosa installed
```

## Notes on what's real vs. simulated in this build

This was built in a sandboxed environment without live mic hardware, a
running MongoDB instance, an API key, or network access to Meta's model
hub — so those integration points are implemented for real and tested
against **graceful, documented fallbacks**, not stubbed out:

- **Ambient + content audio**: `perception/synth_scenarios.py` synthesizes
  representative audio (speech-like envelope for podcast, stable harmonic
  stack for music, dialogue+FX bursts for movie) standing in for real mic
  capture / media playback, for the simulated scenarios. Real mic capture
  is also implemented for real (`perception/live_capture.py`, via
  `sounddevice`) and wired into the dashboard's **🎙️ Real-time (10s mic
  capture)** mode — content audio (what's *playing*) still isn't captured,
  since that needs OS-level loopback/virtual-cable setup that's
  platform-specific; ambient room audio is captured for real.
- **Content-type classification**: the noise-level detector is fully
  signal-derived and tested (`tests/test_classifier.py`). Content-type
  classification uses a real heuristic (syllable-rate envelope modulation,
  spectral flatness, harmonic/percussive ratio) but, per the project's own
  roadmap ("agents wired end-to-end on mock context" before a trained
  classifier exists), the demo scenarios pass a `content_type_hint` — the
  same way a real player would hand over its own metadata rather than
  guess. The heuristic itself is exercised directly in the test suite.
- **Demucs**: real integration, with an automatic fallback to
  `librosa.effects.hpss` if pretrained weights can't be fetched.
- **Claude calls**: real `anthropic` SDK calls when `ANTHROPIC_API_KEY` is
  set; deterministic rule-based fallbacks otherwise, so the graph always
  produces a usable result.
- **MongoDB**: real `pymongo` integration with a transparent local-JSON
  fallback for offline dev.
- **Genre/mood classifier**: real models trained on a real, large,
  third-party dataset (114k tracks) -- not stubbed or mocked. The one
  approximation, clearly documented in `perception/genre_classifier.py`,
  is that the model's input features come from Spotify's own private
  audio-analysis pipeline, so at inference time this project computes
  signal-derived *proxies* for them from the actual audio via `librosa`
  (11 of 13 are genuinely estimated from the signal; 2 -- liveness and
  time signature -- aren't reliably estimable from a short buffer and are
  pinned to typical values rather than guessed). Same "real feature,
  honestly-documented approximation" pattern as Demucs -> HPSS above.
- **Noise-type classifier**: real models trained on a real, third-party
  dataset (2,000 ESC-50 clips) -- no proxy-feature gap at all, since both
  training and inference call the exact same `extract_noise_features()`
  (see `ml/README.md`).

Everything else — the DSP, the classifier's noise detection, the LangGraph
wiring, the EQ decision logic, the Streamlit UI — runs for real, no mocking.

## Deploy it live (free)

Puts a real `https://` link in front of the dashboard — anyone can open it,
no install needed.

1. Go to [share.streamlit.io](https://share.streamlit.io) and **sign in with
   GitHub** (you'll be asked to authorize Streamlit Cloud — that's between
   you and GitHub).
2. **New app** → pick this repo (`auratune`) → branch `main` → main file
   path `app.py` → **Deploy**.
3. Optional — add secrets for the full pipeline: app **Settings → Secrets**,
   paste e.g.
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   GEMINI_API_KEY = "AIza..."
   ```
   Without them the app still runs end-to-end on its rule-based fallbacks
   (see [Setup](#setup)).
4. Copy the `https://<something>.streamlit.app` link it gives you and put
   it at the top of this README (the **Live app** line).
