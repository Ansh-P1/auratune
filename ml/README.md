# AuraTune ML: local genre/mood classifier

A locally-trained ML component that upgrades the EQ Decision agent's
context handling: when the currently playing content is music, a small
classifier predicts one of 8 EQ-relevant genre/mood buckets from the audio
itself, and the predicted bucket's hand-tuned EQ deltas
(`dsp/genre_curves.py`) get blended into the curve, weighted by the
classifier's own confidence. Everything runs locally, offline, after the
one-time training step below -- no API calls, no network access at
inference time.

## Dataset

**Spotify Tracks Dataset** — 114,000 real tracks, 114 genres, with audio
features pulled from Spotify's own audio-analysis API.
<https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset>

Not committed to the repo (20MB, and it's just a public download — see
`.gitignore`). Fetch it yourself:
```bash
mkdir -p ml/data
curl -L "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv" \
  -o ml/data/spotify_tracks_raw.csv
```

## What gets predicted, and why 8 buckets

The dataset's 114 raw genre tags are far more granular than anything the
EQ Decision agent could usefully act on. `ml/train.py`'s `GENRE_TO_BUCKET`
maps every tag down to one of 8 EQ-relevant buckets: `electronic_dance`,
`rock_metal`, `hiphop_rnb`, `pop`, `acoustic_folk`, `classical_jazz`,
`chill_ambient`, `world_latin`. Each bucket has its own small, hand-tuned
set of bass/presence/treble deltas in `dsp/genre_curves.py` (e.g.
`hiphop_rnb` gets a bass boost + presence lift for vocals;
`classical_jazz` gets almost no EQ, preserving dynamic range). The bucket
mapping is a simplification for EQ-tuning purposes, not a musicological
taxonomy — documented inline in `train.py` where every judgment call is made.

## Models trained (`ml/train.py`)

Three models, same train/val/test split, same 13 input features
(`danceability, energy, key_sin, key_cos, loudness, mode, speechiness,
acousticness, instrumentalness, liveness, valence, tempo, time_signature`
— `key` is Spotify's categorical 0-11 pitch class, cyclically encoded as
`key_sin`/`key_cos` rather than fed in as a raw ordinal number):

| Model | Type | Test accuracy | Test F1 (macro) |
|---|---|---|---|
| Logistic Regression | linear baseline | ~40% | ~0.32 |
| HistGradientBoostingClassifier | classical ML ensemble | **~53%** | ~0.49 |
| GenreMLP (`ml/model_def.py`) | neural network (the "AI model"), 200 epochs, PyTorch, trained on GPU when available | ~48% | ~0.42 |

(Exact numbers vary slightly run to run; see `ml/models/metrics.json`
after training for the numbers actually saved.) Random baseline for 8
balanced classes is 12.5% — all three models learn real signal from just
13 numeric features, with gradient boosting currently the strongest. This
is a genuinely hard task: even Spotify's own 114-genre labels overlap
heavily in audio-feature space (a "chill" synth-pop track and an
"acoustic" ballad can have very similar energy/valence/acousticness), so
these numbers are in line with published benchmarks on this exact dataset.

**Accuracy-tuning pass** (see `ml/model_review.ipynb` for the full
comparison): the original build used a plain `StandardScaler`, a raw
ordinal `key` feature, and a 150-tree Random Forest, scoring ~50%
accuracy. Three changes pushed the best model to ~53%:
1. **Cyclical key encoding** (`key_sin`/`key_cos`) instead of ordinal 0-11.
2. **Yeo-Johnson `PowerTransformer`** instead of `StandardScaler` — several
   features (speechiness, acousticness, instrumentalness) are heavily
   right-skewed (most tracks score near 0), which a plain scaler leaves
   untouched.
3. **`HistGradientBoostingClassifier`** swapped in for Random Forest, with
   hyperparameters (`learning_rate`, `max_leaf_nodes`, `l2_regularization`)
   picked via a small grid search scored against the *validation* set only
   (test set touched exactly once, at the end).

One thing that was tried and measurably **didn't** help: `class_weight=
"balanced"` for all three models, to compensate for the 8 buckets ranging
from ~6k to ~23k rows. Measured on the validation set, it made every
model *worse* on both accuracy and macro F1 — left unweighted based on
that evidence, not assumed as a default best practice.

Run training yourself (~2 minutes on a GPU, ~10-15 min on CPU, dominated
by the gradient-boosting grid the code already has hyperparameters
resolved for — no search happens at train time):
```bash
python ml/train.py
```
Saves `ml/models/{scaler,label_encoder}.joblib`, one artifact per model
(`logistic_regression.joblib`, `gradient_boosting.joblib`, `neural_net.pt`),
and `metrics.json` (which model won, and every model's numbers, including
the neural net's full per-epoch loss history). Not committed to the repo
either — regenerate locally, same as the dataset.

## Reviewing the models (`ml/model_review.ipynb`)

A Jupyter notebook that loads the already-trained models and walks
through: raw dataset exploration, the genre-bucket mapping, the cyclical
key encoding, feature correlations, a model comparison chart, a confusion
matrix, the neural net's training curve, gradient-boosting permutation
feature importance, and a live demo classifying AuraTune's own synthetic
audio. Open it with `jupyter notebook ml/model_review.ipynb` (or in
VS Code / JupyterLab) — it's saved with all outputs already populated, so
it's readable without re-running anything.

---

# AuraTune ML #2: local ambient-noise-type classifier

A second, independent local ML component: for the Noise agent
(`agents/noise_agent.py`), a model that classifies *what kind* of ambient
noise is in the room (not just how loud — the RMS-based quiet/moderate/
noisy read already handles that, see `perception/context_classifier.py`)
and refines the EQ curve accordingly. A steady vacuum-cleaner drone and a
sudden door slam are both "noisy" by RMS alone, but call for different
compensation — this is what tells them apart.

## Dataset

**ESC-50** — 2,000 real labeled 5-second environmental sound clips, 50
classes. <https://github.com/karolpiczak/ESC-50> (CC BY-NC 3.0).

Not committed to the repo (public download, same reasoning as the genre
classifier's dataset). Fetch it yourself:
```bash
mkdir -p ml/data
curl -L "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip" \
  -o ml/data/esc50_master.zip
unzip ml/data/esc50_master.zip -d ml/data/
```
(Ends up at `ml/data/ESC-50-master/audio/*.wav` + `ml/data/ESC-50-master/meta/esc50.csv`.)

## No proxy-feature problem this time

The genre classifier has to *approximate* Spotify's private audio
features from raw audio at inference time (see above). The noise
classifier doesn't have this gap at all: `perception/noise_classifier.py`'s
`extract_noise_features()` is the single, real function used both to
build the training table (on ESC-50 clips, in `ml/train_noise.py`) and to
classify a live/synth ambient buffer at inference time. Train and
inference literally call the same code — not two different
approximations of the same idea.

## What gets predicted, and why 6 buckets

ESC-50's 50 raw classes are mapped down to 6 EQ-relevant buckets
(`ml/train_noise.py`'s `ESC50_TO_BUCKET`): `calm_nature`,
`domestic_ambient`, `human_activity`, `mechanical_drone`,
`impulsive_transient`, `traffic_urban`. Each has its own small,
hand-tuned bass/presence/treble deltas in `dsp/noise_curves.py` (e.g.
`mechanical_drone` — vacuum cleaners, engines, chainsaws — gets a hard
bass cut since a steady low hum muddies the mix; `traffic_urban` gets the
strongest adjustment of the six, matching the classic "noisy street"
case). Blended in *on top of* the existing RMS noise-level deltas, not
instead of them — the loudness read is a reliable always-on floor; the
noise-type model is a confidence-weighted refinement layer.

## Models trained (`ml/train_noise.py`)

Three models, same split, same 13 real audio features (`rms_db,
spectral_centroid, spectral_bandwidth, spectral_flatness,
spectral_rolloff, zero_crossing_rate, harmonic_ratio, onset_rate, mfcc1-5`):

| Model | Type | Notes |
|---|---|---|
| Logistic Regression | linear baseline | |
| Random Forest | **bagging** ensemble | |
| HistGradientBoostingClassifier | **boosting** ensemble | typically the strongest of the three |

The best of the three (by test accuracy) is used by default; see
`ml/models/noise_metrics.json` after training for the actual numbers —
this is a genuinely hard 6-way task from 5-second clips and the numbers
will be honest, not cherry-picked.

Run training yourself (first extracts real audio features from all 2,000
clips via librosa, which takes a couple of minutes, then trains):
```bash
python ml/train_noise.py
```
Saves `ml/models/noise_{scaler,label_encoder}.joblib`, one artifact per
model (`noise_logistic_regression.joblib`, `noise_random_forest.joblib`,
`noise_gradient_boosting.joblib`), and `noise_metrics.json`. Prefixed
`noise_` so they sit alongside the genre classifier's artifacts in the
same `ml/models/` folder without clashing. Not committed to the repo —
regenerate locally, same as the dataset.

## Real-time mode

The Streamlit app's Scenario picker has a **"🎙️ Real-time (10s mic
capture)"** option alongside the synthetic demo scenarios. Selecting it
and clicking **Run adaptation** records 10 seconds from your actual
default microphone (`perception/live_capture.py`, built for exactly this)
and runs it through the real noise classifier above — genuinely live,
not simulated. There's no separate "what's playing" audio feed in this
mode (a mic can't isolate content from room noise the way two separate
synthetic buffers can), so you tell it what to expect (podcast/music/
movie) via a dropdown, and genre classification is skipped for this run
(it needs a real content signal, which real-time mode doesn't have) —
noise-type classification runs at full fidelity regardless.

## Inference on real audio (`perception/genre_classifier.py`)

The trained models expect Spotify's audio features as input (see above)
— those come from Spotify's own internal audio-analysis pipeline, which
isn't public. So at inference time, `perception/genre_classifier.py`
computes signal-derived **proxies** for each feature from the actual
audio buffer via `librosa` — reusing techniques already in this codebase
(the syllable-rate envelope-modulation speech detector from
`perception/context_classifier.py`, the harmonic/percussive split used
for stem separation). Two features (`liveness`, `time_signature`) aren't
reliably estimable from a short buffer at all, so they're pinned to
typical training-set values rather than guessed — see the module
docstring for the honest per-feature accuracy notes on all of them.

This is the same "real feature, documented approximation" pattern the
project already uses for Demucs → HPSS and the vision-based EQ screenshot
reader (see the root `README.md`'s "what's real vs. simulated" section) —
never silently wrong, always a working, labeled fallback.

## Where it plugs into the pipeline

```
profile_agent -> genre_agent -> eq_decision_agent -> projection_agent -> explainer_agent
```
`agents/genre_agent.py` is a no-op unless `context.content_type == "music"`
*and* `ml/models/` has been generated (i.e. `python ml/train.py` has been
run at least once) — omit either and the pipeline behaves exactly as it
did before this feature existed. When it does run, `eq_decision_agent.py`
blends in the predicted bucket's `dsp/genre_curves.py` deltas
(scaled by `GENRE_BLEND_WEIGHT * confidence`), and
`explainer_agent.py` adds a clause naming the detected genre and the
local model that made the call.
