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
