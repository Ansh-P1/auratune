# Model Card — Genre/Mood Classifier

**Component:** `agents/genre_agent.py` (via `perception/genre_classifier.py`)
**Trained by:** `ml/train.py` · **Reviewed in:** `ml/model_review.ipynb`

> **Note on the numbers below:** this card documents the numbers already
> recorded in `ml/README.md` and `ml/model_review.ipynb`'s saved outputs.
> Unlike the noise-classifier card, these were **not re-verified by a fresh
> training run in this pass** — the dataset (huggingface.co) sits outside
> this environment's network allowlist, only `github.com`/`pypi.org`-class
> hosts were reachable here. Re-run `python ml/train.py` locally (after
> fetching the dataset per `ml/README.md`) to regenerate `ml/models/metrics.json`
> and confirm these first-hand; they should match closely run to run given
> the fixed seed, with the usual small numeric variance the README notes.

## What it does
When the currently playing content is music, predicts one of 8 EQ-relevant
genre/mood buckets from the audio itself; the predicted bucket's hand-tuned
EQ deltas (`dsp/genre_curves.py`) blend into the curve, weighted by the
classifier's own confidence.

## Dataset
- **Spotify Tracks Dataset** — 114,000 real tracks, 114 raw genre tags,
  audio features from Spotify's own audio-analysis API.
  [huggingface.co/datasets/maharshipandya/spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
- 114 raw tags mapped to **8 EQ-relevant buckets** (`GENRE_TO_BUCKET` in
  `ml/train.py`): `electronic_dance`, `rock_metal`, `hiphop_rnb`, `pop`,
  `acoustic_folk`, `classical_jazz`, `chill_ambient`, `world_latin`.
- Standard train/val/test split, seed-controlled.

## The 13 features
`danceability, energy, key_sin, key_cos, loudness, mode, speechiness,
acousticness, instrumentalness, liveness, valence, tempo, time_signature`
— Spotify's own audio-analysis fields, with categorical `key` (0–11)
cyclically encoded as `key_sin`/`key_cos` instead of fed in as a raw ordinal
number (a real accuracy fix made during the project's own tuning pass, not
a hypothetical one — see `ml/README.md`).

**Important gap the noise classifier doesn't have:** these 13 features come
from Spotify's own private analysis pipeline, unavailable at inference
time on raw audio. `perception/genre_classifier.py` computes **signal-derived
proxies** for each feature via `librosa` instead. Two features (`liveness`,
`time_signature`) aren't reliably estimable from a short buffer at all and
are pinned to typical training-set values rather than guessed — see that
module's docstring for the honest per-feature accuracy notes.

## Models trained & results (from `ml/README.md` / saved `ml/model_review.ipynb`)

| Model | Type | Test accuracy | F1 (macro) |
|---|---|---|---|
| Logistic Regression | linear baseline | ~40% | ~0.32 |
| **HistGradientBoosting** | **classical ML ensemble — best** | **~53%** | ~0.49 |
| GenreMLP (PyTorch, 200 epochs) | neural network | ~48% | ~0.42 |

Random baseline for 8 balanced-ish classes ≈ 12.5%.

**Accuracy-tuning pass documented in `ml/README.md`:** cyclical key
encoding + Yeo-Johnson `PowerTransformer` (fixes right-skew in
speechiness/acousticness/instrumentalness) + a tuned
`HistGradientBoostingClassifier` took the best model from ~50% to ~53%.
`class_weight="balanced"` was tried for all 3 models and **measured worse**
on both accuracy and macro F1 for every one on the validation set — left
off based on that evidence.

## Known weak spots
- This is a genuinely hard task: even Spotify's own 114-genre labels
  overlap heavily in audio-feature space (a "chill" synth-pop track and an
  "acoustic" ballad can score similarly on energy/valence/acousticness).
- The proxy-feature approximation at inference time is the single biggest
  source of accuracy loss vs. the training-time numbers above — the model
  is only ever as good as `perception/genre_classifier.py`'s approximation
  of Spotify's real features from raw audio.
- `liveness` and `time_signature` are pinned constants at inference time,
  not measured — any track whose true values differ a lot from the
  training-set typical values loses signal on those two features
  specifically.

## Where it plugs in
```
profile_agent → genre_agent (this model) → eq_decision_agent → projection_agent → explainer_agent
```
No-op unless `context.content_type == "music"` *and* `ml/models/` exists
(i.e. `python ml/train.py` has run at least once); the pipeline behaves
exactly as it did before this feature existed if either is missing.
`eq_decision_agent.py` blends in the predicted bucket's deltas scaled by
`GENRE_BLEND_WEIGHT * confidence`.
