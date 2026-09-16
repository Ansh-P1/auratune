# Model Card — Ambient Noise-Type Classifier

**Component:** `agents/noise_agent.py` (via `perception/noise_classifier.py`)
**Trained by:** `ml/train_noise.py` · **Reviewed in:** `ml/model_review_noise.ipynb`

## What it does
Classifies *what kind* of ambient noise is in the room — not just how loud
(the RMS-based quiet/moderate/noisy read in `perception/context_classifier.py`
already handles loudness). A steady vacuum-cleaner drone and a sudden door
slam can both read as "noisy" by RMS alone; this model tells them apart so
`eq_decision_agent.py` can apply a more targeted EQ compensation on top of
the always-on loudness-based deltas.

## Dataset
- **ESC-50** — 2,000 real labeled 5-second environmental sound clips, 50
  balanced classes (40 clips each). [github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50), CC BY-NC 3.0.
- 50 raw classes mapped to **6 EQ-relevant buckets** (`ESC50_TO_BUCKET` in
  `ml/train_noise.py`): `calm_nature`, `domestic_ambient`, `human_activity`,
  `mechanical_drone`, `impulsive_transient`, `traffic_urban`. The mapping is
  intentionally **not balanced** — 520 clips landed in `calm_nature` and
  `domestic_ambient` each, only 160 in `traffic_urban` — because it reflects
  how ESC-50's 50 classes cluster, not an artificial rebalancing.
- 80/20 stratified train/test split, seed 42. No train/inference gap:
  `extract_noise_features()` in `perception/noise_classifier.py` is the
  *exact same function* used to build the training table and to classify a
  live/synth buffer at inference time — not two different approximations of
  the same idea (unlike the genre classifier, see its own card).

## The 13 features
`rms_db, spectral_centroid, spectral_bandwidth, spectral_flatness,
spectral_rolloff, zero_crossing_rate, harmonic_ratio, onset_rate,
mfcc1–mfcc5` — all computed directly from the waveform via `librosa`, real
signal properties, no external metadata.

## Models trained & results (this run, `ml/models/noise_metrics.json`)

| Model | Type | Test accuracy | F1 (macro) |
|---|---|---|---|
| Logistic Regression | linear baseline | 47.5% | 0.423 |
| Random Forest | bagging ensemble | 62.5% | 0.606 |
| **HistGradientBoosting** | **boosting ensemble — best** | **64.25%** | **0.624** |

Random baseline for 6 unbalanced classes ≈ 17–26% depending on how it's
computed; all three models learn real signal, gradient boosting is
strongest, consistent with the genre classifier's models.

### Per-class F1 (best model)
| Bucket | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| domestic_ambient | 0.67 | 0.72 | **0.69** | 104 |
| impulsive_transient | 0.68 | 0.64 | 0.66 | 72 |
| calm_nature | 0.62 | 0.65 | 0.64 | 104 |
| mechanical_drone | 0.64 | 0.62 | 0.63 | 48 |
| traffic_urban | 0.71 | 0.47 | **0.57** | 32 |
| human_activity | 0.55 | 0.57 | 0.56 | 40 |

## Known weak spots
- **Most confused pair:** `traffic_urban → mechanical_drone` (21.9% of true
  `traffic_urban` test clips misclassified this way), followed by
  `human_activity → domestic_ambient` (17.5%). Both pairs share real
  acoustic similarity (steady broadband low-frequency energy; indoor
  activity timbre) *and* both involve the two smallest buckets in the
  dataset — likely a class-imbalance signature stacked on a genuine
  acoustic-similarity one.
- **Improvement lever, ranked by evidence:** gather more clips for
  `traffic_urban` and `human_activity` specifically (ESC-50 fixes 40/class,
  so this means a different source dataset, not re-sampling the same 2,000).
  `class_weight="balanced"` is *not* assumed to help — the genre
  classifier measured it makes every model's macro F1 worse on its own
  validation set — so it should be tested empirically here before being
  turned on, not applied as a default fix.
- Hard 6-way task from 5-second clips; 64% accuracy against a much-lower
  random baseline is a genuinely useful signal, not a "solved" classifier.
- Only trained/tested on ESC-50's curated clips, same real-world
  domain-shift caveat Track 1 (real-time mic capture) is checking for the
  noise classifier specifically.

## Where it plugs in
```
context_classifier (RMS loudness) ──┐
                                     ├──▶ eq_decision_agent ──▶ ...
noise_agent (this model)   ─────────┘
```
Runs whenever `ml/models/` has been generated (`python ml/train_noise.py`);
omitted otherwise with a documented fallback message — the pipeline never
breaks either way.
