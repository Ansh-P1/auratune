"""
Train AuraTune's ambient-noise-type classifier.

Dataset: ESC-50 (2,000 real labeled 5-second environmental sound clips,
50 classes) -- https://github.com/karolpiczak/ESC-50
Downloaded once into ml/data/ESC-50-master/ (see ml/README.md).

Pipeline:
  1. Load ESC-50's metadata, map its 50 raw sound classes down to 6
     EQ-relevant noise buckets (ESC50_TO_BUCKET below) -- a noise
     classifier is only useful to AuraTune if its output space matches
     something the EQ Decision agent can act on (dsp/noise_curves.py).
  2. Extract the 13 real audio features (perception/noise_classifier.py's
     extract_noise_features -- the SAME function used at inference time,
     no proxy-feature gap like the genre classifier has) from every clip.
  3. Train/val/test split (stratified), a PowerTransformer to fix
     skewed features.
  4. Train 3 models against the same split: Logistic Regression (linear
     baseline), Random Forest (bagging ensemble), and
     HistGradientBoostingClassifier (boosting ensemble) -- pick the best
     by test accuracy.
  5. Save every artifact (transformer, label encoder, all 3 trained
     models, a noise_metrics.json report) to ml/models/, prefixed
     noise_ so they sit alongside the genre classifier's artifacts
     without clashing.

Run with:  python ml/train_noise.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import librosa
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PowerTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from perception.noise_classifier import extract_noise_features, NOISE_FEATURE_COLUMNS, NOISE_BUCKETS

_HERE = Path(__file__).parent
ESC50_DIR = _HERE / "data" / "ESC-50-master"
ESC50_META = ESC50_DIR / "meta" / "esc50.csv"
ESC50_AUDIO = ESC50_DIR / "audio"
MODELS_DIR = _HERE / "models"
SEED = 42

# ---------------------------------------------------------------------------
# ESC-50's 50 raw sound classes -> 6 EQ-relevant noise buckets. Every one
# of ESC-50's classes is assigned exactly once; see dsp/noise_curves.py
# for what each bucket actually changes about the EQ curve. A
# simplification for EQ-tuning purposes, not an acoustic-scene taxonomy.
# ---------------------------------------------------------------------------
ESC50_TO_BUCKET = {
    # Animals -- ambient wildlife vs. sudden vocalizations
    "pig": "calm_nature", "cow": "calm_nature", "frog": "calm_nature",
    "cat": "calm_nature", "hen": "calm_nature", "insects": "calm_nature",
    "sheep": "calm_nature", "dog": "impulsive_transient",
    "rooster": "impulsive_transient", "crow": "impulsive_transient",
    # Natural soundscapes & water
    "rain": "calm_nature", "sea_waves": "calm_nature", "crackling_fire": "calm_nature",
    "crickets": "calm_nature", "chirping_birds": "calm_nature", "wind": "calm_nature",
    "water_drops": "domestic_ambient", "pouring_water": "domestic_ambient",
    "toilet_flush": "domestic_ambient", "thunderstorm": "impulsive_transient",
    # Human, non-speech sounds
    "crying_baby": "human_activity", "sneezing": "human_activity",
    "clapping": "human_activity", "coughing": "human_activity",
    "laughing": "human_activity", "breathing": "domestic_ambient",
    "footsteps": "domestic_ambient", "brushing_teeth": "domestic_ambient",
    "snoring": "domestic_ambient", "drinking_sipping": "domestic_ambient",
    # Interior/domestic sounds
    "washing_machine": "mechanical_drone", "vacuum_cleaner": "mechanical_drone",
    "door_wood_knock": "impulsive_transient", "clock_alarm": "impulsive_transient",
    "glass_breaking": "impulsive_transient", "mouse_click": "domestic_ambient",
    "keyboard_typing": "domestic_ambient", "door_wood_creaks": "domestic_ambient",
    "can_opening": "domestic_ambient", "clock_tick": "domestic_ambient",
    # Exterior/urban noises
    "siren": "traffic_urban", "car_horn": "traffic_urban", "engine": "traffic_urban",
    "train": "traffic_urban", "helicopter": "mechanical_drone", "chainsaw": "mechanical_drone",
    "airplane": "mechanical_drone", "hand_saw": "mechanical_drone",
    "church_bells": "impulsive_transient", "fireworks": "impulsive_transient",
}


def build_feature_table() -> pd.DataFrame:
    """Extract the 13 real audio features for every ESC-50 clip. Slow
    (2000 files) but one-time; nothing here is cached across runs since
    it's already fast enough (a couple of minutes)."""
    meta = pd.read_csv(ESC50_META)
    unmapped = set(meta["category"].unique()) - set(ESC50_TO_BUCKET.keys())
    if unmapped:
        raise ValueError(f"ESC50_TO_BUCKET is missing a mapping for: {sorted(unmapped)}")

    rows = []
    t0 = time.time()
    for i, row in meta.iterrows():
        path = ESC50_AUDIO / row["filename"]
        y, sr = librosa.load(path, sr=None, mono=True)
        feats = extract_noise_features(y, sr)
        feats["noise_bucket"] = ESC50_TO_BUCKET[row["category"]]
        feats["category"] = row["category"]
        rows.append(feats)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  extracted {i + 1}/{len(meta)} clips ({elapsed:.0f}s elapsed)")

    df = pd.DataFrame(rows)
    missing_buckets = set(NOISE_BUCKETS) - set(df["noise_bucket"].unique())
    if missing_buckets:
        raise ValueError(f"No rows ended up in bucket(s): {missing_buckets}")
    return df


def main():
    np.random.seed(SEED)
    MODELS_DIR.mkdir(exist_ok=True)

    if not ESC50_META.exists():
        raise SystemExit(
            f"ESC-50 metadata not found at {ESC50_META}. "
            f"See ml/README.md for the one-time download + unzip step."
        )

    print(f"Extracting features from {ESC50_AUDIO} ...")
    df = build_feature_table()
    print(f"\n{len(df)} clips, across {df['noise_bucket'].nunique()} noise buckets:")
    print(df["noise_bucket"].value_counts().to_string())

    X = df[NOISE_FEATURE_COLUMNS].values.astype(np.float32)
    label_enc = LabelEncoder().fit(NOISE_BUCKETS)
    y = label_enc.transform(df["noise_bucket"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"\nSplit: {len(X_train)} train / {len(X_test)} test")

    scaler = PowerTransformer(method="yeo-johnson").fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # --- Model 1: Logistic Regression (linear baseline) --------------------
    print("\n[1/3] Logistic Regression ...")
    t0 = time.time()
    logreg = LogisticRegression(max_iter=2000)
    logreg.fit(X_train_s, y_train)
    pred = logreg.predict(X_test_s)
    results["logistic_regression"] = {
        "accuracy": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "train_seconds": round(time.time() - t0, 1),
    }
    joblib.dump(logreg, MODELS_DIR / "noise_logistic_regression.joblib")
    print(f"  acc={results['logistic_regression']['accuracy']:.4f}  "
          f"f1_macro={results['logistic_regression']['f1_macro']:.4f}")

    # --- Model 2: Random Forest (bagging ensemble) --------------------------
    print("\n[2/3] Random Forest (bagging) ...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=2,
                                n_jobs=-1, random_state=SEED)
    rf.fit(X_train_s, y_train)
    pred = rf.predict(X_test_s)
    results["random_forest"] = {
        "accuracy": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "train_seconds": round(time.time() - t0, 1),
    }
    joblib.dump(rf, MODELS_DIR / "noise_random_forest.joblib")
    print(f"  acc={results['random_forest']['accuracy']:.4f}  "
          f"f1_macro={results['random_forest']['f1_macro']:.4f}")

    # --- Model 3: Gradient boosting (boosting ensemble) ----------------------
    print("\n[3/3] HistGradientBoostingClassifier (boosting) ...")
    t0 = time.time()
    hgb = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.08, max_depth=None, max_leaf_nodes=63,
        l2_regularization=0.2, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=20, random_state=SEED,
    )
    hgb.fit(X_train_s, y_train)
    pred = hgb.predict(X_test_s)
    results["gradient_boosting"] = {
        "accuracy": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "train_seconds": round(time.time() - t0, 1),
        "n_iterations": hgb.n_iter_,
    }
    joblib.dump(hgb, MODELS_DIR / "noise_gradient_boosting.joblib")
    print(f"  acc={results['gradient_boosting']['accuracy']:.4f}  "
          f"f1_macro={results['gradient_boosting']['f1_macro']:.4f}  "
          f"({hgb.n_iter_} boosting rounds)")

    # --- save shared artifacts + report -------------------------------------
    joblib.dump(scaler, MODELS_DIR / "noise_scaler.joblib")
    joblib.dump(label_enc, MODELS_DIR / "noise_label_encoder.joblib")

    best_model = max(results, key=lambda k: results[k]["accuracy"])
    report = {
        "dataset": "ESC-50 (github.com/karolpiczak/ESC-50), 2000 real labeled clips",
        "n_rows_used": len(df),
        "feature_columns": NOISE_FEATURE_COLUMNS,
        "noise_buckets": NOISE_BUCKETS,
        "results": results,
        "best_model": best_model,
    }
    (MODELS_DIR / "noise_metrics.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print(f"{'Model':<22}{'Accuracy':<12}{'F1 (macro)':<12}{'Train time'}")
    for name, r in results.items():
        marker = " <- best" if name == best_model else ""
        print(f"{name:<22}{r['accuracy']:<12.4f}{r['f1_macro']:<12.4f}{r['train_seconds']:.1f}s{marker}")
    print("=" * 60)

    pred_best = {"logistic_regression": logreg, "random_forest": rf,
                 "gradient_boosting": hgb}[best_model].predict(X_test_s)
    print(f"\nDetailed classification report ({best_model}):")
    print(classification_report(y_test, pred_best, target_names=label_enc.classes_))
    print(f"\nSaved all artifacts to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
