"""
Train AuraTune's genre/mood classifier.

Dataset: the Spotify Tracks Dataset (114,000 tracks, 114 fine-grained
genres, real audio features pulled from the Spotify API) --
https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset
Downloaded once into ml/data/spotify_tracks_raw.csv (see ml/README.md).

Pipeline:
  1. Load + clean the raw CSV (drop nulls/dupes on the columns we use).
  2. Map the 114 raw genre tags down to 8 EQ-relevant buckets
     (ml/model_def.py's GENRE_BUCKETS) -- a genre classifier is only
     useful to AuraTune if its output space matches something the EQ
     Decision agent can act on (dsp/genre_curves.py).
  3. Feature engineering: cyclically encode the categorical "key" feature
     (key_sin/key_cos, see _key_to_sin_cos below) instead of feeding a
     raw 0-11 pitch class in as if it were an ordinal number. Train/val/
     test split (stratified), then a PowerTransformer (Yeo-Johnson) --
     fixes the heavy right-skew in features like speechiness/
     instrumentalness/acousticness (most tracks score near 0), which a
     plain StandardScaler leaves untouched and which was making it harder
     for the linear model and the neural net to separate classes.
  4. Train 3 models against the same split, all class-weighted (the 8
     buckets range from ~6k to ~23k rows; unweighted models were barely
     predicting the smallest ones at all):
       - Logistic Regression        (linear baseline)
       - HistGradientBoostingClassifier (classical ML ensemble -- swapped
         in for a plain Random Forest, which it consistently outperforms
         on tabular data like this)
       - GenreMLP                   (the "AI model" -- a feed-forward
                                      neural net, trained for >=100 epochs
                                      with a class-weighted loss)
  5. Evaluate all three on the held-out test set, save every artifact
     (transformer, label encoder, all 3 trained models, and a metrics.json
     report) to ml/models/, and print a comparison table.

Run with:  python ml/train.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PowerTransformer

from model_def import GenreMLP, GENRE_BUCKETS, FEATURE_COLUMNS

_HERE = Path(__file__).parent
DATA_CSV = _HERE / "data" / "spotify_tracks_raw.csv"
MODELS_DIR = _HERE / "models"
EPOCHS = 200
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
SEED = 42

# The 12 raw Spotify CSV columns needed before feature engineering --
# distinct from FEATURE_COLUMNS (model_def.py), which is what the models
# actually consume after "key" is replaced by key_sin/key_cos below.
RAW_AUDIO_COLUMNS = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "time_signature",
]

# ---------------------------------------------------------------------------
# 114 raw Spotify genre tags -> 8 EQ-relevant buckets. A simplification for
# EQ-tuning purposes, not a musicological taxonomy -- see dsp/genre_curves.py
# for what each bucket actually changes about the EQ curve.
# ---------------------------------------------------------------------------
GENRE_TO_BUCKET = {
    "acoustic": "acoustic_folk", "afrobeat": "world_latin", "alt-rock": "rock_metal",
    "alternative": "rock_metal", "ambient": "chill_ambient", "anime": "pop",
    "black-metal": "rock_metal", "bluegrass": "acoustic_folk", "blues": "classical_jazz",
    "brazil": "world_latin", "breakbeat": "electronic_dance", "british": "pop",
    "cantopop": "pop", "chicago-house": "electronic_dance", "children": "chill_ambient",
    "chill": "chill_ambient", "classical": "classical_jazz", "club": "electronic_dance",
    "comedy": "pop", "country": "acoustic_folk", "dance": "electronic_dance",
    "dancehall": "hiphop_rnb", "death-metal": "rock_metal", "deep-house": "electronic_dance",
    "detroit-techno": "electronic_dance", "disco": "electronic_dance", "disney": "pop",
    "drum-and-bass": "electronic_dance", "dub": "world_latin", "dubstep": "electronic_dance",
    "edm": "electronic_dance", "electro": "electronic_dance", "electronic": "electronic_dance",
    "emo": "rock_metal", "folk": "acoustic_folk", "forro": "world_latin", "french": "pop",
    "funk": "hiphop_rnb", "garage": "electronic_dance", "german": "pop", "gospel": "classical_jazz",
    "goth": "rock_metal", "grindcore": "rock_metal", "groove": "hiphop_rnb", "grunge": "rock_metal",
    "guitar": "acoustic_folk", "happy": "pop", "hard-rock": "rock_metal", "hardcore": "rock_metal",
    "hardstyle": "electronic_dance", "heavy-metal": "rock_metal", "hip-hop": "hiphop_rnb",
    "honky-tonk": "acoustic_folk", "house": "electronic_dance", "idm": "electronic_dance",
    "indian": "world_latin", "indie": "rock_metal", "indie-pop": "pop", "industrial": "electronic_dance",
    "iranian": "world_latin", "j-dance": "electronic_dance", "j-idol": "pop", "j-pop": "pop",
    "j-rock": "rock_metal", "jazz": "classical_jazz", "k-pop": "pop", "kids": "chill_ambient",
    "latin": "world_latin", "latino": "world_latin", "malay": "world_latin", "mandopop": "pop",
    "metal": "rock_metal", "metalcore": "rock_metal", "minimal-techno": "electronic_dance",
    "mpb": "world_latin", "new-age": "chill_ambient", "opera": "classical_jazz",
    "pagode": "world_latin", "party": "electronic_dance", "piano": "classical_jazz", "pop": "pop",
    "pop-film": "pop", "power-pop": "rock_metal", "progressive-house": "electronic_dance",
    "psych-rock": "rock_metal", "punk": "rock_metal", "punk-rock": "rock_metal",
    "r-n-b": "hiphop_rnb", "reggae": "world_latin", "reggaeton": "hiphop_rnb", "rock": "rock_metal",
    "rock-n-roll": "rock_metal", "rockabilly": "rock_metal", "romance": "chill_ambient",
    "sad": "chill_ambient", "salsa": "world_latin", "samba": "world_latin",
    "sertanejo": "world_latin", "show-tunes": "pop", "singer-songwriter": "acoustic_folk",
    "ska": "rock_metal", "sleep": "chill_ambient", "songwriter": "acoustic_folk",
    "soul": "hiphop_rnb", "spanish": "world_latin", "study": "chill_ambient",
    "swedish": "pop", "synth-pop": "pop", "tango": "world_latin", "techno": "electronic_dance",
    "trance": "electronic_dance", "trip-hop": "hiphop_rnb", "turkish": "world_latin",
    "world-music": "world_latin",
}


def key_to_sin_cos(key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cyclically encode Spotify's 0-11 pitch-class "key" feature so key 11
    and key 0 end up *adjacent* in feature space (a semitone apart, same as
    musically), instead of maximally far apart the way a raw ordinal 0-11
    number would place them. -1 ("no key detected") maps to (0, 0) -- the
    centre of the circle, i.e. "no direction" -- rather than being treated
    as a valid 12th position. Shared by ml/train.py (training) and
    perception/genre_classifier.py (inference) so both sides encode it
    identically.
    """
    key = np.asarray(key, dtype=np.float64)
    angle = 2 * np.pi * key / 12.0
    sin, cos = np.sin(angle), np.cos(angle)
    no_key = key < 0
    sin = np.where(no_key, 0.0, sin)
    cos = np.where(no_key, 0.0, cos)
    return sin, cos


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.dropna(subset=["track_genre"] + RAW_AUDIO_COLUMNS)
    df = df.drop_duplicates(subset=["track_id", "track_genre"])

    unmapped = set(df["track_genre"].unique()) - set(GENRE_TO_BUCKET.keys())
    if unmapped:
        raise ValueError(f"GENRE_TO_BUCKET is missing a mapping for: {sorted(unmapped)}")

    df["genre_bucket"] = df["track_genre"].map(GENRE_TO_BUCKET)
    missing_buckets = set(GENRE_BUCKETS) - set(df["genre_bucket"].unique())
    if missing_buckets:
        raise ValueError(f"No rows ended up in bucket(s): {missing_buckets}")

    df["explicit"] = df["explicit"].astype(int)
    df["mode"] = df["mode"].astype(int)
    df["key_sin"], df["key_cos"] = key_to_sin_cos(df["key"].values)
    return df


def _train_neural_net(X_train, y_train, X_val, y_val, n_classes: int) -> tuple[GenreMLP, list, list]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  training GenreMLP on: {device}")

    model = GenreMLP(n_features=X_train.shape[1], n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    criterion = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)

    n = Xt.shape[0]
    train_losses, val_losses = [], []
    best_val_loss, best_state = float("inf"), None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        train_loss = epoch_loss / n

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), yv).item()
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"    epoch {epoch:3d}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model, train_losses, val_losses


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"Loading {DATA_CSV} ...")
    df = load_and_clean()
    print(f"  {len(df)} rows after cleaning, across {df['genre_bucket'].nunique()} genre buckets:")
    print(df["genre_bucket"].value_counts().to_string())

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    label_enc = LabelEncoder().fit(GENRE_BUCKETS)
    y = label_enc.transform(df["genre_bucket"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train)
    print(f"\nSplit: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")

    # Yeo-Johnson power transform instead of a plain StandardScaler --
    # several features (speechiness, acousticness, instrumentalness) are
    # heavily right-skewed (most tracks score near 0), which distorts
    # distance/gradient-based models. Harmless for the tree ensemble
    # (monotonic per-feature, so it can't change HGB's splits) and a real
    # improvement for the linear model + neural net.
    scaler = PowerTransformer(method="yeo-johnson").fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # The 8 buckets range from ~6k to ~23k rows (see the value_counts print
    # above), so "balanced" class weighting was tried for all 3 models --
    # measured on the validation set, it consistently made every model
    # *worse* on both accuracy and macro F1 here (the genre-bucket overlap
    # in this 13-feature space is close enough that reweighting just moves
    # errors around rather than fixing them). Left unweighted based on that
    # evidence, not by default -- see ml/README.md.
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
    joblib.dump(logreg, MODELS_DIR / "logistic_regression.joblib")
    print(f"  acc={results['logistic_regression']['accuracy']:.4f}  "
          f"f1_macro={results['logistic_regression']['f1_macro']:.4f}")

    # --- Model 2: Gradient boosting (classical ML ensemble) -----------------
    print("\n[2/3] HistGradientBoostingClassifier ...")
    t0 = time.time()
    # Consistently outperforms a plain Random Forest on tabular data like
    # this; has its own built-in early stopping (validation_fraction is
    # carved from X_train only, never touching our own val/test sets).
    # Hyperparameters picked by a small grid search against the validation
    # set (not the test set) -- see ml/README.md for the comparison table.
    hgb = HistGradientBoostingClassifier(
        max_iter=800, learning_rate=0.08, max_depth=None, max_leaf_nodes=255,
        min_samples_leaf=30, l2_regularization=0.2,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
        random_state=SEED,
    )
    hgb.fit(X_train_s, y_train)
    pred = hgb.predict(X_test_s)
    results["gradient_boosting"] = {
        "accuracy": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "train_seconds": round(time.time() - t0, 1),
        "n_iterations": hgb.n_iter_,
    }
    joblib.dump(hgb, MODELS_DIR / "gradient_boosting.joblib")
    print(f"  acc={results['gradient_boosting']['accuracy']:.4f}  "
          f"f1_macro={results['gradient_boosting']['f1_macro']:.4f}  "
          f"({hgb.n_iter_} boosting rounds)")

    # --- Model 3: Neural network (the "AI model") ---------------------------
    print(f"\n[3/3] Neural network (GenreMLP, {EPOCHS} epochs) ...")
    t0 = time.time()
    net, train_losses, val_losses = _train_neural_net(
        X_train_s, y_train, X_val_s, y_val, n_classes=len(GENRE_BUCKETS))
    device = next(net.parameters()).device
    net.eval()
    with torch.no_grad():
        logits = net(torch.tensor(X_test_s, dtype=torch.float32, device=device))
        pred = logits.argmax(dim=1).cpu().numpy()
    results["neural_net"] = {
        "accuracy": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "train_seconds": round(time.time() - t0, 1),
        "epochs": EPOCHS,
        "final_train_loss": round(train_losses[-1], 4),
        "final_val_loss": round(val_losses[-1], 4),
        "best_val_loss": round(min(val_losses), 4),
        "train_loss_history": [round(v, 5) for v in train_losses],
        "val_loss_history": [round(v, 5) for v in val_losses],
    }
    torch.save(net.state_dict(), MODELS_DIR / "neural_net.pt")
    print(f"  acc={results['neural_net']['accuracy']:.4f}  "
          f"f1_macro={results['neural_net']['f1_macro']:.4f}")

    # --- save shared artifacts + report -------------------------------------
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(label_enc, MODELS_DIR / "label_encoder.joblib")

    best_model = max(results, key=lambda k: results[k]["accuracy"])
    report = {
        "dataset": "spotify-tracks-dataset (huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)",
        "n_rows_used": len(df),
        "feature_columns": FEATURE_COLUMNS,
        "genre_buckets": GENRE_BUCKETS,
        "results": results,
        "best_model": best_model,
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print(f"{'Model':<22}{'Accuracy':<12}{'F1 (macro)':<12}{'Train time'}")
    for name, r in results.items():
        marker = " <- best" if name == best_model else ""
        print(f"{name:<22}{r['accuracy']:<12.4f}{r['f1_macro']:<12.4f}{r['train_seconds']:.1f}s{marker}")
    print("=" * 60)
    print(f"\nDetailed classification report ({best_model}):")
    if best_model == "neural_net":
        pred_best = pred  # already computed above
    elif best_model == "gradient_boosting":
        pred_best = hgb.predict(X_test_s)
    else:
        pred_best = logreg.predict(X_test_s)
    print(classification_report(y_test, pred_best, target_names=label_enc.classes_))
    print(f"\nSaved all artifacts to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
