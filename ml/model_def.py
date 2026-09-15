"""
Shared model definitions -- imported by both ml/train.py (training) and
perception/genre_classifier.py (inference), so the architecture used to
train is guaranteed to match the architecture used to load weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# -----------------------------------------------------------------------
# The 8 EQ-relevant genre "buckets" the classifier predicts. Coarser than
# the Spotify dataset's 114 raw genre tags on purpose -- AuraTune needs a
# small number of classes it can attach a distinct, hand-tuned target EQ
# curve to (see GENRE_CURVES in dsp/genre_curves.py), not fine-grained
# musicology.
GENRE_BUCKETS = [
    "electronic_dance",
    "rock_metal",
    "hiphop_rnb",
    "pop",
    "acoustic_folk",
    "classical_jazz",
    "chill_ambient",
    "world_latin",
]

# The 13 numeric model inputs, in the exact order both training and
# inference must use. 12 of Spotify's original audio features, minus
# "key" -- which is a *categorical* pitch class (0-11, musically circular:
# 11 and 0 are adjacent, not far apart), so treating it as a plain ordinal
# number was actively misleading the linear/neural models. It's replaced
# with its sin/cos encoding (key_sin, key_cos), which preserves that
# circularity -- see _key_to_sin_cos() in ml/train.py and
# perception/genre_classifier.py for where each side computes it.
FEATURE_COLUMNS = [
    "danceability", "energy", "key_sin", "key_cos", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "time_signature",
]


class GenreMLP(nn.Module):
    """The "AI model" of the three -- a feed-forward neural net classifier.

    13 scaled numeric features in, 8 genre-bucket logits out. Three hidden
    layers (128 -> 64 -> 32) with batchnorm + dropout -- wide enough to
    use the ~82k training rows without badly underfitting, regularized
    enough (0.3 dropout, weight decay in the optimizer) to not just
    memorize them either.
    """

    def __init__(self, n_features: int = len(FEATURE_COLUMNS),
                 n_classes: int = len(GENRE_BUCKETS), hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden // 4),
            nn.Dropout(0.2),
            nn.Linear(hidden // 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
