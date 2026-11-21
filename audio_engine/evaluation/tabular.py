"""22 tabular features per recording, keyed by recording stem (for SVM and tree)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from audio_engine.evaluation.folds import AUDIO_ROOT, PROJECT_ROOT, Fold, list_recordings
from audio_engine.features.audio_features import TABULAR_FEATURE_ORDER, extract_tabular_features

TABULAR_CACHE = PROJECT_ROOT / "Data" / "processed" / "features" / "tabular_duration_snr_mel20_v1.csv"


def build_tabular_cache(path: Path = TABULAR_CACHE) -> pd.DataFrame:
    """Same extraction as the API (librosa 2000 Hz mono + extract_tabular_features)."""
    import librosa

    from audio_engine.preprocessing.audio import TARGET_SAMPLE_RATE

    rows = []
    for stem, label, class_name in list_recordings():
        signal, sr = librosa.load(AUDIO_ROOT / class_name / f"{stem}.wav", sr=TARGET_SAMPLE_RATE, mono=True)
        rows.append([stem, label, *extract_tabular_features(signal, sr).tolist()])
    df = pd.DataFrame(rows, columns=["recording_stem", "label", *TABULAR_FEATURE_ORDER])
    df.to_csv(path, index=False, float_format="%.17g")
    return df


def load_tabular(path: Path = TABULAR_CACHE) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} missing; call build_tabular_cache() once.")
    return pd.read_csv(path).set_index("recording_stem")


def fold_xy(table: pd.DataFrame, fold: Fold, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X, y and group array for one split, mapped through the recording stem."""
    rows = fold.split(split)
    stems = [r.recording_stem for r in rows]
    sub = table.loc[stems]
    y = np.array([r.label for r in rows])
    if not (sub["label"].to_numpy() == y).all():
        raise AssertionError("Label mismatch between feature table and manifest")
    return sub[list(TABULAR_FEATURE_ORDER)].to_numpy(float), y, np.array([r.group for r in rows])
