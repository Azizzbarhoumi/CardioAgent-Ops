"""Feature extraction matching the SVM, tree, and CNN notebook code."""

from __future__ import annotations

from io import BytesIO

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TABULAR_FEATURE_ORDER = (
    "duration",
    "snr",
    *(f"mel_{index}" for index in range(1, 21)),
)
TABULAR_MEL_BANDS = 20
CNN_MEL_BANDS = 40
CNN_IMAGE_SIZE = (224, 224)
CNN_FEATURE_SETTINGS = {
    "n_fft": 512,
    "hop_length": 128,
    "n_mels": CNN_MEL_BANDS,
    "fmin": 20,
    "fmax": 1000,
}


def signal_to_noise_ratio(signal: np.ndarray) -> float:
    """Replicate the notebook's normalized-signal mean/std ratio."""
    signal = np.asarray(signal)
    peak = max(float(np.max(signal)), float(-np.min(signal)))
    normalized = signal / peak if peak > 0 else signal
    mean = np.mean(normalized)
    std = np.std(normalized, ddof=0)
    return float(np.where(std == 0, 0, mean / std))


def extract_tabular_features(signal: np.ndarray, sample_rate: int = 2000) -> np.ndarray:
    """Return duration, SNR, and 20 log-Mel means in training order."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("Expected a non-empty mono signal.")
    duration = float(librosa.get_duration(y=signal, sr=sample_rate))
    mel = librosa.feature.melspectrogram(
        y=signal, sr=sample_rate, n_fft=512, hop_length=128, n_mels=TABULAR_MEL_BANDS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    vector = np.asarray([duration, signal_to_noise_ratio(signal), *np.mean(mel_db, axis=1)], dtype=np.float64)
    if vector.shape != (22,):
        raise ValueError(f"Expected 22 tabular features, found {vector.shape}.")
    return vector


def extract_cnn_image(signal: np.ndarray, sample_rate: int = 2000, *, return_mel: bool = False):
    """Create the CNN's exact magma-rendered RGB input image and PNG bytes."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("Expected a non-empty mono signal.")
    mel = librosa.feature.melspectrogram(y=signal, sr=sample_rate, **CNN_FEATURE_SETTINGS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    span = float(np.ptp(mel_db))
    unit_image = (mel_db - mel_db.min()) / span if span > 0 else np.zeros_like(mel_db)
    buffer = BytesIO()
    plt.imsave(buffer, unit_image, format="png", cmap="magma", vmin=0, vmax=1)
    png_bytes = buffer.getvalue()
    import tensorflow as tf

    image = tf.io.decode_png(png_bytes, channels=3)
    image = tf.image.resize(image, CNN_IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    result = np.asarray(image.numpy(), dtype=np.float32), png_bytes
    if return_mel:
        return (*result, mel_db)
    return result


# --------------------------------------------------------------------------- benchmark v2 image inputs
# Exactly the "mel40_sr2000_gray_v2" PNGs of TP7-CNN_16Nov_grouped.ipynb (make_mel_image):
# 40 log-Mel bands, min-max scaled to uint8, one PNG pixel per frame. Both v2 image models
# (custom CNN and VGG16) read this same array, so preprocessing runs once per upload.
GRAY_MEL_FEATURE_VERSION = "mel40_sr2000_gray_v2"
GRAY_MEL_SETTINGS = {"n_fft": 512, "hop_length": 128, "n_mels": 40, "fmin": 20, "fmax": 1000}
CNN_V2_IMAGE_SIZE = (40, 256)
VGG_IMAGE_SIZE = (224, 224)


def extract_gray_mel(signal: np.ndarray, sample_rate: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Return (uint8 image of shape (40, frames), mel_db) as saved in the training PNGs."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("Expected a non-empty mono signal.")
    mel = librosa.feature.melspectrogram(y=signal, sr=sample_rate, **GRAY_MEL_SETTINGS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    span = float(np.ptp(mel_db))
    unit = (mel_db - mel_db.min()) / span if span > 0 else np.zeros_like(mel_db)
    return np.rint(unit * 255).astype(np.uint8), mel_db


def cnn_v2_input(gray: np.ndarray) -> np.ndarray:
    """(40, 256, 1) float32 in [0, 1]: same tf ops as training (decode -> resize -> /255)."""
    import tensorflow as tf

    image = tf.image.resize(tf.convert_to_tensor(gray[..., None]), CNN_V2_IMAGE_SIZE)
    return (tf.cast(image, tf.float32) / 255.0).numpy()


def vgg_input(gray: np.ndarray) -> np.ndarray:
    """(224, 224, 3) VGG16 input: same function as training (audio_engine.evaluation.runners.vgg_preprocess)."""
    import tensorflow as tf

    image = tf.image.resize(tf.cast(tf.convert_to_tensor(gray[..., None]), tf.float32), VGG_IMAGE_SIZE)
    return tf.keras.applications.vgg16.preprocess_input(tf.image.grayscale_to_rgb(image)).numpy()
