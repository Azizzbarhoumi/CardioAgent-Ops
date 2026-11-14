"""Shape and preprocessing checks for inference inputs."""

from io import BytesIO

import numpy as np
import soundfile as sf

from audio_engine.features.audio_features import (
    CNN_IMAGE_SIZE,
    TABULAR_FEATURE_ORDER,
    extract_cnn_image,
    extract_tabular_features,
)
from audio_engine.preprocessing.audio import TARGET_SAMPLE_RATE, decode_wav


def make_wav(sample_rate: int = 4000, duration: float = 1.5) -> bytes:
    time = np.arange(int(sample_rate * duration)) / sample_rate
    signal = 0.2 * np.sin(2 * np.pi * 90 * time)
    buffer = BytesIO()
    sf.write(buffer, signal, sample_rate, format="WAV")
    return buffer.getvalue()


def test_decode_wav_resamples_to_training_rate_and_preserves_duration() -> None:
    audio = decode_wav(make_wav())
    assert audio.sample_rate == TARGET_SAMPLE_RATE
    assert audio.source_sample_rate == 4000
    assert audio.signal.ndim == 1
    assert abs(audio.duration_s - 1.5) < 0.02


def test_tabular_features_have_documented_order_and_22_values() -> None:
    audio = decode_wav(make_wav())
    features = extract_tabular_features(audio.signal, audio.sample_rate)
    assert len(TABULAR_FEATURE_ORDER) == 22
    assert features.shape == (22,)
    assert TABULAR_FEATURE_ORDER[:2] == ("duration", "snr")
    assert TABULAR_FEATURE_ORDER[-1] == "mel_20"
    assert abs(features[0] - audio.duration_s) < 0.02


def test_cnn_image_matches_training_model_input_shape() -> None:
    audio = decode_wav(make_wav())
    image, png = extract_cnn_image(audio.signal, audio.sample_rate)
    assert image.shape == (*CNN_IMAGE_SIZE, 3)
    assert image.dtype == np.float32
    assert np.isfinite(image).all()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
