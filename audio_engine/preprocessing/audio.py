"""Decode WAV data as mono float audio resampled to the training rate."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import librosa
import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 2000


@dataclass(frozen=True)
class AudioInput:
    signal: np.ndarray
    sample_rate: int
    source_sample_rate: int
    duration_s: float


def decode_wav(
    wav_bytes: bytes,
    *,
    minimum_seconds: float | None = None,
    maximum_seconds: float | None = None,
) -> AudioInput:
    """Decode a WAV byte string as mono float32 at 2 kHz.

    Training did not crop or pad recordings, so inference preserves the full
    recording duration. Amplitude normalization is performed only where the
    training feature/image pipelines performed it.
    """
    if not wav_bytes:
        raise ValueError("The uploaded WAV file is empty.")
    try:
        info = sf.info(BytesIO(wav_bytes))
    except (RuntimeError, OSError, sf.LibsndfileError, ValueError) as exc:
        raise ValueError(f"The uploaded file is not a readable WAV recording: {exc}") from exc
    if info.frames < 1 or info.samplerate < 1:
        raise ValueError("The WAV file contains no audio samples.")
    source_duration = info.frames / info.samplerate
    if minimum_seconds is not None and source_duration < minimum_seconds:
        raise ValueError(f"Audio is too short. Minimum duration is {minimum_seconds:g} seconds.")
    if maximum_seconds is not None and source_duration > maximum_seconds:
        raise ValueError(f"Audio is too long. Maximum duration is {maximum_seconds:g} seconds.")
    try:
        signal, sample_rate = librosa.load(
            BytesIO(wav_bytes), sr=TARGET_SAMPLE_RATE, mono=True
        )
    except (RuntimeError, OSError, sf.LibsndfileError, ValueError) as exc:
        raise ValueError(f"The uploaded file is not a readable WAV recording: {exc}") from exc
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("The WAV file contains no usable audio samples.")
    return AudioInput(
        signal=np.asarray(signal, dtype=np.float32),
        sample_rate=int(sample_rate),
        source_sample_rate=int(info.samplerate),
        duration_s=float(signal.size / sample_rate),
    )
