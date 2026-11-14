"""Audio analysis orchestration and visualization artifacts."""

from __future__ import annotations

import base64
import time
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from api.services.mlflow_logging import InferenceLogger
from audio_engine.preprocessing.audio import decode_wav


def waveform_png(signal: np.ndarray, sample_rate: int) -> bytes:
    figure, axis = plt.subplots(figsize=(10, 2.5))
    times = np.arange(signal.size) / sample_rate
    stride = max(1, signal.size // 6000)
    axis.plot(times[::stride], signal[::stride], linewidth=0.55, color="#147d82")
    axis.set(xlabel="Time (s)", ylabel="Amplitude", title="Resampled waveform")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=120)
    plt.close(figure)
    return buffer.getvalue()


def analyze_recording(*, filename: str, wav_bytes: bytes, registry, logger: InferenceLogger, minimum_seconds: float, maximum_seconds: float) -> dict:
    started = time.perf_counter()
    audio = decode_wav(
        wav_bytes,
        minimum_seconds=minimum_seconds,
        maximum_seconds=maximum_seconds,
    )
    prediction = registry.predict_all(audio)
    mel_png = prediction["mel_png"]
    wave_png = waveform_png(audio.signal, audio.sample_rate)
    response = {
        "filename": filename,
        "duration_s": audio.duration_s,
        "source_sample_rate": audio.source_sample_rate,
        "sample_rate_after_resampling": audio.sample_rate,
        "results": prediction["results"],
        "total_processing_ms": (time.perf_counter() - started) * 1000.0,
        "mel_spectrogram_png_base64": base64.b64encode(mel_png).decode("ascii"),
        "waveform_png_base64": base64.b64encode(wave_png).decode("ascii"),
        "mlflow_run_id": None,
        "mlflow_run_url": None,
        "warning": None,
    }
    run_id, run_url, warning = logger.log(
        filename=filename,
        wav_bytes=wav_bytes,
        response=response,
        mel_png=mel_png,
        waveform_png=wave_png,
    )
    response["mlflow_run_id"] = run_id
    response["mlflow_run_url"] = run_url
    response["warning"] = warning
    return response
