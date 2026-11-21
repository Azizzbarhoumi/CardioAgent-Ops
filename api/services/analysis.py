"""Audio analysis orchestration and visualization artifacts."""

from __future__ import annotations

import base64
import time
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import librosa

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


def mel_display_png(mel_db: np.ndarray, duration_s: float, sample_rate: int, axes_box: dict | None = None) -> bytes:
    """Render the exact CNN log-Mel matrix with its time and Mel-frequency axes.

    If ``axes_box`` is given it receives the plot area as image fractions (left/right/top/bottom, origin top-left),
    so the UI can draw a playhead on the time axis. Display metadata only; the data are unchanged."""
    from audio_engine.features.audio_features import CNN_FEATURE_SETTINGS

    settings = CNN_FEATURE_SETTINGS
    times = librosa.frames_to_time(
        np.arange(mel_db.shape[1]), sr=sample_rate, hop_length=settings["hop_length"]
    )
    frequencies = librosa.mel_frequencies(
        n_mels=settings["n_mels"], fmin=settings["fmin"], fmax=settings["fmax"]
    )
    figure, axis = plt.subplots(figsize=(8, 3.2), facecolor="#0e1117")
    axis.set_facecolor("#0e1117")
    mesh = axis.pcolormesh(times, frequencies, mel_db, shading="auto", cmap="magma")
    axis.set(xlabel="Time (s)", ylabel="Mel frequency (Hz)", title="Mel spectrogram (CNN and VGG16 input)")
    axis.set_xlim(0, duration_s)
    axis.tick_params(colors="#e6e6e6")
    axis.xaxis.label.set_color("#e6e6e6")
    axis.yaxis.label.set_color("#e6e6e6")
    axis.title.set_color("#e6e6e6")
    for spine in axis.spines.values():
        spine.set_color("#8b949e")
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.02)
    colorbar.set_label("Power (dB)", color="#e6e6e6")
    colorbar.ax.tick_params(colors="#e6e6e6")
    figure.tight_layout()
    if axes_box is not None:
        box = axis.get_position()
        axes_box.update(left=float(box.x0), right=float(box.x1), top=float(1 - box.y1), bottom=float(1 - box.y0),
                        time_start_s=0.0, time_end_s=float(duration_s))
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=130, facecolor=figure.get_facecolor())
    plt.close(figure)
    return buffer.getvalue()


def analyze_recording(*, filename: str, wav_bytes: bytes, registry, logger: InferenceLogger, minimum_seconds: float, maximum_seconds: float, benchmark=None) -> dict:
    started = time.perf_counter()
    audio = decode_wav(
        wav_bytes,
        minimum_seconds=minimum_seconds,
        maximum_seconds=maximum_seconds,
    )
    prediction = registry.predict_all(audio)
    if benchmark is not None:
        for result in prediction["results"]:
            result.update(benchmark.summary_for(result["model_key"]))
    mel_axes: dict = {}
    mel_png = (
        mel_display_png(prediction["mel_db"], audio.duration_s, audio.sample_rate, mel_axes)
        if "mel_db" in prediction else prediction["mel_png"]
    )
    wave_png = waveform_png(audio.signal, audio.sample_rate)
    response = {
        "filename": filename,
        "duration_s": audio.duration_s,
        "source_sample_rate": audio.source_sample_rate,
        "sample_rate_after_resampling": audio.sample_rate,
        "results": prediction["results"],
        "model_groups": [
            {"key": "svm", "name": "SVM (2 kernels)", "models": [r for r in prediction["results"] if r["model_key"].startswith("svm_")]},
            {"key": "decision_tree", "name": "Decision Tree", "models": [r for r in prediction["results"] if r["model_key"] == "decision_tree"]},
            {"key": "cnn", "name": "CNN", "models": [r for r in prediction["results"] if r["model_key"] == "cnn_custom"]},
            {"key": "transfer_learning", "name": "Transfer Learning (VGG16)", "models": [r for r in prediction["results"] if r["model_key"] == "vgg16_transfer"]},
        ],
        "waveform_points": [
            [float(index / audio.sample_rate), float(audio.signal[index])]
            for index in range(0, audio.signal.size, max(1, audio.signal.size // 1200))
        ],
        "total_processing_ms": (time.perf_counter() - started) * 1000.0,
        "mel_spectrogram_png_base64": base64.b64encode(mel_png).decode("ascii"),
        "mel_spectrogram_axes": mel_axes or None,
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
