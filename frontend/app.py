"""Streamlit client for independent heart-sound model analyses."""

from __future__ import annotations

import base64
from io import BytesIO
import os
import sys
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# Keep the project root importable when Streamlit is launched from frontend/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_engine.features.audio_features import CNN_FEATURE_SETTINGS
from audio_engine.preprocessing.audio import TARGET_SAMPLE_RATE

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Heart Sound Model Lab", page_icon="🫀", layout="wide")
st.title("Heart Sound Model Lab")
st.caption("Run independent model analyses on one phonocardiogram recording.")
st.warning("Educational project only. This is not a medical device and must not be used for diagnosis or medical decisions.", icon="⚠️")
st.markdown("""<style>
[data-testid="stVerticalBlock"] [data-testid="stMarkdownContainer"] { overflow-wrap: anywhere; word-break: break-word; }
.model-version { overflow-wrap: anywhere; word-break: break-word; }
@media (max-width: 700px) {
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
  [data-testid="stHorizontalBlock"] > [data-testid="column"] { min-width: 100% !important; flex: 1 1 100% !important; }
}
</style>""", unsafe_allow_html=True)

analyze_tab, history_tab = st.tabs(["Analyze recording", "History"])


def show_result_card(result: dict, kernel_label: str | None = None) -> None:
    color = "#16835f" if result["prediction"] == 0 else "#bd4c43"
    with st.container(border=True):
        if kernel_label:
            st.caption(kernel_label)
        st.markdown(f"**{result['model_name']}**")
        st.markdown(f"<div style='font-size:24px;font-weight:750;color:{color};margin:10px 0'>{result['prediction_label']}</div>", unsafe_allow_html=True)
        st.write(f"{result['confidence_type'].replace('_', ' ').title()}: {result['confidence_or_score']:.4f}")
        st.write(f"Inference: {result['latency_ms']:.2f} ms")
        version = str(result["model_version"])
        short_version = version if len(version) <= 30 else version[:27] + "..."
        st.caption(f"Version: {short_version}")
        with st.expander("Model details"):
            st.write(version)


def mel_spectrogram_figure(wav_bytes: bytes):
    signal, sample_rate = librosa.load(BytesIO(wav_bytes), sr=TARGET_SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(y=signal, sr=sample_rate, **CNN_FEATURE_SETTINGS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    hop_length = CNN_FEATURE_SETTINGS["hop_length"]
    times = librosa.frames_to_time(np.arange(mel_db.shape[1]), sr=sample_rate, hop_length=hop_length)
    figure, axis = plt.subplots(figsize=(10, 2.5), facecolor="#0e1117")
    axis.set_facecolor("#0e1117")
    mesh = axis.pcolormesh(times, librosa.mel_frequencies(n_mels=CNN_FEATURE_SETTINGS["n_mels"], fmin=CNN_FEATURE_SETTINGS["fmin"], fmax=CNN_FEATURE_SETTINGS["fmax"]), mel_db, shading="auto", cmap="magma")
    axis.set(xlabel="Time (s)", ylabel="Mel frequency (Hz)", title="Mel spectrogram used by the CNN")
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
    return figure


with analyze_tab:
    uploaded = st.file_uploader("Choose a WAV recording", type=["wav"], help="Mono or stereo PCM WAV; audio is resampled to 2,000 Hz by the API.")
    if uploaded is not None:
        st.audio(uploaded.getvalue(), format="audio/wav")
    if st.button("Analyze", type="primary", disabled=uploaded is None, use_container_width=False):
        try:
            with st.spinner("Analyzing the recording with each model…"):
                response = requests.post(
                    f"{API_URL}/analyze_pcg",
                    files={"file": (uploaded.name, uploaded.getvalue(), "audio/wav")},
                    timeout=180,
                )
            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                st.error(f"Analysis failed: {detail}")
            else:
                result = response.json()
                st.session_state["latest_analysis"] = result
                st.session_state["latest_wav_bytes"] = uploaded.getvalue()
        except requests.RequestException as exc:
            st.error(f"Could not reach the API at {API_URL}. Start the API and try again. Details: {exc}")

    result = st.session_state.get("latest_analysis")
    if result:
        st.subheader("Recording details")
        st.write(
            f"**{result['filename']}** · {result['duration_s']:.2f} s · "
            f"{result['source_sample_rate']} Hz → {result['sample_rate_after_resampling']} Hz · "
            f"{result['total_processing_ms']:.1f} ms processing"
        )
        if result.get("warning"):
            st.warning(result["warning"])
        st.subheader("Model results")
        results = result["results"]
        svm_results = [item for item in results if "svm" in item["model_name"].lower()]
        tree_result = next((item for item in results if "tree" in item["model_name"].lower()), None)
        cnn_result = next((item for item in results if "cnn" in item["model_name"].lower()), None)
        st.markdown("### SVM (2 kernels)")
        with st.container(border=True):
            svm_cols = st.columns(2)
            for column, model_result in zip(svm_cols, svm_results):
                with column:
                    kernel = "Linear kernel" if "linear" in model_result["model_name"].lower() else "RBF kernel"
                    show_result_card(model_result, kernel)
        group_cols = st.columns(2)
        if tree_result:
            with group_cols[0]:
                st.markdown("### Decision Tree")
                show_result_card(tree_result)
        if cnn_result:
            with group_cols[1]:
                st.markdown("### CNN")
                show_result_card(cnn_result)
        st.caption("Educational project only. These outputs are for learning and must not guide diagnosis or medical decisions.")
        st.subheader("Audio analysis")
        waveform_col, mel_col = st.columns(2)
        with waveform_col:
            st.image(base64.b64decode(result["waveform_png_base64"]), caption="Waveform after resampling", use_container_width=True)
        with mel_col:
            wav_bytes = st.session_state.get("latest_wav_bytes")
            if wav_bytes:
                figure = mel_spectrogram_figure(wav_bytes)
                st.pyplot(figure, use_container_width=True)
                plt.close(figure)
        if result.get("mlflow_run_url"):
            st.markdown(f"[Open this analysis in MLflow]({result['mlflow_run_url']})")

with history_tab:
    st.subheader("Recent analyses")
    limit = st.selectbox("Number of records", [10, 20, 50, 100], index=1)
    try:
        history_response = requests.get(f"{API_URL}/history", params={"limit": limit}, timeout=15)
        if history_response.status_code >= 400:
            st.error(f"History request failed: {history_response.text}")
        else:
            payload = history_response.json()
            if payload.get("warning"):
                st.warning(payload["warning"])
            analyses = payload.get("analyses", [])
            if not analyses:
                st.info("No analysis runs were found yet.")
            else:
                rows = []
                def label_for(value):
                    if value is None or pd.isna(value):
                        return "—"
                    return "Abnormal" if int(value) == 1 else "Normal"

                for analysis in analyses:
                    metrics = analysis.get("metrics", {})
                    rows.append({
                        "Filename": analysis.get("filename") or "—",
                        "Run": analysis.get("run_name") or analysis["run_id"][:8],
                        "Duration (s)": analysis.get("duration_s"),
                        "Linear SVM": label_for(metrics.get("svm_linear.prediction")),
                        "RBF SVM": label_for(metrics.get("svm_rbf.prediction")),
                        "Decision Tree": label_for(metrics.get("decision_tree.prediction")),
                        "CNN": label_for(metrics.get("cnn_custom.prediction")),
                        "MLflow run": analysis.get("run_url", analysis["run_id"]),
                    })
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={"MLflow run": st.column_config.LinkColumn("MLflow run", display_text="Open run")},
                )
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_URL}. Details: {exc}")
