"""Train the custom CNN benchmark and log its three-fold evaluation to MLflow."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = PROJECT_ROOT / "ourdataset" / "train"
MEL_ROOT = PROJECT_ROOT / "Data" / "processed" / "features" / "mel_spectrograms"
SPLITS_ROOT = PROJECT_ROOT / "Data" / "processed" / "splits" / "cnn"
MODEL_ROOT = PROJECT_ROOT / "audio_engine" / "artifacts" / "models" / "cnn_custom"
EVALUATION_ROOT = PROJECT_ROOT / "audio_engine" / "evaluation" / "cnn_custom"
REPORT_ROOT = PROJECT_ROOT / "docs" / "report" / "benchmarks" / "cnn_custom"
CLASS_TO_LABEL = {"normal": 0, "abnormal": 1}
FEATURE_VERSION = "mel40_sr2000_v1"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


def make_mel_image(wav_path: Path, output_path: Path) -> None:
    signal, sample_rate = librosa.load(wav_path, sr=2000, mono=True)
    mel = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=512, hop_length=128, n_mels=40, fmin=20, fmax=1000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    value_range = float(np.ptp(mel_db))
    image = (mel_db - mel_db.min()) / value_range if value_range else np.zeros_like(mel_db)
    plt.imsave(output_path, image, cmap="magma", vmin=0, vmax=1)


def prepare_records() -> list[tuple[Path, int, str]]:
    records = []
    for class_name, label in CLASS_TO_LABEL.items():
        wav_files = sorted((AUDIO_ROOT / class_name).glob("*.wav"))
        if not wav_files:
            raise FileNotFoundError(f"No WAV files found in {AUDIO_ROOT / class_name}")
        output_dir = MEL_ROOT / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for wav_path in wav_files:
            image_path = output_dir / f"{wav_path.stem}_{FEATURE_VERSION}.png"
            if not image_path.exists() or image_path.stat().st_mtime < wav_path.stat().st_mtime:
                make_mel_image(wav_path, image_path)
            records.append((image_path, label, class_name))
    return records


def write_splits(records: list[tuple[Path, int, str]]) -> list[Path]:
    labels = np.asarray([record[1] for record in records], dtype=np.int32)
    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)
    split_files = []
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=10)
    for fold, (train_indices, test_indices) in enumerate(splitter.split(np.zeros(len(records)), labels), start=1):
        split_path = SPLITS_ROOT / f"cnn_fold_{fold}.csv"
        with split_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=("split", "relative_path", "label", "class_name"))
            writer.writeheader()
            for split_name, indices in (("train", train_indices), ("test", test_indices)):
                for index in indices:
                    image_path, label, class_name = records[int(index)]
                    writer.writerow({"split": split_name, "relative_path": image_path.relative_to(PROJECT_ROOT).as_posix(), "label": label, "class_name": class_name})
        split_files.append(split_path)
    return split_files


def load_fold_rows(split_path: Path, split_name: str) -> tuple[list[str], np.ndarray]:
    paths, labels = [], []
    with split_path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["split"] == split_name:
                paths.append(str(PROJECT_ROOT / row["relative_path"]))
                labels.append(int(row["label"]))
    return paths, np.asarray(labels, dtype=np.int32)


def make_dataset(paths: list[str], labels: np.ndarray, training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.decode_png(tf.io.read_file(path), channels=3)
        return tf.cast(tf.image.resize(image, IMAGE_SIZE), tf.float32) / 255.0, label

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(len(paths), seed=10, reshuffle_each_iteration=True)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def make_model(learning_rate: float) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(*IMAGE_SIZE, 3)),
        layers.Conv2D(32, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"), layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4), layers.Dense(1, activation="sigmoid"),
    ], name="cardio_custom_cnn")
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])
    return model


def save_curve(history: tf.keras.callbacks.History, fold: int) -> Path:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric, title in ((axes[0], "accuracy", "accuracy"), (axes[1], "loss", "loss")):
        axis.plot(history.history[metric], label="train")
        axis.plot(history.history[f"val_{metric}"], label="test")
        axis.set(title=f"Fold {fold} {title}", xlabel="Epoch")
        axis.legend()
    figure.tight_layout()
    path = REPORT_ROOT / f"fold_{fold}_training.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(10)
    records = prepare_records()
    split_files = write_splits(records)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    folds = []

    for fold, split_path in enumerate(split_files, start=1):
        train_paths, train_labels = load_fold_rows(split_path, "train")
        test_paths, test_labels = load_fold_rows(split_path, "test")
        train_data, test_data = make_dataset(train_paths, train_labels, True), make_dataset(test_paths, test_labels, False)
        tf.keras.backend.clear_session()
        model = make_model(args.learning_rate)
        model_path = MODEL_ROOT / f"fold_{fold}" / "best.keras"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        history = model.fit(train_data, validation_data=test_data, epochs=args.epochs, callbacks=[callbacks.ModelCheckpoint(str(model_path), monitor="val_recall", mode="max", save_best_only=True), callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)], verbose=1)
        best_model = tf.keras.models.load_model(model_path)
        probabilities = best_model.predict(test_data, verbose=0).reshape(-1)
        predictions = (probabilities >= 0.5).astype(np.int32)
        metrics = {"accuracy": float(accuracy_score(test_labels, predictions)), "precision": float(precision_score(test_labels, predictions, zero_division=0)), "recall": float(recall_score(test_labels, predictions, zero_division=0)), "f1": float(f1_score(test_labels, predictions, zero_division=0))}
        matrix = confusion_matrix(test_labels, predictions, labels=[0, 1]).tolist()
        curve_path = save_curve(history, fold)
        report_path = EVALUATION_ROOT / f"fold_{fold}_report.txt"
        report_path.write_text(f"Metrics: {metrics}\nConfusion matrix (normal, abnormal):\n{np.asarray(matrix)}\n", encoding="utf-8")
        folds.append({"fold": fold, "metrics": metrics, "matrix": matrix, "curve_path": curve_path, "report_path": report_path, "model_path": model_path})
        print(f"Fold {fold}: {metrics}")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("audio_benchmark")
    mlflow.set_system_metrics_sampling_interval(1)
    mlflow.enable_system_metrics_logging()
    with mlflow.start_run(run_name="cnn_custom", log_system_metrics=True) as run:
        mlflow.set_tags({"model": "custom_cnn", "task": "pcg_normal_vs_abnormal", "evaluation": "3_fold_cross_validation", "feature_set": FEATURE_VERSION})
        mlflow.log_params({"n_folds": 3, "epochs_requested": args.epochs, "batch_size": BATCH_SIZE, "learning_rate": args.learning_rate, "image_size": "224x224", "mel_bands": 40, "sample_rate": 2000})
        metric_names = ("accuracy", "precision", "recall", "f1")
        for metric in metric_names:
            mlflow.log_metric(f"mean_{metric}", float(np.mean([item["metrics"][metric] for item in folds])))
        for item in folds:
            for metric in metric_names:
                mlflow.log_metric(f"fold_{metric}", item["metrics"][metric], step=item["fold"])
            mlflow.log_artifact(str(item["curve_path"]), artifact_path=f"figures/fold_{item['fold']}")
            mlflow.log_artifact(str(item["report_path"]), artifact_path=f"reports/fold_{item['fold']}")
        mlflow.log_dict({"folds": [{"fold": item["fold"], **item["metrics"]} for item in folds]}, "reports/cv_metrics.json")
        mlflow.log_dict({"labels": ["normal", "abnormal"], "folds": [{"fold": item["fold"], "matrix": item["matrix"]} for item in folds]}, "reports/confusion_matrices.json")
        best_fold = max(folds, key=lambda item: (item["metrics"]["recall"], item["metrics"]["f1"]))
        mlflow.set_tag("best_fold", str(best_fold["fold"]))
        mlflow.log_artifact(str(best_fold["model_path"]), artifact_path="models/best_fold")
        print(f"Logged CNN run {run.info.run_id}; best fold: {best_fold['fold']}")


if __name__ == "__main__":
    main()
