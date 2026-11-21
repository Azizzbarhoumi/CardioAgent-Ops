"""Train/evaluate every model on the shared folds. Notebooks call these functions.

Protocol per fold (identical for every model):
* fit on ``train`` only;
* early stopping (neural nets), hyperparameter search (sklearn: inner grouped CV
  inside ``train``) and threshold selection use ``train``/``validation`` only;
* ``test`` is touched once, at the end, for reporting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from audio_engine.evaluation.folds import PROJECT_ROOT, Fold, load_folds, manifest_sha
from audio_engine.evaluation.metrics import choose_threshold, compute_metrics, summarize

MODELS_DIR = PROJECT_ROOT / "audio_engine" / "artifacts" / "models" / "v3"
RESULTS_DIR = PROJECT_ROOT / "audio_engine" / "evaluation" / "results" / "v3"
SEED = 1


def _save_results(model_key: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{model_key}.json"
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return path


def _finish(model_key, display_name, framework, score_type, params, folds, fold_records):
    payload = {
        "model": model_key, "display_name": display_name, "framework": framework,
        "score_type": score_type, "label_convention": {"0": "normal", "1": "abnormal"},
        "evaluation": "patient_grouped_3_fold_cv", "split_manifest_sha": manifest_sha(),
        "fold_manifest_shas": {f.number: f.sha for f in folds},
        "params": params, "folds": fold_records, "summary": summarize(fold_records),
    }
    payload["results_path"] = str(_save_results(model_key, payload).relative_to(PROJECT_ROOT))
    return payload


# --------------------------------------------------------------------------- sklearn
SKLEARN_SPECS = {
    "svm_linear": dict(display="SVM (linear)", grid={"svc__C": [0.1, 1.0, 10.0]}, kernel="linear"),
    "svm_rbf": dict(display="SVM (RBF)", grid={"svc__C": [0.1, 1.0, 10.0], "svc__gamma": ["scale", 0.01, 0.1]}, kernel="rbf"),
    "decision_tree": dict(display="Decision Tree", grid={"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 5, 20]}),
}


def _sklearn_estimator(key):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    if key.startswith("svm"):
        return make_pipeline(StandardScaler(), SVC(kernel=SKLEARN_SPECS[key]["kernel"], class_weight="balanced"))
    return DecisionTreeClassifier(class_weight="balanced", random_state=SEED)


def run_sklearn(model_key: str, folds: list[Fold] | None = None) -> dict:
    """Inner grouped 3-fold grid search on ``train``; SVM label = decision > 0,
    tree label = P(abnormal) >= threshold chosen on ``validation``."""
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

    from audio_engine.evaluation.tabular import fold_xy, load_tabular

    spec = SKLEARN_SPECS[model_key]
    folds = folds or load_folds()
    table = load_tabular()
    records = []
    for fold in folds:
        X_tr, y_tr, g_tr = fold_xy(table, fold, "train")
        search = GridSearchCV(
            _sklearn_estimator(model_key), spec["grid"], scoring="balanced_accuracy",
            cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED), n_jobs=-1,
        )
        start = time.time()
        search.fit(X_tr, y_tr, groups=g_tr)
        model = search.best_estimator_
        is_svm = model_key.startswith("svm")

        def score(X):
            return model.decision_function(X) if is_svm else model.predict_proba(X)[:, 1]

        X_val, y_val, _ = fold_xy(table, fold, "validation")
        threshold = 0.0 if is_svm else choose_threshold(y_val, score(X_val), split_name="validation")
        rec = {"fold": fold.number, "sizes": fold.sizes(), "best_params": {k: (str(v) if v is None else v) for k, v in search.best_params_.items()},
               "inner_cv_balanced_accuracy": float(search.best_score_), "threshold": threshold,
               "threshold_rule": "decision_score > 0" if is_svm else "P(abnormal) >= threshold (chosen on validation)",
               "fit_seconds": round(time.time() - start, 2)}
        for split in ("train", "validation", "test"):
            X, y, _ = fold_xy(table, fold, split)
            s = score(X)
            pred = (s > threshold) if is_svm else (s >= threshold)
            rec[split] = compute_metrics(y, pred, s)
        out = MODELS_DIR / model_key / f"fold_{fold.number}" / "model.joblib"
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out)
        rec["model_path"] = str(out.relative_to(PROJECT_ROOT))
        records.append(rec)
    params = {"features": "duration, snr, mel_1..mel_20", "class_weight": "balanced", "random_state": SEED,
              "search_grid": spec["grid"], "search_scoring": "balanced_accuracy", "inner_cv": "StratifiedGroupKFold(3) on train"}
    return _finish(model_key, spec["display"], "scikit-learn", "decision_score" if model_key.startswith("svm") else "probability",
                   params, folds, records)


# --------------------------------------------------------------------------- keras
def _keras_eval(model_key, display, folds, build, make_ds, params, epochs, batch_size):
    import tensorflow as tf
    from tensorflow.keras import callbacks

    records = []
    for fold in folds:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        y = {s: fold.labels(s) for s in ("train", "validation", "test")}
        counts = np.bincount(y["train"], minlength=2)
        class_weight = {i: float(len(y["train"]) / (2 * counts[i])) for i in (0, 1)}
        model = build()
        out = MODELS_DIR / model_key / f"fold_{fold.number}" / "best.keras"
        out.parent.mkdir(parents=True, exist_ok=True)
        cbs = [
            callbacks.ModelCheckpoint(str(out), monitor="val_loss", mode="min", save_best_only=True),
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        ]
        start = time.time()
        history = model.fit(make_ds(fold, "train", training=True, batch_size=batch_size), validation_data=make_ds(fold, "validation", training=False, batch_size=batch_size),
                            epochs=epochs, callbacks=cbs, class_weight=class_weight, verbose=2)
        best = tf.keras.models.load_model(out)
        prob = {s: best.predict(make_ds(fold, s, training=False, batch_size=batch_size), verbose=0).reshape(-1) for s in y}
        threshold = choose_threshold(y["validation"], prob["validation"], split_name="validation")
        rec = {"fold": fold.number, "sizes": fold.sizes(), "threshold": threshold,
               "threshold_rule": "P(abnormal) >= threshold (chosen on validation)",
               "epochs_run": len(history.history["loss"]), "fit_seconds": round(time.time() - start, 1),
               "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
               "model_path": str(out.relative_to(PROJECT_ROOT))}
        for s in y:
            rec[s] = compute_metrics(y[s], prob[s] >= threshold, prob[s])
        print(f"{model_key} fold {fold.number}: test recall={rec['test']['recall']:.3f} bal_acc={rec['test']['balanced_accuracy']:.3f}")
        records.append(rec)
    params.update({"epochs_max": epochs, "batch_size": batch_size, "early_stopping": "val_loss patience 5 (validation split)",
                   "class_weight": "balanced per fold", "seed": SEED})
    return _finish(model_key, display, "tensorflow-keras", "probability", params, folds, records)


CNN_IMAGE_SIZE = (40, 256)


def run_cnn(folds=None, epochs=30, batch_size=16, learning_rate=1e-4) -> dict:
    """Custom CNN of TP7-CNN_16Nov_grouped.ipynb (same architecture, inputs and augmentation)."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    folds = folds or load_folds()

    def augment(image):
        for axis, width in ((0, 4), (1, 24)):
            n = tf.shape(image)[axis]
            w = tf.random.uniform([], 0, width + 1, dtype=tf.int32)
            start = tf.random.uniform([], 0, n - w + 1, dtype=tf.int32)
            pos = tf.range(n)
            keep = tf.cast(tf.logical_or(pos < start, pos >= start + w), image.dtype)
            image = image * tf.reshape(keep, tf.stack([n, 1, 1]) if axis == 0 else tf.stack([1, n, 1]))
        return image

    def make_ds(fold, split, training, batch_size):
        ds = tf.data.Dataset.from_tensor_slices((fold.paths(split), fold.labels(split)))
        ds = ds.map(lambda p, l: (tf.cast(tf.image.resize(tf.io.decode_png(tf.io.read_file(p), channels=1), CNN_IMAGE_SIZE), tf.float32) / 255.0, l),
                    num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.shuffle(len(fold.split(split)), seed=10).map(lambda i, l: (augment(i), l), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def block(f):
        return [layers.Conv2D(f, 3, padding="same", use_bias=False), layers.BatchNormalization(), layers.Activation("relu")]

    def build():
        m = models.Sequential([layers.Input(shape=(*CNN_IMAGE_SIZE, 1)), *block(32), *block(32), layers.MaxPooling2D(),
                               *block(64), *block(64), layers.MaxPooling2D(), *block(128), layers.GlobalAveragePooling2D(),
                               layers.Dropout(0.35), layers.Dense(1, activation="sigmoid")], name="cardio_custom_cnn")
        m.compile(optimizer=optimizers.Adam(learning_rate), loss="binary_crossentropy", metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")])
        return m

    params = {"input": "mel40_sr2000_gray_v2 PNG resized 40x256x1, /255", "learning_rate": learning_rate,
              "augmentation": "time/frequency masking (train only)", "source_notebook": "TP7-CNN_16Nov_grouped.ipynb"}
    return _keras_eval("cnn_custom", "CNN", folds, build, make_ds, params, epochs, batch_size)


VGG_IMAGE_SIZE = (224, 224)


def vgg_preprocess(image_uint8_gray):
    """gray PNG tensor (H, W, 1) -> VGG16 input (224, 224, 3), ImageNet BGR mean-centred."""
    import tensorflow as tf

    image = tf.image.resize(tf.cast(image_uint8_gray, tf.float32), VGG_IMAGE_SIZE)
    return tf.keras.applications.vgg16.preprocess_input(tf.image.grayscale_to_rgb(image))


def run_vgg16(folds=None, epochs=30, batch_size=20, learning_rate=1e-6) -> dict:
    """VGG16 (ImageNet, frozen) + the TL notebook's head: Dropout .3, GAP, Dense 1024/512/128 relu,
    BatchNorm, sigmoid. Backbone features are computed once per image (frozen = identical maths)."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    folds = folds or load_folds()
    backbone = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(*VGG_IMAGE_SIZE, 3))
    backbone.trainable = False
    cache_path = PROJECT_ROOT / "Data" / "processed" / "features" / "vgg16_block5_mel40_gray_v2.npz"
    stems = sorted({r.recording_stem: PROJECT_ROOT / r.image_path for r in folds[0].rows}.items())
    if cache_path.is_file():
        cache = np.load(cache_path)
        feats = dict(zip(cache["stems"].tolist(), cache["features"]))
    else:
        ds = tf.data.Dataset.from_tensor_slices([str(p) for _, p in stems]).map(
            lambda p: vgg_preprocess(tf.io.decode_png(tf.io.read_file(p), channels=1))).batch(32)
        arr = backbone.predict(ds, verbose=1).astype(np.float16)
        np.savez_compressed(cache_path, stems=np.array([s for s, _ in stems]), features=arr)
        feats = dict(zip([s for s, _ in stems], arr))

    def make_ds(fold, split, training, batch_size):
        x = np.stack([feats[s] for s in fold.stems(split)]).astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((x, fold.labels(split)))
        if training:
            ds = ds.shuffle(len(x), seed=10)
        return ds.batch(batch_size)

    def build():
        inp = layers.Input(shape=backbone.output_shape[1:])
        x = layers.Dropout(0.3)(inp)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        m = models.Model(inp, out, name="vgg16_head")
        m.compile(optimizer=optimizers.Adam(learning_rate), loss="binary_crossentropy", metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")])
        return m

    params = {"backbone": "VGG16 imagenet, include_top=False, frozen", "input": "mel40_sr2000_gray_v2 PNG -> 224x224 RGB, vgg16.preprocess_input",
              "head": "Dropout(.3)-GAP-Dense1024-Dense512-Dense128-BN-sigmoid", "learning_rate": learning_rate,
              "source_notebook": "TP_heartbeatTransferLearning.ipynb"}
    result = _keras_eval("vgg16_transfer", "Transfer Learning (VGG16)", folds, build, make_ds, params, epochs, batch_size)
    # full end-to-end model per fold (raw gray PNG tensor -> probability) for the app
    for rec in result["folds"]:
        head = tf.keras.models.load_model(PROJECT_ROOT / rec["model_path"])
        inp = layers.Input(shape=(*VGG_IMAGE_SIZE, 3))
        full = models.Model(inp, head(backbone(inp)), name="vgg16_transfer_full")
        full_path = (PROJECT_ROOT / rec["model_path"]).with_name("full.keras")
        full.save(full_path)
        rec["full_model_path"] = str(full_path.relative_to(PROJECT_ROOT))
    _save_results("vgg16_transfer", result)
    return result


# --------------------------------------------------------------------------- notebook helpers
RUNNERS = {"svm_linear": lambda: run_sklearn("svm_linear"), "svm_rbf": lambda: run_sklearn("svm_rbf"),
           "decision_tree": lambda: run_sklearn("decision_tree"), "cnn_custom": run_cnn, "vgg16_transfer": run_vgg16}


def load_or_run(model_key: str, retrain: bool = False) -> dict:
    """Reuse saved fold results unless retrain=True (results are deterministic per manifest)."""
    path = RESULTS_DIR / f"{model_key}.json"
    if retrain or not path.is_file():
        return RUNNERS[model_key]()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["split_manifest_sha"] != manifest_sha():
        raise RuntimeError(f"{path} was computed on other folds; rerun with retrain=True")
    return payload


def fold_table(payload: dict):
    """One row per fold and split, plus mean/std rows, for display in notebooks."""
    import pandas as pd

    from audio_engine.evaluation.metrics import METRIC_NAMES

    rows = [{"fold": f["fold"], "split": s, "threshold": f["threshold"], **{m: f[s][m] for m in METRIC_NAMES}}
            for f in payload["folds"] for s in ("train", "validation", "test")]
    df = pd.DataFrame(rows)
    stats = df.groupby("split")[list(METRIC_NAMES)].agg(["mean", "std"])
    return df.round(3), stats.round(3)
