"""Copy each model's app fold (best VALIDATION balanced accuracy) into audio_engine/artifacts/models
and write model_manifest.json (v3, PhysioNet labels fixed). The v2 manifest is kept as model_manifest_v2.json."""
import hashlib, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audio_engine.evaluation.folds import SPLITS_DIR
from audio_engine.evaluation.mlflow_logging import select_app_fold
from audio_engine.evaluation.runners import RESULTS_DIR
from audio_engine.features.audio_features import GRAY_MEL_FEATURE_VERSION, GRAY_MEL_SETTINGS, TABULAR_FEATURE_ORDER

MODELS = ROOT / "audio_engine" / "artifacts" / "models"
server = json.loads((RESULTS_DIR / "mlflow_runs_v3_server.json").read_text())
INPUTS = {
    "svm_linear": ("tabular_duration_snr_mel20_v1", "22 features: " + ", ".join(TABULAR_FEATURE_ORDER), "svm_linear_v3.joblib", "model_path"),
    "svm_rbf": ("tabular_duration_snr_mel20_v1", "22 features: " + ", ".join(TABULAR_FEATURE_ORDER), "svm_rbf_v3.joblib", "model_path"),
    "decision_tree": ("tabular_duration_snr_mel20_v1", "22 features: " + ", ".join(TABULAR_FEATURE_ORDER), "decision_tree_v3.joblib", "model_path"),
    "cnn_custom": (GRAY_MEL_FEATURE_VERSION, "gray log-Mel uint8 (40 x frames) resized to 40x256x1, /255", "cnn_custom_v3.keras", "model_path"),
    "vgg16_transfer": (GRAY_MEL_FEATURE_VERSION, "gray log-Mel uint8 resized to 224x224, gray->RGB, vgg16.preprocess_input", "vgg16_transfer_v3.keras", "full_model_path"),
}
manifest = {"benchmark_version": "v3", "experiment": "audio_benchmark", "evaluation": "patient_grouped_3_fold_cv",
            "split_manifest": SPLITS_DIR.relative_to(ROOT).as_posix(), "label_convention": {"0": "normal", "1": "abnormal"},
            "positive_class": "abnormal", "gray_mel_settings": GRAY_MEL_SETTINGS, "models": {}}
for key, (feature_version, input_desc, filename, path_key) in INPUTS.items():
    res = json.loads((RESULTS_DIR / f"{key}.json").read_text())
    fold = select_app_fold(res)
    src = ROOT / fold[path_key]
    dst = MODELS / filename
    shutil.copy2(src, dst)
    runs = server["runs"][key]
    manifest["split_manifest_sha"] = res["split_manifest_sha"]
    manifest["models"][key] = {
        "display_name": res["display_name"], "file": dst.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(), "framework": res["framework"],
        "input": input_desc, "feature_version": feature_version, "score_type": res["score_type"],
        "threshold": fold["threshold"], "label_rule": fold["threshold_rule"],
        "app_fold": fold["fold"], "app_fold_selection": "best validation balanced_accuracy",
        "training_run_id": runs["child_run_ids"][str(fold["fold"])] if str(fold["fold"]) in runs["child_run_ids"] else runs["child_run_ids"][fold["fold"]],
        "benchmark_parent_run_id": runs["parent_run_id"], "experiment_id": runs["experiment_id"],
        "kernel": {"svm_linear": "linear", "svm_rbf": "rbf"}.get(key),
        "app_fold_test_metrics": {m: fold["test"][m] for m in ("recall", "accuracy", "balanced_accuracy", "f1", "roc_auc", "specificity")},
        "cv_summary": {k: v for k, v in res["summary"].items() if k.startswith(("mean_test", "std_test"))},
    }
old = MODELS / "model_manifest.json"
if json.loads(old.read_text()).get("benchmark_version") == "v2" and not (MODELS / "model_manifest_v2.json").exists():
    shutil.copy2(old, MODELS / "model_manifest_v2.json")
(MODELS / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({k: (v["file"], v["app_fold"], round(v["threshold"], 4)) for k, v in manifest["models"].items()}, indent=1))
