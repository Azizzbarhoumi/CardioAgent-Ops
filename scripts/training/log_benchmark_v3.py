"""Phase 4 of the label fix: log the v3 results to MLflow (parent + 3 child runs per model),
mark them selected_for_app=true, and tag every v2 run label_inverted=true / selected_for_app=false.
Nothing is deleted. Usage: python scripts/training/log_benchmark_v3.py [--tracking-uri URI]
Default store is the local sqlite:///mlflow.db (backed up in backups/label_fix_20261121_123413).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from audio_engine.evaluation.mlflow_logging import EXPERIMENT, log_benchmark  # noqa: E402
from audio_engine.evaluation.runners import RESULTS_DIR  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
uri = sys.argv[sys.argv.index("--tracking-uri") + 1] if "--tracking-uri" in sys.argv else f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
mlflow.set_tracking_uri(uri)
client = MlflowClient()
exp = client.get_experiment_by_name(EXPERIMENT)

if client.search_runs([exp.experiment_id], "tags.benchmark_version = 'v3'", max_results=1):
    sys.exit("v3 runs already exist in this store; not logging twice.")

old = client.search_runs([exp.experiment_id], "tags.benchmark_version LIKE 'v2%'", max_results=1000)
print(len(old), "v2 runs to retag")
for r in old:
    client.set_tag(r.info.run_id, "label_inverted", "true")
    client.set_tag(r.info.run_id, "label_inverted_note", "PhysioNet -1/1 mapping was reversed; superseded by benchmark v3")
    if "selected_for_app" in r.data.tags:
        client.set_tag(r.info.run_id, "selected_for_app", "false")

runs = {}
for key in ["svm_linear", "svm_rbf", "decision_tree", "cnn_custom", "vgg16_transfer"]:
    payload = json.loads((RESULTS_DIR / f"{key}.json").read_text(encoding="utf-8"))
    runs[key] = log_benchmark(payload, selected_for_app=True)
    client.set_tag(runs[key]["parent_run_id"], "label_inverted", "false")
    print(key, runs[key]["parent_run_id"])
(RESULTS_DIR / ("mlflow_runs_v3_server.json" if uri.startswith("http") else "mlflow_runs_v3.json")).write_text(json.dumps({"tracking_uri": uri, "runs": runs}, indent=2), encoding="utf-8")
print("done")
