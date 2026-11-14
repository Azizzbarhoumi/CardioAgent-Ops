"""Resolve a Logged Model URI from an MLflow experiment when local exports are absent."""

from __future__ import annotations

import mlflow


def resolve_best_logged_model(
    model_name: str,
    *,
    tracking_uri: str | None = None,
    experiment_name: str = "audio_benchmark",
    metric: str = "mean_recall",
) -> dict:
    """Return the highest-metric READY Logged Model with the requested name.

    Models without the requested metric rank after models that have it; ties
    resolve to the most recently created model.
    """
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise LookupError(f"MLflow experiment {experiment_name!r} was not found.")
    models = mlflow.search_logged_models(
        experiment_ids=[experiment.experiment_id], output_format="list"
    )
    candidates = [
        item for item in models if item.name == model_name and item.status == "READY"
    ]
    if not candidates:
        raise LookupError(f"No READY Logged Model named {model_name!r} was found.")

    def rank(item):
        run = client.get_run(item.source_run_id)
        return (run.data.metrics.get(metric, float("-inf")), item.creation_timestamp)

    selected = max(candidates, key=rank)
    return {
        "name": selected.name,
        "model_id": selected.model_id,
        "run_id": selected.source_run_id,
        "model_uri": f"models:/{selected.model_id}",
        "artifact_location": selected.artifact_location,
    }
