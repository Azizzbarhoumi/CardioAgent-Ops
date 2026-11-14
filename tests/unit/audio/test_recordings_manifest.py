import csv
from pathlib import Path

import pytest

from scripts.data.build_recordings_manifest import build_manifest


def _create_dataset(root: Path) -> None:
    dataset_root = root / "Data" / "dataset"
    for subset in ("training-a", "training-b", "training-c", "training-d", "training-e", "training-f"):
        subset_root = dataset_root / "physionet" / subset
        subset_root.mkdir(parents=True)
        records = [("a0001", "1"), ("a0002", "-1")] if subset == "training-a" else []
        with (subset_root / "REFERENCE.csv").open("w", encoding="utf-8", newline="") as reference_file:
            csv.writer(reference_file).writerows(records)
        for recording, _ in records:
            (subset_root / f"{recording}.wav").touch()

    for category in ("normal", "murmur", "extra"):
        class_root = dataset_root / "pascal" / category
        class_root.mkdir(parents=True)
        (class_root / f"{category}_sample.wav").touch()


def test_build_manifest_maps_labels_and_uses_relative_paths(tmp_path: Path) -> None:
    _create_dataset(tmp_path)

    rows = build_manifest(tmp_path)

    by_id = {row["recording_id"]: row for row in rows}
    assert len(rows) == 5
    assert by_id["physionet:training-a:a0001"]["label"] == "normal"
    assert by_id["physionet:training-a:a0002"]["label"] == "abnormal"
    assert by_id["pascal:normal:normal_sample.wav"]["label"] == "normal"
    assert by_id["pascal:murmur:murmur_sample.wav"]["label"] == "abnormal"
    assert by_id["pascal:extra:extra_sample.wav"]["label"] == "abnormal"
    assert all(not Path(row["relative_path"]).is_absolute() for row in rows)


def test_build_manifest_rejects_missing_labeled_audio(tmp_path: Path) -> None:
    _create_dataset(tmp_path)
    (tmp_path / "Data" / "dataset" / "physionet" / "training-a" / "a0001.wav").unlink()

    with pytest.raises(FileNotFoundError, match="Labeled recording is missing"):
        build_manifest(tmp_path)
