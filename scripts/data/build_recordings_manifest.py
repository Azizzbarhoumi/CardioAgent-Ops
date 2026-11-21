"""Build a deterministic index of the PASCAL and PhysioNet source recordings."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


PHYSIONET_SUBSETS = ("training-a", "training-b", "training-c", "training-d", "training-e", "training-f")
# PhysioNet/CinC 2016 REFERENCE.csv convention: -1 = normal, 1 = abnormal
# (matches RECORDS-normal / RECORDS-abnormal). Fixed 2026-11-21; was inverted.
PHYSIONET_LABELS = {"-1": "normal", "1": "abnormal"}
PASCAL_LABELS = {"normal": "normal", "murmur": "abnormal", "extra": "abnormal"}
MANIFEST_FIELDS = ("recording_id", "source", "subset", "source_label", "label", "relative_path")


def build_manifest(project_root: Path) -> list[dict[str, str]]:
    """Validate source labels and return one manifest row per labeled WAV."""
    project_root = project_root.resolve()
    dataset_root = project_root / "Data" / "dataset"
    physionet_root = dataset_root / "physionet"
    pascal_root = dataset_root / "pascal"
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()

    for subset in PHYSIONET_SUBSETS:
        subset_root = physionet_root / subset
        reference_path = subset_root / "REFERENCE.csv"
        if not subset_root.is_dir():
            raise FileNotFoundError(f"PhysioNet subset not found: {subset_root}")
        if not reference_path.is_file():
            raise FileNotFoundError(f"PhysioNet label file not found: {reference_path}")

        labeled_stems: set[str] = set()
        with reference_path.open("r", encoding="utf-8-sig", newline="") as reference_file:
            for line_number, record in enumerate(csv.reader(reference_file), start=1):
                if len(record) != 2:
                    raise ValueError(f"{reference_path}:{line_number}: expected recording,label")
                recording, source_label = (value.strip() for value in record)
                if not recording or Path(recording).name != recording or Path(recording).suffix:
                    raise ValueError(f"{reference_path}:{line_number}: invalid recording ID {recording!r}")
                if source_label not in PHYSIONET_LABELS:
                    raise ValueError(
                        f"{reference_path}:{line_number}: unsupported PhysioNet label {source_label!r}"
                    )
                if recording in labeled_stems:
                    raise ValueError(f"{reference_path}:{line_number}: duplicate recording {recording!r}")
                labeled_stems.add(recording)

                wav_path = subset_root / f"{recording}.wav"
                if not wav_path.is_file():
                    raise FileNotFoundError(f"Labeled recording is missing: {wav_path}")
                rows.append(
                    _manifest_row(
                        project_root,
                        recording_id=f"physionet:{subset}:{recording}",
                        source="physionet",
                        subset=subset,
                        source_label=source_label,
                        label=PHYSIONET_LABELS[source_label],
                        wav_path=wav_path,
                    )
                )

        actual_stems = {path.stem for path in subset_root.rglob("*.wav")}
        if actual_stems != labeled_stems:
            unlisted = sorted(actual_stems - labeled_stems)
            absent = sorted(labeled_stems - actual_stems)
            raise ValueError(
                f"{subset}: REFERENCE.csv and WAV files differ; "
                f"unlisted WAVs={unlisted[:5]}, missing WAVs={absent[:5]}"
            )

    for subset, label in PASCAL_LABELS.items():
        class_root = pascal_root / subset
        if not class_root.is_dir():
            raise FileNotFoundError(f"PASCAL class folder not found: {class_root}")
        wav_paths = sorted(
            (path for path in class_root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"),
            key=lambda path: path.relative_to(class_root).as_posix().casefold(),
        )
        if not wav_paths:
            raise ValueError(f"No WAV recordings found in PASCAL class folder: {class_root}")
        for wav_path in wav_paths:
            relative_recording = wav_path.relative_to(class_root).as_posix()
            rows.append(
                _manifest_row(
                    project_root,
                    recording_id=f"pascal:{subset}:{relative_recording}",
                    source="pascal",
                    subset=subset,
                    source_label=subset,
                    label=label,
                    wav_path=wav_path,
                )
            )

    rows.sort(key=lambda row: (row["source"], row["subset"], row["relative_path"]))
    for row in rows:
        if row["recording_id"] in seen_ids:
            raise ValueError(f"Duplicate recording ID in dataset: {row['recording_id']}")
        if row["relative_path"] in seen_paths:
            raise ValueError(f"Recording path appears more than once: {row['relative_path']}")
        seen_ids.add(row["recording_id"])
        seen_paths.add(row["relative_path"])
    if not rows:
        raise ValueError(f"No source recordings found under {dataset_root}")
    return rows


def _manifest_row(
    project_root: Path,
    *,
    recording_id: str,
    source: str,
    subset: str,
    source_label: str,
    label: str,
    wav_path: Path,
) -> dict[str, str]:
    return {
        "recording_id": recording_id,
        "source": source,
        "subset": subset,
        "source_label": source_label,
        "label": label,
        "relative_path": wav_path.resolve().relative_to(project_root).as_posix(),
    }


def write_manifest(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing Data/dataset (defaults to this script's project).",
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    rows = build_manifest(project_root)
    output_path = project_root / "Data" / "labels" / "recordings.csv"
    write_manifest(rows, output_path)

    class_counts = Counter(row["label"] for row in rows)
    source_counts = Counter(row["source"] for row in rows)
    print(f"Validated {len(rows)} recordings: {dict(source_counts)}")
    print(f"Class counts: {dict(class_counts)}")
    print(f"Manifest written to {output_path}")


if __name__ == "__main__":
    main()
