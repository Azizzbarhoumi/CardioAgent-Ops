"""Patient-grouped 3-fold manifests shared by every benchmark model.

The CSV manifests in ``Data/processed/splits/v2_patient_grouped`` are the single
source of truth. Notebooks only *load* them (``load_fold``); they are built once
by ``scripts/build_fold_manifests.py``.

Group rule (one rule for every recording):
* PASCAL dataset B, ``pascal_<cat>_<patientId>_<timestamp>_<position>``: the
  group is the patient number, ``pascal:patient:<patientId>``, whatever the category.
* PASCAL dataset A, ``pascal_<cat>_<YYYYMMDDhhmm>``: there is no patient number.
  Recordings made within ``PASCAL_A_SESSION_MINUTES`` of the previous one are put in
  the same session group, ``pascal:session:<first timestamp>``. This is a
  conservative stand-in for "same person".
* PhysioNet, ``physio_training-x_<record>``: there is no patient metadata, so the
  group is the recording, ``physionet:<set>:<record>``.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# v3: same rule as v2, rebuilt after the PhysioNet label fix (2026-11-21).
# v2_patient_grouped is kept for reference; its PhysioNet labels are inverted.
SPLITS_DIR = PROJECT_ROOT / "Data" / "processed" / "splits" / "v3_labels_fixed"
AUDIO_ROOT = PROJECT_ROOT / "ourdataset" / "train"
MEL_ROOT = PROJECT_ROOT / "Data" / "processed" / "features" / "mel_spectrograms"
IMAGE_FEATURE_VERSION = "mel40_sr2000_gray_v2"
CLASS_TO_LABEL = {"normal": 0, "abnormal": 1}
N_SPLITS = 3
N_VALIDATION_SPLITS = 5
OUTER_SEED = 10
PASCAL_A_SESSION_MINUTES = 15
SPLIT_NAMES = ("train", "validation", "test")
MANIFEST_FIELDS = ("split", "recording_stem", "group", "label", "class_name", "wav_path", "image_path")

_PASCAL_B = re.compile(r"^pascal_(normal|murmur|extra)_(\d+)_(\d+)_([A-Za-z0-9_ ]+)$")  # e.g. "_B", "_B_1", "_D 2"
_PASCAL_A = re.compile(r"^pascal_(normal|murmur|extra)_(\d{12})$")
_PHYSIO = re.compile(r"^physio_(training-[a-f])_(.+)$")


def parse_stem(stem: str) -> tuple[str, str]:
    """Return (kind, key) for one recording stem; raise if no rule matches."""
    if m := _PASCAL_B.match(stem):
        return "pascal_patient", m.group(2)
    if m := _PASCAL_A.match(stem):
        return "pascal_a", m.group(2)
    if m := _PHYSIO.match(stem):
        return "physionet", f"{m.group(1)}:{m.group(2)}"
    raise ValueError(f"No grouping rule for recording {stem!r}; refusing to split it at random.")


def assign_groups(stems: list[str]) -> dict[str, str]:
    """Map every recording stem to its patient/session/recording group."""
    groups: dict[str, str] = {}
    pascal_a: list[tuple[datetime, str]] = []
    for stem in stems:
        kind, key = parse_stem(stem)
        if kind == "pascal_patient":
            groups[stem] = f"pascal:patient:{key}"
        elif kind == "physionet":
            groups[stem] = f"physionet:{key}"
        else:
            pascal_a.append((datetime.strptime(key, "%Y%m%d%H%M"), stem))
    session_start, previous = None, None
    for when, stem in sorted(pascal_a):
        if previous is None or (when - previous).total_seconds() / 60 > PASCAL_A_SESSION_MINUTES:
            session_start = when
        groups[stem] = f"pascal:session:{session_start:%Y%m%d%H%M}"
        previous = when
    return groups


def list_recordings(audio_root: Path = AUDIO_ROOT) -> list[tuple[str, int, str]]:
    """(stem, label, class_name) for every wav, sorted for determinism."""
    out = []
    for class_name, label in CLASS_TO_LABEL.items():
        for wav in sorted((audio_root / class_name).glob("*.wav")):
            out.append((wav.stem, label, class_name))
    return out


def _rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def build_manifests(out_dir: Path = SPLITS_DIR) -> list[Path]:
    """Build the outer 3 folds plus one inner grouped validation split per fold."""
    from sklearn.model_selection import StratifiedGroupKFold

    records = list_recordings()
    stems = [r[0] for r in records]
    groups_map = assign_groups(stems)
    labels = np.array([r[1] for r in records])
    groups = np.array([groups_map[s] for s in stems])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    outer = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=OUTER_SEED)
    for fold, (train_val, test) in enumerate(outer.split(np.zeros(len(labels)), labels, groups), start=1):
        inner = StratifiedGroupKFold(n_splits=N_VALIDATION_SPLITS, shuffle=True, random_state=OUTER_SEED + fold)
        inner_train, inner_val = next(inner.split(np.zeros(len(train_val)), labels[train_val], groups[train_val]))
        split_of = {}
        for name, idx in (("train", train_val[inner_train]), ("validation", train_val[inner_val]), ("test", test)):
            for i in idx:
                split_of[int(i)] = name
        path = out_dir / f"fold_{fold}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
            writer.writeheader()
            for i, (stem, label, class_name) in enumerate(records):
                writer.writerow({
                    "split": split_of[i], "recording_stem": stem, "group": groups[i],
                    "label": label, "class_name": class_name,
                    "wav_path": _rel(AUDIO_ROOT / class_name / f"{stem}.wav"),
                    "image_path": _rel(MEL_ROOT / class_name / f"{stem}_{IMAGE_FEATURE_VERSION}.png"),
                })
        validate_fold(read_manifest(path))
        paths.append(path)
    return paths


@dataclass(frozen=True)
class FoldRow:
    split: str
    recording_stem: str
    group: str
    label: int
    class_name: str
    wav_path: str
    image_path: str


def read_manifest(path: Path) -> list[FoldRow]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return [FoldRow(**{**row, "label": int(row["label"])}) for row in csv.DictReader(fh)]


def manifest_path(fold: int, splits_dir: Path = SPLITS_DIR) -> Path:
    return Path(splits_dir) / f"fold_{fold}.csv"


def manifest_sha(fold: int | None = None, splits_dir: Path = SPLITS_DIR) -> str:
    """SHA-256 of one manifest, or of the 3 manifests concatenated in order."""
    digest = hashlib.sha256()
    for f in ([fold] if fold else range(1, N_SPLITS + 1)):
        digest.update(manifest_path(f, splits_dir).read_bytes())
    return digest.hexdigest()


def validate_fold(rows: list[FoldRow]) -> None:
    """Fail loudly on any group leakage, missing class or unknown split."""
    by_split: dict[str, list[FoldRow]] = {name: [] for name in SPLIT_NAMES}
    for row in rows:
        if row.split not in by_split:
            raise ValueError(f"Unknown split {row.split!r}")
        by_split[row.split].append(row)
    stems = [r.recording_stem for r in rows]
    if len(stems) != len(set(stems)):
        raise ValueError("A recording appears more than once in a fold manifest")
    group_sets = {name: {r.group for r in rs} for name, rs in by_split.items()}
    patient_sets = {
        name: {parse_stem(r.recording_stem)[1] for r in rs if parse_stem(r.recording_stem)[0] == "pascal_patient"}
        for name, rs in by_split.items()
    }
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if overlap := group_sets[a] & group_sets[b]:
            raise AssertionError(f"Group leakage between {a} and {b}: {sorted(overlap)[:5]}")
        if overlap := patient_sets[a] & patient_sets[b]:
            raise AssertionError(f"PASCAL patient leakage between {a} and {b}: {sorted(overlap)[:5]}")
    for name, rs in by_split.items():
        counts = np.bincount([r.label for r in rs], minlength=2)
        if (counts == 0).any():
            raise AssertionError(f"Split {name} is missing a class: {counts.tolist()}")


@dataclass
class Fold:
    number: int
    sha: str
    rows: list[FoldRow]

    def split(self, name: str) -> list[FoldRow]:
        return [r for r in self.rows if r.split == name]

    def stems(self, name: str) -> list[str]:
        return [r.recording_stem for r in self.split(name)]

    def labels(self, name: str) -> np.ndarray:
        return np.array([r.label for r in self.split(name)], dtype=np.int32)

    def paths(self, name: str, kind: str = "image_path") -> list[str]:
        return [str(PROJECT_ROOT / getattr(r, kind)) for r in self.split(name)]

    def sizes(self) -> dict[str, int]:
        return {name: len(self.split(name)) for name in SPLIT_NAMES}


def load_fold(fold: int, splits_dir: Path = SPLITS_DIR) -> Fold:
    """Load and validate one manifest. Never regenerates anything."""
    path = manifest_path(fold, splits_dir)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Run scripts/build_fold_manifests.py once.")
    rows = read_manifest(path)
    validate_fold(rows)
    return Fold(fold, manifest_sha(fold, splits_dir), rows)


def load_folds(splits_dir: Path = SPLITS_DIR) -> list[Fold]:
    return [load_fold(f, splits_dir) for f in range(1, N_SPLITS + 1)]


def fold_report(folds: list[Fold]) -> list[dict]:
    """Patients/groups, recordings and class balance per split, per fold."""
    out = []
    for fold in folds:
        for name in SPLIT_NAMES:
            rs = fold.split(name)
            labels = np.bincount([r.label for r in rs], minlength=2)
            out.append({
                "fold": fold.number, "split": name, "recordings": len(rs),
                "groups": len({r.group for r in rs}),
                "pascal_patients": len({r.group for r in rs if r.group.startswith("pascal:patient:")}),
                "normal": int(labels[0]), "abnormal": int(labels[1]),
                "abnormal_share": round(float(labels[1] / max(len(rs), 1)), 3),
            })
    return out
