"""One-off fix (2026-11-21): put PhysioNet files in the class folder that matches
the corrected Data/labels/recordings.csv (-1 = normal, 1 = abnormal).

* Moves (never deletes) WAVs in ourdataset/train/<class>/ and mel PNGs in
  Data/processed/features/mel_spectrograms/<class>/. Every move is logged to
  backups/label_fix_moves.csv so it can be undone with --undo.
* Rewrites only the label column of the tabular feature cache (features unchanged).
Dry run by default; pass --apply to change files.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIO = ROOT / "ourdataset" / "train"
MEL = ROOT / "Data" / "processed" / "features" / "mel_spectrograms"
TABULAR = ROOT / "Data" / "processed" / "features" / "tabular_duration_snr_mel20_v1.csv"
LOG = ROOT / "backups" / "label_fix_moves.csv"
LABEL_INT = {"normal": 0, "abnormal": 1}


def correct_classes() -> dict[str, str]:
    with (ROOT / "Data" / "labels" / "recordings.csv").open(encoding="utf-8") as fh:
        return {
            f"physio_{r['subset']}_{Path(r['relative_path']).stem}": r["label"]
            for r in csv.DictReader(fh) if r["source"] == "physionet"
        }


def plan_moves(truth: dict[str, str]) -> list[tuple[Path, Path]]:
    moves = []
    for root in (AUDIO, MEL):
        for cls in ("normal", "abnormal"):
            for f in sorted((root / cls).iterdir()):
                if not f.name.startswith("physio_"):
                    continue
                stem = f.stem if root is AUDIO else f.name.split("_mel")[0]
                want = truth.get(stem)
                if want is None:
                    sys.exit(f"No label for {f}")
                if want != cls:
                    dst = root / want / f.name
                    if dst.exists():
                        sys.exit(f"Refusing to overwrite {dst}")
                    moves.append((f, dst))
    return moves


def main() -> None:
    apply = "--apply" in sys.argv
    if "--undo" in sys.argv:
        with LOG.open(encoding="utf-8") as fh:
            for r in reversed(list(csv.DictReader(fh))):
                shutil.move(ROOT / r["dst"], ROOT / r["src"])
        print("undone")
        return
    truth = correct_classes()
    moves = plan_moves(truth)
    print(f"{len(moves)} files to move "
          f"({sum(m[0].suffix == '.wav' for m in moves)} wav, {sum(m[0].suffix == '.png' for m in moves)} png)")

    import pandas as pd
    tab = pd.read_csv(TABULAR, dtype={"recording_stem": str}, float_precision="round_trip")
    phys = tab["recording_stem"].str.startswith("physio_")
    new = tab.loc[phys, "recording_stem"].map(lambda s: LABEL_INT[truth[s]])
    print(f"tabular: {int((tab.loc[phys, 'label'] != new).sum())} of {int(phys.sum())} PhysioNet labels change")
    if not apply:
        print("dry run; pass --apply")
        return
    LOG.parent.mkdir(exist_ok=True)
    if LOG.exists():
        sys.exit(f"{LOG} exists; already applied?")
    with LOG.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["src", "dst"])
        for src, dst in moves:
            shutil.move(src, dst)
            w.writerow([src.relative_to(ROOT).as_posix(), dst.relative_to(ROOT).as_posix()])
    tab.loc[phys, "label"] = new
    tab.to_csv(TABULAR, index=False, float_format="%.17g")
    print("applied; move log at", LOG)


if __name__ == "__main__":
    main()
