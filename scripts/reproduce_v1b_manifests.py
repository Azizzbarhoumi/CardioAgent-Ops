"""Re-create the 16 Nov grouped-by-filename folds (MLflow run d0b60123) as a REFERENCE only.

Exact copy of the split cell of TP7-CNN_16Nov_grouped.ipynb; needs scikit-learn 1.9.1.
Writes Data/processed/splits/v1b_grouped_by_filename/cnn_fold_*.csv. Not used by v2 models.
"""
import csv, re, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
MEL = ROOT / "Data/processed/features/mel_spectrograms"; V = "mel40_sr2000_gray_v2"
OUT = ROOT / "Data/processed/splits/v1b_grouped_by_filename"
recs = []
for cn, l in (("normal", 0), ("abnormal", 1)):
    for p in sorted((MEL / cn).glob(f"*_{V}.png")):
        s = p.name[: -len(V) - 5]
        m = re.match(r"^pascal_(?:normal|murmur|extra)_(\d+)_", s)
        m1 = re.match(r"^pascal_(?:normal|murmur|extra)_(\d+)$", s)
        m2 = re.match(r"^physio_(training-[a-f])_(.+)$", s)
        g = f"pascal:patient:{m.group(1)}" if m else (f"pascal:recording:{m1.group(1)}" if m1 else f"physionet:{m2.group(1)}:{m2.group(2)}")
        recs.append((p, l, cn, g))
y = np.array([r[1] for r in recs]); g = np.array([r[3] for r in recs])
OUT.mkdir(parents=True, exist_ok=True)
for f, (tv, te) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=10).split(np.zeros(len(y)), y, g), 1):
    it, iv = next(StratifiedGroupKFold(5, shuffle=True, random_state=10 + f).split(np.zeros(len(tv)), y[tv], g[tv]))
    with (OUT / f"cnn_fold_{f}.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=("split", "relative_path", "label", "class_name", "group"))
        w.writeheader()
        for name, idx in (("train", tv[it]), ("validation", tv[iv]), ("test", te)):
            for i in idx:
                p, l, cn, gg = recs[int(i)]
                w.writerow({"split": name, "relative_path": p.relative_to(ROOT).as_posix(), "label": l, "class_name": cn, "group": gg})
    print(f, len(it), len(iv), len(te), np.bincount(y[te]))
