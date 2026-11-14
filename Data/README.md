# Dataset layout and status

The source audio is kept in place and treated as read-only. Do not mix it with generated features,
model inputs, or train/validation/test splits.

## Current source data

```text
Data/
├── dataset/                         # original source files
│   ├── pascal/
│   │   ├── normal/                  # 351 WAV
│   │   ├── murmur/                  # 129 WAV
│   │   └── extra/                   # 65 WAV (extrastole)
│   └── physionet/
│       ├── training-a/ ... training-f/  # WAV + REFERENCE.csv
├── labels/
│   └── recordings.csv               # generated index and harmonized labels
├── interim/                         # reserved for temporary transformations
├── processed/
│   ├── audio/                       # reserved for normalized/resampled WAV
│   ├── features/                    # reserved for MFCC/Mel/ZCR features
│   └── splits/                      # reserved for reproducible split definitions
└── validation/                      # reserved for data-quality reports
```

`Data/__MACOSX/` and `.DS_Store` files are archive/operating-system metadata and are not training
data. The manifest builder ignores them. Original recordings are not moved or copied by the
builder.

## Label mapping and counts

- PhysioNet `REFERENCE.csv`: `1` maps to `normal`; `-1` maps to `abnormal`.
- PASCAL `normal` maps to `normal`; `murmur` and `extra` map to `abnormal`.
- Current source total: 3,240 PhysioNet + 545 PASCAL = 3,785 WAV files.
- Harmonized total: 1,016 normal + 2,769 abnormal.

`ourdataset/train/{normal,abnormal}` is a legacy, copied training layout used by existing notebooks.
It is not a held-out evaluation set. Do not use it as train and test data simultaneously.
The current SVM result uses random recording-level folds; because some sources contain multiple
recordings per patient, use patient-grouped splits before treating its cross-validation score as
an estimate of performance on unseen patients.

## Rebuild and verify the manifest

From the project root:

```powershell
python scripts/data/build_recordings_manifest.py
python -m pytest tests/unit/audio/test_recordings_manifest.py
```

The generated CSV uses project-relative paths and includes the source, source subset, original label,
and binary label. It is an index only; the recordings remain under `Data/dataset/`.
