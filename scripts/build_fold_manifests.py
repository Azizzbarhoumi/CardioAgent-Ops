"""Build the patient-grouped fold manifests ONCE. Notebooks only load them.

Run with .venv (scikit-learn 1.9.1):  .venv\\Scripts\\python scripts\\build_fold_manifests.py
Refuses to overwrite existing manifests unless --force is given.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sklearn  # noqa: E402

from audio_engine.evaluation.folds import SPLITS_DIR, build_manifests, fold_report, load_folds, manifest_sha  # noqa: E402

if __name__ == "__main__":
    if any(SPLITS_DIR.glob("fold_*.csv")) and "--force" not in sys.argv:
        sys.exit(f"Manifests already exist in {SPLITS_DIR}; they are the source of truth. Use --force to rebuild.")
    print("scikit-learn", sklearn.__version__)
    for p in build_manifests():
        print("wrote", p)
    for row in fold_report(load_folds()):
        print(row)
    print("split_manifest_sha", manifest_sha())
