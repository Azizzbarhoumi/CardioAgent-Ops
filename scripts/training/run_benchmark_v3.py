"""Phase 3 of the label fix: retrain + evaluate all 5 models on the v3 folds.
Writes models to audio_engine/artifacts/models/v3 and results to audio_engine/evaluation/results/v3.
Does NOT log to MLflow (phase 4). Usage: python scripts\training\run_benchmark_v3.py [model ...]
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audio_engine.evaluation.runners import RUNNERS, load_or_run

keys = sys.argv[1:] or list(RUNNERS)
for k in keys:
    t = time.time(); p = load_or_run(k, retrain=True)
    s = p["summary"]
    print(f"{k}: {time.time()-t:.0f}s  " + "  ".join(f"{m}={v:.3f}" for m, v in s.items() if m.startswith("mean_test_")), flush=True)
