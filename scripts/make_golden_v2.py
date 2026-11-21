"""Golden snapshot v2: expected outputs computed by the TRAINING-side pipeline
(saved PNG / tabular cache + exported model files), independent of the API code path."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import joblib, tensorflow as tf
from audio_engine.evaluation.folds import MEL_ROOT, IMAGE_FEATURE_VERSION, CLASS_TO_LABEL, AUDIO_ROOT
from audio_engine.evaluation.runners import vgg_preprocess
from audio_engine.evaluation.tabular import load_tabular
from audio_engine.features.audio_features import TABULAR_FEATURE_ORDER

manifest = json.loads((ROOT / "audio_engine/artifacts/models/model_manifest.json").read_text())
table = load_tabular()
models = {k: (joblib.load(ROOT / v["file"]) if v["framework"] == "scikit-learn" else tf.keras.models.load_model(ROOT / v["file"], compile=False)) for k, v in manifest["models"].items()}
out = {"description": __doc__, "label_convention": manifest["label_convention"], "tolerance": {"decision_score": 1e-6, "probability": 1e-4},
       "split_manifest_sha": manifest["split_manifest_sha"], "recordings": {}}
for wav in sorted((ROOT / "tests/fixtures/golden_v2/wavs").glob("*.wav")):
    stem = wav.stem
    cls = next(c for c in CLASS_TO_LABEL if (AUDIO_ROOT / c / wav.name).is_file())
    x = table.loc[[stem], list(TABULAR_FEATURE_ORDER)].to_numpy(float)
    png = tf.io.decode_png(tf.io.read_file(str(MEL_ROOT / cls / f"{stem}_{IMAGE_FEATURE_VERSION}.png")), channels=1)
    inputs = {"cnn_custom": (tf.cast(tf.image.resize(png, (40, 256)), tf.float32) / 255.0)[None], "vgg16_transfer": vgg_preprocess(png)[None]}
    rec = {"true_label": CLASS_TO_LABEL[cls], "source": stem.split("_")[0], "models": {}}
    import soundfile as sf
    rec["duration_s"] = sf.info(str(wav)).duration
    for k, info in manifest["models"].items():
        m = models[k]
        if k.startswith("svm"):
            s = float(m.decision_function(x)[0]); pred = int(s > info["threshold"])
        elif k == "decision_tree":
            s = float(m.predict_proba(x)[0][list(m.classes_).index(1)]); pred = int(s >= info["threshold"])
        else:
            s = float(np.asarray(m.predict(inputs[k], verbose=0)).reshape(-1)[0]); pred = int(s >= info["threshold"])
        rec["models"][k] = {"prediction": pred, "score": s, "score_type": info["score_type"], "threshold": info["threshold"]}
    out["recordings"][wav.name] = rec
(ROOT / "tests/fixtures/golden_v2/expected.json").write_text(json.dumps(out, indent=2))
for n, r in out["recordings"].items():
    print(f"{n:45} true={r['true_label']} dur={r['duration_s']:.1f}", {k: v["prediction"] for k, v in r["models"].items()})
