"""
Quick preflight to ensure model artifacts exist and load correctly.

Checks:
- artifacts/pa_model.joblib can be loaded and returns probabilities.
- artifacts/pa_calibrator.joblib (optional) can be loaded.
- If config/settings.example.yaml (or configured path) has use_onnx: true,
  artifacts/pa_model.onnx and pa_feature_names.json are present and usable.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure 'src' is importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import json
import numpy as np
from joblib import load as joblib_load

from baseball_sim.config import load_settings
from baseball_sim.models.pa_model import PaOutcomeModel
from baseball_sim.serve.onnx_runtime import OnnxRunner


def main() -> int:
    settings = load_settings()  # uses config/settings.example.yaml by default

    ok = True
    # Joblib model
    mdl_path = ROOT / settings.pa_model_path
    if not mdl_path.exists():
        print(f"[FAIL] Missing model artifact: {mdl_path}")
        ok = False
    else:
        try:
            m = PaOutcomeModel()
            m.load(str(mdl_path))
            # Quick smoke prediction on zero features
            p = m.predict_proba({})
            if not isinstance(p, dict) or not p:
                print("[FAIL] Model returned empty probabilities")
                ok = False
            else:
                print(f"[OK] Loaded model: {mdl_path}")
        except Exception as e:
            print(f"[FAIL] Could not load/use model: {e}")
            ok = False

    # Calibrator (optional)
    calib_path = ROOT / settings.calibrator_path
    if calib_path.exists():
        try:
            _ = joblib_load(str(calib_path))
            print(f"[OK] Loaded calibrator: {calib_path}")
        except Exception as e:
            print(f"[FAIL] Could not load calibrator {calib_path}: {e}")
            ok = False
    else:
        print(f"[WARN] Calibrator not found (optional): {calib_path}")

    # ONNX path (if configured)
    if settings.use_onnx:
        onnx_path = ROOT / settings.onnx_path
        feats_json = ROOT / "artifacts" / "pa_feature_names.json"
        if not onnx_path.exists():
            print(f"[FAIL] ONNX enabled but file missing: {onnx_path}")
            ok = False
        if not feats_json.exists():
            print(f"[FAIL] Missing feature name json: {feats_json}")
            ok = False
        if onnx_path.exists() and feats_json.exists():
            try:
                data = json.loads(feats_json.read_text(encoding="utf-8"))
                runner = OnnxRunner(str(onnx_path), feature_names=data.get("feature_names", []), class_names=data.get("class_names", []))
                probs = runner.predict_proba({})
                if not isinstance(probs, dict) or not probs:
                    print("[FAIL] ONNX runner produced empty probabilities")
                    ok = False
                else:
                    print(f"[OK] ONNX runner loaded: {onnx_path}")
            except Exception as e:
                print(f"[FAIL] Could not init/use ONNX runner: {e}")
                ok = False

    print("[RESULT]", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

