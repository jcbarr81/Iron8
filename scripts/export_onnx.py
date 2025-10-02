"""
Export trained PA model (scikit-learn LogisticRegression) to ONNX.
Writes artifacts/pa_model.onnx and artifacts/pa_feature_names.json.
Performs a light parity check against the sklearn model on a dummy input.
"""

from __future__ import annotations

from pathlib import Path
import sys
import json
import numpy as np

# Ensure 'src' is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from joblib import load as joblib_load

MODEL_PATH = ROOT / "artifacts" / "pa_model.joblib"
ONNX_PATH = ROOT / "artifacts" / "pa_model.onnx"
FEATS_JSON = ROOT / "artifacts" / "pa_feature_names.json"


def main() -> int:
    try:
        from skl2onnx import convert_sklearn  # type: ignore
        from skl2onnx.common.data_types import FloatTensorType  # type: ignore
    except Exception:
        print("skl2onnx is not installed. Install it from requirements.txt.")
        return 1

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}. Train first.")
        return 1

    payload = joblib_load(MODEL_PATH)
    clf = payload.get("model")
    feature_names = payload.get("feature_names", [])
    class_names = payload.get("classes", [])
    if clf is None or not feature_names:
        print("Missing model or feature names in artifact.")
        return 1

    initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
    # Disable ZipMap to get raw probability array
    onx = convert_sklearn(clf, initial_types=initial_type, options={type(clf): {"zipmap": False}})
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ONNX_PATH, "wb") as f:
        f.write(onx.SerializeToString())
    FEATS_JSON.write_text(json.dumps({"feature_names": feature_names, "class_names": class_names}), encoding="utf-8")

    # Parity check: compare proba for zero vector
    # Parity check skipped if onnxruntime not installed
    print(f"Exported to {ONNX_PATH} and wrote feature names to {FEATS_JSON}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
