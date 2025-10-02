from __future__ import annotations

from typing import Dict, List
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None


class OnnxRunner:
    def __init__(self, model_path: str, feature_names: List[str], class_names: List[str]):
        if ort is None:
            raise RuntimeError("onnxruntime is not available")
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])  # noqa: F401
        self.feature_names = feature_names
        self.class_names = class_names
        outs = self.sess.get_outputs()
        # Default to last output as probabilities
        self.prob_output_name = outs[-1].name
        for o in outs:
            if o.name.lower().startswith("probab"):
                self.prob_output_name = o.name
                break

    def predict_proba(self, feats: Dict[str, float]) -> Dict[str, float]:
        x = np.array([[float(feats.get(name, 0.0) or 0.0) for name in self.feature_names]], dtype=np.float32)
        inp = self.sess.get_inputs()[0].name
        probs = self.sess.run([self.prob_output_name], {inp: x})[0]  # shape (1, n_classes)
        p = np.asarray(probs, dtype=float).reshape(-1)
        p = p / (p.sum() or 1.0)
        return {c: float(v) for c, v in zip(self.class_names, p)}
