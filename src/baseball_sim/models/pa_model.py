from typing import Dict, List
import numpy as np

CLASSES = ["BB","K","HBP","IP_OUT","1B","2B","3B","HR"]

class PaOutcomeModel:
    def __init__(self, version: str = "pa-1.0.0"):
        self.version = version
        self._fitted = False
        self._classes = CLASSES
        self._model = None  # sklearn LogisticRegression or similar
        self._feature_names: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        # Placeholder; training occurs in scripts/train_pa.py. This stores metadata if provided.
        self._feature_names = feature_names
        self._fitted = True

    def predict_proba(self, x: Dict[str, float]) -> Dict[str, float]:
        # Use trained model if available; otherwise baseline heuristic.
        if self._model is not None and self._feature_names:
            # Build feature vector in saved order; default missing to 0.0
            row = np.array([float(x.get(name, 0.0) or 0.0) for name in self._feature_names], dtype=float)[
                np.newaxis, :
            ]
            try:
                p = self._model.predict_proba(row)[0]
                # Model trained with integer labels aligned to class indices
                p = np.asarray(p, dtype=float)
                p = p / (p.sum() or 1.0)
                return {c: float(v) for c, v in zip(self._classes, p)}
            except Exception:
                # Fall back to baseline if anything goes wrong
                pass

        # Baseline heuristic
        base = np.array([0.085, 0.24, 0.008, 0.25, 0.21, 0.095, 0.012, 0.10], dtype=float)
        pit = float(x.get("pit_stuff_z", 0.0) or 0.0)
        powr = float(x.get("bat_power_z", 0.0) or 0.0)
        gb_bias = float(x.get("pit_gb_bias_z", 0.0) or 0.0)
        pl = float(x.get("bat_pull_z", 0.0) or 0.0)
        risp = float(x.get("bat_risp_z", 0.0) or 0.0)  # already gated by context

        base[1] += 0.03 * pit  # K
        base[7] += 0.04 * powr + 0.01 * pl  # HR
        base[3] += 0.01 * gb_bias  # IP_OUT up if GB bias
        base[3] -= 0.02 * powr  # fewer outs with power
        # Modest RISP effect: slightly lower K, slightly higher 1B when RISP and batter has high sc
        base[1] -= 0.02 * risp  # reduce K a bit with RISP bonus
        base[4] += 0.03 * risp  # increase 1B a bit with RISP bonus
        base = np.clip(base, 1e-4, None)
        base = base / base.sum()
        return {c: float(v) for c, v in zip(self._classes, base)}

    def save(self, path: str):
        # Import lazily to avoid hard dependency during tests without artifacts
        from joblib import dump  # type: ignore

        payload = {
            "version": self.version,
            "feature_names": self._feature_names,
            "classes": self._classes,
            "model": self._model,
        }
        dump(payload, path)

    def load(self, path: str):
        # Import lazily to avoid hard dependency during tests without artifacts
        from joblib import load  # type: ignore

        payload = load(path)
        self.version = payload.get("version", self.version)
        self._feature_names = payload.get("feature_names", [])
        self._classes = payload.get("classes", CLASSES)
        self._model = payload.get("model", None)
        self._fitted = True if self._model is not None else False
