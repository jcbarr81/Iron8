from typing import Dict, List
import numpy as np

CLASSES = ["BB","K","HBP","IP_OUT","1B","2B","3B","HR"]

class PaOutcomeModel:
    def __init__(self, version: str = "pa-1.0.0"):
        self.version = version
        self._fitted = False
        self._classes = CLASSES
        self._model = None  # replace with sklearn/gbdt
        self._feature_names: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        # TODO(Codex): Implement sklearn multinomial model or GBDT with monotonic constraints
        self._feature_names = feature_names
        self._fitted = True

    def predict_proba(self, x: Dict[str, float]) -> Dict[str, float]:
        # TODO(Codex): replace with real model; simple feature-sensitive baseline for now
        base = np.array([0.085,0.24,0.008,0.25,0.21,0.095,0.012,0.10], dtype=float)
        pit = float(x.get("pit_stuff_z", 0.0) or 0.0)
        powr = float(x.get("bat_power_z", 0.0) or 0.0)
        gb_bias = float(x.get("pit_gb_bias_z", 0.0) or 0.0)
        pl = float(x.get("bat_pull_z", 0.0) or 0.0)

        base[1] += 0.03 * pit                    # K
        base[7] += 0.04 * powr + 0.01 * pl       # HR
        base[3] += 0.01 * gb_bias                # IP_OUT up if GB bias
        base[3] -= 0.02 * powr                   # fewer outs with power
        base = np.clip(base, 1e-4, None)
        base = base / base.sum()
        return {c: float(v) for c, v in zip(self._classes, base)}

    def save(self, path: str):
        # TODO(Codex): joblib.dump(...)
        ...

    def load(self, path: str):
        # TODO(Codex): joblib.load(...)
        self._fitted = True
