from typing import Dict, List, Optional
import numpy as np


class IdentityCalibrator:
    """No-op calibrator that simply renormalizes probabilities."""

    def apply(
        self, probs: Dict[str, float], slice_meta: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        s = float(sum(probs.values()) or 1.0)
        return {k: (float(v) / s) for k, v in probs.items()}


class MultiClassPlattCalibrator:
    """Per-class Platt scaling using LogisticRegression on predicted probs.

    For each class k, fits a binary logistic model on (p_k -> y==k).
    At inference, transforms each p_k via its logistic model, then renormalizes.
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names: List[str] = class_names or []
        self.models: Dict[str, object] = {}

    def fit(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> "MultiClassPlattCalibrator":
        if class_names is not None:
            self.class_names = list(class_names)
        if not self.class_names:
            raise ValueError("class_names must be provided for calibration")

        # Lazy import to avoid hard dep during tests when unused
        from sklearn.linear_model import LogisticRegression  # type: ignore

        y_true = np.asarray(y_true, dtype=int)
        P = np.asarray(y_pred_proba, dtype=float)
        if P.ndim != 2 or P.shape[1] != len(self.class_names):
            raise ValueError("y_pred_proba shape does not match class_names length")

        self.models = {}
        for idx, cname in enumerate(self.class_names):
            y_bin = (y_true == idx).astype(int)
            Xk = P[:, idx].reshape(-1, 1)
            clf = LogisticRegression(solver="lbfgs", max_iter=200)
            clf.fit(Xk, y_bin)
            self.models[cname] = clf
        return self

    def apply(
        self, probs: Dict[str, float], slice_meta: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        if not self.models or not self.class_names:
            # Fallback to identity normalization
            s = float(sum(probs.values()) or 1.0)
            return {k: (float(v) / s) for k, v in probs.items()}

        # Calibrate each probability with its per-class model
        calibrated = {}
        for cname in self.class_names:
            p = float(probs.get(cname, 0.0) or 0.0)
            mdl = self.models.get(cname)
            if mdl is None:
                calibrated[cname] = p
                continue
            # Predict probability of the positive class (label=1)
            pr = mdl.predict_proba(np.array([[p]], dtype=float))[0]
            # Find index of positive class in model
            classes = getattr(mdl, "classes_", np.array([0, 1]))
            pos_idx = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
            calibrated[cname] = float(pr[pos_idx])

        s = float(sum(calibrated.values()) or 1.0)
        return {k: (v / s) for k, v in calibrated.items()}
