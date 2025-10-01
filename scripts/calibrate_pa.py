"""
Fit per-class Platt calibrators on a synthetic validation set using
the trained PA model's predicted probabilities. Writes artifacts/pa_calibrator.joblib.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import sys

# Ensure 'src' is importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from joblib import dump

from baseball_sim.models.pa_model import PaOutcomeModel, CLASSES
from baseball_sim.calibration.calibrators import MultiClassPlattCalibrator


RNG = np.random.default_rng(24680)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def _make_features(n: int) -> tuple[np.ndarray, List[str]]:
    feats = {}
    # Batter
    feats["bat_contact_z"] = RNG.normal(0.0, 1.0, size=n)
    feats["bat_power_z"] = RNG.normal(0.0, 1.0, size=n)
    feats["bat_gbfb_z"] = RNG.normal(0.0, 1.0, size=n)
    feats["bat_pull_z"] = RNG.normal(0.0, 1.0, size=n)
    feats["bat_risp_z"] = RNG.normal(0.0, 1.0, size=n)
    feats["bat_field_z"] = RNG.normal(0.0, 1.0, size=n)
    feats["bat_arm_z"] = RNG.normal(0.0, 1.0, size=n)
    feats["bat_isLHB"] = RNG.integers(0, 2, size=n).astype(float)
    # Pitcher
    feats["pit_stamina_z"] = RNG.normal(0.0, 1.0, size=n)
    feats["pit_control_z"] = RNG.normal(0.0, 1.0, size=n)
    feats["pit_move_z"] = RNG.normal(0.0, 1.0, size=n)
    feats["pit_hold_z"] = RNG.normal(0.0, 1.0, size=n)
    feats["pit_field_z"] = RNG.normal(0.0, 1.0, size=n)
    feats["pit_arm_z"] = RNG.normal(0.0, 1.0, size=n)
    feats["pit_stuff_z"] = RNG.normal(0.0, 1.0, size=n)
    feats["pit_gb_bias_z"] = RNG.normal(0.0, 1.0, size=n)
    # Context
    feats["ctx_count_b"] = RNG.integers(0, 4, size=n).astype(float)
    feats["ctx_count_s"] = RNG.integers(0, 3, size=n).astype(float)
    feats["ctx_outs"] = RNG.integers(0, 3, size=n).astype(float)
    feats["ctx_on_first"] = RNG.integers(0, 2, size=n).astype(float)
    feats["ctx_on_second"] = RNG.integers(0, 2, size=n).astype(float)
    feats["ctx_on_third"] = RNG.integers(0, 2, size=n).astype(float)
    feats["ctx_is_risp"] = ((feats["ctx_on_second"] + feats["ctx_on_third"]) > 0).astype(float)
    feats["ctx_tto"] = RNG.integers(1, 4, size=n).astype(float)

    names = list(feats.keys())
    X = np.column_stack([feats[k] for k in names]).astype(float)
    return X, names


def _make_logits(X: np.ndarray, names: List[str]) -> np.ndarray:
    n = X.shape[0]
    base = np.array([0.0, 0.3, -1.5, 0.6, 0.2, -0.4, -1.0, -1.2], dtype=float)
    Z = np.tile(base, (n, 1))
    name_to_idx = {k: i for i, k in enumerate(names)}

    def v(k: str) -> np.ndarray:
        return X[:, name_to_idx[k]]

    Z[:, 1] += 0.9 * v("pit_stuff_z") + 0.4 * v("ctx_count_s")
    Z[:, 0] += -0.8 * v("pit_control_z") + 0.5 * v("ctx_count_b")
    Z[:, 7] += 0.9 * v("bat_power_z") + 0.3 * v("bat_pull_z")
    Z[:, 3] += 0.6 * v("pit_gb_bias_z")
    Z[:, 4] += 0.2 * v("bat_contact_z")
    Z[:, 5] += 0.15 * v("bat_contact_z")
    Z[:, 6] += 0.10 * v("bat_contact_z")
    Z += RNG.normal(0.0, 0.25, size=Z.shape)
    return Z


def main():
    model_path = ROOT / "artifacts" / "pa_model.joblib"
    if not model_path.exists():
        print(f"Model artifact not found: {model_path}. Run training first.")
        return

    m = PaOutcomeModel()
    m.load(str(model_path))

    # Validation set
    n = 20000
    X, names = _make_features(n)
    logits = _make_logits(X, names)
    P_true = _softmax(logits)
    y = np.array([RNG.choice(len(CLASSES), p=P_true[i]) for i in range(n)], dtype=int)

    # Predicted (raw) probabilities from the trained model
    if getattr(m, "_model", None) is not None and m._feature_names:
        # Ensure columns align to model's expected feature order
        idx_map = [names.index(f) for f in m._feature_names]
        X_aligned = X[:, idx_map]
        P_raw = m._model.predict_proba(X_aligned)
    else:
        # Fallback: slower path via per-row predict_proba
        P_raw = np.zeros((n, len(CLASSES)), dtype=float)
        for i in range(n):
            feats = {k: float(X[i, j]) for j, k in enumerate(names)}
            d = m.predict_proba(feats)
            P_raw[i] = np.array([d[c] for c in CLASSES], dtype=float)

    # Fit Platt calibrators
    calib = MultiClassPlattCalibrator(class_names=CLASSES)
    calib.fit(y_true=y, y_pred_proba=P_raw, class_names=CLASSES)

    # Save
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pa_calibrator.joblib"
    dump(calib, out_path)

    # Summary
    # Vectorized apply: per-class calibration then row-normalize
    P_cal = np.zeros_like(P_raw)
    for idx, cname in enumerate(CLASSES):
        mdl = calib.models.get(cname)
        if mdl is None:
            P_cal[:, idx] = P_raw[:, idx]
        else:
            classes = getattr(mdl, "classes_", np.array([0, 1]))
            pos_idx = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
            P_cal[:, idx] = mdl.predict_proba(P_raw[:, idx].reshape(-1, 1))[:, pos_idx]
    P_cal = P_cal / P_cal.sum(axis=1, keepdims=True)

    print("Calibration summary (mean raw -> mean cal | true rate):")
    for idx, cname in enumerate(CLASSES):
        raw_mean = P_raw[:, idx].mean()
        cal_mean = P_cal[:, idx].mean()
        true_rate = (y == idx).mean()
        print(f"  {cname}: {raw_mean:.3f} -> {cal_mean:.3f} | true={true_rate:.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
