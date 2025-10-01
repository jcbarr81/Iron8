"""
Synthetic trainer for PA outcome model.

Generates synthetic samples with features aligned to live inference features,
builds logits with intuitive relationships, samples labels, trains multinomial
LogisticRegression, and saves to artifacts/pa_model.joblib.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import sys

# Ensure 'src' is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump

from baseball_sim.models.pa_model import CLASSES


RNG = np.random.default_rng(12345)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def _make_features(n: int) -> tuple[np.ndarray, List[str]]:
    # Construct features resembling live adapters
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
    # Baseline logits for 8 classes: BB, K, HBP, IP_OUT, 1B, 2B, 3B, HR
    n = X.shape[0]
    base = np.array([0.0, 0.3, -1.5, 0.6, 0.2, -0.4, -1.0, -1.2], dtype=float)
    Z = np.tile(base, (n, 1))

    name_to_idx = {k: i for i, k in enumerate(names)}
    def v(k: str) -> np.ndarray:
        return X[:, name_to_idx[k]]

    # Intuitive relationships
    Z[:, 1] += 0.9 * v("pit_stuff_z") + 0.4 * v("ctx_count_s")        # K
    Z[:, 0] += -0.8 * v("pit_control_z") + 0.5 * v("ctx_count_b")      # BB
    Z[:, 7] += 0.9 * v("bat_power_z") + 0.3 * v("bat_pull_z")          # HR
    Z[:, 3] += 0.6 * v("pit_gb_bias_z")                                 # IP_OUT

    # Small influences for hits
    Z[:, 4] += 0.2 * v("bat_contact_z")
    Z[:, 5] += 0.15 * v("bat_contact_z")
    Z[:, 6] += 0.10 * v("bat_contact_z")

    # Noise
    Z += RNG.normal(0.0, 0.25, size=Z.shape)
    return Z


def main():
    n = 20000
    X, names = _make_features(n)
    logits = _make_logits(X, names)
    P = _softmax(logits)

    # Sample labels according to synthetic probabilities
    y = np.array([RNG.choice(len(CLASSES), p=P[i]) for i in range(n)], dtype=int)

    # Train multinomial logistic regression
    clf = LogisticRegression(
        multi_class="multinomial",
        max_iter=500,
        solver="lbfgs",
        n_jobs=None,
    )
    clf.fit(X, y)

    # Save artifact
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pa_model.joblib"

    payload = {
        "version": "pa-1.0.0",
        "feature_names": names,
        "classes": CLASSES,
        "model": clf,
    }
    dump(payload, out_path)

    # Print class frequencies and path
    counts = np.bincount(y, minlength=len(CLASSES))
    freqs = counts / counts.sum()
    print("Class frequencies:")
    for c, f in zip(CLASSES, freqs):
        print(f"  {c}: {f:.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
