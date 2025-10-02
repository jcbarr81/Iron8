"""
Train PA outcome model.

Behavior:
- If artifacts/pa_table.parquet exists, trains on that real PA table
  using ratings + context-derived features.
- Otherwise, falls back to synthetic training (for dev).

Writes artifacts/pa_model.joblib and prints class frequencies and validation
log loss (when real data path is used).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import sys

# Ensure 'src' is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from joblib import dump

from baseball_sim.models.pa_model import CLASSES, PaOutcomeModel
from baseball_sim.features.ratings_adapter import batter_features, pitcher_features
from baseball_sim.features.context_features import context_features
from baseball_sim.utils.players import load_players_csv


RNG = np.random.default_rng(12345)
PA_TABLE = ROOT / "artifacts" / "pa_table.parquet"
PLAYERS_CSV = ROOT / "data" / "players.csv"


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
    if PA_TABLE.exists():
        print(f"Loading PA table: {PA_TABLE}")
        df = pd.read_parquet(PA_TABLE)
        # Join ratings if available (for bats/throws and ratings)
        players = load_players_csv(str(PLAYERS_CSV)) if PLAYERS_CSV.exists() else {}

        feat_rows: List[Dict[str, float]] = []
        labels: List[int] = []

        def _safe_int(val, default: int = 0) -> int:
            try:
                return int(val) if not pd.isna(val) else default
            except Exception:
                return default

        def _safe_str(val, default: str = "") -> str:
            try:
                return str(val) if not pd.isna(val) else default
            except Exception:
                return default

        def _safe_base(val):
            # Return None if NaN/empty
            if pd.isna(val):
                return None
            return val

        def to_state_row(r: pd.Series) -> Dict:
            inning = _safe_int(r.get("inning", 1), 1)
            half = _safe_str(r.get("half", "T"), "T") or "T"
            if half and half.lower().startswith("b"):
                half = "B"
            elif half and half.lower().startswith("t"):
                half = "T"
            else:
                half = "T"
            outs = _safe_int(r.get("outs", 0), 0)
            balls = _safe_int(r.get("balls", 0), 0)
            strikes = _safe_int(r.get("strikes", 0), 0)
            park_id = r.get("park_id")
            state = {
                "inning": inning,
                "half": half,
                "outs": outs,
                "bases": {
                    "1B": _safe_base(r.get("base_1B")),
                    "2B": _safe_base(r.get("base_2B")),
                    "3B": _safe_base(r.get("base_3B")),
                },
                "score": {"away": 0, "home": 0},
                "count": {"balls": balls, "strikes": strikes},
                "park_id": park_id if not pd.isna(park_id) else None,
            }
            return state

        for _, r in df.iterrows():
            batter = {"player_id": r.get("batter_id")}
            pitcher = {"player_id": r.get("pitcher_id"), "throws": "R"}
            # Attach ratings if known
            if batter["player_id"] in players:
                batter["ratings"] = players[batter["player_id"]]
            if pitcher["player_id"] in players:
                pr = players[pitcher["player_id"]]
                # normalize 'as' -> 'arm'
                if "arm" not in pr and "as" in pr:
                    pr = {**pr, "arm": pr.get("as")}
                    pr.pop("as", None)
                else:
                    pr = {k: v for k, v in pr.items() if k != "as"}
                pitcher["ratings"] = pr

            b_feats = batter_features(batter)
            p_feats = pitcher_features(pitcher)
            c_feats = context_features(to_state_row(r), pitcher)
            feats = {**b_feats, **p_feats, **c_feats}
            feat_rows.append(feats)

            # Map label
            cls = str(r.get("outcome_class")).upper()
            try:
                y_idx = CLASSES.index(cls)
            except ValueError:
                y_idx = CLASSES.index("IP_OUT")
            labels.append(y_idx)

        # Vectorize features
        # Build consistent feature order
        all_keys = sorted({k for row in feat_rows for k in row.keys()})
        X = np.array([[float(row.get(k, 0.0) or 0.0) for k in all_keys] for row in feat_rows], dtype=float)
        y = np.array(labels, dtype=int)

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        clf = LogisticRegression(multi_class="multinomial", max_iter=500, solver="lbfgs")
        clf.fit(X_train, y_train)

        # Eval
        P_val = clf.predict_proba(X_val)
        ll = log_loss(y_val, P_val, labels=list(range(len(CLASSES))))
        counts = np.bincount(y, minlength=len(CLASSES))
        freqs = counts / counts.sum()
        print("Validation log loss:", f"{ll:.4f}")
        print("Class frequencies:")
        for c, f in zip(CLASSES, freqs):
            print(f"  {c}: {f:.3f}")

        # Save artifact
        out_dir = Path("artifacts")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "pa_model.joblib"
        payload = {
            "version": "pa-1.0.0",
            "feature_names": all_keys,
            "classes": CLASSES,
            "model": clf,
        }
        dump(payload, out_path)
        print(f"Saved: {out_path}")
        return

    # Fallback: synthetic
    n = 20000
    X, names = _make_features(n)
    logits = _make_logits(X, names)
    P = _softmax(logits)
    y = np.array([RNG.choice(len(CLASSES), p=P[i]) for i in range(n)], dtype=int)
    clf = LogisticRegression(multi_class="multinomial", max_iter=500, solver="lbfgs")
    clf.fit(X, y)
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pa_model.joblib"
    payload = {"version": "pa-1.0.0", "feature_names": names, "classes": CLASSES, "model": clf}
    dump(payload, out_path)
    counts = np.bincount(y, minlength=len(CLASSES))
    freqs = counts / counts.sum()
    print("Synthetic training path used.")
    print("Class frequencies:")
    for c, f in zip(CLASSES, freqs):
        print(f"  {c}: {f:.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
