"""
Fit per-class Platt calibrators on a held-out set from the real PA table using
the trained PA model's predicted probabilities. Writes artifacts/pa_calibrator.joblib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import sys

# Ensure 'src' is importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from baseball_sim.models.pa_model import PaOutcomeModel, CLASSES
from baseball_sim.features.ratings_adapter import batter_features, pitcher_features
from baseball_sim.features.context_features import context_features
from baseball_sim.utils.players import load_players_csv
from baseball_sim.calibration.calibrators import MultiClassPlattCalibrator


PA_TABLE = ROOT / "artifacts" / "pa_table.parquet"
PLAYERS_CSV = ROOT / "data" / "players.csv"
MODEL_PATH = ROOT / "artifacts" / "pa_model.joblib"


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


def main():
    if not MODEL_PATH.exists():
        print(f"Model artifact not found: {MODEL_PATH}. Run training first.")
        return 1
    if not PA_TABLE.exists():
        print(f"PA table not found: {PA_TABLE}. Build it first.")
        return 1

    m = PaOutcomeModel()
    m.load(str(MODEL_PATH))
    feature_order: List[str] = getattr(m, "_feature_names", [])

    df = pd.read_parquet(PA_TABLE)
    players = load_players_csv(str(PLAYERS_CSV)) if PLAYERS_CSV.exists() else {}

    feat_rows: List[Dict[str, float]] = []
    labels: List[int] = []
    for _, r in df.iterrows():
        batter = {"player_id": r.get("batter_id")}
        pitcher = {"player_id": r.get("pitcher_id"), "throws": "R"}
        if batter["player_id"] in players:
            batter["ratings"] = players[batter["player_id"]]
        if pitcher["player_id"] in players:
            pr = players[pitcher["player_id"]]
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

        cls = str(r.get("outcome_class")).upper()
        try:
            y_idx = CLASSES.index(cls)
        except ValueError:
            y_idx = CLASSES.index("IP_OUT")
        labels.append(y_idx)

    if not feature_order:
        feature_order = sorted({k for row in feat_rows for k in row.keys()})
    X = np.array([[float(row.get(k, 0.0) or 0.0) for k in feature_order] for row in feat_rows], dtype=float)
    y = np.array(labels, dtype=int)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if getattr(m, "_model", None) is None:
        print("Trained model is missing. Cannot calibrate.")
        return 1

    P_raw = m._model.predict_proba(X_val)
    calib = MultiClassPlattCalibrator(class_names=CLASSES)
    calib.fit(y_true=y_val, y_pred_proba=P_raw, class_names=CLASSES)

    out_path = ROOT / "artifacts" / "pa_calibrator.joblib"
    dump(calib, out_path)

    # Evaluate
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

    raw_ll = log_loss(y_val, P_raw, labels=list(range(len(CLASSES))))
    cal_ll = log_loss(y_val, P_cal, labels=list(range(len(CLASSES))))
    print(f"Log loss: raw={raw_ll:.4f} cal={cal_ll:.4f}")
    print("Calibration summary (mean raw -> mean cal):")
    for idx, cname in enumerate(CLASSES):
        raw_mean = P_raw[:, idx].mean()
        cal_mean = P_cal[:, idx].mean()
        print(f"  {cname}: {raw_mean:.3f} -> {cal_mean:.3f}")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
