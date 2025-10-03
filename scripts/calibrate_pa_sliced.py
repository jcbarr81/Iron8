"""
Fit per-slice MultiClass Platt calibrators from the held-out PA table
using the trained PA model's predicted probabilities. Writes
artifacts/pa_calibrator.joblib as a SlicedPlattCalibrator with a default
global calibrator and slice-specific overrides when sufficient data exists.

Slices: matchup (bats-throws) x roof (open/dome/retractable).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split

from baseball_sim.models.pa_model import PaOutcomeModel, CLASSES
from baseball_sim.features.ratings_adapter import batter_features, pitcher_features
from baseball_sim.features.context_features import context_features
from baseball_sim.utils.players import load_players_csv
from baseball_sim.calibration.calibrators import MultiClassPlattCalibrator, SlicedPlattCalibrator
from baseball_sim.calibration.slices import build_slice_meta, slice_key_from_meta
import yaml


PA_TABLE = ROOT / "artifacts" / "pa_table.parquet"
PLAYERS_CSV = ROOT / "data" / "players.csv"
PARKS_YAML = ROOT / "config" / "parks.yaml"
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


def main() -> int:
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
    parks = {}
    try:
        if PARKS_YAML.exists():
            parks = yaml.safe_load(PARKS_YAML.read_text(encoding="utf-8"))
            if not isinstance(parks, dict):
                parks = {}
    except Exception:
        parks = {}

    feat_rows: List[Dict[str, float]] = []
    slice_keys: List[str] = []
    labels: List[int] = []
    for _, r in df.iterrows():
        batter = {"player_id": r.get("batter_id"), "bats": None}
        pitcher = {"player_id": r.get("pitcher_id"), "throws": "R"}
        if batter["player_id"] in players:
            bpr = players[batter["player_id"]]
            batter["ratings"] = bpr
            # pick bats if available
            bats = bpr.get("bats") if isinstance(bpr, dict) else None
            if bats in ("L", "R"):
                batter["bats"] = bats
        if pitcher["player_id"] in players:
            pr = players[pitcher["player_id"]]
            if "arm" not in pr and "as" in pr:
                pr = {**pr, "arm": pr.get("as")}
                pr.pop("as", None)
            else:
                pr = {k: v for k, v in pr.items() if k != "as"}
            pitcher["ratings"] = pr

        # Build features
        b_feats = batter_features(batter)
        p_feats = pitcher_features(pitcher)
        s = to_state_row(r)
        c_feats = context_features(s, pitcher)
        feats = {**b_feats, **p_feats, **c_feats}
        feat_rows.append(feats)

        # Slice meta
        meta = build_slice_meta(s, batter, pitcher, parks)
        skey = f"matchup={meta.get('matchup')}|roof={meta.get('roof')}"
        slice_keys.append(skey)

        # Label
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
    slices = np.array(slice_keys)

    # Simple random split
    X_train, X_val, y_train, y_val, slices_train, slices_val = train_test_split(
        X, y, slices, test_size=0.2, random_state=42, stratify=y
    )

    if getattr(m, "_model", None) is None:
        print("Trained model is missing. Cannot calibrate.")
        return 1

    P_raw = m._model.predict_proba(X_val)

    # Fit default (global) calibrator
    default_cal = MultiClassPlattCalibrator(class_names=CLASSES)
    default_cal.fit(y_true=y_val, y_pred_proba=P_raw, class_names=CLASSES)

    # Fit per-slice calibrators for slices with sufficient support
    sliced = SlicedPlattCalibrator(class_names=CLASSES)
    sliced.default_model = default_cal

    # Aggregate by slice
    df_val = pd.DataFrame({"slice": slices_val, "y": y_val})
    # Indices for rows in each slice
    min_count = 500
    for skey, grp in df_val.groupby("slice"):
        idx = grp.index.values
        if len(idx) < min_count:
            continue
        cal = MultiClassPlattCalibrator(class_names=CLASSES)
        cal.fit(y_true=y_val[idx], y_pred_proba=P_raw[idx, :], class_names=CLASSES)
        sliced.models_by_slice[str(skey)] = cal

    out_path = ROOT / "artifacts" / "pa_calibrator.joblib"
    dump(sliced, out_path)
    print(f"Saved sliced calibrator with {len(sliced.models_by_slice)} slice models -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

