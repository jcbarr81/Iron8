"""
Evaluation & Reporting

Computes overall metrics (log loss, Brier score) and reliability bins for the
trained PA outcome model (with and without calibrator if present).

Writes a text summary to artifacts/eval_report.txt and prints a brief summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import sys

# Ensure 'src' is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from baseball_sim.models.pa_model import PaOutcomeModel, CLASSES
from baseball_sim.features.ratings_adapter import batter_features, pitcher_features
from baseball_sim.features.context_features import context_features
from baseball_sim.utils.players import load_players_csv
from joblib import load as joblib_load


PA_TABLE = ROOT / "artifacts" / "pa_table.parquet"
PLAYERS_CSV = ROOT / "data" / "players.csv"
MODEL_PATH = ROOT / "artifacts" / "pa_model.joblib"
CAL_PATH = ROOT / "artifacts" / "pa_calibrator.joblib"
OUT_PATH = ROOT / "artifacts" / "eval_report.txt"


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


def brier_score(y_true: np.ndarray, P: np.ndarray) -> float:
    # y_true shape: (n,), class indices; P shape: (n, n_classes)
    n = y_true.shape[0]
    k = P.shape[1]
    onehot = np.zeros((n, k), dtype=float)
    onehot[np.arange(n), y_true] = 1.0
    return float(np.mean((P - onehot) ** 2))


def reliability_bins(y_true: np.ndarray, P: np.ndarray, n_bins: int = 10) -> List[str]:
    # Overall reliability on the winning class probability
    confid = P.max(axis=1)
    preds = P.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    lines: List[str] = ["Reliability bins (by max prob):"]
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confid >= lo) & (confid < hi) if i < n_bins - 1 else (confid >= lo) & (confid <= hi)
        count = int(mask.sum())
        if count == 0:
            lines.append(f"  [{lo:.1f},{hi:.1f}): n=0")
            continue
        acc = float((preds[mask] == y_true[mask]).mean())
        avg_conf = float(confid[mask].mean())
        lines.append(f"  [{lo:.1f},{hi:.1f}): n={count} acc={acc:.3f} conf={avg_conf:.3f}")
    return lines


def main() -> int:
    if not MODEL_PATH.exists() or not PA_TABLE.exists():
        print("Missing model or PA table. Train first.")
        return 1

    m = PaOutcomeModel()
    m.load(str(MODEL_PATH))
    feature_order: List[str] = getattr(m, "_feature_names", [])
    df = pd.read_parquet(PA_TABLE)
    players = load_players_csv(str(PLAYERS_CSV)) if PLAYERS_CSV.exists() else {}

    # Sample for speed if huge
    nmax = 200000
    if len(df) > nmax:
        df = df.sample(n=nmax, random_state=42)

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

    # Raw model probs
    if getattr(m, "_model", None) is None:
        print("Model missing.")
        return 1
    P_raw = m._model.predict_proba(X)

    # Calibrated probs (if calibrator exists)
    P_cal = None
    if CAL_PATH.exists():
        calib = joblib_load(str(CAL_PATH))
        # Apply per-class Platt scaler then renormalize
        P_tmp = np.zeros_like(P_raw)
        for idx, cname in enumerate(CLASSES):
            mdl = getattr(calib, 'models', {}).get(cname)
            if mdl is None:
                P_tmp[:, idx] = P_raw[:, idx]
            else:
                classes = getattr(mdl, 'classes_', np.array([0, 1]))
                pos_idx = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
                P_tmp[:, idx] = mdl.predict_proba(P_raw[:, idx].reshape(-1, 1))[:, pos_idx]
        P_cal = P_tmp / P_tmp.sum(axis=1, keepdims=True)

    # Metrics
    lines: List[str] = []
    def add_block(title: str, P: np.ndarray):
        ll = log_loss(y, P, labels=list(range(len(CLASSES))))
        bs = brier_score(y, P)
        lines.append(f"== {title} ==")
        lines.append(f"Log loss: {ll:.4f}")
        lines.append(f"Brier score: {bs:.4f}")
        # Reliability bins
        lines.extend(reliability_bins(y, P))
        lines.append("")

    add_block("Raw", P_raw)
    if P_cal is not None:
        add_block("Calibrated", P_cal)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print("\n".join(lines[:10]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
