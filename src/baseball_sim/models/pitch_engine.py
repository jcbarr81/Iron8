from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np


PITCH_TYPES = ["fb", "sl", "si", "cu", "cb", "scb", "kn"]


class PitchEngine:
    def __init__(self):
        ...

    @staticmethod
    def _select_pitch_type(pitcher_ratings: Dict[str, Any]) -> str:
        weights = []
        names = []
        for k in PITCH_TYPES:
            v = pitcher_ratings.get(k)
            if v is not None:
                names.append(k)
                weights.append(max(0.0, float(v)))
        if not names:
            return "fb"
        w = np.array(weights, dtype=float)
        if (w <= 0).all():
            return names[0]
        idx = int(np.argmax(w))
        return names[idx]

    @staticmethod
    def _probs_for_outcomes(features: Dict[str, float], count: Dict[str, int], umpire: Optional[Dict[str, float]] = None, edge: bool = False) -> Dict[str, float]:
        # Base distribution
        p = {
            "ball": 0.35,
            "called_strike": 0.20,
            "swing_miss": 0.12,
            "foul": 0.13,
            "in_play": 0.18,
            "hbp": 0.02,
        }
        ctrl = float(features.get("pit_control_z", 0.0) or 0.0)
        stuff = float(features.get("pit_stuff_z", 0.0) or 0.0)
        contact = float(features.get("bat_contact_z", 0.0) or 0.0)

        # Control reduces balls/HBP and increases called strikes
        p["ball"] += -0.07 * ctrl
        p["hbp"] += -0.01 * max(0.0, ctrl)
        p["called_strike"] += 0.04 * ctrl

        # Stuff pushes swing_miss up and in_play down
        p["swing_miss"] += 0.06 * stuff
        p["in_play"] += -0.05 * stuff

        # Batter contact reduces swing_miss, increases in_play
        p["swing_miss"] += -0.05 * contact
        p["in_play"] += 0.05 * contact

        # Count influence (simple): more strikes in fastball counts; more balls in 3-0
        balls = int(count.get("balls", 0) or 0)
        strikes = int(count.get("strikes", 0) or 0)
        if balls >= 3 and strikes == 0:
            p["ball"] += 0.05
        if balls == 0 and strikes == 2:
            p["swing_miss"] += 0.02

        # Umpire bias: overall and edge-specific shift between ball and called_strike
        overall = float((umpire or {}).get("overall_bias", 0.0) or 0.0)
        edge_bias = float((umpire or {}).get("edge_bias", 0.0) or 0.0)
        bias = overall + (edge_bias if edge else 0.0)
        if bias != 0.0:
            # shift small mass from ball to called_strike or vice versa
            shift = 0.06 * bias
            p["called_strike"] = max(1e-5, p["called_strike"] + shift)
            p["ball"] = max(1e-5, p["ball"] - shift)

        # Normalize
        arr = np.array(list(p.values()), dtype=float)
        arr = np.clip(arr, 1e-5, None)
        arr = arr / arr.sum()
        return {k: float(v) for k, v in zip(p.keys(), arr)}

    @staticmethod
    def _next_count(result: str, count: Dict[str, int]) -> Dict[str, int]:
        b = int(count.get("balls", 0) or 0)
        s = int(count.get("strikes", 0) or 0)
        if result == "ball":
            b = min(3, b + 1)
        elif result in ("called_strike", "swing_miss"):
            s = min(2, s + 1)
        elif result == "foul":
            s = min(2, max(s, 1))
        elif result in ("in_play", "hbp"):
            # Count resets on ball in play or HBP
            b, s = 0, 0
        return {"balls": b, "strikes": s}

    @staticmethod
    def _pre_pitch_event(state: Dict[str, Any], features: Dict[str, float]) -> Tuple[str, Dict[str, Any]]:
        # Simple: if runner on 1B and strong hold -> pickoff
        bases = state.get("bases", {}) if isinstance(state, dict) else {}
        has_1b = bases.get("1B") is not None
        hold = float(features.get("pit_hold_z", 0.0) or 0.0)
        if has_1b and hold > 1.5:  # z ~ >1.5 ~ rating ~ 72+
            bases_end = dict(bases)
            bases_end["1B"] = None
            return "PO", {"bases": bases_end, "outs_recorded": 1}
        return "no_play", {}

    def pitch(self,
              state: Dict[str, Any],
              batter: Dict[str, Any],
              pitcher: Dict[str, Any],
              features: Dict[str, float],
              seed: Optional[int] = None,
              umpire: Optional[Dict[str, float]] = None,
              edge: bool = False) -> Dict[str, Any]:
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        # Pre-pitch event
        ev, info = self._pre_pitch_event(state, features)
        if ev != "no_play":
            return {
                "pre_pitch": ev,
                "pre_info": info,
            }

        count = state.get("count", {"balls": 0, "strikes": 0})
        probs = self._probs_for_outcomes(features, count, umpire=umpire, edge=edge)
        outcomes = list(probs.keys())
        p = np.array([probs[k] for k in outcomes], dtype=float)
        idx = rng.choice(len(outcomes), p=p)
        result = outcomes[idx]

        pr = pitcher.get("ratings") or {}
        pitch_type = self._select_pitch_type(pr)
        next_count = self._next_count(result, count)

        return {
            "pitch_type": pitch_type,
            "result": result,
            "next_count": next_count,
            "probs": probs,
        }
