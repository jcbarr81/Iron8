from __future__ import annotations

from typing import Dict, Any, Optional
from .wp import leverage_index
from ..features.transforms import z_0_99


def _get_rating(entity: Dict[str, Any], key: str) -> Optional[int]:
    r = entity.get("ratings") if isinstance(entity, dict) else None
    if isinstance(r, dict):
        return r.get(key)
    return None


def _batter_score(batter: Dict[str, Any], pitcher: Dict[str, Any], state: Dict[str, Any]) -> float:
    # Simple expected value proxy: power + contact + platoon + RISP gate
    ph = z_0_99(_get_rating(batter, "ph") or 50) or 0.0
    ch = z_0_99(_get_rating(batter, "ch") or 50) or 0.0
    pl_adv = 1.0 if (str(batter.get("bats", "R")).upper() != str(pitcher.get("throws", "R")).upper()) else 0.0
    # RISP
    bases = state.get("bases", {}) if isinstance(state, dict) else {}
    risp = 1.0 if (bases.get("2B") is not None or bases.get("3B") is not None) else 0.0
    sc = z_0_99(_get_rating(batter, "sc") or 50) or 0.0
    return float(0.6 * ph + 0.3 * ch + 0.2 * pl_adv + 0.1 * sc * risp)


def advise_pinch_hit(state: Dict[str, Any], current_batter: Dict[str, Any], pitcher: Dict[str, Any], candidates: list[Dict[str, Any]]) -> Dict[str, Any]:
    cur = _batter_score(current_batter, pitcher, state)
    best_id = None
    best_score = cur
    for c in candidates:
        s = _batter_score(c, pitcher, state)
        if s > best_score:
            best_score = s
            best_id = c.get("player_id")
    delta = float(best_score - cur)
    recommend = bool(delta > 0.15)  # threshold
    return {"recommend": recommend, "best_candidate_id": best_id, "score_delta": delta}


def _pitcher_effective_stuff(pitcher: Dict[str, Any]) -> float:
    r = pitcher.get("ratings", {}) or {}
    arm = z_0_99(r.get("arm") or r.get("as") or 50) or 0.0
    fb = z_0_99(r.get("fb") or 50) or 0.0
    sl = z_0_99(r.get("sl") or 50) or 0.0
    ctrl = z_0_99(r.get("control") or 50) or 0.0
    return float(0.5 * arm + 0.3 * max(fb, sl) + 0.2 * ctrl)


def advise_bullpen(state: Dict[str, Any], upcoming_batter: Dict[str, Any], candidates: list[Dict[str, Any]]) -> Dict[str, Any]:
    bats = str(upcoming_batter.get("bats", "R")).upper()
    best = None
    best_score = -1e9
    for p in candidates:
        throws = str(p.get("throws", "R")).upper()
        platoon = 0.2 if (bats != throws) else -0.05
        eff = _pitcher_effective_stuff(p)
        score = eff + platoon
        if score > best_score:
            best_score = score
            best = p
    return {"recommend": True, "best_pitcher_id": (best or {}).get("player_id"), "rationale": "stuff+platoon"}


def advise_bunt(state: Dict[str, Any], batter: Dict[str, Any]) -> Dict[str, Any]:
    # Heuristic: 0 outs, runner on 1B (and not on 2B/3B), late innings, close score, weak power batter
    bases = state.get("bases", {}) if isinstance(state, dict) else {}
    outs = int(state.get("outs", 0) or 0)
    inning = int(state.get("inning", 1) or 1)
    score = state.get("score", {}) or {}
    diff = abs(int(score.get("home", 0) or 0) - int(score.get("away", 0) or 0))
    runner1 = bases.get("1B") is not None
    runner2 = bases.get("2B") is not None
    runner3 = bases.get("3B") is not None
    ph = (_get_rating(batter, "ph") or 50)

    li = leverage_index(state)
    recommend = bool(outs == 0 and runner1 and not (runner2 or runner3) and inning >= 7 and diff <= 1 and ph <= 55 and li >= 1.2)
    return {"recommend": recommend, "rationale": "0 outs, R1, late, close, weak power" if recommend else "skip"}


def advise_ibb(state: Dict[str, Any], batter: Dict[str, Any]) -> Dict[str, Any]:
    bases = state.get("bases", {}) if isinstance(state, dict) else {}
    inning = int(state.get("inning", 1) or 1)
    outs = int(state.get("outs", 0) or 0)
    score = state.get("score", {}) or {}
    diff = int(score.get("home", 0) or 0) - int(score.get("away", 0) or 0)
    first_open = bases.get("1B") is None
    risp = (bases.get("2B") is not None or bases.get("3B") is not None)
    power = (_get_rating(batter, "ph") or 50)

    # Recommend IBB if: RISP, 1B open, late innings, power slugger, tie/small deficit for defense
    recommend = bool(risp and first_open and inning >= 7 and power >= 80 and abs(diff) <= 1)
    return {"recommend": recommend, "rationale": "RISP, 1B open, late, slugger" if recommend else "skip"}

