from __future__ import annotations

from typing import Any


def win_prob_home(state: Any) -> float:
    """Return a crude home-team win probability in [0,1].

    Factors: score diff, inning/half, outs, runners (RISP bonus).
    This is a placeholder heuristic suitable for testing integration only.
    """
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    inning = int(_get(state, "inning", 1) or 1)
    half = str(_get(state, "half", "T") or "T")
    outs = int(_get(state, "outs", 0) or 0)
    score = _get(state, "score", {}) or {}
    home_runs = int(score.get("home", 0) or 0)
    away_runs = int(score.get("away", 0) or 0)

    # Base prob ~50% + score diff effect
    diff = home_runs - away_runs
    p = 0.5 + 0.06 * diff

    # Inning leverage: later innings weight more
    late = max(0, inning - 5) * 0.02
    p += late * (1 if diff > 0 else (-1 if diff < 0 else 0))

    # Half-inning: bottom inning slight home edge when tied in late innings
    if inning >= 9 and diff == 0 and half == "B":
        p += 0.05

    # Outs: fewer outs with RISP improves chance modestly
    bases = _get(state, "bases") or {}
    risp = 1 if (bases.get("2B") is not None or bases.get("3B") is not None) else 0
    p += 0.02 * risp
    p -= 0.01 * outs

    # Clamp
    return max(0.0, min(1.0, float(p)))


def leverage_index(state: Any) -> float:
    """Crude leverage index proxy based on inning and score closeness.

    Returns a value ~[0.5, 3.0].
    """
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    inning = int(_get(state, "inning", 1) or 1)
    score = _get(state, "score", {}) or {}
    home_runs = int(score.get("home", 0) or 0)
    away_runs = int(score.get("away", 0) or 0)
    margin = abs(home_runs - away_runs)

    base = 0.8 + 0.1 * max(0, inning - 3)
    close = 0.6 if margin <= 1 else (0.3 if margin == 2 else 0.0)
    li = base + close
    return float(max(0.5, min(3.0, li)))

