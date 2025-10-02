from __future__ import annotations

from typing import Dict, Optional


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_fatigue_z(pitch_count: Optional[int], tto_index: Optional[int], entered_cold: Optional[bool], batters_faced_in_stint: Optional[int]) -> float:
    pc = int(pitch_count or 0)
    tto = int(tto_index or 1)
    cold = bool(entered_cold)
    bf = int(batters_faced_in_stint or 0)

    # Base fatigue from pitch count (0..100+ maps to ~0..2.0)
    pc_term = max(0.0, (pc - 30) / 35.0)  # starts after ~30 pitches
    # TTO term (after first time through order)
    tto_term = max(0.0, (tto - 1) * 0.3)
    # Warm-up penalty if cold and first batter or two
    wu_term = 0.0
    if cold and bf < 2:
        wu_term = 0.8 if bf == 0 else 0.4

    fatigue = pc_term + tto_term + wu_term
    return _clamp(fatigue, 0.0, 3.0)


def apply_fatigue(features: Dict[str, float], pitch_count: Optional[int], tto_index: Optional[int], entered_cold: Optional[bool], batters_faced_in_stint: Optional[int]) -> Dict[str, float]:
    """Apply fatigue adjustments to pitcher features.

    - Reduces `pit_control_z` and `pit_stuff_z` by scaled fatigue.
    - Adds `pit_fatigue_z` for downstream models.
    """
    fz = compute_fatigue_z(pitch_count, tto_index, entered_cold, batters_faced_in_stint)
    out = dict(features)
    if "pit_control_z" in out:
        out["pit_control_z"] = out["pit_control_z"] - 0.35 * fz
    if "pit_stuff_z" in out:
        out["pit_stuff_z"] = out["pit_stuff_z"] - 0.40 * fz
    out["pit_fatigue_z"] = fz
    return out

