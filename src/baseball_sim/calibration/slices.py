from __future__ import annotations

from typing import Any, Dict


def build_slice_meta(state: Any, batter: Any, pitcher: Any, parks_cfg: Dict[str, Dict]) -> Dict[str, str]:
    """Construct a stable slice descriptor from context for calibration.

    Slices aim to be broad enough to have coverage but specific enough to capture
    systematic calibration differences.

    Keys:
    - matchup: "L-R", "L-L", "R-L", "R-R"
    - roof: "dome" | "retractable" | "open"
    """

    def _get(obj: Any, key: str, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    bats = _get(batter, "bats")
    throws = _get(pitcher, "throws")
    bats_u = str(bats).upper() if bats else "R"
    throws_u = str(throws).upper() if throws else "R"
    if bats_u not in {"L", "R"}:
        bats_u = "R"
    if throws_u not in {"L", "R"}:
        throws_u = "R"
    matchup = f"{bats_u}-{throws_u}"

    park_id = _get(state, "park_id")
    roof = "open"
    if park_id and isinstance(parks_cfg, dict):
        park = parks_cfg.get(str(park_id)) or {}
        r = str((park or {}).get("roof", "")).strip().lower()
        if r in {"dome", "retractable", "open"}:
            roof = r
    return {"matchup": matchup, "roof": roof}


def slice_key_from_meta(meta: Dict[str, str]) -> str:
    matchup = meta.get("matchup", "R-R")
    roof = meta.get("roof", "open")
    return f"matchup={matchup}|roof={roof}"

