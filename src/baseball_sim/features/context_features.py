from typing import Dict, Any, Optional


def context_features(state: Any, pitcher: Any, batter: Optional[Any] = None) -> Dict[str, int]:
    """Return flat numeric features for the current PA.

    Keys: ctx_count_b, ctx_count_s, ctx_outs, ctx_on_first, ctx_on_second,
    ctx_on_third, ctx_is_risp, ctx_tto, ctx_matchup_is_platoon_adv,
    ctx_shift_restrictions, ctx_temp_f, ctx_wind_mph, ctx_wind_dir_deg.

    - ctx_on_first/second/third are 0/1 based on Bases occupancy
    - ctx_is_risp is 1 if 2B or 3B is occupied
    - ctx_tto comes from pitcher.tto_index (default 1 if None)
    - ctx_matchup_is_platoon_adv is 1 if batter handedness is opposite of
      pitcher throws (e.g., L vs R), else 0

    Only uses stdlib and typing; no training/runtime libraries.
    """

    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # Count
    count = _get(state, "count")
    balls = _get(count, "balls", 0)
    strikes = _get(count, "strikes", 0)

    # Outs
    outs = _get(state, "outs", 0)

    # Bases occupancy; support both Pydantic attrs (B1/B2/B3) and dict keys ("1B"/"2B"/"3B")
    bases = _get(state, "bases")

    def _base_occ(names: list[str]) -> int:
        if bases is None:
            return 0
        if isinstance(bases, dict):
            for n in names:
                if bases.get(n) is not None:
                    return 1
            return 0
        # object with attributes
        for n in names:
            if hasattr(bases, n) and getattr(bases, n) is not None:
                return 1
        return 0

    on_first = _base_occ(["B1", "1B"])
    on_second = _base_occ(["B2", "2B"])
    on_third = _base_occ(["B3", "3B"])
    is_risp = 1 if (on_second or on_third) else 0

    # TTO index from pitcher
    tto = _get(pitcher, "tto_index")
    if tto is None:
        tto = 1
    try:
        tto_val = int(tto)
    except Exception:
        tto_val = 1

    # Platoon advantage: opposite-handed (L vs R or R vs L) -> 1, else 0
    bats = _get(batter, "bats", None) if batter is not None else None
    throws = _get(pitcher, "throws", None)
    try:
        bats_u = str(bats).upper() if bats is not None else None
        throws_u = str(throws).upper() if throws is not None else None
        platoon_adv = 1 if (bats_u in {"L","R"} and throws_u in {"L","R"} and bats_u != throws_u) else 0
    except Exception:
        platoon_adv = 0

    # Rules: shift restrictions
    rules = _get(state, "rules")
    shift_restr = _get(rules, "shift_restrictions", True)
    try:
        shift_flag = 1 if bool(shift_restr) else 0
    except Exception:
        shift_flag = 1

    # Weather
    weather = _get(state, "weather")
    temp_f = _get(weather, "temp_f", None)
    wind_mph = _get(weather, "wind_mph", None)
    wind_dir = _get(weather, "wind_dir_deg", None)

    return {
        "ctx_count_b": int(balls) if balls is not None else 0,
        "ctx_count_s": int(strikes) if strikes is not None else 0,
        "ctx_outs": int(outs) if outs is not None else 0,
        "ctx_on_first": int(on_first),
        "ctx_on_second": int(on_second),
        "ctx_on_third": int(on_third),
        "ctx_is_risp": int(is_risp),
        "ctx_tto": int(tto_val),
        "ctx_matchup_is_platoon_adv": int(platoon_adv),
        "ctx_shift_restrictions": int(shift_flag),
        "ctx_temp_f": float(temp_f) if temp_f is not None else 70.0,
        "ctx_wind_mph": float(wind_mph) if wind_mph is not None else 0.0,
        "ctx_wind_dir_deg": float(wind_dir) if wind_dir is not None else 0.0,
    }
