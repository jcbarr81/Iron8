import csv
from pathlib import Path
from typing import Dict, Optional


_RATING_KEYS = {
    # Batter
    "ch",
    "ph",
    "gf",
    "pl",
    "sc",
    "sp",  # speed is used to derive bat_speed_z
    "fa",
    "arm",
    "as",  # alias for arm in some sources
    # Pitcher
    "endurance",
    "control",
    "movement",
    "hold_runner",
    "fb",
    "sl",
    "si",
    "cu",
    "cb",
    "scb",
    "kn",
}


def _to_int(val: object) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, int):
        return val
    s = str(val).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def load_players_csv(path: str) -> Dict[str, Dict]:
    """
    Load players.csv into a dict: player_id -> {ratings...}.
    Normalize keys: if a pitcher uses 'as' for arm, map to 'arm'.
    Return {} on missing file; do not raise.
    """
    p = Path(path)
    if not p.exists():
        return {}

    cache: Dict[str, Dict] = {}
    try:
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = (row.get("player_id") or "").strip()
                if not pid:
                    continue
                ratings: Dict[str, int] = {}
                for k in _RATING_KEYS:
                    if k in row:
                        v = _to_int(row.get(k))
                        if v is not None:
                            ratings[k] = v
                # Normalize alias 'as' -> 'arm' if present
                if "arm" not in ratings and "as" in ratings:
                    ratings["arm"] = ratings.pop("as")
                else:
                    # Remove 'as' if both are present to avoid confusion
                    ratings.pop("as", None)

                cache[pid] = ratings
    except Exception:
        # On any read/parse error, return best-effort cache built so far
        return cache

    return cache


def get_ratings_for(player_id: str, cache: Dict[str, Dict]) -> Optional[Dict]:
    return cache.get(player_id)
