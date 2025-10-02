from typing import Dict, Type


# Canonical Retrosheet-like event columns for downstream processing
CANON_COLS: Dict[str, Type] = {
    "game_id": str,
    "date": str,
    "inning": int,
    "half": str,  # "T" or "B"
    "batter_retro_id": str,
    "pitcher_retro_id": str,
    "outs_before": int,
    "outs_after": int,
    "b1_start": object,
    "b2_start": object,
    "b3_start": object,
    "rbi": int,
    "runs_scored": int,
    "event_cd": int,
    "event_tx": str,
    "home_team": str,
    "away_team": str,
    "park_id": object,
    "bat_event_fl": object,
    "pa_new_fl": object,
    "event_num": int,
}

