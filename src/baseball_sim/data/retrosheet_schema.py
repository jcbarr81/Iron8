"""
Canonical Retrosheet event schema used by scripts/ingest_retrosheet.py.

Only the column names are used by the ingester to normalize chunks and to
ensure all expected columns are present before writing. Types are informative.
"""

from __future__ import annotations

from typing import Dict


CANON_COLS: Dict[str, str] = {
    # identifiers
    "game_id": "string",
    "date": "string",
    "inning": "Int64",
    "half": "string",  # 'T'/'B'

    # actors
    "batter_retro_id": "string",
    "pitcher_retro_id": "string",

    # base/out state
    "outs_before": "Int64",
    "outs_after": "Int64",
    "b1_start": "object",
    "b2_start": "object",
    "b3_start": "object",

    # event info
    "event_cd": "Int64",
    "event_tx": "string",
    "rbi": "Int64",
    "runs_scored": "Int64",
    "event_num": "Int64",

    # teams/venue
    "home_team": "string",
    "away_team": "string",
    "park_id": "object",

    # flags (may be bool/str/object in source)
    "bat_event_fl": "object",
    "pa_new_fl": "object",
}

