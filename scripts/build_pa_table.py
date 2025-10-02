"""
Build a Plate Appearance (PA) training table from Retrosheet events.

Input: data/processed/retrosheet_events.csv
Output: artifacts/pa_table.parquet (one row per PA)

Columns in output:
- game_date, park_id, inning, half (T/B)
- batter_id, pitcher_id
- balls, strikes, outs
- base_1B, base_2B, base_3B (runner IDs or None)
- outcome_class in {BB,K,HBP,IP_OUT,1B,2B,3B,HR}

This script is resilient to schema differences; it attempts to detect
likely input columns.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd


# Resolve repo root so the script can be run from any working directory
ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "retrosheet_events.csv"
OUT_DIR = ROOT / "artifacts"
OUT_PATH = OUT_DIR / "pa_table.parquet"


CLASS_MAP_TEXT = {
    "walk": "BB",
    "intentional walk": "BB",
    "strikeout": "K",
    "hit by pitch": "HBP",
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "home run": "HR",
}


def to_half(row: pd.Series) -> str:
    # Try common forms: 'top/bottom', 'T/B', batting team vs home/away
    for k in ("inning_topbot", "half", "batting_team_half", "TOP_BOTTOM", "TB"):
        if k in row and pd.notna(row[k]):
            v = str(row[k]).strip().upper()
            if v.startswith("T") or v.startswith("TOP"):
                return "T"
            if v.startswith("B") or v.startswith("BOT"):
                return "B"
    # Fallback; if batting team is home -> B else T
    # cwevent default has BAT_HOME_ID: 1 when home is batting
    for key in ("BAT_HOME_ID", "bat_home_id", "bat_home_fl"):
        if key in row and pd.notna(row[key]):
            try:
                return "B" if int(row[key]) == 1 else "T"
            except Exception:
                pass
    bt = str(row.get("batting_team", "")).lower()
    if bt == "home":
        return "B"
    return "T"


def map_outcome(row: pd.Series) -> str:
    # 1) Use explicit numeric codes if present
    evcd = row.get("event_cd") if "event_cd" in row else row.get("EVENT_CD")
    try:
        if pd.notna(evcd):
            evcd = int(evcd)
            if evcd == 14:
                return "BB"
            if evcd == 3:
                return "K"
            if evcd == 16:
                return "HBP"
            if evcd in (20, 21, 22, 23):
                return {20: "1B", 21: "2B", 22: "3B", 23: "HR"}[evcd]
    except Exception:
        pass

    # 2) hit value column
    hv = row.get("hit_value") if "hit_value" in row else row.get("H_CD")
    try:
        if pd.notna(hv):
            hv = int(hv)
            if hv in (1, 2, 3, 4):
                return {1: "1B", 2: "2B", 3: "3B", 4: "HR"}[hv]
    except Exception:
        pass

    # 3) textual event type
    et = str(row.get("event_type", row.get("EVENT_TX", ""))).strip().lower()
    if et:
        for key, cls in CLASS_MAP_TEXT.items():
            if key in et:
                return cls

    # Default
    return "IP_OUT"


def safe_get(obj, names: list[str], default=None):
    """Return column/field from a DataFrame/Series by first matching name.

    - If obj is a DataFrame: return the entire Series if the column exists.
    - If obj is a Series (row): return the scalar value if present and not NA.
    - Otherwise: default.
    """
    for n in names:
        if isinstance(obj, pd.DataFrame):
            if n in obj.columns:
                return obj[n]
        elif isinstance(obj, pd.Series):
            if n in obj and not pd.isna(obj[n]):
                return obj[n]
    return default


def main() -> int:
    if not IN_PATH.exists():
        print(f"Input not found: {IN_PATH}")
        print("Tip: Generate it via one of these paths:\n"
              "  - PowerShell: scripts/convert_retrosheet.ps1 (requires 'cwevent' in PATH)\n"
              "  - Python:     python scripts/ingest_retrosheet.py (reads data/raw/retrosheet/*.csv|.zip)")
        return 1
    df = pd.read_csv(IN_PATH)

    out = pd.DataFrame()
    out["game_date"] = safe_get(df, ["game_date", "date", "DATE"], None)
    out["park_id"] = safe_get(df, ["park_id", "home_park_id", "park", "PARK_ID"], None)
    out["inning"] = safe_get(df, ["inning", "INN_CT"], 1)
    out["half"] = df.apply(to_half, axis=1)
    out["batter_id"] = safe_get(df, ["batter_id", "bat_id", "batter", "BAT_ID"], None)
    out["pitcher_id"] = safe_get(df, ["pitcher_id", "pit_id", "pitcher", "PIT_ID"], None)
    out["balls"] = safe_get(df, ["balls", "b", "BALLS_CT"], 0)
    out["strikes"] = safe_get(df, ["strikes", "s", "STRIKES_CT"], 0)
    out["outs"] = safe_get(df, ["outs", "o", "OUTS_CT"], 0)
    out["base_1B"] = safe_get(df, ["base1_runner_id", "runner_on_1b", "on_1b", "BASE1_RUN_ID"], None)
    out["base_2B"] = safe_get(df, ["base2_runner_id", "runner_on_2b", "on_2b", "BASE2_RUN_ID"], None)
    out["base_3B"] = safe_get(df, ["base3_runner_id", "runner_on_3b", "on_3b", "BASE3_RUN_ID"], None)
    out["outcome_class"] = df.apply(map_outcome, axis=1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        out.to_parquet(OUT_PATH, index=False)
        print(f"Wrote {OUT_PATH} with {len(out)} rows")
    except Exception as e:
        # Fall back to CSV if parquet engine missing
        alt = OUT_DIR / "pa_table.csv"
        out.to_csv(alt, index=False)
        print(f"Parquet write failed ({e}); wrote CSV fallback: {alt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
