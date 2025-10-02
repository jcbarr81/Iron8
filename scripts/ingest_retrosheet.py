"""
Ingest Retrosheet-like raw CSVs (and zipped CSVs), normalize to canonical schema,
and write a single processed CSV to data/processed/retrosheet_events.csv.

Robust to differing input schemas: missing columns become None/NA; multiple common
column name variants are accepted. Uses chunked reads for large files.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import zipfile

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Ensure 'src' is importable when running as a script
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
RAW_DIR = ROOT / "data" / "raw" / "retrosheet"
OUT_DIR = ROOT / "data" / "processed"
OUT_PATH = OUT_DIR / "retrosheet_events.csv"

# Canonical schema
from baseball_sim.data.retrosheet_schema import CANON_COLS


def _pick_col(df: pd.DataFrame, names: Iterable[str]) -> Optional[pd.Series]:
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def _to_tb(val) -> Optional[str]:
    if pd.isna(val):
        return None
    try:
        s = str(val).strip().lower()
    except Exception:
        return None
    if s in {"t", "top"}:
        return "T"
    if s in {"b", "bot", "bottom"}:
        return "B"
    # numeric encodings
    try:
        v = int(float(s))
        # Heuristic: 0->Top, 1->Bottom
        return "B" if v == 1 else "T"
    except Exception:
        return None


def _norm_half(df: pd.DataFrame) -> pd.Series:
    # Prefer explicit top/bottom indicator
    s = _pick_col(df, ["half", "top_bottom", "tb", "topbot"])
    if s is not None:
        return s.map(_to_tb)
    # Fallback: bat_home_id (1 => home batting => Bottom)
    bha = _pick_col(df, ["bat_home_id", "bat_home_fl"])  # 1/0 or True/False
    if bha is not None:
        return bha.apply(lambda x: "B" if (pd.notna(x) and (int(x) == 1)) else "T")
    return pd.Series([None] * len(df))


def _coerce_int(s: Optional[pd.Series]) -> pd.Series:
    if s is None:
        return pd.Series([pd.NA])
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _coerce_str(s: Optional[pd.Series]) -> pd.Series:
    if s is None:
        return pd.Series([None])
    return s.astype("string")


def _coerce_obj(s: Optional[pd.Series]) -> pd.Series:
    if s is None:
        return pd.Series([None])
    return s.astype("object")


def _normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    out: Dict[str, pd.Series] = {}

    out["game_id"] = _coerce_str(_pick_col(df, ["game_id", "g_id", "game_pk"]))
    out["date"] = _coerce_str(_pick_col(df, ["date", "game_date", "start_date"]))
    out["inning"] = _coerce_int(_pick_col(df, ["inning", "inn_ct", "inn"]))
    out["half"] = _norm_half(df)

    out["batter_retro_id"] = _coerce_str(
        _pick_col(df, ["batter_retro_id", "bat_id", "batter_id", "batter"])
    )
    out["pitcher_retro_id"] = _coerce_str(
        _pick_col(df, ["pitcher_retro_id", "pit_id", "pitcher_id", "pitcher"])
    )

    outs_before = _pick_col(df, ["outs_before", "outs_ct", "outs"])
    out["outs_before"] = _coerce_int(outs_before)

    # outs_after: prefer provided, else outs_before + event_outs_ct bounded to 3
    outs_after = _pick_col(df, ["outs_after"])
    if outs_after is None:
        eouts = _pick_col(df, ["event_outs_ct", "outs_on_play", "outs_made"])
        if outs_before is not None and eouts is not None:
            tmp = pd.to_numeric(outs_before, errors="coerce") + pd.to_numeric(
                eouts, errors="coerce"
            )
            outs_after = tmp.clip(upper=3)
    out["outs_after"] = _coerce_int(outs_after)

    out["b1_start"] = _coerce_obj(
        _pick_col(df, [
            "b1_start",
            "base1_run_id",
            "b1_run_id",
            "runner_on_1b",
            "on_first_id",
        ])
    )
    out["b2_start"] = _coerce_obj(
        _pick_col(df, [
            "b2_start",
            "base2_run_id",
            "b2_run_id",
            "runner_on_2b",
            "on_second_id",
        ])
    )
    out["b3_start"] = _coerce_obj(
        _pick_col(df, [
            "b3_start",
            "base3_run_id",
            "b3_run_id",
            "runner_on_3b",
            "on_third_id",
        ])
    )

    out["rbi"] = _coerce_int(_pick_col(df, ["rbi", "rbi_ct"]))
    out["runs_scored"] = _coerce_int(
        _pick_col(df, ["runs_scored", "runs", "runs_ct", "event_runs_ct"])
    )
    out["event_cd"] = _coerce_int(_pick_col(df, ["event_cd", "event_code"]))
    out["event_tx"] = _coerce_str(
        _pick_col(df, ["event_tx", "event_text", "play_desc", "event"])
    )

    out["home_team"] = _coerce_str(
        _pick_col(df, ["home_team", "home_team_id", "home", "home_team_code"])
    )
    out["away_team"] = _coerce_str(
        _pick_col(df, ["away_team", "away_team_id", "vis", "away_team_code"])
    )
    out["park_id"] = _coerce_obj(_pick_col(df, ["park_id", "park", "venue_id", "park_code"]))

    out["bat_event_fl"] = _coerce_obj(
        _pick_col(df, ["bat_event_fl", "is_atbat", "ab_flag", "ab_fl"])
    )
    out["pa_new_fl"] = _coerce_obj(
        _pick_col(df, ["pa_new_fl", "pa_flag", "start_pa", "new_pa_fl"])
    )
    out["event_num"] = _coerce_int(
        _pick_col(df, ["event_num", "event_number", "event_id", "seq"])
    )

    out_df = pd.DataFrame(out)
    # Ensure all canonical columns exist (in case any missing Series fallback created only 1-row)
    for k in CANON_COLS.keys():
        if k not in out_df.columns:
            out_df[k] = pd.Series([None] * len(df))
    return out_df[list(CANON_COLS.keys())]


def _iter_source_frames(path: Path, chunksize: int = 250_000):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                with zf.open(name) as f:
                    reader = pd.read_csv(f, low_memory=False, chunksize=chunksize)
                    if isinstance(reader, pd.DataFrame):
                        yield reader
                    else:
                        for chunk in reader:
                            yield chunk
    else:
        reader = pd.read_csv(path, low_memory=False, chunksize=chunksize)
        if isinstance(reader, pd.DataFrame):
            yield reader
        else:
            for chunk in reader:
                yield chunk


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        print(f"No raw retrosheet folder found at {RAW_DIR}. Nothing to ingest.")
        return 0

    sources = [p for p in RAW_DIR.glob("*.csv")] + [p for p in RAW_DIR.glob("*.zip")]
    if not sources:
        print(f"No retrosheet CSVs/zips found in {RAW_DIR}. Nothing to ingest.")
        return 0

    # Prepare write: remove existing output to avoid stale content
    if OUT_PATH.exists():
        OUT_PATH.unlink()

    total_rows = 0
    game_ids: set = set()
    evt_counts: Dict[int, int] = {}

    header_written = False
    for src in sources:
        print(f"Ingesting: {src}")
        try:
            for df in _iter_source_frames(src, chunksize=250_000):
                if df is None or df.empty:
                    continue
                n_before = len(df)
                norm = _normalize_chunk(df)

                # Append to output
                mode = "a" if header_written else "w"
                norm.to_csv(OUT_PATH, index=False, mode=mode, header=not header_written)
                header_written = True

                total_rows += len(norm)
                # Stats
                # game_id string; dropna then update set
                if "game_id" in norm.columns:
                    game_ids.update(norm["game_id"].dropna().unique().tolist())
                if "event_cd" in norm.columns:
                    vc = norm["event_cd"].dropna().value_counts()
                    for k, v in vc.items():
                        k_int = int(k)
                        evt_counts[k_int] = evt_counts.get(k_int, 0) + int(v)
        except Exception as e:
            print(f"Warning: failed to ingest {src}: {e}")
            continue

    if total_rows == 0:
        print("No rows ingested. Output not created.")
        return 0

    # Summary
    print(f"Ingested rows: {total_rows}")
    print(f"Distinct games: {len(game_ids)}")
    if evt_counts:
        print("event_cd value_counts (top 20):")
        for k in sorted(evt_counts.keys())[:20]:
            print(f"  {k}: {evt_counts[k]}")
    print(f"Wrote: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
