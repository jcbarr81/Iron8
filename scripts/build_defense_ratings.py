from __future__ import annotations

"""
Build per-player, per-position defense ratings (0-99) from Baseball Savant exports.

Inputs (directories of CSVs; pass via CLI or leave defaults):
- OAA directory (outs above average): data/raw/savant/oaa
- IAS directory (infield arm strength): data/raw/savant/ias
- OAS directory (outfield arm strength, optional): data/raw/savant/oas
- POP directory (catcher pop time): data/raw/savant/pop

Output:
- data/derived/defense_ratings.csv with columns:
  player_id, player_name, position, oaa_rating, arm_rating, arm_rating_filled

Notes:
- Ratings are percentiles within position, scaled to 0-99 and rounded.
- POP time is inverted (faster is better) before percentile.
- If a player lacks an arm metric for a given position, arm_rating_filled defaults to 50.
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd


def _list_csvs(d: Optional[str]) -> List[Path]:
    if not d:
        return []
    p = Path(d)
    if not p.exists():
        return []
    return [q for q in p.glob("*.csv") if q.is_file()]


def _norm_pos(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().upper()
    # Sometimes 'Catcher' etc; trim to primary code
    mapping = {
        "CATCHER": "C",
        "FIRST BASE": "1B",
        "SECOND BASE": "2B",
        "THIRD BASE": "3B",
        "SHORTSTOP": "SS",
        "LEFT FIELD": "LF",
        "CENTER FIELD": "CF",
        "RIGHT FIELD": "RF",
    }
    return mapping.get(s, s)


def _col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for nm in cands:
        if nm.lower() in cols:
            return cols[nm.lower()]
    return None


def _load_oaa(dir_path: Optional[str], start_year: Optional[int], end_year: Optional[int]) -> pd.DataFrame:
    rows = []
    for f in _list_csvs(dir_path):
        df = pd.read_csv(f)
        pid = _col(df, "player_id", "playerid", "mlbam_id", "id")
        name = _col(df, "player_name", "name")
        pos = _col(df, "position", "pos", "primary_pos_formatted")
        year = _col(df, "season", "year", "game_year")
        oaa = _col(df, "outs_above_average", "oaa")
        if not (pid and pos and oaa):
            continue
        sub = df[[pid, pos, oaa] + ([name] if name else []) + ([year] if year else [])].copy()
        sub.columns = ["player_id", "position", "oaa"] + (["player_name"] if name else []) + (["season"] if year else [])
        if "season" in sub.columns and (start_year or end_year):
            try:
                if start_year:
                    sub = sub[sub["season"] >= int(start_year)]
                if end_year:
                    sub = sub[sub["season"] <= int(end_year)]
            except Exception:
                pass
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["player_id", "player_name", "position", "oaa"])    
    df = pd.concat(rows, ignore_index=True)
    df["position"] = df["position"].map(_norm_pos)
    # Aggregate by player+position (mean OAA across rows)
    agg = df.groupby(["player_id", "position"], as_index=False).agg({
        "oaa": "mean",
        **({"player_name": "first"} if "player_name" in df.columns else {}),
    })
    return agg


def _load_arm(dir_path: Optional[str], start_year: Optional[int], end_year: Optional[int]) -> pd.DataFrame:
    rows = []
    for f in _list_csvs(dir_path):
        df = pd.read_csv(f)
        pid = _col(df, "player_id", "playerid", "mlbam_id", "id")
        name = _col(df, "player_name", "fielder_name", "name")
        year = _col(df, "season", "year")
        # If this is a wide IAS export with per-position columns (arm_1b, arm_ss, arm_lf, ...)
        wide_cols = [c for c in df.columns if str(c).lower().startswith("arm_")]
        if pid and wide_cols:
            # Melt arms into long per-position mph
            uses = [pid] + ([name] if name else []) + ([year] if year else []) + wide_cols
            sub = df[uses].copy()
            sub = sub.melt(id_vars=[pid] + ([name] if name else []) + ([year] if year else []), value_vars=wide_cols, var_name="arm_pos", value_name="mph")
            sub = sub.dropna(subset=["mph"])  # keep rows with mph
            # Map arm_pos -> position code
            def map_pos(x: str) -> Optional[str]:
                s = str(x).lower().replace("arm_", "")
                m = {
                    "1b": "1B",
                    "2b": "2B",
                    "3b": "3B",
                    "ss": "SS",
                    "lf": "LF",
                    "cf": "CF",
                    "rf": "RF",
                    "inf": None,
                    "of": None,
                    "overall": None,
                }
                return m.get(s, None)
            sub["position"] = sub["arm_pos"].map(map_pos)
            sub = sub.dropna(subset=["position"]).copy()
            # Drop arm_pos and rename id columns
            sub = sub.drop(columns=["arm_pos"])  
            # Reorder columns to [player_id, position, mph, player_name?, season?]
            cols = [pid, "position", "mph"] + ([name] if name else []) + ([year] if year else [])
            sub = sub[cols]
            sub.columns = ["player_id", "position", "mph"] + (["player_name"] if name else []) + (["season"] if year else [])
        else:
            # Generic arm export (single mph column, maybe with position)
            pos = _col(df, "position", "pos")
            mph = _col(df, "avg_arm_strength", "arm_strength", "average_arm_strength", "mph", "avg_arm_str", "arm_overall")
            if not (pid and mph):
                continue
            sub = df[[pid] + ([pos] if pos else []) + [mph] + ([name] if name else []) + ([year] if year else [])].copy()
            cols = ["player_id"] + (["position"] if pos else []) + ["mph"] + (["player_name"] if name else []) + (["season"] if year else [])
            sub.columns = cols
        if "season" in sub.columns and (start_year or end_year):
            try:
                if start_year:
                    sub = sub[sub["season"] >= int(start_year)]
                if end_year:
                    sub = sub[sub["season"] <= int(end_year)]
            except Exception:
                pass
        if "position" in sub.columns:
            sub["position"] = sub["position"].map(_norm_pos)
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["player_id", "position", "mph"])    
    df = pd.concat(rows, ignore_index=True)
    # If no position present (some exports may be OF-only or IF-only), leave position as-is
    # Aggregate mean mph by player+position (if present), else by player
    keys = ["player_id"] + (["position"] if "position" in df.columns else [])
    agg = df.groupby(keys, as_index=False).agg({
        "mph": "mean",
        **({"player_name": "first"} if "player_name" in df.columns else {}),
    })
    return agg


def _load_pop(dir_path: Optional[str], start_year: Optional[int], end_year: Optional[int]) -> pd.DataFrame:
    rows = []
    for f in _list_csvs(dir_path):
        df = pd.read_csv(f)
        pid = _col(df, "player_id", "playerid", "mlbam_id", "id")
        name = _col(df, "player_name", "name")
        year = _col(df, "season", "year")
        pop2b = _col(df, "pop_time_2b", "pop_time_s_2b", "poptime_2b_s", "avg_pop_time_2b")
        if not (pid and pop2b):
            continue
        sub = df[[pid, pop2b] + ([name] if name else []) + ([year] if year else [])].copy()
        sub.columns = ["player_id", "pop2b"] + (["player_name"] if name else []) + (["season"] if year else [])
        if "season" in sub.columns and (start_year or end_year):
            try:
                if start_year:
                    sub = sub[sub["season"] >= int(start_year)]
                if end_year:
                    sub = sub[sub["season"] <= int(end_year)]
            except Exception:
                pass
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["player_id", "pop2b"])    
    df = pd.concat(rows, ignore_index=True)
    agg = df.groupby(["player_id"], as_index=False).agg({
        "pop2b": "mean",
        **({"player_name": "first"} if "player_name" in df.columns else {}),
    })
    # Tag position as catcher
    agg["position"] = "C"
    return agg


def _percentile_by_pos(df: pd.DataFrame, value_col: str, pos_col: str = "position", invert: bool = False) -> pd.Series:
    def _rank(s: pd.Series) -> pd.Series:
        pct = s.rank(pct=True, method="average")
        return (1.0 - pct) if invert else pct
    return df.groupby(pos_col)[value_col].transform(_rank)


def main():
    ap = argparse.ArgumentParser(description="Build defense ratings (0-99) from Savant exports")
    ap.add_argument("--oaa-dir", default=str(Path("data/raw/savant/oaa")))
    ap.add_argument("--ias-dir", default=str(Path("data/raw/savant/ias")))
    ap.add_argument("--oas-dir", default=str(Path("data/raw/savant/oas")))
    ap.add_argument("--pop-dir", default=str(Path("data/raw/savant/pop")))
    ap.add_argument("--start-year", type=int, default=None)
    ap.add_argument("--end-year", type=int, default=None)
    ap.add_argument("--out", default=str(Path("data/derived/defense_ratings.csv")))
    args = ap.parse_args()

    oaa = _load_oaa(args.oaa_dir, args.start_year, args.end_year)
    ias = _load_arm(args.ias_dir, args.start_year, args.end_year)
    oas = _load_arm(args.oas_dir, args.start_year, args.end_year)  # may be empty
    pop = _load_pop(args.pop_dir, args.start_year, args.end_year)

    # Compute OAA percentiles within position
    if not oaa.empty:
        oaa["oaa_pct"] = _percentile_by_pos(oaa, "oaa")
        oaa["oaa_rating"] = (oaa["oaa_pct"] * 99).round().astype("Int64")

    # Arm ratings
    arm_frames = []
    if not ias.empty:
        ias["arm_pct"] = _percentile_by_pos(ias, "mph", pos_col="position") if "position" in ias.columns else ias["mph"].rank(pct=True)
        ias["arm_rating"] = (ias["arm_pct"] * 99).round().astype("Int64")
        arm_frames.append(ias)
    if not oas.empty:
        oas["arm_pct"] = _percentile_by_pos(oas, "mph", pos_col="position") if "position" in oas.columns else oas["mph"].rank(pct=True)
        oas["arm_rating"] = (oas["arm_pct"] * 99).round().astype("Int64")
        arm_frames.append(oas)
    arm = pd.DataFrame(columns=["player_id", "position", "arm_rating"])
    if arm_frames:
        arm = pd.concat([f[[c for c in ("player_id", "position", "arm_rating") if c in f.columns]] for f in arm_frames], ignore_index=True)
        # If we have per-player entries without position, drop (can't use for pos-specific block)
        if "position" in arm.columns:
            arm = arm.dropna(subset=["position"])  # keep only with position

    # Catcher arm from POP (invert: faster pop -> better rating)
    if not pop.empty:
        pop["pop_pct"] = pop["pop2b"].rank(pct=True)  # lower better; invert later
        pop["arm_rating"] = ((1.0 - pop["pop_pct"]) * 99).round().astype("Int64")
        # Keep only C position
        pop = pop[["player_id", "position", "arm_rating"]]

    # Merge OAA and arm; prefer position-specific arm
    # Build master list of positions
    positions = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]
    base = pd.DataFrame(columns=["player_id", "position"]).dropna()
    if not oaa.empty:
        base = pd.concat([base, oaa[["player_id", "position"]]], ignore_index=True)
    if not arm.empty:
        base = pd.concat([base, arm[["player_id", "position"]]], ignore_index=True)
    if not pop.empty:
        base = pd.concat([base, pop[["player_id", "position"]]], ignore_index=True)
    if base.empty:
        print("No input rows found; nothing to write.")
        return 0
    base = base.dropna().drop_duplicates()
    base = base[base["position"].isin(positions)]

    out = base.copy()
    if not oaa.empty:
        out = out.merge(oaa[["player_id", "position", "oaa_rating"]], on=["player_id", "position"], how="left")
    # Merge arm sources; catcher POP overrides any other C arm metric if present
    if not arm.empty:
        out = out.merge(arm[["player_id", "position", "arm_rating"]], on=["player_id", "position"], how="left")
    if not pop.empty:
        out = out.merge(pop[["player_id", "position", "arm_rating"]].rename(columns={"arm_rating": "arm_rating_pop"}), on=["player_id", "position"], how="left")
        # If position C, prefer pop arm
        is_c = out["position"] == "C"
        out.loc[is_c, "arm_rating"] = out.loc[is_c, "arm_rating_pop"]
        out = out.drop(columns=["arm_rating_pop"])

    # Fill missing arm with neutral 50
    out["arm_rating_filled"] = out["arm_rating"].fillna(50).round().astype("Int64")
    # Keep optional player_name where available (from OAA or arm)
    if "player_name" in oaa.columns:
        out = out.merge(oaa[["player_id", "player_name"]].drop_duplicates(), on="player_id", how="left")
    elif "player_name" in arm.columns:
        out = out.merge(arm[["player_id", "player_name"]].drop_duplicates(), on="player_id", how="left")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
