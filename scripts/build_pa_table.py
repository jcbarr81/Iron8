import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH  = Path("data/processed/retrosheet_events.csv")
OUT_PATH = Path("data/processed/pa_table.csv")

# ---- helpers ------------------------------------------------------------

EVENT_MAP = {
    2: "BB",   # walk
    3: "BB",   # intentional walk
    4: "K",    # strikeout
    14: "HBP", # hit by pitch
    20: "1B",
    21: "2B",
    22: "3B",
    23: "HR",
}

def read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Did you run convert_retrosheet.ps1?")
    return pd.read_csv(path, low_memory=False)

def col(df: pd.DataFrame, *candidates: str) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    get = lambda *xs: col(df, *xs)

    out = pd.DataFrame(index=df.index)

    out["game_id"] = df[get("game_id")] if get("game_id") else pd.NA
    out["date"]    = df[get("game_dt","date","start_date")] if get("game_dt","date","start_date") else pd.NA
    out["inning"]  = pd.to_numeric(df[get("inn_ct","inning")] if get("inn_ct","inning") else np.nan, errors="coerce").astype("Int64")

    # half from bat_home_id (1=Bottom, 0=Top)
    if get("bat_home_id"):
        out["half"] = np.where(pd.to_numeric(df[get("bat_home_id")], errors="coerce")==1, "B", "T")
    else:
        out["half"] = "T"

    out["batter_retro_id"]  = df[get("bat_id","batter")] if get("bat_id","batter") else pd.NA
    out["pitcher_retro_id"] = df[get("pit_id","pitcher")] if get("pit_id","pitcher") else pd.NA

    out["outs_before"] = pd.to_numeric(df[get("outs_ct","outs_before")] if get("outs_ct","outs_before") else np.nan, errors="coerce").astype("Int64")

    out["b1_start"] = df[get("base1_run_id","runner_on_1b","b1_start")] if get("base1_run_id","runner_on_1b","b1_start") else pd.NA
    out["b2_start"] = df[get("base2_run_id","runner_on_2b","b2_start")] if get("base2_run_id","runner_on_2b","b2_start") else pd.NA
    out["b3_start"] = df[get("base3_run_id","runner_on_3b","b3_start")] if get("base3_run_id","runner_on_3b","b3_start") else pd.NA

    out["home_team"] = df[get("hometeam","home_team","home_team_id")] if get("hometeam","home_team","home_team_id") else pd.NA
    out["away_team"] = df[get("visteam","away_team","away_team_id")] if get("visteam","away_team","away_team_id") else pd.NA
    out["park_id"]   = df[get("park_id","park")] if get("park_id","park") else pd.NA

    # outcomes
    out["event_cd"] = pd.to_numeric(df[get("event_cd")] if get("event_cd") else np.nan, errors="coerce").astype("Int64")
    out["event_tx"] = df[get("event_tx","event")] if get("event_tx","event") else ""

    # RBI per event/PA
    out["rbi_ct"] = pd.to_numeric(df[get("rbi_ct","rbi","runs_batted_in")] if get("rbi_ct","rbi","runs_batted_in") else 0, errors="coerce").fillna(0).astype(int)

    # game-local ordering to build pa_id
    # prefer a sequence if present; else use natural order within game
    seq_col = get("bat_event_seq","event_num","seq")
    if seq_col:
        # ensure int order
        out["_order"] = pd.to_numeric(df[seq_col], errors="coerce")
    else:
        out["_order"] = np.arange(len(df))

    # stabilize types
    for s in ["game_id","date","batter_retro_id","pitcher_retro_id","home_team","away_team","park_id","b1_start","b2_start","b3_start"]:
        if s in out:
            out[s] = out[s].astype("string")

    return out

def map_outcome(event_cd: pd.Series, event_tx: pd.Series) -> pd.Series:
    # primary: numeric code
    out = event_cd.map(EVENT_MAP)

    # fallback: text contains keywords
    tx = event_tx.fillna("").str.upper()
    out = out.fillna(
        np.where(tx.str.contains("HIT BY PITCH|HBP"), "HBP",
        np.where(tx.str.contains("STRIKEOUT|STRUCK OUT|\\bK\\b"), "K",
        np.where(tx.str.contains("INTENTIONAL|IBB|WALK"), "BB",
        np.where(tx.str.contains("HOME RUN|HOMERUN|\\bHR\\b"), "HR",
        np.where(tx.str.contains("\\bTRIPLE\\b|\\b3B\\b"), "3B",
        np.where(tx.str.contains("\\bDOUBLE\\b|\\b2B\\b"), "2B",
        np.where(tx.str.contains("\\bSINGLE\\b|\\b1B\\b"), "1B", None)))))))
    )

    return out.fillna("IP_OUT")

def build_pa(df: pd.DataFrame) -> pd.DataFrame:
    # PA rows = any row with a valid event_cd (most robust across years)
    mask = df["event_cd"].notna()
    pa = df.loc[mask].copy()

    # per-game running index to make a unique pa_id (ordered)
    pa = pa.sort_values(["game_id","_order"], kind="mergesort")
    pa["pa_idx"] = pa.groupby("game_id").cumcount()
    pa["pa_id"]  = pa["game_id"].fillna("UNK") + "-" + pa["pa_idx"].astype(str)

    # map outcomes
    pa["final_outcome_class"] = map_outcome(pa["event_cd"], pa["event_tx"])

    # runs_scored for the PA
    pa["runs_scored"] = pa["rbi_ct"]

    # rename outs_before -> outs_start
    pa["outs_start"] = pa["outs_before"]
    keep = [
        "pa_id","game_id","date","inning","half","outs_start",
        "b1_start","b2_start","b3_start",
        "batter_retro_id","pitcher_retro_id",
        "home_team","away_team","park_id",
        "final_outcome_class","runs_scored"
    ]
    for k in keep:
        if k not in pa.columns:
            pa[k] = pd.NA
    pa = pa[keep]

    # tidy types
    pa["inning"] = pa["inning"].astype("Int64")
    pa["outs_start"] = pa["outs_start"].astype("Int64")

    return pa

def main():
    raw = read_events(IN_PATH)
    df  = normalize(raw)
    pa  = build_pa(df)

    print(f"Plate appearances: {len(pa)}")
    if len(pa):
        dist = pa["final_outcome_class"].value_counts().sort_index()
        for k, v in dist.items():
            print(f"  {k:7s}: {v:>8d} ({v/len(pa):5.1%})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pa.to_csv(OUT_PATH, index=False)
    print(f"Wrote: {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
