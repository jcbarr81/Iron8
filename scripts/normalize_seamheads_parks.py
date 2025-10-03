from __future__ import annotations

"""
Normalize Seamheads Ballparks downloads into two CSVs used by our builder:

Inputs (default paths under data/raw/seamheads):
- Parks.csv: park metadata with Retrosheet-style codes and Altitude
- ParkConfig.csv: per-year park configuration (dimensions, wall heights, surface, cover, foul area)

Outputs:
- data/raw/seamheads_ballparks.csv with columns:
    retro_park_id, altitude_ft, foul_territory_sqft, roof, surface
- data/raw/park_dimensions.csv with columns:
    park_id, lf_line_ft, lf_gap_ft, cf_ft, rf_gap_ft, rf_line_ft,
    lf_wall_ft, lf_gap_wall_ft, cf_wall_ft, rf_gap_wall_ft, rf_wall_ft

Usage:
  python scripts/normalize_seamheads_parks.py \
      --in-dir data/raw/seamheads \
      --out-seamheads data/raw/seamheads_ballparks.csv \
      --out-dims data/raw/park_dimensions.csv

Notes:
- For ParkConfig, the most recent Year per park_id is used.
- Foul area appears in thousands of square feet; we scale by 1000.
- Surface codes: N -> grass (natural), A -> turf (artificial); others passed through.
- Cover codes: O -> open, R -> retractable, D -> dome; others passed through.
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd


def _map_surface(code: Optional[str]) -> Optional[str]:
    if code is None:
        return None
    c = str(code).strip().upper()
    if not c:
        return None
    if c == "N":
        return "grass"
    if c == "A":
        return "turf"
    return c.lower()


def _map_roof(code: Optional[str]) -> Optional[str]:
    if code is None:
        return None
    c = str(code).strip().upper()
    if not c:
        return None
    if c == "O":
        return "open"
    if c == "R":
        return "retractable"
    if c == "D":
        return "dome"
    return c.lower()


def load_parks_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect columns: PARKID, Altitude; normalize
    cols = {c.lower(): c for c in df.columns}
    pid = cols.get("parkid") or cols.get("retro_park_id") or cols.get("retroparkid")
    alt = cols.get("altitude") or cols.get("elevation") or cols.get("altitude_ft")
    if not pid:
        raise ValueError("Parks.csv missing PARKID column")
    out = df[[pid] + ([alt] if alt else [])].copy()
    out.columns = ["retro_park_id"] + (["altitude_ft"] if alt else [])
    out["retro_park_id"] = out["retro_park_id"].astype(str).str.strip().str.upper()
    if "altitude_ft" in out.columns:
        out["altitude_ft"] = pd.to_numeric(out["altitude_ft"], errors="coerce")
    return out


def load_parkconfig_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Use most recent year row per parkID
    if "Year" in df.columns and "parkID" in df.columns:
        df = df.sort_values(["parkID", "Year"]).groupby("parkID", as_index=False).tail(1)
    return df


def build_seamheads_ballparks(parks: pd.DataFrame, cfg: pd.DataFrame) -> pd.DataFrame:
    # Join on park code
    # parks: retro_park_id, altitude_ft
    # cfg: parkID, Foul, Surface, Cover
    cols = {c.lower(): c for c in cfg.columns}
    pid = cols.get("parkid")
    foul = cols.get("foul")
    surface = cols.get("surface")
    cover = cols.get("cover")

    take_cols = [pid]
    for c in (foul, surface, cover):
        if c and c not in take_cols:
            take_cols.append(c)
    cc = cfg[take_cols].copy()
    cc.columns = ["retro_park_id"] + ["Foul", "Surface", "Cover"][: len(take_cols) - 1]

    cc["retro_park_id"] = cc["retro_park_id"].astype(str).str.strip().str.upper()
    if "Foul" in cc.columns:
        cc["foul_territory_sqft"] = pd.to_numeric(cc["Foul"], errors="coerce") * 1000.0
    else:
        cc["foul_territory_sqft"] = pd.NA

    cc["surface"] = cc["Surface"].map(_map_surface) if "Surface" in cc.columns else None
    cc["roof"] = cc["Cover"].map(_map_roof) if "Cover" in cc.columns else None

    merged = pd.merge(parks, cc[["retro_park_id", "foul_territory_sqft", "surface", "roof"]], on="retro_park_id", how="left")
    # Keep only relevant columns
    cols_out = ["retro_park_id"]
    if "altitude_ft" in merged.columns:
        cols_out.append("altitude_ft")
    cols_out += ["foul_territory_sqft", "roof", "surface"]
    return merged[cols_out]


def build_dimensions(cfg: pd.DataFrame) -> pd.DataFrame:
    # Map ParkConfig dimension columns into our concise schema
    c = {k.lower(): k for k in cfg.columns}
    def g(*names: str) -> Optional[str]:
        for n in names:
            if n in c:
                return c[n]
        return None

    pid = g("parkid")
    lf = g("lf_dim")
    lfa = g("lfa_dim")
    lc = g("lc_dim")
    lcc = g("lcc_dim")
    cf = g("cf_dim")
    rc = g("rc_dim")
    rcc = g("rcc_dim")
    rfa = g("rfa_dim")
    rf = g("rf_dim")
    # Walls
    lf_w = g("lf_w")
    lc_w = g("lc_w")
    cf_w = g("cf_w")
    rc_w = g("rc_w")
    rf_w = g("rf_w")

    take = [pid]
    for k in [lf, lfa, lc, lcc, cf, rfa, rc, rcc, rf, lf_w, lc_w, cf_w, rc_w, rf_w]:
        if k and k not in take:
            take.append(k)
    sub = cfg[take].copy()
    sub.columns = (["park_id"] + [
        "LF_Dim", "LFA_Dim", "LC_Dim", "LCC_Dim", "CF_Dim", "RFA_Dim", "RC_Dim", "RCC_Dim", "RF_Dim",
        "LF_W", "LC_W", "CF_W", "RC_W", "RF_W"
    ][: len(take)-1])

    # Choose gap distances preferring alley then LC/RC
    def coalesce_row(row, keys):
        for k in keys:
            v = row.get(k)
            if pd.notna(v):
                try:
                    return float(v)
                except Exception:
                    continue
        return None

    rows: list[Dict[str, Any]] = []
    for _, r in sub.iterrows():
        rid = str(r.get("park_id", "")).strip().upper()
        if not rid:
            continue
        lf_line = r.get("LF_Dim")
        rf_line = r.get("RF_Dim")
        cf_ft = r.get("CF_Dim")
        lf_gap = coalesce_row(r, ["LFA_Dim", "LC_Dim", "LCC_Dim"])
        rf_gap = coalesce_row(r, ["RFA_Dim", "RC_Dim", "RCC_Dim"])

        def num(x):
            try:
                return float(x)
            except Exception:
                return None

        row_out: Dict[str, Any] = {
            "park_id": rid,
            "lf_line_ft": num(lf_line),
            "lf_gap_ft": lf_gap,
            "cf_ft": num(cf_ft),
            "rf_gap_ft": rf_gap,
            "rf_line_ft": num(rf_line),
        }
        # Wall heights map: use LC_W/RC_W for gaps if present
        row_out.update({
            "lf_wall_ft": num(r.get("LF_W")),
            "lf_gap_wall_ft": num(r.get("LC_W")),
            "cf_wall_ft": num(r.get("CF_W")),
            "rf_gap_wall_ft": num(r.get("RC_W")),
            "rf_wall_ft": num(r.get("RF_W")),
        })
        rows.append(row_out)

    dims = pd.DataFrame(rows)
    # Drop rows with no core distances
    core = ["lf_line_ft", "cf_ft", "rf_line_ft"]
    if not dims.empty:
        all_na = dims[core].isna().all(axis=1)
        dims = dims.loc[~all_na].reset_index(drop=True)
    return dims


def main():
    ap = argparse.ArgumentParser(description="Normalize Seamheads ballparks into builder-friendly CSVs")
    ap.add_argument("--in-dir", default=str(Path("data") / "raw" / "seamheads"))
    ap.add_argument("--out-seamheads", default=str(Path("data") / "raw" / "seamheads_ballparks.csv"))
    ap.add_argument("--out-dims", default=str(Path("data") / "raw" / "park_dimensions.csv"))
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    parks_path = in_dir / "Parks.csv"
    cfg_path = in_dir / "ParkConfig.csv"

    if not parks_path.exists():
        raise FileNotFoundError(f"Missing {parks_path}; please place Seamheads Parks.csv in {in_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}; please place Seamheads ParkConfig.csv in {in_dir}")

    parks = load_parks_csv(parks_path)
    cfg = load_parkconfig_csv(cfg_path)

    seamheads_out = build_seamheads_ballparks(parks, cfg)
    dims_out = build_dimensions(cfg)

    out_seamheads = Path(args.out_seamheads)
    out_dims = Path(args.out_dims)
    out_seamheads.parent.mkdir(parents=True, exist_ok=True)
    out_dims.parent.mkdir(parents=True, exist_ok=True)

    seamheads_out.to_csv(out_seamheads, index=False)
    dims_out.to_csv(out_dims, index=False)

    print(f"Wrote {len(seamheads_out)} rows to {out_seamheads}")
    print(f"Wrote {len(dims_out)} rows to {out_dims}")


if __name__ == "__main__":
    main()

