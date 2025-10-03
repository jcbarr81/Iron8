"""
Build config/parks.yaml from public datasets.

Sources used (manual downloads may be required for some):
- Retrosheet park codes (required): https://www.retrosheet.org/parkcode.txt
- Optional: Seamheads Ballparks CSV (register + download): https://www.seamheads.com/ballparks/
  - Used for elevation (altitude), orientation, foul territory, and sometimes dimensions.
- Optional: Dimensions CSV curated by you (columns documented below).
- Optional: Overrides YAML to fill roof/surface or correct specific parks.

Outputs a YAML keyed by Retrosheet PARKID with fields:
- name, city, state, year_from, year_to
- rf_short_porches, lf_short_porches (if dims available)
- roof, surface (if provided via overrides)
- altitude_ft (if available from Seamheads or overrides)
- dims: {lf_line_ft, lf_gap_ft, cf_ft, rf_gap_ft, rf_line_ft} (if available)
- wall_heights_ft: {lf, lf_gap, cf, rf_gap, rf} (if available)

Usage examples:
  python scripts/build_parks_yaml.py \
    --output config/parks.yaml

  python scripts/build_parks_yaml.py \
    --seamheads data/raw/seamheads_ballparks.csv \
    --dims data/raw/park_dimensions.csv \
    --overrides config/park_overrides.yaml \
    --output config/parks.yaml

Dimensions CSV schema (wide, simple):
  park_id, lf_line_ft, lf_gap_ft, cf_ft, rf_gap_ft, rf_line_ft, lf_wall_ft?, lf_gap_wall_ft?, cf_wall_ft?, rf_gap_wall_ft?, rf_wall_ft?

Overrides YAML example:
  BOS05:
    roof: open
    surface: grass
    altitude_ft: 20

Notes:
- This script is conservative: it only writes fields present in inputs.
- It never overwrites with nulls; overrides take precedence.
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import httpx
import yaml


RETRO_PARKCODES_URL = "https://www.retrosheet.org/parkcode.txt"


def fetch_retrosheet_parks(url: str = RETRO_PARKCODES_URL, timeout: int = 30) -> pd.DataFrame:
    """Download and parse Retrosheet park codes CSV into a DataFrame.

    Columns: PARKID, NAME, AKA, CITY, STATE, START, END, LEAGUE, NOTES
    """
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url)
        resp.raise_for_status()
        txt = resp.text
    df = pd.read_csv(StringIO(txt))
    # Normalize columns
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def load_seamheads_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Load Seamheads Ballparks CSV if provided.

    Tries to normalize likely columns for elevation/orientation/foul territory and park id.
    Because Seamheads schema can vary by export, we match columns by case-insensitive names.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[build_parks_yaml] Seamheads CSV not found at {path}; continuing without it.")
        return None
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    # Identify candidate columns
    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    # Common possibilities across exports
    park_id_col = pick("park_id", "parkid", "retro_park_id", "retroparkid", "retrosheet_id")
    elev_col = pick("elevation", "elev_ft", "altitude", "altitude_ft")
    orient_col = pick("orientation", "orientation_deg", "bearing")
    foul_col = pick("foul_territory", "foul_territory_sqft", "foul_sqft")
    roof_col = pick("roof", "cover")
    surface_col = pick("surface")

    if not park_id_col:
        # We can still return the df, but we cannot merge on park without an id
        return None

    keep = {park_id_col: "PARKID"}
    if elev_col:
        keep[elev_col] = "ALT_FT"
    if orient_col:
        keep[orient_col] = "ORIENTATION_DEG"
    if foul_col:
        keep[foul_col] = "FOUL_TERRITORY_SQFT"
    if roof_col:
        keep[roof_col] = "ROOF"
    if surface_col:
        keep[surface_col] = "SURFACE"
    out = df[list(keep.keys())].rename(columns=keep)
    # Normalize PARKID to Retrosheet code style (uppercase)
    out["PARKID"] = out["PARKID"].astype(str).str.strip().str.upper()
    return out


def load_dims_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[build_parks_yaml] Dimensions CSV not found at {path}; continuing without it.")
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    required = ["park_id", "lf_line_ft", "lf_gap_ft", "cf_ft", "rf_gap_ft", "rf_line_ft"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Dimensions CSV missing required column: {col}")
    # Normalize park id
    df["park_id"] = df["park_id"].astype(str).str.strip().str.upper()
    return df


def load_overrides_yaml(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[build_parks_yaml] Overrides YAML not found at {path}; continuing without it.")
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def build_record(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Merge dicts but drop keys with None values to keep YAML clean."""
    rec: Dict[str, Any] = {}
    for src in (base, extra):
        for k, v in src.items():
            if v is None:
                continue
            rec[k] = v
    return rec


def main():
    ap = argparse.ArgumentParser(description="Build parks.yaml from external sources")
    ap.add_argument("--output", default=str(Path("config") / "parks.yaml"))
    ap.add_argument("--seamheads", default=None, help="Path to Seamheads Ballparks CSV (optional)")
    ap.add_argument("--dims", default=None, help="Path to curated dimensions CSV (optional)")
    ap.add_argument("--overrides", default=None, help="Path to overrides YAML (optional)")
    args = ap.parse_args()

    retro = fetch_retrosheet_parks()
    sh = load_seamheads_csv(args.seamheads)
    dims = load_dims_csv(args.dims)
    overrides = load_overrides_yaml(args.overrides)

    # Index optional frames by PARKID for quick lookups
    sh_by = sh.set_index("PARKID").to_dict(orient="index") if sh is not None else {}
    dims_by = dims.set_index("park_id").to_dict(orient="index") if dims is not None else {}

    out: Dict[str, Dict[str, Any]] = {}

    # Add a neutral default GEN entry if not present via overrides
    out["GEN"] = {
        "rf_short_porches": False,
        "lf_short_porches": False,
        "roof": "open",
        "surface": "grass",
        "altitude_ft": 0,
    }

    for _, r in retro.iterrows():
        park_id = str(r.get("PARKID", "")).strip().upper()
        if not park_id:
            continue
        name = str(r.get("NAME", "")).strip() or None
        city = str(r.get("CITY", "")).strip() or None
        state = str(r.get("STATE", "")).strip() or None
        start = str(r.get("START", "")).strip() or None
        end = str(r.get("END", "")).strip() or None

        base = {
            "name": name,
            "city": city,
            "state": state,
        }
        # Normalize year bounds if present as MM/DD/YYYY
        def _year(s: Optional[str]) -> Optional[int]:
            if not s or not isinstance(s, str):
                return None
            s = s.strip()
            if not s:
                return None
            try:
                if len(s) == 4 and s.isdigit():
                    return int(s)
                # Formats like 04/14/2017
                parts = s.split("/")
                if len(parts) == 3 and parts[2].isdigit():
                    return int(parts[2])
            except Exception:
                return None
            return None

        years = {
            "year_from": _year(start),
            "year_to": _year(end),
        }

        extra: Dict[str, Any] = {}
        # Merge in Seamheads attributes if available
        if park_id in sh_by:
            sh_row = sh_by[park_id]
            if "ALT_FT" in sh_row and pd.notna(sh_row["ALT_FT"]):
                try:
                    extra["altitude_ft"] = float(sh_row["ALT_FT"])  # type: ignore
                except Exception:
                    pass
            if "ORIENTATION_DEG" in sh_row and pd.notna(sh_row["ORIENTATION_DEG"]):
                try:
                    extra["orientation_deg"] = float(sh_row["ORIENTATION_DEG"])  # type: ignore
                except Exception:
                    pass
            if "FOUL_TERRITORY_SQFT" in sh_row and pd.notna(sh_row["FOUL_TERRITORY_SQFT"]):
                try:
                    extra["foul_territory_sqft"] = float(sh_row["FOUL_TERRITORY_SQFT"])  # type: ignore
                except Exception:
                    pass
            # Optional roof/surface passthrough if present
            try:
                rv = sh_row.get("ROOF")
                if pd.notna(rv):
                    extra["roof"] = str(rv).strip().lower()
            except Exception:
                pass
            try:
                sv = sh_row.get("SURFACE")
                if pd.notna(sv):
                    extra["surface"] = str(sv).strip().lower()
            except Exception:
                pass

        # Merge in dimensions if provided
        if park_id in dims_by:
            d = dims_by[park_id]
            dims_obj: Dict[str, Any] = {}
            for k in ("lf_line_ft", "lf_gap_ft", "cf_ft", "rf_gap_ft", "rf_line_ft"):
                v = d.get(k)
                if pd.notna(v):
                    try:
                        dims_obj[k] = float(v)
                    except Exception:
                        pass
            if dims_obj:
                extra["dims"] = dims_obj

            walls_obj: Dict[str, Any] = {}
            wall_keys = {
                "lf_wall_ft": "lf",
                "lf_gap_wall_ft": "lf_gap",
                "cf_wall_ft": "cf",
                "rf_gap_wall_ft": "rf_gap",
                "rf_wall_ft": "rf",
            }
            for src, dst in wall_keys.items():
                if src in d and pd.notna(d.get(src)):
                    try:
                        walls_obj[dst] = float(d.get(src))
                    except Exception:
                        pass
            if walls_obj:
                extra["wall_heights_ft"] = walls_obj

            # Compute simple short-porch flags
            try:
                if "lf_line_ft" in d and pd.notna(d["lf_line_ft"]) and float(d["lf_line_ft"]) <= 330:
                    extra["lf_short_porches"] = True
                if "rf_line_ft" in d and pd.notna(d["rf_line_ft"]) and float(d["rf_line_ft"]) <= 330:
                    extra["rf_short_porches"] = True
            except Exception:
                pass

        # Apply overrides last
        ovr = overrides.get(park_id, {}) if isinstance(overrides, dict) else {}
        rec = build_record({**base, **years}, {**extra, **ovr})
        if rec:
            out[park_id] = rec

    # Ensure output directory exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort keys for stable diffs
    sorted_out = {k: out[k] for k in sorted(out.keys())}
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(sorted_out, f, sort_keys=False, allow_unicode=True)

    print(f"Wrote {len(sorted_out)} parks to {out_path}")


if __name__ == "__main__":
    main()
