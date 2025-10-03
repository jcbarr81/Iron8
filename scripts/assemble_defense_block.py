from __future__ import annotations

"""
Assemble a defense block (oaa_by_pos, arm_by_pos) for API requests from
derived ratings and a lineup file.

Inputs:
- --ratings: data/derived/defense_ratings.csv (from build_defense_ratings.py)
- --lineup: CSV with columns: position, player_id (one row per position)

Output:
- Prints JSON with keys: oaa_by_pos, arm_by_pos

Usage:
  python scripts/assemble_defense_block.py \
      --ratings data/derived/defense_ratings.csv \
      --lineup my_lineup.csv
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description="Assemble defense block from ratings and a lineup")
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--lineup", required=True, help="CSV with columns: position, player_id")
    args = ap.parse_args()

    r = pd.read_csv(args.ratings)
    r["position"] = r["position"].astype(str).str.upper()
    posset = set(["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"])  
    r = r[r["position"].isin(posset)]
    line = pd.read_csv(args.lineup)
    line["position"] = line["position"].astype(str).str.upper()

    oaa_by_pos = {}
    arm_by_pos = {}
    for _, row in line.iterrows():
        pos = str(row.get("position")).upper()
        pid = row.get("player_id")
        if pos not in posset or pd.isna(pid):
            continue
        sub = r[(r["position"] == pos) & (r["player_id"].astype(str) == str(pid))]
        if not sub.empty:
            oaa = sub.iloc[0].get("oaa_rating")
            arm = sub.iloc[0].get("arm_rating_filled")
            try:
                if pd.notna(oaa):
                    oaa_by_pos[pos] = int(oaa)
            except Exception:
                pass
            try:
                if pd.notna(arm):
                    arm_by_pos[pos] = int(arm)
            except Exception:
                pass
        else:
            # default neutral
            arm_by_pos[pos] = 50

    out = {"oaa_by_pos": oaa_by_pos, "arm_by_pos": arm_by_pos}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

