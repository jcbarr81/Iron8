"""
Merge season-split Retrosheet event CSVs into a single file used by
scripts/build_pa_table.py.

Input directory: data/processed/retrosheet_by_season/
Output file:     data/processed/retrosheet_events.csv

The script concatenates all files named retrosheet_events_*.csv in the
season directory, preserving header from the first file and using a
simple streaming append to handle large datasets.
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data" / "processed" / "retrosheet_by_season"
OUT_DIR = ROOT / "data" / "processed"
OUT_PATH = OUT_DIR / "retrosheet_events.csv"


def main() -> int:
    if not IN_DIR.exists():
        print(f"Input directory not found: {IN_DIR}")
        return 1
    files = sorted(IN_DIR.glob("retrosheet_events_*.csv"))
    if not files:
        print(f"No retrosheet_events_*.csv files found in {IN_DIR}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT_PATH.open("w", encoding="utf-8", newline="") as out:
        # Write header from first file
        with files[0].open("r", encoding="utf-8") as f0:
            header = f0.readline()
            out.write(header)
            for line in f0:
                out.write(line)
                written += 1
        # Append remaining files skipping their header
        for fp in files[1:]:
            with fp.open("r", encoding="utf-8") as fi:
                _ = fi.readline()  # skip header
                for line in fi:
                    out.write(line)
                    written += 1
    print(f"Wrote {OUT_PATH} with {written} rows (excluding header)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

