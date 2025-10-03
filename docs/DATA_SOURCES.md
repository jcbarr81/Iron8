# Data Sources & Licensing

- Retrosheet
  - Required notice: “The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at 20 Sunset Rd., Newark, DE 19711.”
  - Website: https://www.retrosheet.org/
  - Usage: We convert Retrosheet event files (EVN/EVA via `cwevent`) or ingest Retrosheet-like CSVs into training/evaluation tables.

- MLBAM / Baseball Savant (Statcast)
  - Terms of use apply as surfaced by MLB/Statcast. Use responsibly and attribute appropriately.
  - Website: https://baseballsavant.mlb.com/

- Lahman Baseball Database
  - https://www.seanlahman.com/baseball-archive/statistics/

- Chadwick Bureau Tools
  - `cwevent` used to convert Retrosheet event files.
  - https://chadwick.sourceforge.net/

## Ballparks (parks.yaml)
- Retrosheet Park Codes (park metadata; authoritative `PARKID` keys)
  - https://www.retrosheet.org/parkcode.txt
  - CSV columns: `PARKID,NAME,AKA,CITY,STATE,START,END,LEAGUE,NOTES`.
- Seamheads Ballparks Database (elevation, orientation, foul territory, historical changes)
  - https://www.seamheads.com/ballparks/
  - Registration may be required; respect their licensing/attribution.
- MLB Ballpark pages (dimensions/features, roof type, surface)
  - Example: https://www.mlb.com/ballparks (navigate to specific park pages)
- Ballparks of Baseball (aggregated dimensions and descriptive details)
  - https://www.ballparksofbaseball.com/
- Altitude (if not in Seamheads):
  - Open-Elevation API: https://api.open-elevation.com/

Script to build parks.yaml:
- `python scripts/build_parks_yaml.py --output config/parks.yaml`
- Optional inputs:
  - `--seamheads data/raw/seamheads_ballparks.csv`
  - `--dims data/raw/park_dimensions.csv` (see script header for schema)
  - `--overrides config/park_overrides.yaml` (roof/surface/altitude fixes)

Notes
- This project is for simulation/modeling. Ensure you follow the license terms of any datasets you use.
- Include this notice in product documentation or about pages where applicable.
