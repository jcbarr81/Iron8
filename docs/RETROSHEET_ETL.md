# Retrosheet ETL + Training Quickstart

This guide shows two supported paths to build a Plate Appearance (PA) training table from Retrosheet data and train the PA outcome model the service will load.

You can use either the PowerShell + cwevent path or the Python ingestion path to produce `data/processed/retrosheet_events.csv`. Then convert to a PA table and train.

## Prerequisites
- Python 3.10+ and pip
- Install deps in your environment:
  - `pip install -r requirements.txt`
  - If parquet write fails, the scripts fall back to CSV; `pyarrow` is recommended but optional.
- Optional (PowerShell path): Chadwick `cwevent` in your PATH (https://chadwick.sourceforge.net/)

## Directory layout
- Raw input (PowerShell or Python path):
  - `data/raw/retrosheet/` … place your source Retrosheet files here
- Intermediates and outputs:
  - `data/processed/retrosheet_events.csv` … combined events CSV (from either ingestion path)
  - `artifacts/pa_table.parquet` … PA table for training (or `pa_table.csv` fallback)
  - `artifacts/pa_model.joblib` … trained PA model

## Path A — PowerShell + cwevent (recommended for EVN/EVA)
1) Place files
   - Under `data/raw/retrosheet/`, extract per-season zips so that each year folder contains:
     - `TEAMYYYY` file and `YYYY*.EVN` and `YYYY*.EVA` files in the same directory
2) Convert per-season CSVs and combine
   - Run in PowerShell:
     - `scripts/convert_retrosheet.ps1`
   - What it does:
     - Uses `cwevent` to produce `data/processed/retrosheet_by_season/retrosheet_events_YYYY.csv`
     - Combines them (single header) into `data/processed/retrosheet_events.csv`

Troubleshooting cwevent
- Make sure you run `cwevent` from a directory that contains BOTH `TEAMYYYY` and `YYYY*.EVN`/`YYYY*.EVA` files.
- Some builds require no space between `-f` and the field spec and support only up to field 63. For a quick manual test:
  - `cwevent -x -y 2010 -f0-63 2010*.EV? > test_2010.csv`
  - If that works, the PowerShell script should also succeed.

## Path B — Python ingestion (CSV or ZIP inputs)
1) Place files
   - Under `data/raw/retrosheet/`, drop CSVs or ZIPs with Retrosheet-like event data
2) Ingest and normalize
   - Run:
     - `python scripts/ingest_retrosheet.py`
   - Output:
     - `data/processed/retrosheet_events.csv` (canonical columns; tolerant of schema variants)

## Build the PA table
- Convert the processed events into a PA-level table (one row per PA):
  - `python scripts/build_pa_table.py`
- Output: `artifacts/pa_table.parquet` (or `artifacts/pa_table.csv` if parquet engine unavailable)
- If you see “Input not found,” generate `data/processed/retrosheet_events.csv` first using Path A or B.

## Train the PA model
- Train on real data (uses the PA table when present):
  - `python scripts/train_pa.py`
- Expected output:
  - Prints validation log loss and class frequencies
  - Writes `artifacts/pa_model.joblib`
- The server autoloads this artifact at startup.

## Serve and smoke-test
- Start the API:
  - `uvicorn run_server:app --reload --port 8000`
- Health check:
  - GET `http://localhost:8000/health`
- Minimal request (plate appearance):
  - POST `http://localhost:8000/v1/sim/plate-appearance`
  - Example body (see `docs/API_REFERENCE.md`), e.g.:
    ```json
    {
      "game_id": "X",
      "state": {
        "inning": 1, "half": "T", "outs": 0,
        "bases": {"1B": null, "2B": null, "3B": null},
        "score": {"away": 0, "home": 0},
        "count": {"balls": 0, "strikes": 0},
        "park_id": "GEN",
        "rules": {"dh": true}
      },
      "batter": {"player_id": "B1", "bats": "R", "ratings": {"ph": 60, "gf": 50}},
      "pitcher": {"player_id": "P1", "throws": "L", "ratings": {"arm": 80, "fb": 75, "sl": 70}},
      "return_probabilities": true,
      "return_contact": false,
      "seed": 42
    }
    ```
- Before training, responses use the baseline heuristic; after training, probabilities come from your trained model.

## Troubleshooting
- `data/processed/retrosheet_events.csv` missing:
  - Run `scripts/convert_retrosheet.ps1` (PowerShell) or `python scripts/ingest_retrosheet.py` (Python)
- `pandas` or `pyarrow` not installed:
  - `pip install -r requirements.txt`
  - If parquet write fails, the builder falls back to CSV and training still works
- `cwevent` not found (PowerShell path):
  - Install Chadwick tools and ensure `cwevent` is on PATH; verify with `cwevent -V`

## What gets saved
- `data/processed/retrosheet_events.csv` — event-level combined CSV
- `artifacts/pa_table.parquet` — PA-level table used for training
- `artifacts/pa_model.joblib` — trained model loaded by the server

Happy simming!
