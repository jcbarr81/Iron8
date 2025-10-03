# Integration Guide

This engine is a stateless microservice you drive from your game. You own the authoritative game state (inning/half/outs, bases, score, count, actors, rules) and call the API to get the next baseball event and details to advance the sim.

- Pitch-by-pitch: Use `/v1/sim/pitch` to evolve count, handle pickoffs/step‑offs, and (when in_play) generate contact, fielding, and advancement.
- Plate-appearance: Use `/v1/sim/plate-appearance` for a single PA outcome (optionally with contact + advancement).
- Micro-events: Use `/v1/sim/steal` to attempt SB/CS/PO/WP/PB outside pitch flow.

See `docs/API_REFERENCE.md` for complete JSON contracts and examples.

## Required/Optional Inputs
- `state` (required): inning (int), half ("T"|"B"), outs (0–2), bases (object keys "1B","2B","3B" as IDs or null), score (away/home ints), count (balls 0–3, strikes 0–2)
  - Optional: `park_id` (string), `weather` ({`temp_f`,`wind_mph`,`wind_dir_deg`}), `rules` ({`dh`,`shift_restrictions`,`extras_runner_on_2nd`,`challenges_enabled`,`overturn_rate`,`three_batter_min`})
- `batter` (required): `{ player_id, bats: "L"|"R", ratings?: {...} }`
- `pitcher` (required): `{ player_id, throws: "L"|"R", pitch_count?, tto_index?, batters_faced_in_stint?, entered_cold?, ratings?: {...} }`
- `defense` (optional): `{ oaa_by_pos: {...}, arm_by_pos?: {...}, alignment?: { infield?: "in"|"dp"|"corners"|"normal", outfield?: "shallow"|"normal"|"deep" } }`
- Ratings 0–99 inline are preferred for determinism. If omitted, the service tries to look up by `player_id` in `data/players.csv`.

### Player Ratings Mapping (0–99)
- Batter ratings in `players.csv` (0–99): `ch` (contact), `ph` (power), `gf` (GB tilt), `pl` (pull), `sc` (situational), `sp` (speed), `fa` (field), `arm` (throwing arm)
- Pitcher ratings in `players.csv` (0–99): `arm` (velo), `endurance` (stamina), `control`, `movement`, `hold_runner`, pitch mix usage flags `fb, sl, si, cu, cb, scb, kn`
- The engine converts 0–99 ratings to z-like features via `z_0_99` centered at 50 with scale 15 and clamps to [-3,3]. See `src/baseball_sim/features/transforms.py`.
- If `ratings` are omitted in requests, the server auto-fills from `data/players.csv` by `player_id`.

## Pitch-by-Pitch Loop (Suggested)
1) Build `state`, `batter`, `pitcher`, `defense`.
2) Optional: call `/v1/sim/steal` and update bases/outs (handle PO/WP/PB).
3) Call `/v1/sim/pitch` with `return_contact=true`.
   - Result in {ball, called_strike, swing_miss, foul, in_play, hbp}.
   - If `pre_pitch` (e.g., `PO`), update outs/bases and skip pitch result.
   - If `in_play`, use `bip_detail` (EV/LA/spray/contact_type + fielder/out_subtype/dp/error/sf) and `advancement` (outs_recorded/bases_ending/rbi) to advance state; PA ends.
   - Otherwise, update `count` from `next_count` and repeat.
4) Track fatigue by updating pitcher `pitch_count` each pitch and `batters_faced_in_stint` at PA end. Apply three‑batter minimum via `/v1/rules/check-three-batter-min`.

## Realism Factors
- Pitch selection: inferred from pitcher ratings (fb/sl/si/cu/cb/scb/kn) — strongest pitch favored.
- Count/swing/take: `control`/`contact`/count influence ball/strike/swing_miss/foul/in_play; `umpire` and `edge=true` shift called strike/borderline calls; `challenges_enabled` can overturn edge called strikes to balls with `overturn_rate`.
- Fatigue: `pitch_count`, `tto_index`, `entered_cold` degrade `pit_control_z` and `pit_stuff_z`.
- BIP: `ph`↑EV; `gf`↓LA (more GB); `pl` pulls spray; `pit_stuff_z` modestly ↓EV; `weather` (temp, wind) adjusts EV/LA.
- Fielding & outs: `oaa_by_pos` and `alignment` affect GB/LD/FB/PU conversion; infield‑fly rule; GB DPs; errors/FC/SF; wind affects FB/LD outs.
- Advancement: OF `arm_by_pos` vs runner `sp` controls sends/holds and assists (e.g., 3B→home on flies).
- Steals/pickoffs: `sp` vs `hold_runner` and catcher arm; outcomes SB/CS/PO/WP/PB.

## Seeds & Telemetry
- Pass `seed` to make stochastic sampling deterministic.
- Responses include `trace_id` for correlation; Plate-appearance responses include simple `win_prob_home` and `leverage_index` as telemetry.
- Optional calibration logging: set `PA_LOG_ENABLE=1` (and optionally `PA_LOG_PATH`) to log per-PA predicted probs and sampled events to `artifacts/pa_logs.csv`. These logs can help audit calibration quality over slices.

## Performance (Optional)
- Train model: `python scripts/train_pa.py`
- Calibrate: `python scripts/calibrate_pa.py`
- Calibrate per-slice: `python scripts/calibrate_pa_sliced.py` (fits a sliced Platt calibrator by matchup x roof and writes `artifacts/pa_calibrator.joblib`); the server auto-loads it on startup.
- Export ONNX: `python scripts/export_onnx.py`, set `model.use_onnx: true` in config to enable ONNX runtime.

## Pitch‑by‑Pitch Driver
A minimal driver showing a full PA using the pitch API.

- File: `scripts/drive_pitch_by_pitch.py`
- TestClient mode (no server):
  - `python scripts/drive_pitch_by_pitch.py`
- HTTP mode (requires server running):
  - Start: `uvicorn run_server:app --reload --port 8000`
  - Run: `python scripts/drive_pitch_by_pitch.py --http http://localhost:8000`

The driver:
- Calls `/health`.
- Loops through `/v1/sim/pitch` until PA ends (handles pre‑pitch pickoff, in‑play BIP + fielding + advancement, and count updates).
- Prints full JSON responses and final state.

## API Contracts
See `docs/API_REFERENCE.md` for complete request/response formats and field descriptions for:
- `/v1/sim/pitch`, `/v1/sim/plate-appearance`, `/v1/sim/steal`
- `/v1/rules/check-three-batter-min`, `/v1/game/boxscore`, `/v1/game/log`
- `/admin/reload-players`
