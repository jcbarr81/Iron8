# API Reference

General Notes
- All endpoints accept and return JSON. Fields not listed as required are optional.
- IDs are strings unless noted. Numeric scores/counts/outs use integers.
- Ratings are 0â€“99 if provided inline. If not provided, the service attempts to look up ratings in `data/players.csv` by `player_id`.
- Pass a `seed` to make stochastic sampling deterministic.

## POST /v1/sim/plate-appearance
Request:
```json
{
  "game_id": "2025-06-01-TOR@BOS",
  "state": {
    "inning": 7, "half": "T", "outs": 2,
    "bases": {"1B": "641856", "2B": null, "3B": "673548"},
    "score": {"away": 2, "home": 3},
    "count": {"balls": 1, "strikes": 2},
    "park_id": "BOS05",
    "weather": {"temp_f": 68, "wind_mph": 8, "wind_dir_deg": 45},
    "rules": {"dh": true, "shift_restrictions": true, "extras_runner_on_2nd": true}
  },
  "batter": {
    "player_id": "B123",
    "bats": "L",
    "ratings": {"ch": 68, "ph": 74, "gf": 80, "pl": 85, "sc": 62, "fa": 77, "arm": 66}
  },
  "pitcher": {
    "player_id": "P456",
    "throws": "R", "pitch_count": 78, "tto_index": 2,
    "ratings": {
      "endurance": 55, "control": 52, "movement": 64, "hold_runner": 58, "fa": 70,
      "arm": 88, "fb": 80, "sl": 72, "si": 65, "cu": 50, "cb": 60, "scb": 0, "kn": 0
    }
  },
  "defense": {"oaa_by_pos": {"SS": 7, "2B": 1, "CF": 8}},
  "return_probabilities": true,
  "return_contact": false,
  "seed": 123456789
}
```

Response:
```json
{
  "model_version": "pa-1.0.0",
  "probs": {"BB":0.09,"K":0.27,"HBP":0.01,"IP_OUT":0.24,"1B":0.20,"2B":0.08,"3B":0.01,"HR":0.10},
  "sampled_event": "HR",
  "bip_detail": {
    "ev_mph": 104.7,
    "la_deg": 27.0,
    "spray_deg": 18.0,
    "contact_type": "FB",
    "out_subtype": "FB_OUT",   // present when fielded for an out
    "fielder": "CF",
    "dp": false
  },
  "advancement": {
    "outs_recorded": 0,
    "bases_ending": {"1B": null, "2B": null, "3B": null},
    "rbi": 2
  },
  "rng": {"seed": 123456789}
}
```

Notes
- When `return_contact` is true, the service includes `bip_detail` (EV/LA/spray/contact_type) and merges fielding information (out_subtype, fielder, dp when applicable).
- `advancement` reports outs recorded by the play and the bases after the play; when not an out, baserunning advancement may score runners (e.g., 3B â†’ home) and set `rbi`.

---

## POST /v1/sim/steal
Request:
```json
{
  "state": {
    "inning": 7, "half": "T", "outs": 1,
    "bases": {"1B": "641856", "2B": null, "3B": null},
    "score": {"away": 2, "home": 3},
    "count": {"balls": 1, "strikes": 1}
  },
  "runner_base": "1B",
  "runner": {"player_id": "641856", "ratings": {"sp": 80}},
  "pitcher": {"player_id": "P1", "throws": "R", "ratings": {"hold_runner": 60}},
  "defense": {"arm_by_pos": {"C": 70}},
  "seed": 123
}
```
Response:
```json
{
  "event": "SB", // one of SB, CS, PO, WP, PB
  "outs_recorded": 0,
  "bases_ending": {"1B": null, "2B": "641856", "3B": null},
  "rng": {"seed": 123}
}
```

Notes
- Use this endpoint when your game is attempting a steal. The service evaluates runner speed vs. pitcher hold and catcher arm and returns the result with updated bases/outs.
- For pickoff, WP, or PB outcomes, `event` reflects the specific result and `bases_ending`/`outs_recorded` reflect the change.

---

## POST /v1/sim/pitch
Request:
```json
{
  "game_id": "G1",
  "state": {"inning":1,"half":"T","outs":0,"bases":{"1B":null,"2B":null,"3B":null},"count":{"balls":0,"strikes":0},
            "rules": {"challenges_enabled": false, "overturn_rate": 0.2, "three_batter_min": true}},
  "batter": {"player_id":"B1","bats":"L","ratings":{"ch":60}},
  "pitcher": {"player_id":"P1","throws":"R","pitch_count":22,"tto_index":2, "batters_faced_in_stint":1, "entered_cold": false,
               "ratings": {"control":55,"arm":70,"fb":70}},
  "defense": {"oaa_by_pos":{"CF":6}},
  "edge": false,
  "umpire": {"overall_bias": 0.0, "edge_bias": 0.5},
  "return_contact": true,
  "seed": 123
}
```
Response:
```json
{
  "pitch_type": "fb",
  "result": "called_strike",
  "next_count": {"balls": 0, "strikes": 1},
  "probs": {"ball":0.31,"called_strike":0.22,"swing_miss":0.12,"foul":0.13,"in_play":0.20,"hbp":0.02}
}
```

Notes
- Fatigue: The service applies a pitcher fatigue adjustment based on `pitch_count`, `tto_index`, and optional warm-up penalty when `entered_cold=true` and `batters_faced_in_stint < 2`. Fatigue reduces `pit_control_z` and `pit_stuff_z`, which increases balls and contact quality.
- Three-batter minimum: Use the rules check endpoint below to validate substitutions.

---

## POST /v1/rules/check-three-batter-min
Request:
```json
{ "three_batter_min": true, "batters_faced_in_stint": 1, "inning_ended": false, "injury_exception": false }
```
Response:
```json
{ "allowed": false, "reason": "Three-batter minimum not reached" }
```

---

## Fields & Behavior (Reference)

### /v1/sim/plate-appearance
- Required request fields: `game_id`, `state`, `batter`, `pitcher`
- `state` required members:
  - `inning` (int), `half` ("T"|"B"), `outs` (0–2)
  - `bases` (object with keys "1B","2B","3B": string or null)
  - `score` (object with `away`,`home` ints)
  - `count` (object with `balls` 0–3, `strikes` 0–2)
- Optional `state` members: `park_id`, `weather` ({`temp_f`,`wind_mph`,`wind_dir_deg`}), `rules` ({`dh`,`shift_restrictions`,`extras_runner_on_2nd`})
- `batter`: `{ player_id, bats, ratings? }`
- `pitcher`: `{ player_id, throws, pitch_count?, tto_index?, ratings? }`
- Optional: `defense`, `return_probabilities`, `return_contact`, `seed`
- Response: `probs`, `sampled_event`, optional `bip_detail` + `advancement`, telemetry (`rng`,`trace_id`,`win_prob_home`,`leverage_index`)

### /v1/sim/steal
- Required: `state`, `runner_base` ("1B"|"2B"), `runner`, `pitcher`
- Optional: `defense` (catcher arm/fielding), `seed`
- Response: `event` ("SB"|"CS"|"PO"|"WP"|"PB"), `outs_recorded`, `bases_ending`, `rng`

### /v1/sim/pitch
- Required: `game_id`, `state`, `batter`, `pitcher`
- Optional: `defense`, `edge` (bool), `umpire` ({`overall_bias`,`edge_bias`}), `return_contact`, `seed`
- Response: `pitch_type`, `result`, `next_count`, `probs`. If pre-pitch event: `pre_pitch`, `pre_info`. If `in_play` and `return_contact`: `bip_detail`, `advancement`.

### /v1/rules/check-three-batter-min
- Request: `{ three_batter_min?, batters_faced_in_stint, inning_ended?, injury_exception? }`
- Response: `{ allowed, reason? }`

### /v1/game/boxscore
- Request: `{ game_id, home_team, away_team, log: GameEvent[] }`
- Response: `{ game_id, runs: {home,away}, earned_runs: {home,away}, outs, winner? }`

### /v1/game/log
- Request: `{ game_id, log: GameEvent[] }`
- Response: `{ game_id, events: GameEvent[] }`

### /admin/reload-players
- Reloads `data/players.csv` into memory
- Response: `{ ok, count? | error? }`
- If `ADMIN_TOKEN` is set on the server, a matching token is required.
