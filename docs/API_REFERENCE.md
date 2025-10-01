# API Reference

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
  "rng": {"seed": 123456789}
}
```
