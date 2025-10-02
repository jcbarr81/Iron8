from __future__ import annotations

import json
from fastapi.testclient import TestClient
import os, sys

# Ensure 'src' is importable regardless of CWD
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from baseball_sim.serve.api import app


def main() -> int:
    client = TestClient(app)

    # Health
    r = client.get("/health")
    print("/health:", r.status_code, r.json())

    # Minimal PA request, with inline ratings to avoid lookups
    req = {
        "game_id": "SMOKE-1",
        "state": {
            "inning": 1,
            "half": "T",
            "outs": 0,
            "bases": {"1B": None, "2B": None, "3B": None},
            "score": {"away": 0, "home": 0},
            "count": {"balls": 0, "strikes": 0},
            "park_id": "GEN",
            "rules": {"dh": True, "shift_restrictions": True},
        },
        "batter": {"player_id": "B1", "bats": "R", "ratings": {"ph": 60, "gf": 50, "pl": 55, "sc": 50}},
        "pitcher": {
            "player_id": "P1",
            "throws": "L",
            "pitch_count": 25,
            "tto_index": 1,
            "ratings": {"arm": 80, "fb": 75, "sl": 70, "control": 55, "movement": 55, "hold_runner": 50},
        },
        "return_probabilities": True,
        "return_contact": False,
        "seed": 42,
    }
    r2 = client.post("/v1/sim/plate-appearance", json=req)
    print("/v1/sim/plate-appearance:", r2.status_code)
    print(json.dumps(r2.json(), indent=2))

    # With contact details
    req["return_contact"] = True
    r3 = client.post("/v1/sim/plate-appearance", json=req)
    print("/v1/sim/plate-appearance (contact):", r3.status_code)
    print(json.dumps(r3.json(), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
