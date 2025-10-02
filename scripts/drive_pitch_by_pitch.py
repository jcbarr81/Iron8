from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict


def make_client(http_base: str | None):
    if http_base:
        import httpx

        class HttpClient:
            def __init__(self, base: str):
                self.base = base.rstrip("/")
                self.sess = httpx.Client(timeout=30.0)

            def get(self, path: str):
                return self.sess.get(self.base + path)

            def post(self, path: str, json):
                return self.sess.post(self.base + path, json=json)

        return HttpClient(http_base)
    else:
        # In-process client
        ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        SRC = os.path.join(ROOT, "src")
        if SRC not in sys.path:
            sys.path.insert(0, SRC)
        from fastapi.testclient import TestClient
        from baseball_sim.serve.api import app

        return TestClient(app)


def drive_one_pa(client, seed: int = 42):
    # Minimal initial state
    state: Dict = {
        "inning": 1,
        "half": "T",
        "outs": 0,
        "bases": {"1B": None, "2B": None, "3B": None},
        "score": {"away": 0, "home": 0},
        "count": {"balls": 0, "strikes": 0},
        "park_id": "GEN",
        "rules": {"dh": True, "shift_restrictions": True, "three_batter_min": True},
    }
    batter = {"player_id": "B1", "bats": "R", "ratings": {"ph": 60, "ch": 55, "pl": 55, "gf": 50, "sp": 55}}
    pitcher = {
        "player_id": "P1",
        "throws": "L",
        "pitch_count": 25,
        "tto_index": 1,
        "batters_faced_in_stint": 0,
        "entered_cold": False,
        "ratings": {"arm": 80, "fb": 75, "sl": 70, "control": 55, "movement": 55, "hold_runner": 50},
    }
    defense = {"oaa_by_pos": {"SS": 7, "2B": 1, "CF": 8}, "arm_by_pos": {"C": 70}, "alignment": {"infield": "dp", "outfield": "deep"}}

    print("/health:")
    try:
        r = client.get("/health")
        print(getattr(r, "status_code", "OK"), getattr(r, "json", lambda: r)())
    except Exception:
        pass  # In TestClient, this always exists

    # Pitch loop
    while True:
        req = {
            "game_id": "INTEGRATION-DRIVER",
            "state": state,
            "batter": batter,
            "pitcher": pitcher,
            "defense": defense,
            "return_contact": True,
            "seed": seed,
        }
        r = client.post("/v1/sim/pitch", json=req)
        body = r.json()
        print("/v1/sim/pitch:", r.status_code)
        print(json.dumps(body, indent=2))

        # Pre-pitch event (e.g., pickoff)
        if body.get("pre_pitch") and body["pre_pitch"] != "no_play":
            info = body.get("pre_info", {})
            # Update outs/bases
            state["outs"] = int(info.get("outs_recorded", state["outs"]))
            state["bases"] = info.get("bases", state["bases"])  # already a dict
            break

        result = body["result"]
        if result in ("ball", "called_strike", "swing_miss", "foul"):
            # Update count
            state["count"] = body.get("next_count", state["count"])  # {balls,strikes}
            seed += 1
            continue
        elif result == "hbp":
            # PA ends; batter to 1B
            state["bases"]["1B"] = batter["player_id"]
            break
        elif result == "in_play":
            adv = body.get("advancement", {})
            state["outs"] = state["outs"] + int(adv.get("outs_recorded", 0))
            state["bases"] = adv.get("bases_ending", state["bases"])  # mapping
            # PA ends
            break

    print("Final state:")
    print(json.dumps(state, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Pitch-by-pitch driver")
    ap.add_argument("--http", help="Base URL (e.g., http://localhost:8000) to use HTTP instead of TestClient")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    client = make_client(args.http)
    drive_one_pa(client, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

