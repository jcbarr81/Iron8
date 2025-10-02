from fastapi.testclient import TestClient
from baseball_sim.serve.api import app


client = TestClient(app)


def _req(bases):
    return {
        "game_id": "X",
        "state": {
            "inning": 1,
            "half": "T",
            "outs": 0,
            "bases": bases,
            "score": {"away": 0, "home": 0},
            "count": {"balls": 0, "strikes": 0},
            "park_id": "GEN",
            "rules": {"dh": True},
        },
        "batter": {
            "player_id": "B1",
            "bats": "R",
            # High RISP performance rating to observe effect
            "ratings": {"sc": 80, "pl": 50, "gf": 50, "fa": 50},
        },
        "pitcher": {
            "player_id": "P1",
            "throws": "L",
            # Keep pitcher features neutral to isolate RISP effect
            "ratings": {"arm": 50, "fb": 50, "sl": 50, "fa": 50},
        },
        "return_probabilities": True,
        "return_contact": False,
        "seed": 7,
    }


def test_risp_gates_sc_effect():
    # No RISP: bases empty
    r0 = client.post(
        "/v1/sim/plate-appearance",
        json=_req({"1B": None, "2B": None, "3B": None}),
    )
    assert r0.status_code == 200
    p0 = r0.json()["probs"]

    # RISP: runner on second
    r1 = client.post(
        "/v1/sim/plate-appearance",
        json=_req({"1B": None, "2B": "R2", "3B": None}),
    )
    assert r1.status_code == 200
    p1 = r1.json()["probs"]

    # With RISP, 1B should increase and K should decrease
    assert p1["1B"] > p0["1B"]
    assert p1["K"] < p0["K"]

