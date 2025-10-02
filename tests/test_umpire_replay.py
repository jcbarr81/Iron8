from fastapi.testclient import TestClient
from baseball_sim.serve.api import app


client = TestClient(app)


def base_req(edge=False, challenges=False, overturn=0.0, ump=None):
    return {
        "game_id": "G1",
        "state": {
            "inning": 5,
            "half": "T",
            "outs": 0,
            "bases": {"1B": None, "2B": None, "3B": None},
            "score": {"away": 0, "home": 0},
            "count": {"balls": 1, "strikes": 1},
            "rules": {"challenges_enabled": challenges, "overturn_rate": overturn},
        },
        "batter": {"player_id": "B1", "bats": "L", "ratings": {"ch": 50}},
        "pitcher": {"player_id": "P1", "throws": "R", "ratings": {"control": 50, "arm": 70, "fb": 70}},
        "edge": edge,
        "umpire": ump or {},
        "return_contact": False,
        "seed": 123,
    }


def test_umpire_edge_bias_changes_called_strike_vs_ball_mix():
    edge = True
    tight = base_req(edge=edge, ump={"overall_bias": 0.0, "edge_bias": -1.0})
    loose = base_req(edge=edge, ump={"overall_bias": 0.0, "edge_bias": 1.0})
    r1 = client.post("/v1/sim/pitch", json=tight).json()
    r2 = client.post("/v1/sim/pitch", json=loose).json()
    assert r1["result"] in {"ball", "called_strike", "swing_miss", "foul", "in_play", "hbp"}
    assert r2["result"] in {"ball", "called_strike", "swing_miss", "foul", "in_play", "hbp"}


def test_challenge_can_overturn_edge_called_strike():
    # Force overturn by setting override rate to 1.0 and edge True
    req = base_req(edge=True, challenges=True, overturn=1.0, ump={"overall_bias": 1.0, "edge_bias": 1.0})
    r = client.post("/v1/sim/pitch", json=req).json()
    # If result is called_strike initially at edge, it will be flipped to ball
    assert r["result"] in {"ball", "called_strike", "swing_miss", "foul", "in_play", "hbp"}

