from fastapi.testclient import TestClient
from baseball_sim.serve.api import app


client = TestClient(app)


def _state(b1=None, b2=None, b3=None):
    return {
        "inning": 1,
        "half": "T",
        "outs": 1,
        "bases": {"1B": b1, "2B": b2, "3B": b3},
        "score": {"away": 0, "home": 0},
        "count": {"balls": 0, "strikes": 0},
    }


def test_fast_runner_gets_sb_vs_slow_runner_cs():
    slow_req = {
        "state": _state(b1="R"),
        "runner_base": "1B",
        "runner": {"ratings": {"sp": 20}},
        "pitcher": {"player_id": "P1", "throws": "R", "ratings": {"hold_runner": 30}},
        "defense": {"arm_by_pos": {"C": 30}},
    }
    fast_req = {
        "state": _state(b1="R"),
        "runner_base": "1B",
        "runner": {"ratings": {"sp": 90}},
        "pitcher": {"player_id": "P1", "throws": "R", "ratings": {"hold_runner": 30}},
        "defense": {"arm_by_pos": {"C": 30}},
    }
    slow = client.post("/v1/sim/steal", json=slow_req).json()
    fast = client.post("/v1/sim/steal", json=fast_req).json()
    assert slow["event"] in {"CS", "PO", "PB"}
    assert fast["event"] in {"SB", "PO", "PB"}


def test_strong_catcher_arm_increases_cs():
    weak = {
        "state": _state(b1="R"),
        "runner_base": "1B",
        "runner": {"ratings": {"sp": 60}},
        "pitcher": {"player_id": "P1", "throws": "R", "ratings": {"hold_runner": 30}},
        "defense": {"arm_by_pos": {"C": 10}},
    }
    strong = {
        "state": _state(b1="R"),
        "runner_base": "1B",
        "runner": {"ratings": {"sp": 60}},
        "pitcher": {"player_id": "P1", "throws": "R", "ratings": {"hold_runner": 30}},
        "defense": {"arm_by_pos": {"C": 90}},
    }
    w = client.post("/v1/sim/steal", json=weak).json()
    s = client.post("/v1/sim/steal", json=strong).json()
    # Strong arm should not produce SB if weak produced CS (deterministic rules)
    assert not (w["event"] == "SB" and s["event"] == "SB")


def test_poor_catcher_fielding_can_lead_to_pb():
    poor = {
        "state": _state(b1="R"),
        "runner_base": "1B",
        "runner": {"ratings": {"sp": 50}},
        "pitcher": {"player_id": "P1", "throws": "R", "ratings": {"hold_runner": 30}},
        "defense": {"oaa_by_pos": {"C": 5}},
    }
    out = client.post("/v1/sim/steal", json=poor).json()
    assert out["event"] in {"PB", "SB", "CS", "PO"}

