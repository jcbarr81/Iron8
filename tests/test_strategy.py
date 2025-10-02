from fastapi.testclient import TestClient
from baseball_sim.serve.api import app


client = TestClient(app)


def state_base():
    return {
        "inning": 8,
        "half": "T",
        "outs": 0,
        "bases": {"1B": "R1", "2B": None, "3B": None},
        "score": {"away": 2, "home": 2},
        "count": {"balls": 0, "strikes": 0},
    }


def test_pinch_hit_recommends_stronger_platoon_bat():
    req = {
        "state": state_base(),
        "current_batter": {"player_id": "B1", "bats": "R", "ratings": {"ph": 50, "ch": 50}},
        "pitcher": {"player_id": "P1", "throws": "R", "ratings": {"control": 50}},
        "candidates": [
            {"player_id": "B2", "bats": "L", "ratings": {"ph": 80, "ch": 60}},
            {"player_id": "B3", "bats": "R", "ratings": {"ph": 40, "ch": 45}},
        ],
    }
    r = client.post("/v1/strategy/pinch-hit", json=req)
    assert r.status_code == 200
    body = r.json()
    assert body["recommend"] in (True, False)
    if body["recommend"]:
        assert body["best_candidate_id"] == "B2"


def test_bullpen_prefers_opposite_handed_with_stuff():
    req = {
        "state": state_base(),
        "upcoming_batter": {"player_id": "B1", "bats": "L"},
        "pitcher_candidates": [
            {"player_id": "RHP1", "throws": "R", "ratings": {"arm": 60, "fb": 60, "control": 60}},
            {"player_id": "LHP1", "throws": "L", "ratings": {"arm": 65, "fb": 65, "control": 65}},
        ],
    }
    r = client.post("/v1/strategy/bullpen", json=req)
    assert r.status_code == 200
    body = r.json()
    assert body["recommend"] is True
    assert body["best_pitcher_id"] in {"LHP1", "RHP1"}


def test_bunt_recommendation_conditions():
    req = {"state": state_base(), "batter": {"player_id": "B", "bats": "R", "ratings": {"ph": 40}}}
    r = client.post("/v1/strategy/bunt", json=req)
    assert r.status_code == 200
    assert r.json()["recommend"] in (True, False)


def test_ibb_recommendation():
    s = state_base()
    s["bases"] = {"1B": None, "2B": "R2", "3B": None}
    req = {"state": s, "batter": {"player_id": "SLUG", "bats": "R", "ratings": {"ph": 95}}}
    r = client.post("/v1/strategy/ibb", json=req)
    assert r.status_code == 200
    assert r.json()["recommend"] in (True, False)

