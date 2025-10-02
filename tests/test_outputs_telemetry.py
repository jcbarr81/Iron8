from fastapi.testclient import TestClient
from baseball_sim.serve.api import app


client = TestClient(app)


def test_sim_response_contains_trace_and_wp():
    req = {
        "game_id": "G1",
        "state": {
            "inning": 9,
            "half": "B",
            "outs": 1,
            "bases": {"1B": None, "2B": None, "3B": None},
            "score": {"away": 3, "home": 3},
            "count": {"balls": 1, "strikes": 1},
            "park_id": "GEN",
        },
        "batter": {"player_id": "B1", "bats": "L", "ratings": {"ph": 60}},
        "pitcher": {"player_id": "P1", "throws": "R", "ratings": {"arm": 70, "fb": 70}},
        "return_probabilities": True,
        "return_contact": False,
        "seed": 1,
    }
    r = client.post("/v1/sim/plate-appearance", json=req)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body.get("trace_id"), str)
    assert 0.0 <= body.get("win_prob_home", 0.5) <= 1.0
    assert 0.5 <= body.get("leverage_index", 1.0) <= 3.0


def test_boxscore_from_log():
    log = [
        {"inning": 1, "half": "T", "team": "away", "rbi": 1, "outs_recorded": 1, "error": False},
        {"inning": 1, "half": "B", "team": "home", "rbi": 2, "outs_recorded": 2, "error": False},
        {"inning": 2, "half": "T", "team": "away", "rbi": 0, "outs_recorded": 3, "error": True},
    ]
    payload = {"game_id": "G1", "home_team": "HOME", "away_team": "AWAY", "log": log}
    r = client.post("/v1/game/boxscore", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["runs"]["away"] == 1
    assert body["runs"]["home"] == 2
    assert body["outs"] == 6
    assert body["winner"] == "HOME"


def test_game_log_endpoint_echos_events():
    log = [
        {"inning": 1, "half": "T", "team": "away", "rbi": 0, "outs_recorded": 1},
        {"inning": 1, "half": "B", "team": "home", "rbi": 0, "outs_recorded": 2},
    ]
    r = client.post("/v1/game/log", json={"game_id": "G9", "log": log})
    assert r.status_code == 200
    body = r.json()
    assert body["game_id"] == "G9"
    assert len(body["events"]) == 2

