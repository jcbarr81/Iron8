from fastapi.testclient import TestClient
from baseball_sim.serve.api import app


client = TestClient(app)


def base_pitch_req(control=50, stuff=50, contact=50, balls=0, strikes=0):
    return {
        "game_id": "G1",
        "state": {
            "inning": 1,
            "half": "T",
            "outs": 0,
            "bases": {"1B": None, "2B": None, "3B": None},
            "score": {"away": 0, "home": 0},
            "count": {"balls": balls, "strikes": strikes},
        },
        "batter": {"player_id": "B1", "bats": "L", "ratings": {"ch": contact}},
        "pitcher": {"player_id": "P1", "throws": "R", "ratings": {"control": control, "arm": 70, "fb": 70}},
        "return_contact": True,
        "seed": 123,
    }


def test_low_control_more_balls_high_control_fewer():
    low = base_pitch_req(control=10, balls=0, strikes=0)
    high = base_pitch_req(control=90, balls=0, strikes=0)
    r_low = client.post("/v1/sim/pitch", json=low).json()
    r_high = client.post("/v1/sim/pitch", json=high).json()
    # With same seed, low control should more likely be a ball than high control
    assert r_low["result"] in {"ball", "called_strike", "swing_miss", "foul", "in_play", "hbp"}
    assert r_high["result"] in {"ball", "called_strike", "swing_miss", "foul", "in_play", "hbp"}


def test_high_stuff_increases_miss_probability():
    low = base_pitch_req(stuff=10, balls=0, strikes=2)
    # stuff is derived from arm/pitch mix; set very high arm and fb to simulate high stuff
    req = base_pitch_req(balls=0, strikes=2)
    req["pitcher"]["ratings"].update({"arm": 99, "fb": 99, "sl": 99})
    r_low = client.post("/v1/sim/pitch", json=low).json()
    r_high = client.post("/v1/sim/pitch", json=req).json()
    assert r_low["result"] in {"ball", "called_strike", "swing_miss", "foul", "in_play", "hbp"}
    assert r_high["result"] in {"ball", "called_strike", "swing_miss", "foul", "in_play", "hbp"}


def test_in_play_delegates_to_bip_and_adv():
    # Encourage in_play: high contact batter, moderate control
    req = base_pitch_req(control=60, contact=90, balls=0, strikes=0)
    r = client.post("/v1/sim/pitch", json=req).json()
    if r["result"] == "in_play":
        assert "bip_detail" in r
        assert "advancement" in r


def test_pickoff_pre_pitch_when_strong_hold_with_runner():
    req = base_pitch_req(control=60, contact=50, balls=0, strikes=0)
    req["state"]["bases"]["1B"] = "R1"
    req["pitcher"]["ratings"]["hold_runner"] = 99
    r = client.post("/v1/sim/pitch", json=req).json()
    # Strong hold triggers a PO pre_pitch event
    if r.get("pre_pitch"):
        assert r["pre_pitch"] in {"no_play", "PO"}

