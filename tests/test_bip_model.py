from baseball_sim.models.bip_model import BipModel


def test_bip_ph_increases_ev():
    m = BipModel()
    base_feats = {
        "bat_power_z": 0.0,
        "pit_stuff_z": 0.0,
        "bat_gbfb_z": 0.0,
        "bat_pull_z": 0.0,
        "bat_isLHB": 1,
    }
    low = m.predict({**base_feats, "bat_power_z": -1.0}, seed=123)
    high = m.predict({**base_feats, "bat_power_z": 1.0}, seed=123)
    assert high["ev_mph"] > low["ev_mph"]


def test_bip_gf_more_groundballs_lower_la():
    m = BipModel()
    base_feats = {
        "bat_power_z": 0.0,
        "pit_stuff_z": 0.0,
        "bat_gbfb_z": 0.0,
        "bat_pull_z": 0.0,
        "bat_isLHB": 0,
    }
    low_gf = m.predict({**base_feats, "bat_gbfb_z": -1.0}, seed=7)
    high_gf = m.predict({**base_feats, "bat_gbfb_z": 1.0}, seed=7)
    assert high_gf["la_deg"] < low_gf["la_deg"]
    assert high_gf["contact_type"] in {"GB", "LD", "FB", "PU"}


def test_bip_pl_pulls_by_handedness_sign():
    m = BipModel()
    # LHB: positive spray with higher pull
    lhb_low = m.predict({"bat_pull_z": 0.0, "bat_isLHB": 1}, seed=999)
    lhb_high = m.predict({"bat_pull_z": 1.0, "bat_isLHB": 1}, seed=999)
    assert lhb_high["spray_deg"] > lhb_low["spray_deg"]

    # RHB: negative spray with higher pull
    rhb_low = m.predict({"bat_pull_z": 0.0, "bat_isLHB": 0}, seed=999)
    rhb_high = m.predict({"bat_pull_z": 1.0, "bat_isLHB": 0}, seed=999)
    assert rhb_high["spray_deg"] < rhb_low["spray_deg"]


def test_api_includes_bip_detail_when_requested():
    from fastapi.testclient import TestClient
    from baseball_sim.serve.api import app

    client = TestClient(app)
    req = {
        "game_id": "X",
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
        "batter": {"player_id": "B", "bats": "L", "ratings": {"ph": 70, "pl": 60, "gf": 50}},
        "pitcher": {"player_id": "P", "throws": "R", "ratings": {"arm": 70, "fb": 70, "sl": 60}},
        "return_probabilities": True,
        "return_contact": True,
        "seed": 42,
    }
    r = client.post("/v1/sim/plate-appearance", json=req)
    assert r.status_code == 200
    body = r.json()
    assert "bip_detail" in body
    bd = body["bip_detail"]
    for k in ("ev_mph", "la_deg", "spray_deg", "contact_type"):
        assert k in bd

