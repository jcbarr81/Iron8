from baseball_sim.utils.fatigue import compute_fatigue_z, apply_fatigue
from fastapi.testclient import TestClient
from baseball_sim.serve.api import app


def test_fatigue_increases_with_pitch_count_and_tto():
    f_low = compute_fatigue_z(pitch_count=20, tto_index=1, entered_cold=False, batters_faced_in_stint=3)
    f_high = compute_fatigue_z(pitch_count=90, tto_index=3, entered_cold=False, batters_faced_in_stint=3)
    assert f_high > f_low


def test_apply_fatigue_reduces_control_and_stuff():
    feats = {"pit_control_z": 1.0, "pit_stuff_z": 1.0}
    adj = apply_fatigue(feats, pitch_count=90, tto_index=3, entered_cold=True, batters_faced_in_stint=0)
    assert adj["pit_control_z"] < feats["pit_control_z"]
    assert adj["pit_stuff_z"] < feats["pit_stuff_z"]


def test_three_batter_min_endpoint():
    client = TestClient(app)
    # Not allowed: three_batter_min true, bfis 1, no inning end or injury
    r = client.post("/v1/rules/check-three-batter-min", json={
        "three_batter_min": True,
        "batters_faced_in_stint": 1,
        "inning_ended": False,
        "injury_exception": False
    })
    assert r.status_code == 200 and r.json()["allowed"] is False

    # Allowed: injury exception
    r2 = client.post("/v1/rules/check-three-batter-min", json={
        "three_batter_min": True,
        "batters_faced_in_stint": 1,
        "inning_ended": False,
        "injury_exception": True
    })
    assert r2.status_code == 200 and r2.json()["allowed"] is True

