from baseball_sim.models.bip_model import BipModel
from baseball_sim.models.advancement_model import FieldingConverter


def test_temperature_increases_ev():
    m = BipModel()
    base_feats = {
        "bat_power_z": 0.0,
        "pit_stuff_z": 0.0,
        "bat_gbfb_z": 0.0,
        "bat_pull_z": 0.0,
        "bat_isLHB": 1,
    }
    cold = m.predict({**base_feats, "ctx_temp_f": 50.0}, seed=10)
    hot = m.predict({**base_feats, "ctx_temp_f": 90.0}, seed=10)
    assert hot["ev_mph"] > cold["ev_mph"]


def test_tailwind_reduces_fb_outs_vs_headwind():
    fc = FieldingConverter()
    bip = {"ev_mph": 92.0, "la_deg": 30.0, "spray_deg": 0.0, "contact_type": "FB"}
    feats = {}
    defense = {"oaa_by_pos": {"CF": 60}}
    # Tailwind: wind_dir_deg=0 (out), headwind: 180 (in); use high wind_mph
    tail = fc.convert(bip, feats, defense, {"outs": 0, "bases": {}, "weather": {"wind_mph": 20.0, "wind_dir_deg": 0.0}}, seed=5)
    head = fc.convert(bip, feats, defense, {"outs": 0, "bases": {}, "weather": {"wind_mph": 20.0, "wind_dir_deg": 180.0}}, seed=5)
    assert tail.get("outs_recorded", 0) <= head.get("outs_recorded", 0)

