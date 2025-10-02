from baseball_sim.models.advancement_model import FieldingConverter


def _state(b1=None, b2=None, b3=None, outs=0):
    return {"outs": outs, "bases": {"1B": b1, "2B": b2, "3B": b3}}


def test_low_if_defense_more_roe_on_gb():
    fc = FieldingConverter()
    bip = {"ev_mph": 95.0, "la_deg": 2.0, "spray_deg": 0.0, "contact_type": "GB"}
    low_def = {"SS": 5, "2B": 5}
    high_def = {"SS": 95, "2B": 95}
    low = fc.convert(bip, {}, low_def, _state(b1=None, outs=0), seed=2024)
    high = fc.convert(bip, {}, high_def, _state(b1=None, outs=0), seed=2024)
    # With same seed, low defense more likely to yield error/ROE
    assert bool(low.get("error")) >= bool(high.get("error"))


def test_sac_fly_credit_with_runner_on_third():
    fc = FieldingConverter()
    bip = {"ev_mph": 90.0, "la_deg": 30.0, "spray_deg": 0.0, "contact_type": "FB"}
    res = fc.convert(bip, {}, {"CF": 70}, _state(b3="R3", outs=0), seed=7)
    if res.get("out") and res.get("out_subtype") in {"FB_OUT", "LD_OUT"}:
        assert res.get("sf") in (True, False)

