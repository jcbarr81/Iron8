from baseball_sim.models.advancement_model import FieldingConverter


def _state(b1=None, b2=None, b3=None, outs=0):
    return {"outs": outs, "bases": {"1B": b1, "2B": b2, "3B": b3}}


def test_if_fa_increases_gb_out_and_dp():
    fc = FieldingConverter()
    bip = {"ev_mph": 86.0, "la_deg": 5.0, "spray_deg": 0.0, "contact_type": "GB"}
    feats = {}
    low_def = {"SS": 10, "2B": 10, "3B": 10, "1B": 10}
    high_def = {"SS": 90, "2B": 90, "3B": 90, "1B": 90}

    # Runner on first, <2 outs -> DP possible. Same seed for comparability
    low = fc.convert(bip, feats, low_def, _state(b1="R1", outs=0), seed=123)
    high = fc.convert(bip, feats, high_def, _state(b1="R1", outs=0), seed=123)

    # With higher IF defense, out more likely; with same seed, expect high->out where low may be hit
    assert high["outs_recorded"] >= low.get("outs_recorded", 0)
    # At least one of them should indicate GB_OUT when an out occurs
    if high["outs_recorded"]:
        assert high.get("out_subtype") == "GB_OUT"


def test_of_fa_increases_fb_out():
    fc = FieldingConverter()
    bip = {"ev_mph": 92.0, "la_deg": 30.0, "spray_deg": 0.0, "contact_type": "FB"}
    feats = {}
    low_def = {"CF": 10}
    high_def = {"CF": 90}

    low = fc.convert(bip, feats, low_def, _state(outs=0), seed=777)
    high = fc.convert(bip, feats, high_def, _state(outs=0), seed=777)

    assert high["outs_recorded"] >= low.get("outs_recorded", 0)
    if high["outs_recorded"]:
        assert high.get("out_subtype") in {"FB_OUT", "LD_OUT"}

