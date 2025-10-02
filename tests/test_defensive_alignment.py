from baseball_sim.models.advancement_model import FieldingConverter


def _state(b1=None, b2=None, b3=None, outs=0):
    return {"outs": outs, "bases": {"1B": b1, "2B": b2, "3B": b3}}


def test_infield_in_reduces_gb_single_but_more_ld_hits():
    fc = FieldingConverter()
    bip_gb = {"ev_mph": 86.0, "la_deg": 5.0, "spray_deg": 0.0, "contact_type": "GB"}
    bip_ld = {"ev_mph": 92.0, "la_deg": 20.0, "spray_deg": 0.0, "contact_type": "LD"}
    normal_def = {"oaa_by_pos": {"SS": 50, "2B": 50}, "alignment": {"infield": "normal"}}
    in_def = {"oaa_by_pos": {"SS": 50, "2B": 50}, "alignment": {"infield": "in"}}

    # Grounders: infield-in should result in more outs (less singles)
    n = fc.convert(bip_gb, {}, normal_def, _state(), seed=101)
    i = fc.convert(bip_gb, {}, in_def, _state(), seed=101)
    assert i.get("outs_recorded", 0) >= n.get("outs_recorded", 0)

    # Line drives: infield-in should yield fewer outs (more LD hits)
    n2 = fc.convert(bip_ld, {}, normal_def, _state(), seed=102)
    i2 = fc.convert(bip_ld, {}, in_def, _state(), seed=102)
    assert i2.get("outs_recorded", 0) <= n2.get("outs_recorded", 0)


def test_outfield_deep_changes_short_vs_deep_flies():
    fc = FieldingConverter()
    shallow_fb = {"ev_mph": 88.0, "la_deg": 15.0, "spray_deg": 0.0, "contact_type": "FB"}
    deep_fb = {"ev_mph": 95.0, "la_deg": 30.0, "spray_deg": 0.0, "contact_type": "FB"}
    normal_def = {"oaa_by_pos": {"CF": 60}, "alignment": {"outfield": "normal"}}
    deep_def = {"oaa_by_pos": {"CF": 60}, "alignment": {"outfield": "deep"}}

    # Shallow fly: deep OF should produce fewer outs (more singles)
    n = fc.convert(shallow_fb, {}, normal_def, _state(), seed=201)
    d = fc.convert(shallow_fb, {}, deep_def, _state(), seed=201)
    assert d.get("outs_recorded", 0) <= n.get("outs_recorded", 0)

    # Deep fly: deep OF should produce more outs
    n2 = fc.convert(deep_fb, {}, normal_def, _state(), seed=202)
    d2 = fc.convert(deep_fb, {}, deep_def, _state(), seed=202)
    assert d2.get("outs_recorded", 0) >= n2.get("outs_recorded", 0)

