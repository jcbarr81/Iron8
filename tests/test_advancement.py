from baseball_sim.models.advancement_model import BaserunningAdvancer


def _state(b1=None, b2=None, b3=None, outs=0):
    return {"outs": outs, "bases": {"1B": b1, "2B": b2, "3B": b3}}


def test_of_arm_reduces_3b_send():
    adv = BaserunningAdvancer()
    # Medium fly ball to CF, runner on 3B
    bip = {"ev_mph": 92.0, "la_deg": 28.0, "spray_deg": 0.0, "contact_type": "FB"}
    field = {"out": False, "fielder": "CF", "bases_ending": {"1B": None, "2B": None, "3B": "R3"}}
    feats = {"bat_speed_z": 0.0}
    low_arm_def = {"arm_by_pos": {"CF": 10}}
    high_arm_def = {"arm_by_pos": {"CF": 90}}

    low = adv.advance(bip, field, feats, low_arm_def, _state(b3="R3"))
    high = adv.advance(bip, field, feats, high_arm_def, _state(b3="R3"))

    # With higher OF arm, rbi should not be greater than with low arm (less likely to send)
    assert high.get("rbi", 0) <= low.get("rbi", 0)


def test_runner_speed_increases_send():
    adv = BaserunningAdvancer()
    bip = {"ev_mph": 92.0, "la_deg": 28.0, "spray_deg": 0.0, "contact_type": "FB"}
    field = {"out": False, "fielder": "CF", "bases_ending": {"1B": None, "2B": None, "3B": "R3"}}
    slow = adv.advance(bip, field, {"bat_speed_z": -1.0}, {"arm_by_pos": {"CF": 20}}, _state(b3="R3"))
    fast = adv.advance(bip, field, {"bat_speed_z": 1.0}, {"arm_by_pos": {"CF": 20}}, _state(b3="R3"))
    assert fast.get("rbi", 0) >= slow.get("rbi", 0)

