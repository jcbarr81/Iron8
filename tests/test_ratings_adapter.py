from baseball_sim.features.ratings_adapter import batter_features, pitcher_features

def test_batter_pull_and_gbfb_monotone():
    b = {"bats":"L", "ratings":{"pl": 80, "gf": 80}}
    feats = batter_features(b)
    assert feats["bat_pull_z"] > 0
    assert feats["bat_gbfb_z"] > 0   # higher => more GB (later used to lower LA)

def test_pitcher_stuff_uses_arm_or_as():
    p1 = {"ratings":{"arm": 80, "fb":70}}
    p2 = {"ratings":{"as": 80, "fb":70}}
    f1 = pitcher_features(p1)
    f2 = pitcher_features(p2)
    assert abs(f1["pit_stuff_z"] - f2["pit_stuff_z"]) < 1e-9
