from baseball_sim.features.context_features import context_features


def test_ctx_platoon_advantage_lhb_vs_rhp():
    state = {
        "count": {"balls": 0, "strikes": 0},
        "outs": 0,
        "bases": {"1B": None, "2B": None, "3B": None},
    }
    pitcher = {"throws": "R", "tto_index": 1}
    batter = {"bats": "L"}
    feats = context_features(state, pitcher, batter)
    assert feats["ctx_matchup_is_platoon_adv"] == 1


def test_ctx_platoon_advantage_rhb_vs_rhp():
    state = {
        "count": {"balls": 0, "strikes": 0},
        "outs": 0,
        "bases": {"1B": None, "2B": None, "3B": None},
    }
    pitcher = {"throws": "R", "tto_index": 1}
    batter = {"bats": "R"}
    feats = context_features(state, pitcher, batter)
    assert feats["ctx_matchup_is_platoon_adv"] == 0

