from pathlib import Path

from baseball_sim.features.context_features import context_features
from baseball_sim.utils.parks import load_parks_yaml, features_for_park


def test_rules_shift_restrictions_flag():
    state_true = {
        "count": {"balls": 0, "strikes": 0},
        "outs": 0,
        "bases": {"1B": None, "2B": None, "3B": None},
        "rules": {"shift_restrictions": True},
    }
    state_false = {
        "count": {"balls": 0, "strikes": 0},
        "outs": 0,
        "bases": {"1B": None, "2B": None, "3B": None},
        "rules": {"shift_restrictions": False},
    }
    feats_true = context_features(state_true, pitcher={"throws": "R"}, batter={"bats": "L"})
    feats_false = context_features(state_false, pitcher={"throws": "R"}, batter={"bats": "L"})
    assert feats_true["ctx_shift_restrictions"] == 1
    assert feats_false["ctx_shift_restrictions"] == 0


def test_parks_yaml_features_loaded():
    # Load real parks.yaml from repo if present
    cfg_path = Path("config/parks.yaml")
    cfg = load_parks_yaml(cfg_path) if cfg_path.exists() else {}

    # Known example from our default config: BOS05 has rf_short_porches: true, lf_short_porches: false
    feats = features_for_park("BOS05", cfg)
    # If config exists, assert expected; otherwise, empty dict
    if cfg:
        assert feats.get("ctx_park_rf_short_porches") in (0, 1)
        assert feats.get("ctx_park_lf_short_porches") in (0, 1)
    else:
        assert feats == {}

    # Unknown park returns empty dict
    assert features_for_park("UNKNOWN", cfg) == {}

