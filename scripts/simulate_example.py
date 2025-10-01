"""
End-to-end local example (no HTTP).
"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from baseball_sim.models.pa_model import PaOutcomeModel
from baseball_sim.features.ratings_adapter import batter_features, pitcher_features
from baseball_sim.calibration.calibrators import IdentityCalibrator
from baseball_sim.sampler.rng import sample_event

def main():
    pa = PaOutcomeModel()
    calib = IdentityCalibrator()

    batter = {"bats":"L","ratings":{"gf":80,"pl":85,"sc":60,"fa":70,"arm":65,"ph":72}}
    pitcher = {"ratings":{"arm":88,"fb":80,"sl":72,"si":65,"cu":50,"cb":60,"fa":70,"control":55,"movement":62}}

    feats = batter_features(batter) | pitcher_features(pitcher)
    probs = pa.predict_proba(feats)
    probs = calib.apply(probs)
    ev = sample_event(probs, seed=123)

    print(probs)
    print("EVENT:", ev)

if __name__ == "__main__":
    main()
