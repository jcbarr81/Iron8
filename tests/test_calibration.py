from baseball_sim.calibration.calibrators import IdentityCalibrator

def test_identity_calibrator_normalizes():
    calib = IdentityCalibrator()
    probs = {"A":0.2,"B":0.2,"C":0.0}
    out = calib.apply(probs)
    assert abs(sum(out.values()) - 1.0) < 1e-9
