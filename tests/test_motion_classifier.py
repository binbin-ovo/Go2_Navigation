import numpy as np
from go2_perception.tracking.config import TrackerConfig
from go2_perception.tracking.motion_classifier import classify_motion


def test_evidence_uncertainty_and_hysteresis():
    c = TrackerConfig()
    p = np.eye(2) * 0.001
    assert classify_motion([0, 0], p, 10, 2, 0, c) == "STATIONARY"
    assert classify_motion([1, 0], p, 10, 2, 0, c) == "MOVING"
    assert classify_motion([1, 0], p, 1, 0, 0, c) == "UNCERTAIN"
    assert classify_motion([1, 0], np.eye(2), 10, 2, 0, c) == "UNCERTAIN"
    assert classify_motion([1, 0], p, 10, 2, 1, c) == "UNCERTAIN"
    assert classify_motion([0.23, 0], p, 10, 2, 0, c, "MOVING") == "MOVING"
    assert classify_motion([0.23, 0], p, 10, 2, 0, c) == "UNCERTAIN"
