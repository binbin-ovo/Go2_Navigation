import numpy as np
from go2_perception.core.tracking_types import Detection3D
from go2_perception.tracking.config import TrackerConfig
from go2_perception.tracking.multi_object_tracker import _Track
from go2_perception.tracking.association import associate, deduplicate, box_iou


def detection(x, category=None):
    return Detection3D(
        0,
        [x, 0, 0],
        "odom",
        "sim",
        class_name=category,
        measurement_covariance=np.eye(3) * 0.01,
    )


def test_hungarian_gating_unknown_and_known_classes():
    c = TrackerConfig()
    tracks = [_Track(1, detection(0, "person"), 0, c), _Track(2, detection(5), 0, c)]
    matches, misses, new = associate(
        tracks, [detection(5.02), detection(0.01, "person"), detection(100)], c
    )
    assert set(matches) == {(0, 1), (1, 0)} and misses == [] and new == [2]
    assert associate(tracks, [detection(0, "car")], c)[0] == []


def test_vertical_gate_and_empty_inputs():
    c = TrackerConfig()
    t = _Track(1, detection(0), 0, c)
    d = detection(0)
    d.position[2] = 5
    assert associate([t], [d], c)[0] == []
    assert associate([], [], c) == ([], [], [])


def test_duplicates_and_iou():
    assert len(deduplicate([detection(0), detection(0.001)], 0.02)) == 1
    assert len(deduplicate([detection(0, "car"), detection(0, "person")], 0.02)) == 2
    assert box_iou(np.zeros(3), np.ones(3), np.zeros(3), np.ones(3)) == 1
