import pytest
from go2_perception.core.tracking_types import Detection3D
from go2_perception.tracking.multi_object_tracker import MultiObjectTracker
from go2_perception.tracking.config import TrackerConfig


def d(t, x=2):
    return Detection3D(t, [x, 0, 0.5], "odom", "sim")


def test_lifecycle_and_reacquisition_and_deletion():
    tracker = MultiObjectTracker({"max_misses": 3, "max_unobserved_s": 2})
    assert tracker.update([d(0)], 0)[0].status == "Tentative"
    tracker.update([d(0.1)], 0.1)
    t = tracker.update([d(0.2)], 0.2)[0]
    assert t.status == "Confirmed"
    for time in (0.3, 0.4):
        out = tracker.update([], time, frame_id="odom")
    assert out[0].status == "Lost" and out[0].consecutive_misses == 2
    assert out[0].motion_state == "UNCERTAIN" and not out[0].predicted_positions
    out = tracker.update([d(0.5)], 0.5)
    assert out[0].track_id == t.track_id and out[0].status == "Confirmed"
    for time in (0.6, 0.7, 0.8, 0.9):
        out = tracker.update([], time, frame_id="odom")
    assert out == [] and tracker.deleted_ids == [t.track_id]
    assert tracker.update([d(1)], 1)[0].track_id != t.track_id


def test_long_gap_never_reuses_stale_track():
    tracker = MultiObjectTracker()
    old = tracker.update([d(0)], 0)[0]
    new = tracker.update([d(10)], 10)[0]
    assert old.track_id != new.track_id and tracker.deleted_ids == [old.track_id]


def test_rejected_update_is_atomic_and_output_is_detached():
    tracker = MultiObjectTracker()
    first = tracker.update([d(1)], 1)[0]
    for time in (1, 0.5):
        with pytest.raises(ValueError):
            tracker.update([d(time)], time)
    wrong = d(1.1)
    wrong.clock_domain = "host"
    with pytest.raises(ValueError):
        tracker.update([wrong], 1.1)
    out = tracker.update([d(1.1)], 1.1)[0]
    assert out.hits == 2 and out.track_id == first.track_id
    first.position[:] = 100
    first.state_covariance[:] = 100
    assert out.position[0] < 3


def test_confirmation_requires_recent_hits_and_duplicate_suppression():
    tracker = MultiObjectTracker()
    assert len(tracker.update([d(0), d(0, 0.0 + 2.001)], 0)) == 1
    assert tracker.suppressed_duplicates == 1
    tracker.update([], 0.1, frame_id="odom")
    for time in (1.2, 1.3):
        assert tracker.update([d(time)], time)[0].status == "Tentative"
    assert tracker.update([d(1.4)], 1.4)[0].status == "Confirmed"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "unknown"},
        {"process_noise": -1},
        {"measurement_noise": 0},
        {"min_hits": 0},
        {"max_misses": 1},
        {"min_dt": 1},
        {"trajectory_step_s": 4},
        {"moving_speed": 0.1},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        TrackerConfig(**kwargs)
