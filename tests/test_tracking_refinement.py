import numpy as np
import pytest
from go2_perception.core.tracking_types import Detection3D
from go2_perception.io.config import load_config
from go2_perception.tracking.config import TrackerConfig
from go2_perception.tracking.multi_object_tracker import MultiObjectTracker, _Track
from go2_perception.tracking.track_duplicates import suppress_track_duplicates
from go2_perception.tracking.robustness_scenarios import scenarios, SCENARIOS
from go2_perception.tracking.replay import frame_from_dict
from go2_perception.tracking.motion_classifier import classify_motion


def config():
    return TrackerConfig(**load_config("configs/tracking_refinement.yaml")["tracker"])


def detection(t, x=0, y=0, label=None):
    p = np.array([x, y, 0.5])
    return Detection3D(
        t,
        p,
        "odom",
        "sim",
        bbox_min=p - 0.3,
        bbox_max=p + 0.3,
        class_name=label,
        measurement_covariance=np.eye(3) * 0.01,
    )


def test_duplicate_requires_repeated_overlap_and_preserves_stronger_id():
    c = config()
    a = _Track(1, detection(0), 0, c)
    b = _Track(2, detection(0, 0.05), 0, c)
    a.confirmed = b.confirmed = True
    a.hits = 20
    b.hits = 4
    evidence = {}
    for i in range(3):
        tracks, removed, evidence = suppress_track_duplicates([a, b], evidence, c)
        assert removed == ([2] if i == 2 else [])
    assert tracks[0].track_id == 1


@pytest.mark.parametrize("kind", ["nearby", "velocity", "class", "lost_winner"])
def test_genuine_or_ambiguous_tracks_not_suppressed(kind):
    c = config()
    a = _Track(1, detection(0), 0, c)
    b = _Track(2, detection(0, 0.05 if kind != "nearby" else 0.4), 0, c)
    a.confirmed = b.confirmed = True
    a.hits = 20
    b.hits = 4
    if kind == "velocity":
        b.filter.state[2] = 1
    if kind == "class":
        a.class_name = "person"
        b.class_name = "dog"
    if kind == "lost_winner":
        a.misses = 2
    evidence = {}
    for _ in range(5):
        tracks, removed, evidence = suppress_track_duplicates([a, b], evidence, c)
        assert not removed and len(tracks) == 2


def test_implausible_reactivation_rejected_even_with_large_covariance():
    tracker = MultiObjectTracker(config())
    for i in range(10):
        tracker.update([detection(i * 0.1)], i * 0.1)
    old = tracker._tracks[0].track_id
    tracker.update([], 1.0, frame_id="odom")
    tracker.update([], 1.1, frame_id="odom")
    tracker._tracks[0].filter.P[:2, :2] = np.eye(2) * 100
    rows = tracker.update([detection(1.2, 1.2)], 1.2)
    assert old not in tracker.reactivated_ids
    assert any(t.track_id != old for t in rows)


@pytest.mark.parametrize("gap,retained", [(0.8, True), (1.7, False)])
def test_occlusion_time_bounds(gap, retained):
    tr = MultiObjectTracker(config())
    for i in range(10):
        tr.update([detection(i * 0.1, i * 0.05)], i * 0.1)
    old = tr._tracks[0].track_id
    tr.update([], 1.0, frame_id="odom")
    rows = tr.update([detection(0.9 + gap, 0.45 + 0.5 * gap)], 0.9 + gap)
    assert (
        any(t.track_id == old and t.consecutive_misses == 0 for t in rows)
    ) == retained


def test_motion_release_is_independent_of_stationary_threshold():
    c = config()
    cov = np.eye(2) * 0.01
    assert classify_motion([0.3, 0], cov, 20, 2, 0, c, "MOVING") == "MOVING"
    assert classify_motion([0, 0], cov, 20, 2, 0, c, "MOVING") == "STATIONARY"
    assert classify_motion([0.8, 0], cov, 20, 2, 1, c, "MOVING") == "UNCERTAIN"


@pytest.mark.parametrize("name", SCENARIOS)
def test_refinement_regression_scenarios(name):
    tr = MultiObjectTracker(config())
    ids = set()
    moving = 0
    for row in scenarios()[name]:
        f = frame_from_dict(row)
        if row.get("reset_before"):
            tr.reset()
        outputs = tr.update(
            f.detections,
            f.timestamp,
            f.robot_pose,
            frame_id=f.frame_id,
            clock_domain=f.clock_domain,
        )
        ids.update(t.track_id for t in outputs)
        moving += sum(t.motion_state == "MOVING" for t in outputs)
        assert all(np.isfinite(t.state_covariance).all() for t in outputs)
    if name in ["fragmentation", "occlusion", "rotation_wrap"]:
        assert len(ids) == 1
    if name in ["nearby_objects", "odometry_gap"]:
        assert len(ids) == 2
    if name == "rotation_wrap":
        assert moving == 0
    if name == "brief_stop":
        assert moving > 49


@pytest.mark.parametrize(
    "kw",
    [
        {"reactivation_max_speed": -1},
        {"duplicate_track_hits": -1},
        {"duplicate_track_iou": 2},
        {"moving_dwell_s": -1},
        {"moving_release_speed": 0.4},
    ],
)
def test_invalid_config(kw):
    with pytest.raises(ValueError):
        TrackerConfig(**kw)


def test_coasting_overlap_is_lost_but_keeps_identity():
    c = config()
    a = _Track(1, detection(0), 0, c)
    b = _Track(2, detection(0, 0.05), 0, c)
    a.confirmed = b.confirmed = True
    a.status = b.status = "Confirmed"
    a.hits = 20
    b.hits = 10
    a.misses = 1
    tracks, removed, _ = suppress_track_duplicates([a, b], {}, c)
    assert not removed and len(tracks) == 2
    assert a.status == "Lost" and b.status == "Confirmed"


def test_nonoverlapping_coasting_object_is_not_hidden():
    c = config()
    a = _Track(1, detection(0), 0, c)
    b = _Track(2, detection(0, 0.4), 0, c)
    a.confirmed = b.confirmed = True
    a.status = b.status = "Confirmed"
    a.misses = 1
    _, removed, _ = suppress_track_duplicates([a, b], {}, c)
    assert not removed and a.status == "Confirmed"


def test_displacement_support_rejects_static_noise_motion():
    tr = MultiObjectTracker(config())
    for row in scenarios()["stationary_noise"]:
        f = frame_from_dict(row)
        outputs = tr.update(
            f.detections,
            f.timestamp,
            f.robot_pose,
            frame_id=f.frame_id,
            clock_domain=f.clock_domain,
        )
        assert all(t.motion_state != "MOVING" for t in outputs)


def test_first_miss_is_lost_without_deleting_confirmed_identity():
    tr = MultiObjectTracker(config())
    for i in range(10):
        tr.update([detection(i * 0.1)], i * 0.1)
    identifier = tr._tracks[0].track_id
    row = tr.update([], 1.0, frame_id="odom")[0]
    assert row.track_id == identifier and row.status == "Lost"
    assert not row.predicted_positions and row.motion_state == "UNCERTAIN"
    assert not tr.deleted_ids
