"""Regressions derived from A1/A2/B/C failure mechanisms; no private sensor inputs."""

import numpy as np
import pytest
from go2_perception.io.config import load_config
from go2_perception.tracking.config import TrackerConfig
from go2_perception.tracking.replay import frame_from_dict
from go2_perception.tracking.multi_object_tracker import MultiObjectTracker
from go2_perception.tracking.robustness_scenarios import scenarios, SCENARIOS
from go2_perception.core.tracking_types import Detection3D


def run(name, improved=True):
    config = load_config(
        "configs/tracking_robustness.yaml" if improved else "configs/tracking.yaml"
    )["tracker"]
    tracker = MultiObjectTracker(config)
    outputs = []
    groups = 0
    for row in scenarios()[name]:
        f = frame_from_dict(row)
        if row.get("reset_before"):
            tracker.reset()
        outputs.append(
            tracker.update(
                f.detections,
                f.timestamp,
                f.robot_pose,
                frame_id=f.frame_id,
                clock_domain=f.clock_domain,
            )
        )
        groups += len(tracker.fragment_groups)
    return outputs, groups


def test_fragments_rejoin_only_after_full_object_history():
    base, _ = run("fragmentation", False)
    new, groups = run("fragmentation")
    assert groups > 50
    assert max(sum(t.status == "Confirmed" for t in frame) for frame in new) == 1
    assert len({t.track_id for frame in new for t in frame}) < len(
        {t.track_id for frame in base for t in frame}
    )


def test_two_nearby_objects_are_not_merged():
    outputs, groups = run("nearby_objects")
    assert groups == 0
    assert len(outputs[-1]) == 2
    assert len({t.track_id for frame in outputs for t in frame}) == 2


def test_short_occlusion_reactivates_same_confirmed_id():
    base, _ = run("occlusion", False)
    new, _ = run("occlusion")
    assert base[20][0].track_id != base[60][0].track_id
    assert new[20][0].track_id == new[60][0].track_id
    assert new[40][0].status == "Lost" and new[40][0].motion_state == "UNCERTAIN"
    assert not new[40][0].predicted_positions


def test_time_expiry_still_deletes_stale_tracks_and_tentatives():
    c = load_config("configs/tracking_robustness.yaml")["tracker"]
    c["min_hits"] = 1
    tracker = MultiObjectTracker(c)
    d = Detection3D(0, [0, 0, 0], "odom", "sim")
    old = tracker.update([d], 0)[0].track_id
    assert tracker.update([], 1.6, frame_id="odom") == []
    assert old in tracker.deleted_ids


def test_pose_outage_reset_is_not_undone_by_longer_coasting():
    outputs, _ = run("odometry_gap")
    assert outputs[20][0].track_id != outputs[-1][0].track_id


def test_yaw_wrap_does_not_produce_world_velocity():
    outputs, _ = run("rotation_wrap")
    assert len({t.track_id for frame in outputs for t in frame}) == 1
    assert np.linalg.norm(outputs[-1][0].velocity) < 0.2
    assert outputs[-1][0].motion_state != "MOVING"


def test_stationary_classification_recovers_without_claiming_moving_target_static():
    before, _ = run("stationary_noise", False)
    after, _ = run("stationary_noise")

    def count(rows):
        return sum(t.motion_state == "STATIONARY" for row in rows for t in row)

    assert count(after) > count(before)
    rows, _ = run("brief_stop")
    assert any(t.motion_state == "STATIONARY" for row in rows[50:60] for t in row)
    assert rows[-1][0].motion_state == "MOVING"


@pytest.mark.parametrize("name", SCENARIOS)
def test_all_regressions_are_finite(name):
    rows, _ = run(name)
    assert all(np.isfinite(t.state_covariance).all() for row in rows for t in row)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fragment_min_iou": 0},
        {"fragment_min_hits": 0},
        {"fragment_margin": -1},
        {"adaptive_gate_max_distance": 0.1},
        {"motion_dwell_s": -1},
        {"confirmed_time_expiry": "true"},
    ],
)
def test_invalid_robustness_config(kwargs):
    with pytest.raises(ValueError):
        TrackerConfig(**kwargs)


def test_adaptive_gate_uses_covariance_but_keeps_hard_cap():
    from go2_perception.tracking.association import associate
    from go2_perception.tracking.multi_object_tracker import _Track

    d = Detection3D(
        0, [0, 0, 0], "odom", "sim", measurement_covariance=np.eye(3) * 0.01
    )
    c = TrackerConfig(adaptive_gate_max_distance=2.5)
    track = _Track(1, d, 0, c)
    track.filter.P[:2, :2] = np.eye(2) * 0.5
    near = Detection3D(
        0, [1.8, 0, 0], "odom", "sim", measurement_covariance=np.eye(3) * 0.01
    )
    far = Detection3D(
        0, [3.0, 0, 0], "odom", "sim", measurement_covariance=np.eye(3) * 0.01
    )
    assert associate([track], [near], TrackerConfig())[0] == []
    assert associate([track], [near], c)[0] == [(0, 0)]
    assert associate([track], [far], c)[0] == []


def test_fragments_without_prior_full_observation_remain_separate():
    c = load_config("configs/tracking_robustness.yaml")["tracker"]
    tracker = MultiObjectTracker(c)
    for r in scenarios()["fragmentation"][15:30]:
        f = frame_from_dict(r)
        tracks = tracker.update(
            f.detections, f.timestamp, frame_id="odom", clock_domain="synthetic"
        )
        assert not tracker.fragment_groups
    assert len(tracks) == 2


def test_missing_observation_never_keeps_stationary_prediction_confident():
    c = load_config("configs/tracking_robustness.yaml")["tracker"]
    tracker = MultiObjectTracker(c)
    for i in range(30):
        d = Detection3D(i * 0.1, [0, 0, 0], "odom", "sim")
        tracks = tracker.update([d], i * 0.1)
    assert tracks[0].motion_state == "STATIONARY"
    missing = tracker.update([], 3.0, frame_id="odom")[0]
    assert missing.motion_state == "UNCERTAIN" and not missing.predicted_positions
