import math

import numpy as np
import pytest

from go2_perception.vln.waypoints import (
    PrefixLimits,
    bounded_prefix,
    path_errors,
    to_world,
    wrap_angle,
)


@pytest.mark.parametrize("target", [[1, 0, 0], [0, 0, 1], [0, 0, -1], [0.5, 0.2, 0.7]])
def test_known_endpoint_and_motion_limits(target):
    start = [2, -1, 0.6]
    plan = bounded_prefix([target], start, limits=PrefixLimits(window_s=3))
    poses, times = (
        np.array(plan["planned_world_poses"]),
        np.array(plan["planned_times_s"]),
    )
    assert poses[-1] == pytest.approx(to_world([target], start)[0])
    assert times[-1] == 3
    dt = np.diff(times)
    assert np.all(dt > 0) and dt.max() <= 0.0100000001
    assert (
        np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1) / dt
    ).max() <= 0.80000001
    assert (np.abs(wrap_angle(np.diff(poses[:, 2]))) / dt).max() <= 3.00000001


def test_absolute_curved_path_keeps_one_origin_and_heading():
    # Yaw intentionally differs from the tangent: it is a pose, not a bearing.
    chunk = [[0.1, 0, 0.1], [0.2, 0.1, 0.3], [0.25, 0.2, -0.1]]
    start = [3, 2, math.pi / 2]
    plan = bounded_prefix(chunk, start, limits=PrefixLimits(window_s=2))
    assert plan["planned_endpoint"] == pytest.approx([2.8, 2.25, math.pi / 2 - 0.1])
    assert plan["selected_indices"] == [0, 1, 2]
    errors = path_errors(plan["planned_world_poses"], start, chunk)
    assert errors["max_cross_track_m"] < 1e-12
    assert errors["max_heading_error_rad"] < 1e-10


def test_near_zero_first_row_is_not_stop_and_later_motion_is_selected():
    plan = bounded_prefix([[1e-8, 0, 0], [0.3, 0, 0]], [0, 0, 0])
    assert not plan["native_stop"]
    assert plan["selected_indices"] == [0, 1]
    assert plan["planned_endpoint"] == pytest.approx([0.2, 0, 0])
    assert plan["requested_world_targets"][-1] == [0.3, 0, 0]


def test_native_stop_requires_whole_decoded_chunk_and_has_no_execution():
    plan = bounded_prefix(np.zeros((10, 3)), [1, 2, 3], native_stop=True)
    assert plan["planned_times_s"] == [0]
    assert plan["planned_world_poses"] == [[1, 2, 3]]
    with pytest.raises(ValueError, match="contradicts"):
        bounded_prefix([[0, 0, 0], [1, 0, 0]], [0, 0, 0], native_stop=True)


def test_finished_path_holds_without_claiming_native_stop():
    plan = bounded_prefix([[0, 0, 0]], [1, 2, 3])
    assert not plan["native_stop"]
    assert plan["planned_times_s"][-1] == 0.25
    assert plan["planned_endpoint"] == [1, 2, 3]


def test_heading_wrap_uses_short_turn_from_nonzero_heading():
    plan = bounded_prefix([[0, 0, 0.2]], [2, -1, math.pi - 0.1])
    assert plan["planned_endpoint"][2] == pytest.approx(-math.pi + 0.1)
    turns = wrap_angle(np.diff(np.array(plan["planned_world_poses"])[:, 2]))
    assert turns.sum() == pytest.approx(0.2)
    assert abs(turns).max() <= 0.03000001


@pytest.mark.parametrize("bad", [[], [[float("nan"), 0, 0]], [[1, 2]], [1, 2, 3]])
def test_reject_invalid_chunks(bad):
    with pytest.raises(ValueError):
        bounded_prefix(bad, [0, 0, 0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"linear_mps": 0},
        {"angular_radps": -1},
        {"window_s": float("nan")},
        {"substep_s": 0.1},
    ],
)
def test_invalid_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        PrefixLimits(**kwargs)


def test_inputs_not_mutated_and_no_goal_or_hardware_fields():
    path = np.array([[0.1, 0, 0], [0.2, 0, 0]])
    saved = path.copy()
    plan = bounded_prefix(path, [1, 2, 3])
    assert np.array_equal(path, saved)
    assert not plan["physical_execution_authorized"]
    assert not any("goal" in key for key in plan)


def test_path_error_detects_off_path_execution():
    error = path_errors([[0.5, 0.2, 0.3]], [0, 0, 0], [[1, 0, 0]])
    assert error["max_cross_track_m"] == pytest.approx(0.2)
    assert error["max_heading_error_rad"] == pytest.approx(0.3)
