import numpy as np
import pytest
from go2_perception.fusion.calibration import Calibration
from go2_perception.fusion.projection import project_points
from go2_perception.fusion.association import FusionProcessor
from go2_perception.core.types import RGBResult, LidarResult, Obstacle, Detection2D


def calibration():
    return Calibration(
        np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]]), np.eye(4), (100, 100)
    )


def inputs():
    p = np.array([[0, 0, 2], [0.1, 0, 2], [0, 0.1, 2]])
    o = Obstacle(0, p, p.mean(0), p.min(0), p.max(0))
    rgb = RGBResult(
        [Detection2D(0, "person", 0.9, (40, 40, 60, 60))],
        (100, 100),
        timestamp=1.0,
        clock_domain="sim",
    )
    lidar = LidarResult(
        p,
        np.empty((0, 3)),
        np.empty((0, 3)),
        [o],
        np.zeros(3),
        timestamp=1.01,
        clock_domain="sim",
    )
    return rgb, lidar


def test_projection_invalid_depth_boundaries():
    uv, depth, idx = project_points(
        [[0, 0, 2], [0, 0, -1], [0, 0, 0], [10, 0, 1], [np.nan, 0, 1], [0.5, 0, 1]],
        calibration(),
    )
    np.testing.assert_allclose(uv, [[50, 50]])
    np.testing.assert_array_equal(idx, [0])


def test_camera_to_lidar_direction():
    c = calibration()
    c.T_lidar_camera[0, 3] = 1
    uv, _, _ = project_points([[1, 0, 2]], c)
    np.testing.assert_allclose(uv, [[50, 50]])


def test_empty_projection():
    assert project_points(np.empty((0, 3)), calibration())[0].shape == (0, 2)


def test_association_and_one_to_one():
    rgb, lidar = inputs()
    rgb.detections *= 2
    r = FusionProcessor().fuse(rgb, lidar, calibration())
    assert len(r.detections) == 1 and len(r.unmatched_rgb) == 1
    assert r.detections[0].timestamp == lidar.timestamp
    assert "NOT validated" in r.validation


def test_no_association():
    rgb, lidar = inputs()
    rgb.detections[0].bbox_xyxy = (0, 0, 10, 10)
    r = FusionProcessor().fuse(rgb, lidar, calibration())
    assert not r.detections and r.unmatched_clusters == [0]


@pytest.mark.parametrize("change", ["time", "clock", "missing", "nan", "frame", "size"])
def test_reject_incompatible_observations(change):
    rgb, lidar = inputs()
    if change == "time":
        lidar.timestamp = 2
    if change == "clock":
        lidar.clock_domain = "other"
    if change == "missing":
        lidar.timestamp = None
    if change == "nan":
        lidar.timestamp = np.nan
    if change == "frame":
        lidar.frame_id = "wrong"
    if change == "size":
        rgb.image_size = (200, 100)
    with pytest.raises(ValueError):
        FusionProcessor().fuse(rgb, lidar, calibration())


def test_timestamp_opt_out_requires_both_missing():
    rgb, lidar = inputs()
    rgb.timestamp = lidar.timestamp = None
    assert (
        len(
            FusionProcessor(require_timestamps=False)
            .fuse(rgb, lidar, calibration())
            .detections
        )
        == 1
    )
    lidar.timestamp = 1
    with pytest.raises(ValueError):
        FusionProcessor(require_timestamps=False).fuse(rgb, lidar, calibration())


def test_sync_boundary():
    rgb, lidar = inputs()
    lidar.timestamp = 1.05
    assert FusionProcessor().fuse(rgb, lidar, calibration()).detections


def test_no_detections_or_clusters():
    rgb, lidar = inputs()
    rgb.detections = []
    assert not FusionProcessor().fuse(rgb, lidar, calibration()).detections
    rgb, lidar = inputs()
    lidar.obstacles = []
    assert not FusionProcessor().fuse(rgb, lidar, calibration()).detections


def test_distortion_rejected():
    with pytest.raises(ValueError):
        Calibration(np.eye(3), np.eye(4), (100, 100), distortion=[0.1])
