import numpy as np
import pytest
import open3d as o3d
from go2_perception.io.pointcloud_loader import load_pointcloud
from go2_perception.io.image_loader import load_image
from go2_perception.io.sensor_interface import RGBMessage, LidarMessage


@pytest.mark.parametrize("extension", ["ply", "pcd", "bin", "npy"])
def test_roundtrip(tmp_path, extension):
    p = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    path = tmp_path / f"points.{extension}"
    if extension in ("ply", "pcd"):
        assert o3d.io.write_point_cloud(
            str(path), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        )
    elif extension == "bin":
        np.column_stack((p, [0.1, 0.2])).astype("<f4").tofile(path)
    else:
        np.save(path, p)
    np.testing.assert_allclose(load_pointcloud(path)[:, :3], p)


def test_bad_bin(tmp_path):
    p = tmp_path / "bad.bin"
    p.write_bytes(b"123")
    with pytest.raises(ValueError):
        load_pointcloud(p)


def test_empty_bin(tmp_path):
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert load_pointcloud(p).shape == (0, 4)


@pytest.mark.parametrize("p", [np.zeros(3), np.zeros((3, 5))])
def test_bad_array(p):
    with pytest.raises(ValueError):
        load_pointcloud(p)


def test_image_validation():
    with pytest.raises(ValueError):
        load_image(np.zeros((20, 20)))
    assert load_image(np.zeros((20, 20, 3), dtype=np.uint8)).shape == (20, 20, 3)


def test_sensor_contract():
    m = RGBMessage(
        np.zeros((10, 20, 3), dtype=np.uint8), 1.0, 20, 10, np.eye(3), "cam", "sim"
    )
    assert m.width == 20
    with pytest.raises(ValueError):
        LidarMessage(np.empty((0, 3)), float("nan"), "lidar", "sim")
    with pytest.raises(ValueError):
        RGBMessage(m.image, 1.0, 10, 20, np.eye(3), "cam", "sim")
