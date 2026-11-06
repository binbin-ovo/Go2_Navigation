import numpy as np
import pytest
from go2_perception.io.synthetic import cases, make_scene
from go2_perception.lidar.preprocessing import preprocess
from go2_perception.lidar.ground_segmentation import ground_mask
from go2_perception.lidar.clustering import cluster_points
from go2_perception.lidar.processor import LidarProcessor


def test_two_obstacles_and_bounds():
    r = LidarProcessor().process(make_scene())
    assert len(r.obstacles) == 2
    assert len(r.ground) > 1450
    assert len(r.points) == len(r.ground) + len(r.unclassified) + sum(
        len(o.points) for o in r.obstacles
    )
    centers = sorted(r.obstacles, key=lambda o: o.center[0])
    np.testing.assert_allclose(centers[0].center, [2.25, -0.75, 0.125], atol=0.07)
    for o in centers:
        assert np.all(o.points >= o.bbox_min) and np.all(o.points <= o.bbox_max)


@pytest.mark.parametrize("name", ["empty", "sparse", "nan", "partial", "nearby"])
def test_edge_scenes(name):
    r = LidarProcessor().process(cases()[name])
    assert np.isfinite(r.points).all()
    if name in ("empty", "sparse"):
        assert not r.obstacles
    if name == "partial":
        assert len(r.obstacles) == 1
    if name == "nearby":
        assert len(r.obstacles) == 2


def test_invalid_filtering_and_range():
    p = [[np.nan, 1, 0], [1, 0, 0], [40, 0, 0], [0, 0, 0], [2, 0, 0]]
    np.testing.assert_allclose(
        preprocess(p, min_range=0.5, max_range=3, voxel_size=0), [[1, 0, 0], [2, 0, 0]]
    )


def test_voxel_reduces_duplicates():
    assert len(preprocess(np.tile([1.0, 0, 0], (20, 1)))) == 1


def test_wall_is_not_ground():
    rng = np.random.default_rng(0)
    wall = np.column_stack(
        (np.full(200, 2), rng.uniform(-2, 2, 200), rng.uniform(-0.5, 2, 200))
    )
    assert not ground_mask(wall).any()


def test_dbscan_merge_parameter():
    p = cases()["nearby"]
    assert len(cluster_points(p, eps=0.2)[1]) == 2
    assert len(cluster_points(p, eps=0.7)[1]) == 1


def test_deterministic_ground():
    np.testing.assert_array_equal(ground_mask(make_scene()), ground_mask(make_scene()))
