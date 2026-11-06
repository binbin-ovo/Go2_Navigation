import numpy as np
import pytest
from go2_perception.core.geometry import aabb
from go2_perception.core.transforms import transform_points, validate_transform


def test_aabb():
    low, high, center = aabb([[0, 1, 2], [2, 3, 4]])
    np.testing.assert_allclose(center, [1, 2, 3])
    np.testing.assert_allclose(high - low, [2, 2, 2])


def test_transform_direction_and_inverse():
    t = np.eye(4)
    t[:3, 3] = [1, 2, 3]
    p = np.array([[1.0, 0, 0]])
    np.testing.assert_allclose(transform_points(p, t), [[2, 2, 3]])
    np.testing.assert_allclose(
        transform_points(transform_points(p, t), np.linalg.inv(t)), p
    )


@pytest.mark.parametrize(
    "t", [np.zeros((4, 4)), np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1])]
)
def test_invalid_transform(t):
    with pytest.raises(ValueError):
        validate_transform(t)


def test_empty_aabb():
    with pytest.raises(ValueError):
        aabb(np.empty((0, 3)))
