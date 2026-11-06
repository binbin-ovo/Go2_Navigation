import numpy as np
import pytest
from go2_perception.tracking.kalman_filter import ConstantVelocityFilter, transition


def test_prediction_uses_full_irregular_interval_and_covariance():
    f = ConstantVelocityFilter([0, 0], np.eye(2) * 0.1, max_prediction_interval=0.2)
    f.state[2:] = [2, -1]
    before = f.P.copy()
    f.predict(0.75)
    np.testing.assert_allclose(f.state, [1.5, -0.75, 2, -1])
    a, q = transition(0.75, f.process_noise)
    np.testing.assert_allclose(f.P, a @ before @ a.T + q)
    assert np.linalg.eigvalsh(f.P).min() > 0


def test_velocity_convergence_and_stability():
    f = ConstantVelocityFilter([0, 0], np.eye(2) * 0.01, process_noise=0.04)
    for i in range(1, 200):
        f.predict(0.1)
        f.update([i * 0.1, i * 0.05], np.eye(2) * 0.01)
        assert np.linalg.eigvalsh(f.P).min() > 0
        np.testing.assert_allclose(f.P, f.P.T, atol=1e-12)
    np.testing.assert_allclose(f.state[2:], [1, 0.5], atol=0.01)
    assert f.mahalanobis(f.state[:2], np.eye(2)) == pytest.approx(0)
    assert f.mahalanobis([100, 100], np.eye(2)) > 100


@pytest.mark.parametrize("dt", [0, -1, float("nan"), 1e-8])
def test_invalid_dt(dt):
    f = ConstantVelocityFilter([0, 0], np.eye(2))
    with pytest.raises(ValueError):
        f.predict(dt)


@pytest.mark.parametrize(
    "r", [np.zeros((2, 2)), [[1, 2], [0, 1]], np.eye(3), [[np.nan, 0], [0, 1]]]
)
def test_invalid_covariance(r):
    with pytest.raises(ValueError):
        ConstantVelocityFilter([0, 0], r)
