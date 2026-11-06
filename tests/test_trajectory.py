import numpy as np
import pytest
from go2_perception.tracking.trajectory import predict_trajectory


def test_constant_velocity_prediction_and_uncertainty():
    result = predict_trajectory(np.array([1, 2, 1, -0.5]), np.eye(4) * 0.1, 0.4, 10)
    assert len(result) == 15 and result[-1].timestamp == pytest.approx(13)
    np.testing.assert_allclose(result[-1].position, [4, 0.5, 0.4])
    assert np.trace(result[-1].position_covariance) > np.trace(
        result[0].position_covariance
    )


def test_invalid_horizon():
    with pytest.raises(ValueError):
        predict_trajectory(np.zeros(4), np.eye(4), 0, 0, step=0)
