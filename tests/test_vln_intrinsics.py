"""ChArUco intrinsic calibration recovers a known synthetic fisheye camera from rendered views."""

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from go2_perception.vln.intrinsics import calibrate, detect, make_board, select_views, split

SPEC = {"dictionary": "DICT_5X5_100", "squares_x": 8, "squares_y": 6,
        "square_size_m_nominal": 0.03, "marker_size_m_nominal": 0.022, "legacy_pattern": False}
SIZE = (640, 360)
K_TRUE = np.array([[300.0, 0, 322.0], [0, 300.0, 178.0], [0, 0, 1]])
D_TRUE = np.array([0.12, -0.06, 0.03, -0.008])


def render(board, square_m, rvec, tvec, px_per_square=40):
    tex = board.generateImage((8 * px_per_square, 6 * px_per_square), marginSize=0, borderBits=1)
    s = px_per_square / square_m
    u, v = np.meshgrid(np.arange(SIZE[0], dtype=np.float64), np.arange(SIZE[1], dtype=np.float64))
    norm = cv2.fisheye.undistortPoints(np.c_[u.ravel(), v.ravel()].reshape(-1, 1, 2), K_TRUE, D_TRUE)
    d = np.c_[norm.reshape(-1, 2), np.ones(u.size)]
    R = Rotation.from_rotvec(rvec).as_matrix()
    n = R[:, 2]
    lam = (n @ tvec) / (d @ n)
    q = (d * lam[:, None] - tvec) @ R  # camera -> board coordinates
    mx = (q[:, 0] * s - 0.5).reshape(u.shape).astype(np.float32)
    my = (q[:, 1] * s - 0.5).reshape(u.shape).astype(np.float32)
    img = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return cv2.GaussianBlur(img, (3, 3), 0.6)


@pytest.fixture(scope="module")
def detections():
    board = make_board(SPEC, 0.03)
    rng = np.random.default_rng(3)
    out = []
    centre = np.array([0.12, 0.09, 0.0])
    for i in range(80):
        rv = Rotation.from_euler("xyz", rng.uniform(-35, 35, 3), degrees=True).as_rotvec()
        R = Rotation.from_rotvec(rv).as_matrix()
        z = rng.uniform(0.25, 0.5)
        target = np.array([rng.uniform(-1.1, 1.1) * z, rng.uniform(-0.6, 0.6) * z, z])
        t = target - R @ centre  # put the board centre at target
        d = detect(board, render(board, 0.03, rv, t))
        if d is not None:
            out.append({"file": f"{i:03d}", "obj": d[0], "img": d[1]})
    return out


def test_recovers_known_fisheye(detections):
    assert len(detections) >= 30
    result = calibrate(detections, SIZE)
    best = result["models"][result["recommended"]]
    # Over a ~100 deg field both models fit a mild fisheye equally well, so selection is by held-out
    # error, and accuracy is checked against the true K for whichever model wins.
    for model in ("fisheye_equidistant", "pinhole_radtan"):
        K = result["models"][model]["intrinsics"].K
        assert K[0, 0] == pytest.approx(300.0, rel=0.01) and K[1, 1] == pytest.approx(300.0, rel=0.01)
        assert K[0, 2] == pytest.approx(322.0, abs=1.5) and K[1, 2] == pytest.approx(178.0, abs=1.5)
    held = best["heldout"]
    assert held["rms_px"] < 0.45 and held["p95_px"] < 1.0
    assert held["border_corners"] >= 20 and held["border_rms_px"] < 0.6
    assert best["intrinsics"].status == "measured"
    assert result["heldout_views"] >= 3 and min(map(min, result["coverage_fit"])) > 0


def test_wrong_square_size_scales_only_translation(detections):
    # Intrinsics do not depend on the square size; extrinsics would. Guard against unit mix-ups.
    scaled = [{**d, "obj": d["obj"] * 1.05} for d in detections]
    a = calibrate(detections, SIZE)["models"]["fisheye_equidistant"]["intrinsics"].K
    b = calibrate(scaled, SIZE)["models"]["fisheye_equidistant"]["intrinsics"].K
    assert b == pytest.approx(a, rel=1e-3)


def test_duplicate_views_dropped_and_too_few_refused(detections):
    dup = [detections[0]] * 20
    assert len(select_views(dup)) == 1
    fit, held = split(list(range(8)))
    assert held == [3, 7] and len(fit) == 6
    with pytest.raises(ValueError, match="Too few distinct views"):
        calibrate(dup, SIZE)
