"""Plane-based camera -> base_link extrinsics recover a known transform and refuse degenerate data."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from go2_perception.vln.extrinsics import camera_plane, fit_plane, leave_one_out, plane_errors, solve

T_TRUE = np.eye(4)
T_TRUE[:3, :3] = Rotation.from_euler("xyz", [-98.0, 1.5, -91.0], degrees=True).as_matrix()
T_TRUE[:3, 3] = [0.29, 0.01, 0.05]


def make_pose(rng, yaw, pitch, dist, lateral, noise_m=0.01, n_pts=800):
    # Board in base_link: centre ahead, facing the robot, yawed and leaned back.
    R_b_board = Rotation.from_euler("zy", [yaw, pitch], degrees=True).as_matrix() @ \
        np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # board x=-y_b, y=-z_b, z=+x_b
    c_b = np.array([dist, lateral, 0.0])
    u, v = rng.uniform(-0.2, 0.2, n_pts), rng.uniform(-0.14, 0.14, n_pts)
    pts = c_b + np.c_[u, v, np.zeros(n_pts)] @ R_b_board.T
    pts += rng.normal(0, noise_m, (n_pts, 1)) * R_b_board[:, 2]
    T_cb = np.linalg.inv(T_TRUE)
    R_c_board = T_cb[:3, :3] @ R_b_board
    t_c_board = T_cb[:3, :3] @ c_b + T_cb[:3, 3]
    n_c, d_c = camera_plane(R_c_board, t_c_board)
    n_b, d_b, inl, _ = fit_plane(pts)
    return {"n_c": n_c, "d_c": d_c, "n_b": n_b, "d_b": d_b, "points_b": pts[inl]}


def poses(rng, specs):
    return [make_pose(rng, *s) for s in specs]


GOOD = [(0, 0, 1.2, 0), (30, 0, 1.0, 0.4), (-30, 0, 1.0, -0.4), (0, -25, 1.5, 0), (20, -20, 1.8, 0.3),
        (-20, -20, 1.8, -0.3), (0, 0, 2.0, 0.5), (-35, 0, 1.4, -0.2), (35, -15, 0.9, 0.2), (0, 20, 1.1, 0)]


def errors(T):
    rot = np.degrees(Rotation.from_matrix(T[:3, :3].T @ T_TRUE[:3, :3]).magnitude())
    return rot, np.linalg.norm(T[:3, 3] - T_TRUE[:3, 3])


def test_recovers_known_transform():
    rng = np.random.default_rng(0)
    P = poses(rng, GOOD)
    r = solve(P)
    rot, trans = errors(r["T"])
    assert rot < 0.3 and trans < 0.01
    assert r["residual_abs_median_m"] < 0.01
    loo = leave_one_out(P)
    # 10 mm LiDAR noise on a 0.4 m board limits each held-out plane to ~0.5 deg / ~7 mm.
    assert all(x["abs_median_m"] < 0.012 and abs(x["mean_m"]) < 0.006 and x["normal_angle_deg"] < 1.5
               for x in loo)


def test_camera_noise_in_planes_degrades_gracefully():
    rng = np.random.default_rng(1)
    P = poses(rng, GOOD)
    for p in P:  # 0.5 deg normal and 5 mm offset error in the camera plane
        p["n_c"] = Rotation.from_rotvec(rng.normal(0, np.radians(0.5), 3)).apply(p["n_c"])
        p["d_c"] += rng.normal(0, 0.005)
    r = solve(P)
    rot, trans = errors(r["T"])
    assert rot < 1.0 and trans < 0.02
    assert max(r["std_rotvec_deg"]) < 0.5  # reported uncertainty matches the injected noise


def test_parallel_boards_are_underconstrained():
    rng = np.random.default_rng(2)
    P = poses(rng, [(0, 0, d, y) for d, y in [(1.0, 0), (1.5, 0.3), (2.0, -0.3), (1.2, 0.5)]])
    with pytest.raises(ValueError, match="Underconstrained"):
        solve(P)


def test_yaw_only_cannot_fix_height():
    rng = np.random.default_rng(3)
    P = poses(rng, [(a, 0, 1.3, 0) for a in (-35, -15, 0, 15, 35)])
    with pytest.raises(ValueError, match="Underconstrained"):
        solve(P)


def test_fit_plane_never_builds_full_svd(monkeypatch):
    # A full SVD of N points allocates an N x N matrix (60 k floor points -> ~29 GB, seen on 2026-11-12).
    real = np.linalg.svd

    def guarded(a, full_matrices=True, **kw):
        assert not (full_matrices and min(np.shape(a)) == 3 and max(np.shape(a)) > 3), "full N x N SVD"
        return real(a, full_matrices=full_matrices, **kw)

    monkeypatch.setattr(np.linalg, "svd", guarded)
    rng = np.random.default_rng(5)
    pts = np.c_[rng.uniform(-1, 1, (5000, 2)), rng.normal(0, 0.005, 5000)]
    n, d, inl, _ = fit_plane(pts)
    assert abs(abs(n[2]) - 1) < 1e-3 and inl.mean() > 0.95


def test_plane_errors_detects_wrong_transform():
    rng = np.random.default_rng(4)
    P = poses(rng, GOOD)
    wrong = T_TRUE.copy()
    wrong[2, 3] += 0.05
    assert plane_errors(wrong, P[3])["abs_median_m"] > 0.015
    assert plane_errors(T_TRUE, P[3])["abs_median_m"] < 0.01
