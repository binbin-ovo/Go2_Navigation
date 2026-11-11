"""Target-based camera -> base_link extrinsics from planar board poses (no visual hand-tuning).

Per pose i the camera sees the ChArUco board plane (unit normal ``n_c``, offset ``d_c``: n_c . p = d_c
for p in the camera optical frame, measured intrinsics) and the LiDAR sees the same physical plane in
base_link (``n_b``, ``d_b``, plus its inlier points). For ``T_base_camera = [R | t]``:
    n_b = R n_c          and          n_b . t = d_b - d_c.
Rotation comes from Kabsch on the normals, translation from the stacked offset equations, then both
are refined by minimising LiDAR point-to-camera-plane distances. Normals that do not span three
directions leave the problem underconstrained, which ``solve`` reports instead of hiding.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def oriented_plane(n, d):
    """Flip so the sensor origin lies on the negative side (d > 0): normals point away from it."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    return (n, float(d)) if d >= 0 else (-n, -float(d))


def camera_plane(R_cam_board, t_cam_board):
    """Board z axis in camera; the board origin lies on the plane."""
    n = R_cam_board[:, 2]
    return oriented_plane(n, n @ np.asarray(t_cam_board, float).ravel())


def fit_plane(points, iters=400, tol=0.02, seed=0):
    """RANSAC then SVD refit -> (n, d, inlier mask, residual std)."""
    p = np.asarray(points, float)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        s = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs(p @ n - n @ s[0]) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    c = p[best].mean(0)
    n = np.linalg.svd(p[best] - c)[2][2]
    n, d = oriented_plane(n, n @ c)
    inl = np.abs(p @ n - d) < tol
    return n, d, inl, float(np.std(p[inl] @ n - d))


def _initial(poses):
    Nc = np.array([p["n_c"] for p in poses])
    Nb = np.array([p["n_b"] for p in poses])
    U, _, Vt = np.linalg.svd(Nc.T @ Nb)
    S = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    b = np.array([p["d_b"] - p["d_c"] for p in poses])
    t = np.linalg.lstsq(Nb, b, rcond=None)[0]
    return R, t


def conditioning(poses):
    """Singular values of the stacked LiDAR normals: the smallest one measures how well translation
    along the least-observed direction is constrained (0 = unobservable)."""
    s = np.linalg.svd(np.array([p["n_b"] for p in poses]), compute_uv=False)
    return s.tolist()


def solve(poses, sigma_normal_deg=0.5, sigma_offset_m=0.005):
    """poses: dicts with n_c, d_c, n_b, d_b, points_b (N,3). Returns dict with T and diagnostics.

    Closed-form start (Kabsch + least squares), then a joint plane-to-plane refinement whose normal
    and offset residuals are scaled by their expected noise. LiDAR points enter only through the
    fitted plane: point-to-plane distances over a ~0.4 m patch barely constrain the board's tilt and
    let rotation trade against translation, so they are used for validation, not for fitting.
    """
    if len(poses) < 3:
        raise ValueError("Need at least 3 board poses")
    sv = conditioning(poses)
    if sv[-1] < 0.15:
        raise ValueError(f"Underconstrained: board normals span too few directions (singular values "
                         f"{np.round(sv, 3).tolist()}); add poses tilted about another axis")
    R0, t0 = _initial(poses)
    sn = np.radians(sigma_normal_deg)

    def residuals(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        t = x[3:]
        out = []
        for p in poses:
            n = R @ p["n_c"]
            out.extend((n - p["n_b"]) / sn)
            out.append((p["n_b"] @ t - (p["d_b"] - p["d_c"])) / sigma_offset_m)
        return np.asarray(out)

    x0 = np.r_[Rotation.from_matrix(R0).as_rotvec(), t0]
    sol = least_squares(residuals, x0)
    R = Rotation.from_rotvec(sol.x[:3]).as_matrix()
    T = _matrix(R, sol.x[3:])
    r = residuals(sol.x)
    chi2 = float(r @ r / max(len(r) - 6, 1))  # >1 means the assumed sigmas are optimistic
    try:
        std = np.sqrt(np.diag(np.linalg.inv(sol.jac.T @ sol.jac)) * max(chi2, 1.0))
    except np.linalg.LinAlgError:
        std = np.full(6, np.nan)
    per_pose = [plane_errors(T, p) for p in poses]
    return {"T": T, "init_T": _matrix(R0, t0), "singular_values": sv, "reduced_chi2": chi2,
            "residual_abs_median_m": float(np.median([e["abs_median_m"] for e in per_pose])),
            "per_pose": per_pose,
            "std_rotvec_deg": np.degrees(std[:3]).tolist(), "std_t_m": std[3:].tolist()}


def _matrix(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def plane_errors(T, pose):
    """How well T predicts one pose: LiDAR points against the camera plane mapped into base_link."""
    R, t = T[:3, :3], T[:3, 3]
    n = R @ pose["n_c"]
    d = pose["d_c"] + n @ t
    dist = np.asarray(pose["points_b"]) @ n - d
    angle = np.degrees(np.arccos(np.clip(n @ pose["n_b"], -1, 1)))
    return {"abs_median_m": float(np.median(np.abs(dist))), "mean_m": float(np.mean(dist)),
            "normal_angle_deg": float(angle), "points": int(len(dist))}


def leave_one_out(poses):
    out = []
    for i in range(len(poses)):
        rest = poses[:i] + poses[i + 1:]
        try:
            T = solve(rest)["T"]
        except ValueError as e:
            out.append({"pose": i, "error": str(e)})
            continue
        out.append({"pose": i, **plane_errors(T, poses[i])})
    return out
