"""ChArUco intrinsic calibration with held-out validation (no values are ever assumed).

Detect corners per frame, drop near-duplicate views, split into fit / held-out sets by alternating
distinct views, fit both supported models, and score each on held-out views by re-estimating only the
board pose with the fitted intrinsics. The model with lower held-out RMS is recommended.
"""

import cv2
import numpy as np

from .calibration import CameraIntrinsics

MIN_CORNERS = 12


def make_board(spec, square_m):
    """Board from a target spec; the marker keeps its nominal ratio to the measured square."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
    if spec.get("square_size_m_nominal"):
        ratio = spec["marker_size_m_nominal"] / spec["square_size_m_nominal"]
    else:
        ratio = spec["marker_to_square_ratio"]
    board = cv2.aruco.CharucoBoard((spec["squares_x"], spec["squares_y"]), square_m,
                                   square_m * ratio, dictionary)
    board.setLegacyPattern(bool(spec.get("legacy_pattern", False)))
    return board


def detect(board, gray):
    """Return (object points (N,3) m, image points (N,2) px, ids) or None."""
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or len(ids) < MIN_CORNERS:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    return obj.reshape(-1, 3).astype(np.float64), img.reshape(-1, 2).astype(np.float64), ids.ravel()


def select_views(detections, min_shift_px=25.0, min_scale_change=0.08):
    """Keep a detection only if its corners moved or rescaled enough since the last kept view."""
    kept, last = [], None
    for d in detections:
        img = d["img"]
        c, s = img.mean(0), np.sqrt(np.prod(img.max(0) - img.min(0)))
        if last is None or np.linalg.norm(c - last[0]) > min_shift_px or abs(s / last[1] - 1) > min_scale_change:
            kept.append(d)
            last = (c, s)
    return kept


def split(views, holdout_every=4):
    fit = [v for i, v in enumerate(views) if i % holdout_every != holdout_every - 1]
    held = [v for i, v in enumerate(views) if i % holdout_every == holdout_every - 1]
    return fit, held


def fit_model(model, views, size):
    obj = [v["obj"].astype(np.float32) for v in views]
    img = [v["img"].astype(np.float32) for v in views]
    if model == "pinhole_radtan":
        rms, K, D, _, _ = cv2.calibrateCamera(obj, img, size, None, None)
        return float(rms), K, D.ravel()[:5]
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-9)
    rms, K, D, _, _ = cv2.fisheye.calibrate([o.reshape(-1, 1, 3).astype(np.float64) for o in obj],
                                            [i.reshape(-1, 1, 2).astype(np.float64) for i in img],
                                            size, None, None, flags=flags, criteria=crit)
    return float(rms), K, D.ravel()


def _normalize(intr, img):
    pts = img.reshape(-1, 1, 2)
    if intr.model == "fisheye_equidistant":
        return cv2.fisheye.undistortPoints(pts, intr.K, intr.D).reshape(-1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)
    return cv2.undistortPointsIter(pts, intr.K, intr.D, None, None, crit).reshape(-1, 2)


def heldout_residuals(intr, views):
    """Per-corner pixel residuals after fitting only the board pose with fixed intrinsics."""
    out = []
    for v in views:
        norm = _normalize(intr, v["img"])
        ok, rvec, tvec = cv2.solvePnP(v["obj"], norm, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        ok, rvec, tvec = cv2.solvePnP(v["obj"], norm, np.eye(3), None, rvec, tvec, True,
                                      cv2.SOLVEPNP_ITERATIVE)
        if intr.model == "fisheye_equidistant":
            proj, _ = cv2.fisheye.projectPoints(v["obj"].reshape(-1, 1, 3), rvec, tvec, intr.K, intr.D)
        else:
            proj, _ = cv2.projectPoints(v["obj"], rvec, tvec, intr.K, intr.D)
        out.append((v["img"], proj.reshape(-1, 2) - v["img"]))
    return out


def summarize(intr, residuals, size, border_frac=0.6):
    if not residuals:
        return {"views": 0}
    pts = np.concatenate([p for p, _ in residuals])
    err = np.linalg.norm(np.concatenate([r for _, r in residuals]), axis=1)
    half_diag = np.hypot(*size) / 2
    r = np.linalg.norm(pts - intr.K[:2, 2], axis=1) / half_diag
    border = r > border_frac
    return {
        "views": len(residuals), "corners": int(len(err)),
        "rms_px": float(np.sqrt(np.mean(err ** 2))), "p95_px": float(np.percentile(err, 95)),
        "max_px": float(err.max()),
        "centre_rms_px": float(np.sqrt(np.mean(err[~border] ** 2))) if (~border).any() else None,
        "border_rms_px": float(np.sqrt(np.mean(err[border] ** 2))) if border.any() else None,
        "border_corners": int(border.sum()),
    }


def coverage(views, size, cells=(4, 3)):
    """Corner counts in a coarse image grid (columns x rows), to show unobserved regions."""
    grid = np.zeros(cells[::-1], int)
    for v in views:
        ij = np.floor(v["img"] / np.array(size) * np.array(cells)).astype(int)
        ij = np.clip(ij, 0, np.array(cells) - 1)
        np.add.at(grid, (ij[:, 1], ij[:, 0]), 1)
    return grid.tolist()


def calibrate(detections, size, models=("fisheye_equidistant", "pinhole_radtan"), holdout_every=4):
    views = select_views(detections)
    fit, held = split(views, holdout_every)
    if len(fit) < 10 or len(held) < 3:
        raise ValueError(f"Too few distinct views: {len(fit)} fit / {len(held)} held-out "
                         "(need >= 10 / 3); capture more poses")
    results = {}
    for model in models:
        try:
            rms, K, D = fit_model(model, fit, size)
            intr = CameraIntrinsics(size[0], size[1], K, model, D, "measured", reprojection_rms_px=rms)
        except (cv2.error, ValueError) as e:
            results[model] = {"error": str(e)}
            continue
        results[model] = {"intrinsics": intr, "fit_rms_px": rms,
                          "heldout": summarize(intr, heldout_residuals(intr, held), size)}
    scored = {m: r for m, r in results.items() if "heldout" in r and r["heldout"].get("views")}
    best = min(scored, key=lambda m: scored[m]["heldout"]["rms_px"]) if scored else None
    return {"views_detected": len(detections), "views_distinct": len(views), "fit_views": len(fit),
            "heldout_views": len(held), "coverage_fit": coverage(fit, size),
            "coverage_heldout": coverage(held, size), "models": results, "recommended": best}


def monotonic_to_corner(intr):
    """True if the distortion keeps image radius increasing with ray angle out to the image corner."""
    corner = np.hypot(max(intr.K[0, 2], intr.width - intr.K[0, 2]), max(intr.K[1, 2], intr.height - intr.K[1, 2]))
    th = np.linspace(0, np.radians(89), 4000)
    if intr.model == "fisheye_equidistant":
        k = intr.D
        r = intr.K[0, 0] * th * (1 + k[0] * th**2 + k[1] * th**4 + k[2] * th**6 + k[3] * th**8)
    else:
        k1, k2, _, _, k3 = intr.D
        x = np.tan(th)
        r = intr.K[0, 0] * x * (1 + k1 * x**2 + k2 * x**4 + k3 * x**6)
    increasing = np.diff(r) > 0
    reach = r[np.argmin(increasing)] if not increasing.all() else r[-1]
    return bool(reach >= corner), float(reach), float(corner)


def _rays(intr, pts):
    pts = pts.reshape(-1, 1, 2).astype(np.float64)
    if intr.model == "fisheye_equidistant":
        n = cv2.fisheye.undistortPoints(pts, intr.K, intr.D)
    else:
        n = cv2.undistortPointsIter(pts, intr.K, intr.D, None, None,
                                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10))
    v = np.c_[n.reshape(-1, 2), np.ones(len(pts))]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def ray_angle_difference(a, b, grid=(33, 19)):
    """Per-pixel ray angle (deg) between two intrinsics after removing their best common rotation,
    which board-pose estimation cannot distinguish from a principal-point shift."""
    u, v = np.meshgrid(np.linspace(0, a.width - 1, grid[0]), np.linspace(0, a.height - 1, grid[1]))
    pts = np.c_[u.ravel(), v.ravel()]
    ra, rb = _rays(a, pts), _rays(b, pts)
    U, _, Vt = np.linalg.svd(rb.T @ ra)
    R = (U @ Vt).T
    ang = np.degrees(np.arccos(np.clip(((rb @ R.T) * ra).sum(1), -1, 1)))
    return {"median_deg": float(np.median(ang)), "p95_deg": float(np.percentile(ang, 95)),
            "max_deg": float(ang.max())}


def bootstrap(views, model, size, reference, n=20, seed=0):
    rng = np.random.default_rng(seed)
    params, angles = [], []
    for _ in range(n):
        sample = [views[j] for j in rng.integers(0, len(views), len(views))]
        _, K, D = fit_model(model, sample, size)
        intr = CameraIntrinsics(size[0], size[1], K, model, D, "measured")
        params.append([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D])
        angles.append(ray_angle_difference(reference, intr))
    names = ["fx", "fy", "cx", "cy"] + [f"d{i}" for i in range(len(params[0]) - 4)]
    return {"n": n, "seed": seed, "std": dict(zip(names, np.std(params, axis=0).tolist())),
            "ray_angle_median_deg": float(np.median([a["median_deg"] for a in angles])),
            "ray_angle_p95_deg": float(np.median([a["p95_deg"] for a in angles]))}


def leave_one_group_out(groups, model, size, reference):
    """Fit on all groups but one (e.g. recordings), score the whole left-out group."""
    out = {}
    for name in groups:
        fit = [v for g, vs in groups.items() if g != name for v in vs]
        _, K, D = fit_model(model, fit, size)
        intr = CameraIntrinsics(size[0], size[1], K, model, D, "measured")
        out[name] = {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                     "heldout": summarize(intr, heldout_residuals(intr, groups[name]), size),
                     "ray_angle_vs_all": ray_angle_difference(reference, intr)}
    return out
