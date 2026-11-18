"""Camera optical-axis heading in base_link from existing static plane recordings (read-only, offline).

Only the yaw of the optical axis is estimated. Each static capture holds the same physical plane twice,
as the ChArUco board normal in the camera (measured intrinsics, PnP) and as a LiDAR plane in base_link
(`cloud_base`, accumulated over the stationary hold). For a near-vertical plane facing the robot, the
azimuth of its normal is insensitive to small camera pitch/roll, so per capture
    yaw = azimuth(n_lidar) - azimuth(R_seed n_camera).
A full 3-D rotation fit is reported only as a diagnostic: these captures' normals nearly all face the
camera, so pitch/roll are ill-conditioned. Captures whose camera and LiDAR plane elevations disagree by
more than MAX_ELEVATION_GAP_DEG, or whose LiDAR fit is not near-vertical, are rejected as segmentation
failures, and only an unambiguous large plane (MIN_PLANE_POINTS) is accepted. Translation is not estimated and nothing here is a T_base_camera
(docs/GO2_LIDAR_CAMERA_EXTRINSICS.md).

The coarse seed rotation (optical z = base x, optical x = -base y) is used only to crop LiDAR points near
the camera-predicted board; it never enters the estimate except through that crop.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from go2_perception.vln.calibration import load_calibration  # noqa: E402
from go2_perception.vln.intrinsics import make_board  # noqa: E402
from go2_perception.vln.intrinsics import detect  # noqa: E402
from phase5d_solve_extrinsics import T_SEED, jsonl  # noqa: E402

R_SEED = T_SEED[:3, :3]
FLOOR_CUT_Z = -0.27      # base_link; floor returns lie at about -0.33 to -0.42 near the robot
MAX_ELEVATION_GAP_DEG = 5.0
MAX_ELEVATION_DEG = 15.0
MIN_PLANE_POINTS = 4000     # an unambiguous large plane (wall); the A4 board slab (1-2 k points, 20-26 mm
                            # sigma) merges with its stand/background, the Phase 5D segmentation failure
SF_DATALESS = 0x40000000  # macOS: file content offloaded to iCloud; reading it blocks until restored


def local(path):
    return path.exists() and not (path.stat().st_flags & SF_DATALESS)


def board_poses(root, intr, board):
    """phase5d_solve_extrinsics.board_poses restricted to frames present on disk (static captures)."""
    out = []
    for row in jsonl(root / "rgb" / "index.jsonl"):
        path = root / "rgb" / row["file"]
        if not local(path):
            continue
        d = detect(board, cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
        if d is None:
            continue
        norm = cv2.fisheye.undistortPoints(d[1].reshape(-1, 1, 2), intr.K, intr.D).reshape(-1, 2)
        ok, rv, tv = cv2.solvePnP(d[0], norm, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
        ok, rv, tv = cv2.solvePnP(d[0], norm, np.eye(3), None, rv, tv, True, cv2.SOLVEPNP_ITERATIVE)
        if ok:
            out.append({"R": cv2.Rodrigues(rv)[0], "t": tv.ravel(), "obj": d[0]})
    return out


def load_scans(root):
    base = root / "lidar" / "utlidar_cloud_base"
    return [np.load(base / r["file"])[:, :3].astype(float) for r in jsonl(base / "index.jsonl")
            if local(base / r["file"])]


def tukey_plane(points, c=0.05, iters=10):
    """Weighted PCA plane, Tukey biweight on point-to-plane residuals (robust to slab thickness)."""
    w = np.ones(len(points))
    for _ in range(iters):
        mu = (w[:, None] * points).sum(0) / w.sum()
        cov = ((points - mu) * w[:, None]).T @ (points - mu)
        n = np.linalg.eigh(cov)[1][:, 0]
        r = (points - mu) @ n
        w = np.where(np.abs(r) < c, (1 - (r / c) ** 2) ** 2, 0.0)
        if w.sum() < 50:
            return None
    keep = w > 0
    return n, mu, float(np.std(r[keep])), int(keep.sum())


def orient(n, toward_origin_point):
    """Normal pointing from the plane towards the sensor."""
    return n if n @ (-toward_origin_point) > 0 else -n


def capture(rec, intr, board, radius, band):
    poses = board_poses(rec, intr, board)
    if len(poses) < 5:
        return None, f"{len(poses)} board detections"
    Rs = np.stack([p["R"] for p in poses])
    ts = np.stack([p["t"] for p in poses])
    n_c = np.median(Rs[:, :, 2], axis=0)
    n_c /= np.linalg.norm(n_c)
    t_c = np.median(ts, axis=0)
    spread = np.degrees(np.median(np.arccos(np.clip(Rs[:, :, 2] @ n_c, -1, 1))))
    # Board centre: the PnP origin is a grid corner; use the mean of the detected corners in camera frame.
    obj = np.concatenate([p["obj"].reshape(-1, 3) for p in poses[:1]])
    centre_c = poses[0]["R"] @ obj.mean(0) + poses[0]["t"]
    n_c = orient(n_c, centre_c)

    scans = load_scans(rec)
    if not scans:
        return None, "no local LiDAR scans"
    pts = np.concatenate(scans)
    centre_b = R_SEED @ centre_c + T_SEED[:3, 3]
    n_seed = R_SEED @ n_c
    near = (np.linalg.norm(pts - centre_b, axis=1) < radius) & (pts[:, 2] > FLOOR_CUT_Z)
    near &= np.abs((pts - centre_b) @ n_seed) < band
    fit = tukey_plane(pts[near])
    if fit is None:
        return None, "no LiDAR plane"
    n_b, mu, sigma, inliers = fit
    n_b = orient(n_b, mu)
    return {"capture": rec.name, "frames": len(poses), "scans": len(scans), "n_camera": n_c.tolist(), "n_base": n_b.tolist(),
            "board_dist_m": float(np.linalg.norm(centre_c)), "camera_normal_spread_deg": float(spread),
            "lidar_sigma_m": sigma, "lidar_inliers": inliers,
            # Camera-predicted board centre vs LiDAR plane: a background surface sits behind the board.
            "plane_offset_m": float((mu - centre_b) @ n_b), "seed_angle_deg":
            float(np.degrees(np.arccos(np.clip(n_seed @ n_b, -1, 1))))}, None


def kabsch(nc, nb):
    H = np.asarray(nc).T @ np.asarray(nb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T, S


def azimuth(n):
    return math.degrees(math.atan2(n[1], n[0]))


def elevation(n):
    return math.degrees(math.asin(np.clip(n[2], -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="+", type=Path)
    ap.add_argument("--calibration", type=Path, default=ROOT / "configs/go2_front_camera_calibration.yaml")
    ap.add_argument("--board", type=Path, required=True)
    ap.add_argument("--square-mm", type=float, default=29.0)
    ap.add_argument("--radius", type=float, default=0.25)
    ap.add_argument("--band", type=float, default=0.15)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    intr, _ = load_calibration(a.calibration)
    board = make_board(json.loads(a.board.read_text()), a.square_mm / 1000.0)
    rows = []
    for rec in a.recordings:
        row, why = capture(rec, intr, board, a.radius * (2.0 if "wall" in rec.name else 1.0), a.band)
        if row is None:
            print(f"{rec.name}: skipped ({why})")
            continue
        nc, nb = R_SEED @ np.array(row["n_camera"]), np.array(row["n_base"])
        row.update(yaw_deg=(azimuth(nb) - azimuth(nc) + 180) % 360 - 180,
                   elevation_camera_deg=elevation(nc), elevation_lidar_deg=elevation(nb))
        gap = abs(row["elevation_camera_deg"] - row["elevation_lidar_deg"])
        row["accepted"] = bool(gap <= MAX_ELEVATION_GAP_DEG and abs(row["elevation_lidar_deg"]) <= MAX_ELEVATION_DEG
                               and row["lidar_inliers"] >= MIN_PLANE_POINTS)
        rows.append(row)
        print(f"{rec.name}: frames {row['frames']} scans {row['scans']} dist {row['board_dist_m']:.2f} m, "
              f"lidar sigma {1000 * row['lidar_sigma_m']:.0f} mm ({row['lidar_inliers']} pts), yaw "
              f"{row['yaw_deg']:+.1f} deg, elevation cam/lidar {row['elevation_camera_deg']:+.1f}/"
              f"{row['elevation_lidar_deg']:+.1f}, plane offset {100 * row['plane_offset_m']:+.0f} cm -> {'accepted' if row['accepted'] else 'rejected'}")
    accepted = [r["yaw_deg"] for r in rows if r["accepted"]]
    R, S = kabsch([r["n_camera"] for r in rows], [r["n_base"] for r in rows])
    print(f"accepted yaw: {np.round(accepted, 2).tolist()}")
    print(f"diagnostic only, 3-D fit over all captures: normal singular values {np.round(S, 2).tolist()}")
    if a.json:
        a.json.write_text(json.dumps({"captures": rows, "accepted_yaw_deg": accepted,
                                      "diagnostic_singular_values": S.tolist(),
                                      "criteria": {"max_elevation_gap_deg": MAX_ELEVATION_GAP_DEG,
                                                   "max_elevation_deg": MAX_ELEVATION_DEG,
                                                   "min_plane_points": MIN_PLANE_POINTS}}, indent=1))


if __name__ == "__main__":
    main()
