"""Solve T_base_camera (camera optical -> base_link) from a static multi-pose ChArUco board capture.

Offline. Inputs: a capture where a board was placed in several static poses with nobody in view,
a background capture of the same scene without the board, measured intrinsics and the measured square
size. Each still period (board pose stable in the camera) gives one plane pair: the camera plane from
the ChArUco pose, the LiDAR plane from background-subtracted cloud_base points. The solve and its
checks live in go2_perception.vln.extrinsics. Saves the calibration only with leave-one-pose-out
validation, and never alters intrinsics.

Usage:
  PYTHONPATH=src python scripts/phase5d_solve_extrinsics.py --capture data/recordings/p5d_extr_main \
      --background data/recordings/p5d_extr_background --intrinsics configs/go2_front_camera_calibration.yaml \
      --board data/outputs/phase5d/targets/charuco_a4_9x6_30mm.json --square-mm 29 \
      --output configs/go2_front_camera_calibration.yaml --report data/outputs/phase5d/extrinsics.json
"""

import argparse
import datetime
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from go2_perception.vln.calibration import RigidTransform, load_calibration, save_calibration
from go2_perception.vln.extrinsics import camera_plane, fit_plane, leave_one_out, plane_errors, solve
from go2_perception.vln.intrinsics import detect, make_board

VOXEL = 0.05


def jsonl(path):
    return [json.loads(line) for line in open(path)]


def board_poses(root, intr, board):
    out = []
    for row in jsonl(root / "rgb" / "index.jsonl"):
        gray = cv2.imread(str(root / "rgb" / row["file"]), cv2.IMREAD_GRAYSCALE)
        d = detect(board, gray)
        if d is None:
            continue
        norm = cv2.fisheye.undistortPoints(d[1].reshape(-1, 1, 2), intr.K, intr.D).reshape(-1, 2)
        ok, rv, tv = cv2.solvePnP(d[0], norm, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
        ok, rv, tv = cv2.solvePnP(d[0], norm, np.eye(3), None, rv, tv, True, cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        out.append({"t_host": row["host_recv_first"], "file": row["file"], "R": cv2.Rodrigues(rv)[0],
                    "t": tv.ravel(), "corners": len(d[2]), "img": d[1], "obj": d[0]})
    return out


def still_segments(poses, max_move_m=0.010, max_rot_deg=2.5, min_len_s=3.0, max_gap_s=0.5):
    # Thresholds sit above single-frame PnP noise on a static board (p5d_extr_test: 3.5 mm / 1.4 deg p95).
    # Segment poses are averaged afterwards.
    segs, cur = [], []
    for p in poses:
        if cur:
            ref = cur[0]
            moved = np.linalg.norm(p["t"] - ref["t"]) > max_move_m
            rot = np.degrees(np.arccos(np.clip((np.trace(ref["R"].T @ p["R"]) - 1) / 2, -1, 1))) > max_rot_deg
            if moved or rot or p["t_host"] - cur[-1]["t_host"] > max_gap_s:
                if cur[-1]["t_host"] - cur[0]["t_host"] >= min_len_s:
                    segs.append(cur)
                cur = []
        cur.append(p)
    if cur and cur[-1]["t_host"] - cur[0]["t_host"] >= min_len_s:
        segs.append(cur)
    return segs


def voxel_keys(points):
    return set(map(tuple, np.floor(points / VOXEL).astype(int)))


def background_set(root):
    pts = np.concatenate([np.load(root / "lidar" / "utlidar_cloud_base" / r["file"])[:, :3]
                          for r in jsonl(root / "lidar" / "utlidar_cloud_base" / "index.jsonl")])
    keys = voxel_keys(pts)
    offs = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    return {(a + i, b + j, c + k) for a, b, c in keys for i, j, k in offs}


def load_scans(root):
    """All cloud_base scans with their midpoint time (robot clock), loaded once."""
    out = []
    for r in jsonl(root / "lidar" / "utlidar_cloud_base" / "index.jsonl"):
        base = root / "lidar" / "utlidar_cloud_base"
        mid = r["stamp"] + 0.5 * float(np.load(base / r["file"].replace(".npy", "_time.npy")).max())
        out.append((mid, np.load(base / r["file"])[:, :3].astype(float)))
    return out


def lidar_window(scans, t0, t1):
    parts = [p for mid, p in scans if t0 <= mid <= t1]
    return np.concatenate(parts) if parts else np.empty((0, 3)), len(parts)


# Coarse camera -> base_link guess used ONLY to crop LiDAR points for the first pass (optical z forward
# = base x, optical x right = -base y, optical y down = -base z; camera ~0.3 m ahead). Never reported.
T_SEED = np.array([[0, 0, 1, 0.30], [-1, 0, 0, 0.0], [0, -1, 0, 0.05], [0, 0, 0, 1.0]])


def pick_grid_points(fg, R_cb, t_cb, grid_wh, T, band, margin=0.0):
    """Foreground points inside the printed grid rectangle (board coordinates) and within ``band`` of the
    board plane, using camera board pose (R_cb, t_cb) and the current T. Hands holding the board edges
    and objects next to or below it fall outside the grid and are dropped."""
    R, t = T[:3, :3], T[:3, 3]
    cam = (fg - t) @ R                       # base -> camera
    brd = (cam - t_cb) @ R_cb                # camera -> board (x right, y down, z into board)
    keep = (brd[:, 0] > -margin) & (brd[:, 0] < grid_wh[0] + margin) & \
           (brd[:, 1] > -margin) & (brd[:, 1] < grid_wh[1] + margin) & (np.abs(brd[:, 2]) < band)
    return fg[keep]


def pick_board_points(fg, n_c, d_c, centre_c, T, radius, band):
    """Foreground points near the board as predicted by the camera plane mapped through T: within
    ``radius`` of the grid centre and ``band`` of the plane. Excludes a person holding the board from
    behind, whose body lies off the plane."""
    R, t = T[:3, :3], T[:3, 3]
    centre_b = R @ centre_c + t
    n_b = R @ n_c
    d_b = d_c + n_b @ t
    near = np.linalg.norm(fg - centre_b, axis=1) < radius
    on = np.abs(fg @ n_b - d_b) < band
    return fg[near & on]


def collect(args, intr, board, T=None, margin=0.0, band=0.05):
    root = args.capture
    poses = board_poses(root, intr, board) if not hasattr(args, "_poses") else args._poses
    args._poses = poses
    segs = still_segments(poses, args.still_move_m, args.still_rot_deg, args.still_min_s)
    odom = jsonl(root / "state" / "utlidar_robot_odom.jsonl")
    offset = min(r["host_recv"] - r["stamp"] for r in odom)
    latency = yaml.safe_load(Path("configs/go2_vln_sensors.yaml").read_text())["timing"]["camera_latency_s"]["value"]
    scans = args._scans if hasattr(args, "_scans") else load_scans(root)
    args._scans = scans
    bg = args._bg if hasattr(args, "_bg") else background_set(args.background)
    args._bg = bg
    grid_centre_board = np.r_[board.getChessboardSize()] * board.getSquareLength() / 2
    out = []
    for i, seg in enumerate(segs):
        margin = min(0.5, 0.2 * (seg[-1]["t_host"] - seg[0]["t_host"]))
        h0, h1 = seg[0]["t_host"] + margin, seg[-1]["t_host"] - margin
        r0, r1 = h0 - offset - latency, h1 - offset - latency
        pts, nscans = lidar_window(scans, r0, r1)
        fg = pts[[k not in bg for k in map(tuple, np.floor(pts / VOXEL).astype(int))]] if len(pts) else pts
        R = np.mean([p["R"] for p in seg], axis=0)
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        t = np.median([p["t"] for p in seg], axis=0)
        n_c, d_c = camera_plane(R, t)
        centre_c = R @ np.r_[grid_centre_board, 0.0] + t
        grid_wh = np.r_[board.getChessboardSize()] * board.getSquareLength()
        if T is None:
            cand = pick_board_points(fg, n_c, d_c, centre_c, T_SEED, radius=0.6, band=0.25)
        else:
            cand = pick_grid_points(fg, R, t, grid_wh, T, band=band, margin=margin)
        entry = {"segment": i, "host_start": seg[0]["t_host"], "duration_s": seg[-1]["t_host"] - seg[0]["t_host"],
                 "frames": len(seg), "scans": nscans, "foreground": int(len(fg)), "candidates": int(len(cand)),
                 "camera_distance_m": float(np.linalg.norm(t)), "n_c": n_c, "d_c": d_c,
                 "camera_t_std_mm": (np.std([p["t"] for p in seg], axis=0) * 1000).tolist(),
                 "frame": seg[len(seg) // 2]["file"], "centre_c": centre_c}
        if len(cand) < 40:
            entry["rejected"] = f"only {len(cand)} LiDAR board candidates"
            out.append(entry)
            continue
        n_b, d_b, inl, res = fit_plane(cand)
        if inl.sum() < 40:
            entry["rejected"] = f"only {int(inl.sum())} plane inliers"
            out.append(entry)
            continue
        entry.update(n_b=n_b, d_b=d_b, points_b=cand[inl], lidar_inliers=int(inl.sum()),
                     lidar_plane_std_mm=res * 1000, lidar_extent_m=np.ptp(cand[inl], axis=0).tolist())
        out.append(entry)
    return out, offset, latency


def overlay(root, entry, intr, T, path):
    from go2_perception.vln.calibration import project_points

    img = cv2.imread(str(root / "rgb" / entry["frame"]))
    cam = (np.asarray(entry["points_b"]) - T[:3, 3]) @ T[:3, :3]
    uv, ok = project_points(cam, intr)
    for (u, v), z in zip(uv[ok], cam[ok, 2]):
        cv2.circle(img, (int(round(u)), int(round(v))), 3, (0, 0, 255), -1)
    cv2.putText(img, f"segment {entry['segment']}: LiDAR board inliers projected ({ok.sum()})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--background", type=Path, required=True)
    ap.add_argument("--intrinsics", type=Path, required=True)
    ap.add_argument("--board", type=Path, required=True)
    ap.add_argument("--square-mm", type=float, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--overlays", type=Path)
    ap.add_argument("--still-move-m", type=float, default=0.010)
    ap.add_argument("--still-rot-deg", type=float, default=2.5)
    ap.add_argument("--still-min-s", type=float, default=3.0)
    ap.add_argument("--max-loo-m", type=float, default=0.015, help="acceptance: held-out plane distance")
    ap.add_argument("--max-loo-deg", type=float, default=2.0, help="acceptance: held-out normal angle")
    a = ap.parse_args()
    intr, _ = load_calibration(a.intrinsics)
    if intr.status != "measured" or intr.model != "fisheye_equidistant":
        raise SystemExit("Needs measured fisheye intrinsics")
    board = make_board(json.loads(a.board.read_text()), a.square_mm / 1000)

    entries, offset, latency = collect(a, intr, board)
    good = [e for e in entries if "n_b" in e]
    first = solve(good)
    # Re-select LiDAR points around the board predicted by the previous solution until stable.
    # Coarse-to-fine: a generous grid margin and plane band first, then the exact printed grid.
    T = first["T"]
    schedule = [(0.10, 0.15), (0.06, 0.10), (0.03, 0.06), (0.0, 0.04), (0.0, 0.04), (0.0, 0.04)]
    for margin, band in schedule:
        entries, _, _ = collect(a, intr, board, T, margin, band)
        good = [e for e in entries if "n_b" in e]
        result = solve(good)
        print(f"margin {margin:.2f} band {band:.2f}: poses {len(good)}, "
              f"t {np.round(result['T'][:3, 3], 3).tolist()}, "
              f"median plane err {result['residual_abs_median_m'] * 1000:.1f} mm")
        T = result["T"]
    loo = leave_one_out(good)

    def clean(e):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in e.items() if k != "points_b"}

    pos = lambda root: np.mean([r["position"] for r in jsonl(root / "state" / "utlidar_robot_odom.jsonl")], axis=0)  # noqa: E731
    robot_shift = float(np.linalg.norm(pos(a.capture) - pos(a.background)))
    report = {
        "robot_shift_vs_background_m": robot_shift,
        "capture": a.capture.name, "background": a.background.name, "square_mm_measured": a.square_mm,
        "host_minus_robot_s": offset, "camera_latency_s": latency,
        "segments": [clean(e) for e in entries], "poses_used": len(good),
        "T_base_camera": result["T"].tolist(), "first_pass_T": first["T"].tolist(),
        "singular_values": result["singular_values"], "reduced_chi2": result["reduced_chi2"],
        "std_rotvec_deg": result["std_rotvec_deg"], "std_t_m": result["std_t_m"],
        "fit_per_pose": result["per_pose"], "leave_one_out": loo,
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2))
    if a.overlays:
        a.overlays.mkdir(parents=True, exist_ok=True)
        for e in good:
            overlay(a.capture, e, intr, result["T"], a.overlays / f"segment_{e['segment']:02d}.jpg")
    loo_ok = [x for x in loo if "abs_median_m" in x]
    summary = {"poses": len(good),
               "rejected": {e["segment"]: e["rejected"] for e in entries if "n_b" not in e},
               "translation_m": np.round(result["T"][:3, 3], 4).tolist(),
               "std_t_mm": np.round(np.array(result["std_t_m"]) * 1000, 1).tolist(),
               "std_rot_deg": np.round(result["std_rotvec_deg"], 3).tolist(),
               "singular_values": np.round(result["singular_values"], 3).tolist(),
               "loo_abs_median_mm": [round(x["abs_median_m"] * 1000, 1) for x in loo_ok],
               "loo_mean_mm": [round(x["mean_m"] * 1000, 1) for x in loo_ok],
               "loo_normal_deg": [round(x["normal_angle_deg"], 2) for x in loo_ok]}
    summary["robot_shift_vs_background_m"] = round(robot_shift, 4)
    print(json.dumps(summary, indent=1))
    if len(loo_ok) < 5:
        raise SystemExit("Fewer than 5 leave-one-out checks; nothing saved")
    loo_med = float(np.median([x["abs_median_m"] for x in loo_ok]))
    loo_bias = float(np.median([abs(x["mean_m"]) for x in loo_ok]))
    loo_ang = float(np.median([x["normal_angle_deg"] for x in loo_ok]))
    if loo_med > a.max_loo_m or loo_bias > a.max_loo_m or loo_ang > a.max_loo_deg:
        raise SystemExit(f"Rejected: leave-one-out median {loo_med * 1000:.1f} mm, offset {loo_bias * 1000:.1f} mm, "
                         f"normal {loo_ang:.2f} deg exceed {a.max_loo_m * 1000:.0f} mm / {a.max_loo_deg} deg; "
                         "nothing saved")
    T = RigidTransform(result["T"], intr.camera_frame, "base_link", "measured", {
        "date": datetime.date.today().isoformat(), "method": "ChArUco plane vs background-subtracted "
        "cloud_base plane, Kabsch + plane-to-plane refinement", "capture": a.capture.name,
        "background": a.background.name, "poses": len(good), "square_mm_measured": a.square_mm,
        "std_t_m": result["std_t_m"], "std_rotvec_deg": result["std_rotvec_deg"],
        "singular_values": result["singular_values"],
        "loo_abs_median_m": float(np.median([x["abs_median_m"] for x in loo_ok])),
        "loo_abs_mean_offset_m": float(np.median([abs(x["mean_m"]) for x in loo_ok])),
        "loo_normal_angle_median_deg": float(np.median([x["normal_angle_deg"] for x in loo_ok])),
        "T_base_lidar_status": "firmware (cloud_base frame used directly)",
        "tool": "scripts/phase5d_solve_extrinsics.py"})
    save_calibration(a.output, intr, T)
    print(f"saved T_base_camera -> {a.output}")


if __name__ == "__main__":
    main()
