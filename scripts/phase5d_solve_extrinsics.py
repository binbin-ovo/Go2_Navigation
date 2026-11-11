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


def lidar_window(root, scans, t0, t1):
    parts = []
    for r in scans:
        tpath = root / "lidar" / "utlidar_cloud_base" / r["file"].replace(".npy", "_time.npy")
        mid = r["stamp"] + 0.5 * float(np.load(tpath).max())
        if t0 <= mid <= t1:
            parts.append(np.load(root / "lidar" / "utlidar_cloud_base" / r["file"])[:, :3].astype(float))
    return np.concatenate(parts) if parts else np.empty((0, 3)), len(parts)


def pick_board_cluster(fg, grid_centre_b=None):
    """Board points among the foreground: near the predicted centre when T is known, otherwise the
    connected 10 cm voxel component whose plane fit has the most inliers and a board-like extent."""
    if grid_centre_b is not None:
        return fg[np.linalg.norm(fg - grid_centre_b, axis=1) < 0.40]
    keys = np.floor(fg / 0.10).astype(int)
    lookup = {}
    for i, k in enumerate(map(tuple, keys)):
        lookup.setdefault(k, []).append(i)
    seen, best = set(), np.empty((0, 3))
    for start in lookup:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            k = stack.pop()
            comp.extend(lookup[k])
            for d in [(i, j, l) for i in (-1, 0, 1) for j in (-1, 0, 1) for l in (-1, 0, 1)]:
                nb = (k[0] + d[0], k[1] + d[1], k[2] + d[2])
                if nb in lookup and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        pts = fg[comp]
        if len(pts) < 100:
            continue
        ext = np.ptp(pts, axis=0)
        if np.sort(ext)[-1] > 0.9:
            continue
        if len(pts) > len(best):
            best = pts
    return best


def collect(args, intr, board, T=None):
    root = args.capture
    poses = board_poses(root, intr, board) if not hasattr(args, "_poses") else args._poses
    args._poses = poses
    segs = still_segments(poses)
    odom = jsonl(root / "state" / "utlidar_robot_odom.jsonl")
    offset = min(r["host_recv"] - r["stamp"] for r in odom)
    latency = yaml.safe_load(Path("configs/go2_vln_sensors.yaml").read_text())["timing"]["camera_latency_s"]["value"]
    scans = jsonl(root / "lidar" / "utlidar_cloud_base" / "index.jsonl")
    bg = args._bg if hasattr(args, "_bg") else background_set(args.background)
    args._bg = bg
    grid_centre_board = np.r_[board.getChessboardSize()] * board.getSquareLength() / 2
    out = []
    for i, seg in enumerate(segs):
        h0, h1 = seg[0]["t_host"] + 0.5, seg[-1]["t_host"] - 0.5
        r0, r1 = h0 - offset - latency, h1 - offset - latency
        pts, nscans = lidar_window(root, scans, r0, r1)
        fg = pts[[k not in bg for k in map(tuple, np.floor(pts / VOXEL).astype(int))]] if len(pts) else pts
        R = np.mean([p["R"] for p in seg], axis=0)
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        t = np.median([p["t"] for p in seg], axis=0)
        n_c, d_c = camera_plane(R, t)
        centre_c = R @ np.r_[grid_centre_board, 0.0] + t
        centre_b = None if T is None else T[:3, :3] @ centre_c + T[:3, 3]
        cand = pick_board_cluster(fg, centre_b)
        entry = {"segment": i, "host_start": seg[0]["t_host"], "duration_s": seg[-1]["t_host"] - seg[0]["t_host"],
                 "frames": len(seg), "scans": nscans, "foreground": int(len(fg)), "candidates": int(len(cand)),
                 "camera_distance_m": float(np.linalg.norm(t)), "n_c": n_c, "d_c": d_c,
                 "camera_t_std_mm": (np.std([p["t"] for p in seg], axis=0) * 1000).tolist(),
                 "frame": seg[len(seg) // 2]["file"], "centre_c": centre_c}
        if len(cand) < 60:
            entry["rejected"] = f"only {len(cand)} LiDAR board candidates"
            out.append(entry)
            continue
        n_b, d_b, inl, res = fit_plane(cand)
        if inl.sum() < 60:
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
    a = ap.parse_args()
    intr, _ = load_calibration(a.intrinsics)
    if intr.status != "measured" or intr.model != "fisheye_equidistant":
        raise SystemExit("Needs measured fisheye intrinsics")
    board = make_board(json.loads(a.board.read_text()), a.square_mm / 1000)

    entries, offset, latency = collect(a, intr, board)
    good = [e for e in entries if "n_b" in e]
    first = solve(good)
    # Second pass: re-select LiDAR points around the board centre predicted by the first solution.
    entries, _, _ = collect(a, intr, board, first["T"])
    good = [e for e in entries if "n_b" in e]
    result = solve(good)
    loo = leave_one_out(good)

    def clean(e):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in e.items() if k != "points_b"}

    report = {
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
    print(json.dumps(summary, indent=1))
    if len(loo_ok) < 5:
        raise SystemExit("Fewer than 5 leave-one-out checks; nothing saved")
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
