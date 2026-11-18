"""Offline check of the target-correspondence gate (local_feature_roi_v1) on recorded live sessions.

Pairs the source images of successive model requests in a recorded interactive session (the same kind of
image the supervised app reviews and refreshes) and labels each pair with the robot's odometry change
between the two frames. The gate should pass for a textured ROI when the robot has not moved, and refuse
when the robot turned/moved or when the model point lies outside the ROI. Read-only; no model, no robot.
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from go2_perception.vln.episodes import file_sha256  # noqa: E402
from go2_perception.vln.supervised_target import correspondence  # noqa: E402

ROIS = {"centre": [0.40, 0.35, 0.60, 0.65], "left": [0.15, 0.30, 0.35, 0.70], "right": [0.65, 0.30, 0.85, 0.70]}


def odometry(run):
    t, xy, yaw = [], [], []
    for f in sorted(glob.glob(str(run / "lidar" / "chunk_*.npz"))):
        d = np.load(f)
        q = d["odom_q_xyzw"]
        t.append(d["odom_host_recv_wall"])
        xy.append(d["odom_position"][:, :2])
        yaw.append(np.arctan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)))
    t = np.concatenate(t)
    order = np.argsort(t)
    return t[order], np.concatenate(xy)[order], np.concatenate(yaw)[order]


def pose_at(odom, wall):
    t, xy, yaw = odom
    i = int(np.clip(np.searchsorted(t, wall), 0, len(t) - 1))
    return (xy[i], yaw[i]) if abs(t[i] - wall) < 0.5 else None


def rectified_png(npz, out_dir):
    """Same rectification as model_preprocessing.calibrated_view (fisheye, balance 0, same size)."""
    out = out_dir / (npz.parent.parent.parent.name + "_" + npz.name.replace(".vlnobs.npz", ".png"))
    if not out.exists():
        d = np.load(npz)
        img, K, D = d["visual.image"], d["visual.K"], d["visual.D"]
        size = (img.shape[1], img.shape[0])
        Kn = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, size, np.eye(3), balance=0.0, new_size=size)
        mx, my = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Kn, size, cv2.CV_32FC1)
        cv2.imwrite(str(out), cv2.cvtColor(cv2.remap(img, mx, my, cv2.INTER_LINEAR), cv2.COLOR_RGB2BGR))
    return str(out)


def observation_frames(run, out_dir, min_gap_s):
    frames, last = [], -1e9
    for meta in sorted((run / "session" / "observations").glob("*.vlnobs.json")):
        m = json.loads(meta.read_text())
        if m["observation_s"] - last >= min_gap_s:
            frames.append((rectified_png(meta.with_suffix("").with_suffix(".vlnobs.npz"), out_dir), m["observation_s"]))
            last = m["observation_s"]
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--observations", type=Path, help="also pair recorded observations, rectified into this dir")
    ap.add_argument("--min-gap-s", type=float, default=4.0)
    a = ap.parse_args()
    rows = []
    for run in a.runs:
        odom = odometry(run)
        frames = []
        if a.observations:
            a.observations.mkdir(parents=True, exist_ok=True)
            frames = [(p, t, pose_at(odom, t)) for p, t in observation_frames(run, a.observations, a.min_gap_s)]
        for job in sorted((run / "session" / "requests").glob("*/job.json")):
            last = json.loads(job.read_text())["frames"][-1]
            if Path(last["path"]).exists():
                frames.append((last["path"], last["observation_s"], pose_at(odom, last["observation_s"])))
        frames.sort(key=lambda f: f[1])
        for (pa, ta, ma), (pb, tb, mb) in zip(frames, frames[1:]):
            if ma is None or mb is None:
                continue
            moved_m = float(np.linalg.norm(mb[0] - ma[0]))
            turned_deg = abs(math.degrees(math.atan2(math.sin(mb[1] - ma[1]), math.cos(mb[1] - ma[1]))))
            for name, roi in ROIS.items():
                target = {"instance_id": "check", "reviewed_image_sha256": file_sha256(pa), "roi_xyxy": roi}
                inside = [(roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2]
                outside = [roi[2] + 0.1, inside[1]]
                for label, point in (("point_inside", inside), ("point_outside", outside)):
                    r = correspondence(pa, pb, target, point)
                    rows.append({"run": run.name, "dt_s": tb - ta, "moved_m": moved_m, "turned_deg": turned_deg,
                                 "roi": name, "point": label, "valid": r["valid"], "reason": r["reason"],
                                 "inliers": r.get("inliers"), "target_inliers": r.get("target_inliers"),
                                 "residual_px": r.get("median_residual_px"),
                                 "drift": r.get("max_corner_drift_fraction")})
    still = [r for r in rows if r["moved_m"] < 0.03 and r["turned_deg"] < 2 and r["point"] == "point_inside"]
    moved = [r for r in rows if (r["moved_m"] >= 0.10 or r["turned_deg"] >= 5) and r["point"] == "point_inside"]
    outside = [r for r in rows if r["point"] == "point_outside"]
    def rate(rs):
        return f"{sum(r['valid'] for r in rs)}/{len(rs)}"
    print(f"stationary pairs (<3 cm, <2 deg), point inside ROI: valid {rate(still)}")
    for roi in ROIS:
        print(f"   {roi}: {rate([r for r in still if r['roi'] == roi])}")
    print(f"moved/turned pairs (>=10 cm or >=5 deg), point inside: valid {rate(moved)}  <- must be 0")
    print(f"any pair, point outside ROI: valid {rate(outside)}  <- must be 0")
    fails = {}
    for r in still:
        if not r["valid"]:
            fails[r["reason"]] = fails.get(r["reason"], 0) + 1
    print("stationary refusals by reason:", fails)
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
