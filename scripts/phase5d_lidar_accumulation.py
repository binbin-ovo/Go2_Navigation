"""Compare ego-motion-compensated cloud_base accumulation windows on one recording.

Read-only and offline. For each window length, every cloud_base frame whose acquisition midpoint
falls in (t_ref - W, t_ref] is mapped through odometry into the reference frame's base_link
(cloud_deskewed is never used, so nothing is accumulated twice). Writes BEV and obstacle-grid
comparison images plus a metrics JSON. Metrics on a static scene measure coverage and
repeatability only; they say nothing about smear on moving objects or about accuracy.

Usage:
  PYTHONPATH=src python scripts/phase5d_lidar_accumulation.py \
      --recording data/recordings/p5d_static_02 --output data/outputs/phase5d/accumulation
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from go2_perception.adapters.go2.window import PoseBuffer  # noqa: E402

WINDOWS_S = [0.065, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
CELL_M = 0.10
GROUND_Z = -0.33  # measured base_link floor height while standing (SENSOR_VALIDATION.md)
OBSTACLE_Z = (GROUND_Z + 0.10, GROUND_Z + 1.20)
EXTENT = {"x": (-2.0, 6.0), "y": (-4.0, 4.0)}


def load_frames(root, topic="utlidar_cloud_base"):
    rows = [json.loads(line) for line in open(root / "lidar" / topic / "index.jsonl")]
    frames = []
    for row in rows:
        pts = np.load(root / "lidar" / topic / row["file"])
        tpath = root / "lidar" / topic / row["file"].replace(".npy", "_time.npy")
        per_point = np.load(tpath) if tpath.exists() else np.zeros(len(pts))
        mid = row["stamp"] + 0.5 * float(per_point.max() if len(per_point) else 0.0)
        frames.append({"t": mid, "xyz": pts[:, :3].astype(float), "host_recv": row["host_recv"]})
    return frames


def load_poses(root):
    poses = PoseBuffer(max_age_s=1e9, max_gap_s=0.1)
    for line in open(root / "state" / "utlidar_robot_odom.jsonl"):
        r = json.loads(line)
        poses.add(r["stamp"], r["position"], r["q_xyzw"])
    return poses


def accumulate(frames, poses, t_ref, window_s):
    ref_pose = poses.lookup(t_ref)
    inv = np.linalg.inv(ref_pose)
    parts, used, refused = [], 0, 0
    for f in frames:
        if not (t_ref - window_s < f["t"] <= t_ref):
            continue
        pose = poses.lookup(f["t"])
        if pose is None:
            refused += 1
            continue
        rel = inv @ pose
        parts.append(f["xyz"] @ rel[:3, :3].T + rel[:3, 3])
        used += 1
    pts = np.concatenate(parts) if parts else np.empty((0, 3))
    return pts, used, refused


def obstacle_cells(pts):
    m = (pts[:, 2] > OBSTACLE_Z[0]) & (pts[:, 2] < OBSTACLE_Z[1])
    return {tuple(c) for c in np.floor(pts[m, :2] / CELL_M).astype(int)}


def azimuth_coverage_deg(pts, r_min=0.5, bin_deg=5):
    r = np.hypot(pts[:, 0], pts[:, 1])
    az = np.degrees(np.arctan2(pts[r > r_min, 1], pts[r > r_min, 0]))
    return bin_deg * len(np.unique(np.floor(az / bin_deg)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recording", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--windows", type=float, nargs="+", default=WINDOWS_S)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    frames, poses = load_frames(a.recording), load_poses(a.recording)
    t0, t1 = frames[0]["t"], frames[-1]["t"]
    w_max = max(a.windows)
    # Reference times spaced by the longest window so repeated windows never share frames.
    refs = np.arange(t0 + w_max + 0.2, t1 - 0.1, w_max)
    if len(refs) < 2:
        raise SystemExit("Recording too short for repeatability across the longest window")
    union = obstacle_cells(accumulate(frames, poses, t1, t1 - t0 + 1)[0])

    metrics = {"recording": a.recording.name, "frames": len(frames), "span_s": t1 - t0,
               "cell_m": CELL_M, "obstacle_z_base_link": OBSTACLE_Z, "reference_times": len(refs),
               "union_obstacle_cells_all_frames": len(union), "windows": []}
    shown = {}
    for w in a.windows:
        rows = []
        cell_sets = []
        for t_ref in refs:
            pts, used, refused = accumulate(frames, poses, t_ref, w)
            cells = obstacle_cells(pts)
            cell_sets.append(cells)
            rows.append({"points": len(pts), "frames": used, "refused": refused,
                         "obstacle_cells": len(cells),
                         "completeness_vs_union": len(cells & union) / max(len(union), 1),
                         "azimuth_coverage_deg": azimuth_coverage_deg(pts)})
        jac = [len(x & y) / max(len(x | y), 1) for x, y in zip(cell_sets, cell_sets[1:])]
        summary = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
        summary.update(window_s=w, repeatability_jaccard_median=float(np.median(jac)),
                       repeatability_jaccard_min=float(np.min(jac)))
        metrics["windows"].append(summary)
        shown[w] = accumulate(frames, poses, refs[-1], w)[0]

    (a.output / "accumulation_metrics.json").write_text(json.dumps(metrics, indent=2))
    rgb = _context_image(a.recording, refs[-1], frames)
    render_bev(shown, metrics, a.output / "accumulation_bev.png", rgb)
    render_grid(shown, metrics, a.output / "accumulation_obstacle_grid.png", rgb)
    print(json.dumps(metrics, indent=2))


def _context_image(root, t_ref, frames):
    """Camera frame nearest the reference LiDAR frame by host receipt (context only, unsynced)."""
    rows = [json.loads(line) for line in open(root / "rgb" / "index.jsonl")]
    ref = min(frames, key=lambda f: abs(f["t"] - t_ref))["host_recv"]
    row = min(rows, key=lambda r: abs(r["host_recv_first"] - ref))
    img = plt.imread(root / "rgb" / row["file"])
    return img[..., ::-1] if row.get("encoding") == "bgr8" and row["file"].endswith(".npy") else img


def _axes(n, rgb):
    fig, axes = plt.subplots(2, 4, figsize=(17, 10.5), constrained_layout=True)
    axes = axes.ravel()
    if rgb is not None:
        axes[-1].imshow(rgb)
        axes[-1].set_title("front camera (context only, uncalibrated)", fontsize=9)
    axes[-1].axis("off")
    return fig, axes[:n]


def _label(w, m):
    name = "single frame (~65 ms)" if w < 0.1 else f"{w:g} s"
    return (f"{name}\n{int(m['points'])} pts, {int(m['frames'])} frames\n"
            f"complete {m['completeness_vs_union']:.0%}, repeat J={m['repeatability_jaccard_median']:.2f}")


def render_bev(shown, metrics, path, rgb=None):
    fig, axes = _axes(len(shown), rgb)
    for ax, (w, pts), m in zip(axes, shown.items(), metrics["windows"]):
        ground = pts[:, 2] <= OBSTACLE_Z[0]
        ax.scatter(pts[ground, 1], pts[ground, 0], s=0.4, c="0.75", linewidths=0)
        sc = ax.scatter(pts[~ground, 1], pts[~ground, 0], s=0.6, c=pts[~ground, 2],
                        cmap="turbo", vmin=GROUND_Z, vmax=GROUND_Z + 1.6, linewidths=0)
        ax.plot([0], [0], marker="^", ms=9, color="k")
        ax.set_xlim(EXTENT["y"][1], EXTENT["y"][0])
        ax.set_ylim(*EXTENT["x"])
        ax.set_aspect("equal")
        ax.set_title(_label(w, m), fontsize=9)
        ax.set_xlabel("y (m): robot left  <->  robot right", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[::4]:
        ax.set_ylabel("x forward (m)", fontsize=8)
    fig.colorbar(sc, ax=list(axes), shrink=0.5, label="z in base_link (m); grey = floor")
    fig.suptitle("cloud_base accumulated in base_link (top-down, robot = black triangle, faces up)",
                 fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_grid(shown, metrics, path, rgb=None):
    nx = int((EXTENT["x"][1] - EXTENT["x"][0]) / CELL_M)
    ny = int((EXTENT["y"][1] - EXTENT["y"][0]) / CELL_M)
    fig, axes = _axes(len(shown), rgb)
    for ax, (w, pts), m in zip(axes, shown.items(), metrics["windows"]):
        grid = np.zeros((nx, ny))
        for cx, cy in obstacle_cells(pts):
            i, j = cx - int(EXTENT["x"][0] / CELL_M), int(EXTENT["y"][1] / CELL_M) - 1 - cy
            if 0 <= i < nx and 0 <= j < ny:
                grid[i, j] = 1
        ax.imshow(grid, origin="lower", cmap="Greys", interpolation="nearest",
                  extent=[EXTENT["y"][1], EXTENT["y"][0], *EXTENT["x"]])
        ax.plot([0], [0], marker="^", ms=9, color="tab:red")
        ax.set_title(_label(w, m) + f"\n{int(m['obstacle_cells'])} obstacle cells", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{CELL_M * 100:.0f} cm cells with a return {OBSTACLE_Z[0] - GROUND_Z:.1f}-"
                 f"{OBSTACLE_Z[1] - GROUND_Z:.1f} m above the floor (white = no return, NOT free)",
                 fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
