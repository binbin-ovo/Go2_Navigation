"""Offline Go2 VLN observation demo: replay a recording, build, validate and serialize one
``go2-vln-obs-v1`` observation, render image + geometry, and list missing or uncertain fields.

A LiDAR overlay on the image is drawn only when a calibration file is given; a synthetic calibration
is accepted only with --allow-synthetic (the synthetic fixture). Read-only; sends nothing.

Usage:
  PYTHONPATH=src python scripts/phase5d_vln_replay_demo.py \
      --recording data/recordings/p5d_static_02 --index 200 --output data/outputs/phase5d/demo
  PYTHONPATH=src python scripts/phase5d_vln_replay_demo.py --synthetic --output /tmp/vln_demo
"""

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from go2_perception.vln import load, save, validate  # noqa: E402
from go2_perception.vln.calibration import project_points  # noqa: E402
from go2_perception.vln.replay import Go2RecordingReplay  # noqa: E402
from go2_perception.vln.synthetic import write_recording  # noqa: E402


def field_report(obs):
    v, g, r = obs.visual, obs.geometric, obs.robot
    rows = []

    def add(name, status, detail):
        rows.append({"field": name, "status": status, "detail": detail})

    add("visual.image", "ok" if v is not None else "missing",
        None if v is None else f"{v.height}x{v.width}x3 uint8 RGB, {len(v.history)} history frames")
    if v is not None:
        add("visual.stamp_robot", "estimated", f"sigma {v.stamp_robot.uncertainty_s * 1e3:.0f} ms "
            "(host receipt - offset - measured camera latency)")
        add("visual.intrinsics", v.intrinsics_status if v.intrinsics_status != "unknown" else "missing",
            "no factory calibration published; ChArUco calibration pending" if v.K is None else v.distortion_model)
    add("geometric.points", "ok" if g is not None else "missing",
        None if g is None else f"{len(g.points)} pts, {g.frames_used} scans, {g.window_s} s window")
    if g is not None:
        known = np.isfinite(g.sector_range).sum()
        add("geometric.grid", "ok", {"unknown": int((g.grid == -1).sum()), "free_floor_seen": int((g.grid == 0).sum()),
                                     "occupied": int((g.grid == 1).sum())})
        add("geometric.sector_range", "partial", f"{known}/{len(g.sector_range)} sectors with an obstacle return")
        add("geometric.projected_depth", "ok" if g.projected_depth is not None else "missing",
            "needs measured camera intrinsics + T_base_camera")
    add("robot.T_odom_base", "ok" if r is not None and r.T_odom_base is not None else "missing",
        "odometry; accuracy under motion unverified")
    add("robot.velocity_frame", "uncertain", None if r is None else r.velocity_frame)
    add("calibration.T_base_lidar", "firmware", "recovered from robot output, not surveyed")
    return rows


def render(obs, path, intrinsics=None, extrinsics=None):
    v, g = obs.visual, obs.geometric
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    if v is not None:
        a0.imshow(v.image)
        title = f"camera (RGB), est. robot t = {v.stamp_robot.t:.3f} s +/- {v.stamp_robot.uncertainty_s * 1e3:.0f} ms"
        if intrinsics is not None and extrinsics is not None and g is not None:
            cam = extrinsics.inverse().apply(g.points.astype(float))
            uv, ok = project_points(cam, intrinsics)
            a0.scatter(uv[ok, 0], uv[ok, 1], c=cam[ok, 2], s=3, cmap="turbo", vmin=0, vmax=6)
            title += f"\nLiDAR overlay ({intrinsics.status} calibration)"
        else:
            title += "\nno LiDAR overlay: camera calibration not measured"
        a0.set_title(title, fontsize=9)
    a0.axis("off")
    if g is not None:
        nx, ny = g.grid.shape
        x0, y0 = g.origin_xy
        cmap = matplotlib.colors.ListedColormap(["#d9d9d9", "#ffffff", "#202020"])
        a1.imshow(g.grid + 1, cmap=cmap, vmin=0, vmax=2, origin="lower", interpolation="nearest",
                  extent=[y0, y0 + ny * g.cell_m, x0, x0 + nx * g.cell_m])
        # Sector ranges as rays from the robot.
        for s, (rng, clear) in enumerate(zip(g.sector_range, g.sector_clear_to)):
            a = np.radians((s + 0.5) * g.sector_deg)
            if np.isfinite(clear):
                a1.plot([0, clear * np.sin(a)], [0, clear * np.cos(a)], color="tab:green", lw=0.8)
            if np.isfinite(rng):
                a1.plot(rng * np.sin(a), rng * np.cos(a), "o", color="tab:red", ms=2.5)
        a1.add_patch(plt.Circle((0, 0), g.blind_radius_m, color="tab:orange", fill=False, ls="--"))
        a1.plot(0, 0, "^", color="tab:blue", ms=9)
        a1.set_xlim(y0 + ny * g.cell_m, y0)
        a1.set_title(f"base_link grid {g.cell_m:.2f} m, {g.window_s} s window ({g.frames_used} scans)\n"
                     "grey unknown, white floor seen, black occupied\ngreen floor evidence, "
                     "red nearest obstacle per sector, orange blind zone", fontsize=8)
        a1.set_xlabel("y (m): robot left <-> right")
        a1.set_ylabel("x forward (m)")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recording", type=Path)
    ap.add_argument("--synthetic", action="store_true", help="use a freshly generated synthetic fixture")
    ap.add_argument("--index", type=int, default=None, help="cloud_base scan index (default: middle)")
    ap.add_argument("--history", type=int, default=2)
    ap.add_argument("--calibration", type=Path)
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--instruction", default="Go to the kitchen")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.synthetic:
        a.recording = write_recording(Path(tempfile.mkdtemp()) / "synthetic")
        a.calibration, a.allow_synthetic = a.recording / "synthetic_calibration.yaml", True
    if a.recording is None:
        ap.error("--recording or --synthetic is required")
    a.output.mkdir(parents=True, exist_ok=True)

    replay = Go2RecordingReplay(a.recording, calibration=a.calibration, allow_synthetic=a.allow_synthetic)
    k = len(replay) // 2 if a.index is None else a.index
    obs = replay.observation(k, history=a.history, instruction=a.instruction)
    issues = validate(obs)
    npz = a.output / f"{a.recording.name}_{k:06d}.vlnobs.npz"
    save(obs, npz)
    back = load(npz)
    roundtrip = (validate(back) == issues and np.array_equal(back.geometric.grid, obs.geometric.grid)
                 and (obs.visual is None or np.array_equal(back.visual.image, obs.visual.image)))
    render(obs, a.output / f"{a.recording.name}_{k:06d}.png", replay.intrinsics, replay.extrinsics)
    report = {"recording": a.recording.name, "scan_index": k, "schema": obs.schema_version,
              "clock_epoch": obs.clock_epoch, "validation_issues": issues, "warnings": obs.warnings,
              "serialized": str(npz), "roundtrip_ok": bool(roundtrip), "fields": field_report(obs),
              "calibration": obs.calibration}
    (a.output / f"{a.recording.name}_{k:06d}.report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
