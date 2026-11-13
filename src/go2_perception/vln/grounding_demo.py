"""Local, inspectable replay artifacts. No networking or robot command interface."""

import hashlib
import json
from pathlib import Path
import textwrap

import cv2
import numpy as np

from .model_preprocessing import calibrated_view
from .observation import save, validate


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def replay_source(replay, k, history=2):
    """Identify the exact causal RGB files and LiDAR window used by existing replay."""
    if not 0 <= k < len(replay):
        raise ValueError("Scan index outside recording")
    t = replay.scans[k]["t_mid"]
    idx = int(np.searchsorted(replay.rgb_t_robot, t, side="right")) - 1
    frames = []
    for j in range(max(0, idx - history), idx + 1):
        row = replay.rgb[j]
        path = replay.root / "rgb" / row["file"]
        frames.append({"index": j, "file": str(path.resolve()), "sha256": sha256(path),
                       "index_metadata": row, "used_by_model": j == idx,
                       "exposure_estimated_robot_s": float(replay.rgb_t_robot[j])})
    scans = []
    for s in replay.scans:
        if t - replay.acc["display_window_s"] < s["t_mid"] <= t + 1e-9:
            path = replay.lidar_dir / s["file"]
            scans.append({"file": str(path.resolve()), "sha256": sha256(path),
                          "t_mid": s["t_mid"], "pose_available": replay.poses.lookup(s["t_mid"]) is not None,
                          "in_obstacle_window": s["t_mid"] > t - replay.acc["obstacle_window_s"]})
    odom_path = replay.root / "state" / "utlidar_robot_odom.jsonl"
    return {"recording": str(replay.root.resolve()), "scan_index": k, "reference_robot_s": t,
            "rgb_frames": frames, "lidar_window_sources": scans,
            "odometry_sha256": sha256(odom_path), "recording_manifest": replay.manifest}


def render_demo(obs, view, proposal, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle, Patch, Rectangle

    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True,
                                      gridspec_kw={"width_ratios": [1.65, 1]})
    left.imshow(view.image)
    for n, c in enumerate(proposal["candidates"]):
        x0, y0, x1, y1 = c["box_xyxy"]
        color = plt.get_cmap("tab10")(n % 10)
        left.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=2, color=color))
        left.text(x0, max(12, y0 - 5), f"{n}: {c['label']} {c['score']:.2f}", fontsize=9,
                  color="white", bbox={"facecolor": color, "alpha": 0.9, "pad": 2})
    left.set_title("\n".join(textwrap.wrap(proposal["instruction"], 65)) +
                   f"\n{proposal['status']} | image-space proposals only", fontsize=11)
    left.axis("off")
    g, r, v = obs.geometric, obs.robot, obs.visual
    if g is None:
        right.text(.5, .5, "Geometry unavailable", ha="center", transform=right.transAxes)
    else:
        nx, ny = g.grid.shape
        x0, y0 = g.origin_xy
        colors = ["#abb0b6", "#f8f5e7", "#313e4e"]
        right.imshow(g.grid + 1, cmap=ListedColormap(colors), vmin=0, vmax=2, origin="lower",
                     interpolation="nearest", extent=[y0, y0 + ny*g.cell_m, x0, x0 + nx*g.cell_m])
        right.add_patch(Circle((0, 0), g.blind_radius_m, fill=False, color="darkorange", ls="--"))
        right.plot(0, 0, "^", color="royalblue", ms=10)
        right.set_xlim(y0 + ny*g.cell_m, y0)
        right.set_xlabel("base_link y (m), robot left ← → right")
        right.set_ylabel("base_link x forward (m)")
        right.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors,
                     ["unknown", "floor seen (not traversable)", "occupied"])], fontsize=8, loc="upper right")
        right.set_title(f"Existing LiDAR grid: {g.window_s:.1f}s / {g.frames_used} scans\n"
                        f"robot scan midpoint {g.stamp.t:.3f}s", fontsize=10)
    pose = "pose unavailable"
    if r is not None and r.T_odom_base is not None:
        pose = "odom position m: " + np.array2string(r.T_odom_base[:3, 3], precision=2)
    stamp = "camera robot time unavailable"
    if v.stamp_robot is not None:
        s = v.stamp_robot
        uncertainty = "unknown" if s.uncertainty_s is None else f"{s.uncertainty_s * 1000:.1f} ms"
        stamp = f"camera {s.semantics}: {s.t:.3f}s; uncertainty {uncertainty}"
    registration = ("NO CAMERA–BASE REGISTRATION" if obs.calibration.get("extrinsics_status", "unknown") == "unknown"
                    else "CAMERA–BASE REGISTRATION NOT USED BY THIS ADAPTER")
    caption = (f"Rectified RGB; boxes are not raw fisheye pixels. {stamp}\n{pose}; epoch {obs.clock_epoch}\n"
               f"{registration}: image targets have no metric goal or LiDAR labels.\n"
               "Floor returns do not prove traversability. Odometry under motion remains unverified.")
    if validate(obs):
        caption += "\nOBSERVATION ISSUES: " + "; ".join(validate(obs))
    fig.suptitle("Offline text-conditioned grounding • no robot commands", fontsize=14)
    fig.text(0.04, -0.015, caption, fontsize=9, va="top")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run_observation(obs, model, output, *, source=None, allow_synthetic=False):
    """Save raw observation, calibrated view, processor evidence and parsed proposals.

    Calibration/timestamp validation issues are visible and retained. Geometry is
    displayed independently; no metric grounding occurs even if transforms exist.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing experiment: {output}")
    output.mkdir(parents=True, exist_ok=True)
    view = calibrated_view(obs, allow_synthetic=allow_synthetic)
    if obs.instruction is None:
        raise ValueError("Observation instruction missing")
    result = model.predict(view, obs.instruction)
    p = result.proposal
    p["warnings"] += list(obs.warnings) + validate(obs) + [
        "Image grounding only; no metric localization, clearance decision or executable goal",
        "T_base_camera is missing/unvalidated; RGB and LiDAR are separate views"
        if obs.calibration.get("extrinsics_status", "unknown") == "unknown"
        else "This adapter does not use camera-to-base registration, even when supplied",
        "Current image only; history is preserved but not consumed"]
    p["source"] = view.metadata["source"]
    p["model"] = model.info
    p["timing"] = result.timing
    p["thresholds"] = {k: model.config[k] for k in ("box_threshold", "text_threshold")}
    save(obs, output / "observation.vlnobs.npz")
    np.savez_compressed(output / "raw_model_output.npz", **result.raw)
    for name, img in [("original.png", obs.visual.image), ("rectified.png", view.image),
                      ("model_input.png", result.model_input_rgb)]:
        if not cv2.imwrite(str(output / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Cannot write {name}")
    compare = np.concatenate([obs.visual.image, view.image], axis=1)
    cv2.imwrite(str(output / "original_vs_rectified.png"), cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))
    np.save(output / "rectification_valid_mask.npy", view.valid_mask, allow_pickle=False)
    write_json(output / "preprocessing.json", {**view.metadata, "model_processor": result.processor_metadata})
    write_json(output / "proposal.json", p)
    write_json(output / "source.json", source or {"kind": "in_memory_go2_vln_obs_v1",
                                                 "clock_epoch": obs.clock_epoch})
    render_demo(obs, view, p, output / "demo.png")
    return p
