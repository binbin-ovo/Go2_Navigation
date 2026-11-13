"""Package private go2-vln-episode-v1 episodes from a go2-recording-v1 recording.

Read-only on the recording: selected RGB bytes and the measured calibration are copied
into a new package; nothing in the recording directory is modified. Per episode:

  <out>/<episode_id>/model_input/      episode.json, images/, calibration.yaml  (model side)
  <out>/<episode_id>/evaluation_only/  annotations.json (pending stub), provenance.json,
                                       odom_slice.jsonl, decision_panels/    (never model input)

Times follow docs/VLN_EPISODE_INTERFACE.md: observation_s = host_recv_first - offset - latency
(estimated exposure), available_s = host_recv_last - offset + declared decode margin. Both are
receipt-based estimates in the go2_robot clock of the recording's own boot, not hardware sync.
Spec times (start/end/decisions) are seconds after the recording's first odometry stamp.

Usage:
  python scripts/phase6b_build_episodes.py --spec SPEC.json --recordings DIR --output DIR
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from go2_perception.vln.episodes import file_sha256, load_annotations, load_episode, select_history

DECODE_MARGIN_S = 0.050


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _yaw_deg(q):
    x, y, z, w = q
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def build(spec, rec_root, calibration, sensors_cfg, out_root):
    rec = Path(rec_root) / spec["recording_id"]
    manifest_path = rec / "manifest.json"
    if json.loads(manifest_path.read_text()).get("schema") != "go2-recording-v1":
        raise ValueError(f"{rec}: not a go2-recording-v1 recording")
    timing = yaml.safe_load(Path(sensors_cfg).read_text())["timing"]
    latency, lat_sigma = timing["camera_latency_s"]["value"], timing["camera_latency_s"]["sigma"]
    sigma = float(np.hypot(lat_sigma, timing.get("offset_sigma_s", 0.002)))
    odom = _jsonl(rec / "state" / "utlidar_robot_odom.jsonl")
    # Same estimator as vln.replay.Go2RecordingReplay: lower envelope of host_recv - stamp.
    offset = float(min(r["host_recv"] - r["stamp"] for r in odom))
    epoch = f"go2_boot:host_minus_robot={offset:.1f}"
    t0 = odom[0]["stamp"]
    rgb = _jsonl(rec / "rgb" / "index.jsonl")

    pkg = Path(out_root) / spec["episode_id"]
    if pkg.exists():
        raise FileExistsError(f"{pkg} exists; packages are never overwritten")
    mi, ev = pkg / "model_input", pkg / "evaluation_only"
    (mi / "images").mkdir(parents=True)
    ev.mkdir(parents=True)
    shutil.copyfile(calibration, mi / "calibration.yaml")

    start, end = t0 + spec["start_rel_s"], t0 + spec["end_rel_s"]
    frames, nominal_dt = [], float(np.median(np.diff([r["host_recv_first"] for r in rgb])))
    for i, r in enumerate(rgb):
        obs = r["host_recv_first"] - offset - latency
        if not start <= obs <= end:
            continue
        src = rec / "rgb" / r["file"]
        dst = mi / "images" / f"{i:06d}{src.suffix}"
        image, missing = None, None
        if r.get("complete", False) and src.is_file():
            shutil.copyfile(src, dst)
            image = {"path": f"images/{dst.name}", "sha256": file_sha256(dst)}
        else:
            missing = "incomplete or missing frame in source recording"
        frames.append({"frame_id": f"rgb_{i:06d}", "recording_id": spec["recording_id"],
                       "clock_epoch": epoch, "observation_s": obs,
                       "available_s": r["host_recv_last"] - offset + DECODE_MARGIN_S,
                       "host_receipt_s": r["host_recv_first"], "image": image,
                       "size_wh": [r["width"], r["height"]], "missing_reason": missing,
                       "robot_state_ref": None})
    obs_t = [f["observation_s"] for f in frames]
    gaps = [{"start_s": a, "end_s": b, "reason": f"inter-frame interval {b - a:.3f} s > 1.5x nominal"}
            for a, b in zip(obs_t, obs_t[1:]) if b - a > 1.5 * nominal_dt]
    decisions = [{"decision_id": d["id"], "time_s": t0 + d["rel_s"]} for d in spec["decisions"]]

    episode = {
        "schema_version": "go2-vln-episode-v1", "episode_id": spec["episode_id"],
        "instruction": spec["instruction"],
        "recording": {"recording_id": spec["recording_id"],
                      "source": f"private Go2 recording {spec['recording_id']} (local, not in Git)",
                      "source_manifest_sha256": file_sha256(manifest_path), "kind": "go2-recording-v1"},
        "clock": {"domain": "go2_robot", "epoch": epoch, "units": "seconds",
                  "observation_semantics": "exposure_estimated",
                  "availability_semantics": f"receipt_based_estimate: host_recv_last - offset + {DECODE_MARGIN_S} s declared decode margin",
                  "uncertainty_s": sigma, "host_minus_robot_s": offset},
        "bounds": {"start_s": start, "end_s": end, "end_reason": spec["end_reason"]},
        "calibration": {"path": "calibration.yaml", "sha256": file_sha256(mi / "calibration.yaml")},
        "preprocessing": {"source_rectified": False, "policy": "phase6a_calibrated_view",
                          "provenance": "raw fisheye RGB bytes copied unmodified from the recording"},
        "frames": frames, "decisions": decisions, "gaps": gaps,
    }
    (mi / "episode.json").write_text(json.dumps(episode, indent=1, allow_nan=False))
    ep = load_episode(mi / "episode.json")

    # Evaluation-only material. Odometry here is the operator's recorded route.
    odom_slice = [r for r in odom if start - 1.0 <= r["stamp"] <= end + 1.0]
    (ev / "odom_slice.jsonl").write_text("".join(json.dumps(r) + "\n" for r in odom_slice))
    stub = {"schema_version": "go2-vln-annotations-v1", "episode_id": spec["episode_id"],
            "manifest_sha256": ep.sha256,
            "reviewer": {"name": "pending human reviewer", "type": "human", "status": "pending"},
            "decisions": [{"decision_id": d["decision_id"], "target_visibility": "unknown",
                           "instruction_progress": None, "acceptable_actions": [],
                           "completion_status": "unknown", "ambiguity": False,
                           "insufficient_information": True,
                           "notes": "PENDING human review; placeholder values are not labels."}
                          for d in decisions]}
    (ev / "annotations.json").write_text(json.dumps(stub, indent=1))
    load_annotations(ev / "annotations.json", ep)

    def pose_at(t):
        r = min(odom, key=lambda r: abs(r["stamp"] - t))
        return {"stamp": r["stamp"], "xy_m": r["position"][:2], "yaw_deg": _yaw_deg(r["q_xyzw"])}

    provenance = {
        "episode_id": spec["episode_id"], "instruction": spec["instruction"],
        "instruction_origin": spec["instruction_origin"],
        "recording_path": str(rec), "recording_notes": json.loads(manifest_path.read_text()).get("notes"),
        "spec": spec, "t0_first_odom_stamp": t0, "offset_estimator": "min(odom host_recv - stamp)",
        "camera_latency_s": latency, "camera_latency_sigma_s": lat_sigma,
        "camera_latency_source": timing["camera_latency_s"].get("source"),
        "decode_margin_s": DECODE_MARGIN_S, "nominal_frame_dt_s": nominal_dt,
        "frames": len(frames), "missing_frames": sum(f["image"] is None for f in frames), "gaps": len(gaps),
        "manifest_sha256": ep.sha256,
        "odom_at_bounds": {"start": pose_at(start), "end": pose_at(end)},
        "odom_at_decisions": {d["decision_id"]: pose_at(d["time_s"]) for d in decisions},
        "limitations": spec.get("limitations", []),
    }
    (ev / "provenance.json").write_text(json.dumps(provenance, indent=1))
    return ep


def render_panels(ep, out_dir, max_frames=4, horizon_s=6.0):
    """Causal panels for labelling: only frames select_history would allow at each cutoff."""
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in ep.data["decisions"]:
        sel = select_history(ep, d["decision_id"], max_frames=max_frames, horizon_s=horizon_s)
        tiles = []
        for f in sel["frames"]:
            img = cv2.resize(cv2.imread(str(ep.root / f["image"]["path"])), (480, 270))
            label = f"{f['frame_id']}  t-{d['time_s'] - f['observation_s']:.2f}s"
            cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            tiles.append(img)
        head = np.full((40, 480 * len(tiles), 3), 255, np.uint8)
        text = f"{ep.data['episode_id']} {d['decision_id']}: \"{ep.data['instruction']}\"  (oldest -> current)"
        cv2.putText(head, text, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imwrite(str(out_dir / f"{d['decision_id']}.jpg"), np.vstack([head, np.hstack(tiles)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--calibration", default="configs/go2_front_camera_calibration.yaml")
    ap.add_argument("--sensors", default="configs/go2_vln_sensors.yaml")
    a = ap.parse_args()
    specs = json.loads(Path(a.spec).read_text())["episodes"]
    for spec in specs:
        ep = build(spec, a.recordings, a.calibration, a.sensors, a.output)
        render_panels(ep, Path(a.output) / spec["episode_id"] / "evaluation_only" / "decision_panels")
        print(f"{spec['episode_id']}: {len(ep.data['frames'])} frames, "
              f"{len(ep.data['decisions'])} decisions, {len(ep.data['gaps'])} gaps, manifest {ep.sha256[:12]}")


if __name__ == "__main__":
    main()
