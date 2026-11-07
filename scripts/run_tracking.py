"""Track timestamped detections or a Phase 2 Go2 recording, entirely offline."""

import argparse
import json
from pathlib import Path
from time import perf_counter
import cv2
from go2_perception.io.config import load_config
from go2_perception.io.serialization import jsonable, save_json
from go2_perception.tracking.config import TrackerConfig
from go2_perception.tracking.multi_object_tracker import MultiObjectTracker
from go2_perception.tracking.pose_gate import PoseAlignmentConfig
from go2_perception.tracking.replay import load_detection_sequence, recording_frames
from go2_perception.visualization.bev_tracking import save_bev


def run(args):
    config = load_config(args.config)
    if args.mode:
        config["tracker"]["mode"] = args.mode
    c = TrackerConfig(**config["tracker"])
    tracker = MultiObjectTracker(c)
    path = Path(args.input)
    manifest = path / "manifest.json"
    schema = (
        json.loads(manifest.read_text()).get("schema") if manifest.is_file() else None
    )
    start = perf_counter()
    duplicates = 0
    pose_report = None
    if schema == "go2-recording-v1":
        pose_report = {}
        frames = recording_frames(
            path,
            c,
            load_config(args.lidar_config),
            args.window_s,
            PoseAlignmentConfig(**config.get("pose_alignment", {})),
            pose_report,
        )
    else:
        if args.window_s:
            raise ValueError("--window-s is only supported for raw Go2 recordings")
        frames, duplicates = load_detection_sequence(path, args.duplicate_policy)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.json", config)
    times = []
    count = 0
    writer = None
    try:
        with (out / "tracks.jsonl").open("w") as stream:
            for index, frame in enumerate(frames):
                t0 = perf_counter()
                reset_ids = tracker.reset() if frame.reset_before else []
                tracks = tracker.update(
                    frame.detections,
                    frame.timestamp,
                    frame.robot_pose,
                    frame_id=frame.frame_id,
                    clock_domain=frame.clock_domain,
                )
                latency = (perf_counter() - t0) * 1000
                times.append(latency)
                count += 1
                target = (
                    tracks[0].frame_id
                    if tracks
                    else (
                        c.tracking_frame
                        if c.mode == "odom"
                        else "fixed_sensor:" + c.sensor_frame
                    )
                )
                row = dict(
                    timestamp=frame.timestamp,
                    frame_id=target,
                    clock_domain=frame.clock_domain,
                    tracks=tracks,
                    deleted_ids=tracker.deleted_ids,
                    reset_before=frame.reset_before,
                    reset_ended_ids=reset_ids,
                    pose_outage_s=frame.pose_outage_s,
                    suppressed_duplicates=tracker.suppressed_duplicates,
                    latency_ms=latency,
                )
                stream.write(json.dumps(jsonable(row), allow_nan=False) + "\n")
                if not args.no_render:
                    image = save_bev(
                        out / "frames" / f"{index:06d}.png",
                        tracks,
                        frame.timestamp,
                        target,
                        frame.robot_pose,
                        **config.get("visualization", {}),
                    )
                    if args.video:
                        if writer is None:
                            writer = cv2.VideoWriter(
                                str(out / "tracking.avi"),
                                cv2.VideoWriter_fourcc(*"MJPG"),
                                args.video_fps,
                                (image.shape[1], image.shape[0]),
                            )
                            if not writer.isOpened():
                                raise RuntimeError(
                                    "MJPG video encoder unavailable; use PNG frames"
                                )
                        writer.write(image)
    finally:
        if writer is not None:
            writer.release()
    elapsed = perf_counter() - start
    pose = None
    if pose_report:
        skipped = pose_report["skipped"]
        reasons = {}
        for s in skipped:
            reasons[s.reason] = reasons.get(s.reason, 0) + 1
        with (out / "skipped_frames.jsonl").open("w") as stream:
            for s in skipped:
                stream.write(json.dumps(dict(timestamp=s.timestamp, reason=s.reason)) + "\n")
        pose = dict(
            frames_input=pose_report["frames_input"],
            frames_skipped=len(skipped),
            skipped_by_reason=reasons,
            start_hold_frames=pose_report["start_hold_frames"],
            tracker_resets=pose_report["resets"],
            raw_recording_modified=False,
        )
    report = dict(
        frames=count,
        duplicate_frames_skipped=duplicates,
        pose_alignment=pose,
        wall_s=elapsed,
        end_to_end_throughput_hz=count / elapsed,
        tracking_latency_ms_mean=sum(times) / len(times) if times else None,
        tracking_throughput_hz=len(times) / (sum(times) / 1000)
        if sum(times) > 0
        else None,
        mode=c.mode,
        hardware_validation=False,
        video_timing="fixed display fps, not acquisition intervals"
        if args.video
        else None,
    )
    save_json(out / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/tracking.yaml")
    p.add_argument("--lidar-config", default="configs/go2_lidar.yaml")
    p.add_argument("--mode", choices=["odom", "fixed_sensor"])
    p.add_argument("--window-s", type=float, default=0)
    p.add_argument("--duplicate-policy", choices=["error", "first"], default="error")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--video", action="store_true")
    p.add_argument("--video-fps", type=float, default=10)
    a = p.parse_args()
    if a.video and (a.no_render or a.video_fps <= 0):
        p.error("--video needs rendering and positive fps")
    run(a)
