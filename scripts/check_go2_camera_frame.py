"""Verify real Go2 camera frames and metadata through the advisory bridge, no model.

Starts the read-only bridge child, converts the first frames to Go2VLNObservation,
saves raw/calibrated images and metadata into a new private directory, then closes
the bridge. Nothing is sent to the robot.
"""

import argparse
import json
from pathlib import Path
from queue import Empty
import time

import cv2
import yaml

from go2_perception.vln.calibration import load_calibration
from go2_perception.vln.model_preprocessing import calibrated_view
from go2_perception.vln.observation import save, validate
from go2_perception.vln.shadow import write_json
from go2_perception.vln.shadow_live import LiveCameraSource, latency_model

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hardware-python", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--camera-host-ip", default="192.168.123.222")
    p.add_argument("--calibration", type=Path, default=ROOT / "configs/go2_front_camera_calibration.yaml")
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--timeout-s", type=float, default=15.0)
    a = p.parse_args(argv)
    if a.output.exists():
        p.error("Output must be a new directory")
    intrinsics, _ = load_calibration(a.calibration)
    sensors = yaml.safe_load((ROOT / "configs/go2_vln_sensors.yaml").read_text())
    camera = LiveCameraSource(
        python=a.hardware_python,
        intrinsics=intrinsics,
        calibration_path=a.calibration,
        output=a.output / "camera",
        host_ip=a.camera_host_ip,
        latency=latency_model(sensors["timing"]["camera_latency_s"]),
    )
    started = time.monotonic()
    camera.start()
    got = []
    try:
        while len(got) < a.frames and time.monotonic() - started < a.timeout_s:
            try:
                delivery = camera.get_nowait()
            except Empty:
                if camera.error or not camera.alive:
                    break
                time.sleep(0.01)
                continue
            delivery.provenance["session_receive_monotonic_s"] = time.monotonic()
            got.append(delivery)
    finally:
        camera.close()
    events = camera.drain_events()
    report = {
        "frames_received": len(got),
        "wall_seconds": time.monotonic() - started,
        "camera": camera.status(time.monotonic()),
        "bye": camera.bye,
        "events": events,
        "robot_commands_sent": 0,
    }
    if got:
        first = got[0]
        obs = first.observation
        v = obs.visual
        save(obs, a.output / "first.vlnobs.npz")
        cv2.imwrite(str(a.output / "first_raw.png"), cv2.cvtColor(v.image, cv2.COLOR_RGB2BGR))
        view = calibrated_view(obs)
        cv2.imwrite(str(a.output / "first_calibrated.png"), cv2.cvtColor(view.image, cv2.COLOR_RGB2BGR))
        receipts = [d.received_monotonic_s for d in got]
        report["first"] = {
            "frame_id": first.frame_id,
            "shape": list(v.image.shape),
            "dtype": str(v.image.dtype),
            "rectified": v.rectified,
            "intrinsics_status": v.intrinsics_status,
            "distortion_model": v.distortion_model,
            "intrinsics_size": list(v.intrinsics_size),
            "camera_frame": v.frame_id,
            "stamp_host": vars(v.stamp_host),
            "stamp_robot": None,
            "clock_epoch": obs.clock_epoch,
            "available_s": first.available_s,
            "received_monotonic_s": first.received_monotonic_s,
            "exposure_monotonic_s": first.exposure_monotonic_s,
            "monotonic_mapping": first.monotonic_mapping,
            "pose": obs.robot,
            "lidar": obs.geometric,
            "validation_issues": validate(obs),
            "calibrated_view": view.metadata["view"],
            "provenance": first.provenance,
        }
        report["frame_ids_unique"] = len({d.frame_id for d in got}) == len(got)
        report["receipt_intervals_s"] = [b - c for c, b in zip(receipts, receipts[1:])]
        report["pipeline_s"] = [
            {
                "receipt_to_decoded": d.provenance["decoded_monotonic_s"] - d.received_monotonic_s,
                "decoded_to_parent_read": d.provenance["parent_read_monotonic_s"]
                - d.provenance["decoded_monotonic_s"],
            }
            for d in got
        ]
    write_json(a.output / "frame_check.json", json.loads(json.dumps(report, default=str)))
    print(json.dumps({k: report[k] for k in ("frames_received", "wall_seconds", "robot_commands_sent")}))
    print(f"Report: {a.output / 'frame_check.json'}")
    return 0 if got else 1


if __name__ == "__main__":
    raise SystemExit(main())
