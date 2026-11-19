"""Read-only Go2, replay and synthetic sensor streams for navigation."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, field, fields
import os
from pathlib import Path
import sys
import time

import yaml

# The root-level link also supports a Python prompt in this checkout.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from go2_perception.vln.calibration import load_calibration
from go2_perception.vln.episodes import file_sha256
from go2_perception.vln.interactive_lidar import LidarRecorder, LidarSource
from go2_perception.vln.sensor_buffers import TapInbox, PacedFeeder
from go2_perception.vln.shadow_live import LiveCameraSource, latency_model
from go2_perception.vln.shadow_sources import EpisodeSource, SyntheticSource

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SensorSettings:
    camera: str = "synthetic"
    lidar: str = "none"
    episode: Path | None = None
    lidar_recording: Path | None = None
    hardware_python: Path | None = None
    unitree_sdk: Path | None = None
    camera_host_ip: str = "192.168.123.222"
    iface: str = "en9"
    calibration: Path = ROOT / "configs/go2_front_camera_calibration.yaml"
    camera_stride: int = 2
    camera_latency: str = "configured"
    camera_inbox: int = 16
    camera_silence_s: float = 1.0
    speed: float = 1.0
    synthetic_seconds: float = 600.0
    synthetic_lidar_option: list[str] = field(default_factory=list)
    record_lidar_continuously: bool = False

    @classmethod
    def from_args(cls, args):
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


class Sensors:
    """The parent owns the streams. Live reception runs in two read-only bridge children.

    Replay uses a paced RGB feeder and an optional independent recorded LiDAR child.
    Sensor times, validity and frame names survive both paths.
    """

    def __init__(self, camera, tap, lidar, recorder, calibration, silence_s):
        self.camera, self.tap, self.lidar = camera, tap, lidar
        self.recorder, self.calibration, self.silence_s = (
            recorder,
            calibration,
            silence_s,
        )
        self._cleanup = ExitStack()
        # The recorder owns a bounded writer even before the first scan.
        if recorder is not None:
            self._cleanup.callback(recorder.close)
        self._started = False
        self._closed = False

    def start(self):
        if self._closed:
            raise RuntimeError("closed sensor session")
        if self._started:
            return
        try:
            for source in (self.camera, self.lidar):
                if source is not None:
                    self._cleanup.callback(source.close)
                    source.start()
            self._started = True
        except BaseException:
            self.stop()
            raise

    def stop(self):
        self._closed = True
        self._cleanup.close()

    def latest_frame(self):
        """Return (RGB uint8 image, source and timestamp metadata), or None before delivery."""
        item = self.tap.frame(self.tap.latest_id)
        return (item[0].copy(), deepcopy(item[1])) if item else None

    def latest_cloud(self):
        """Return (XYZ metres, robot/host times and frame) for one scan, or None."""
        return self.lidar.latest_cloud() if self.lidar is not None else None

    def status(self, now=None):
        """Host monotonic ages, original timestamps, drop counters and source errors."""
        now = time.monotonic() if now is None else now
        camera = self.tap.status(now, self.silence_s)
        camera.update(
            error=getattr(self.camera, "error", None),
            alive=self._started and not self._closed and self.camera.alive,
        )
        return {
            "camera": camera,
            "lidar": self.lidar.snapshot(now) if self.lidar is not None else None,
        }

    def pose(self, now=None):
        """Odometry pose and validity within this robot epoch. No pose is inferred from RGB."""
        lidar = self.status(now)["lidar"] or {}
        return (lidar.get("state") or {}).get(
            "pose", {"valid": False, "reason": "no odometry"}
        )


def open_sensors(
    a: SensorSettings, output: Path, synthetic_source_factory=SyntheticSource
):
    """Construct sources from the application parser options; call start() to receive data."""
    if a.camera not in ("go2", "episode", "synthetic") or a.lidar not in (
        "go2",
        "synthetic",
        "recording",
        "none",
    ):
        raise ValueError("Unknown camera or LiDAR source")
    if (a.camera == "go2" or a.lidar == "go2") and not a.hardware_python:
        raise ValueError("Go2 sources need --hardware-python (hardware environment)")
    if a.camera != "go2" and a.lidar == "go2":
        raise ValueError(
            "Live LiDAR may only accompany the live camera (no live/replay mixing)"
        )
    if a.camera == "go2" and a.lidar in ("synthetic", "recording"):
        raise ValueError(
            "Replay LiDAR may not accompany the live camera (no live/replay mixing)"
        )
    if a.camera == "episode" and not a.episode:
        raise ValueError("--camera episode needs --episode")
    if a.lidar == "recording" and not a.lidar_recording:
        raise ValueError("--lidar recording needs --lidar-recording")
    tap = TapInbox(a.camera_inbox)
    if a.camera == "go2":
        intrinsics, _ = load_calibration(a.calibration)
        if intrinsics.status != "measured":
            raise ValueError("Live Go2 input requires the measured camera calibration")
        latency = None
        if a.camera_latency == "configured":
            sensors = yaml.safe_load(
                (ROOT / "configs/go2_vln_sensors.yaml").read_text()
            )
            latency = latency_model(sensors["timing"]["camera_latency_s"])
        camera = LiveCameraSource(
            python=a.hardware_python,
            intrinsics=intrinsics,
            calibration_path=a.calibration,
            output=output / "camera",
            host_ip=a.camera_host_ip,
            stride=a.camera_stride,
            capacity=a.camera_inbox,
            latency=latency,
        )
        camera.inbox = tap  # same bounded inbox contract + a display ring
        calibration = {
            "path": str(a.calibration.resolve()),
            "sha256": file_sha256(a.calibration),
            "status": "measured",
            "T_base_camera": None,
            "camera_latency": latency,
        }
    else:
        frames = int(a.synthetic_seconds * 8)
        source = (
            synthetic_source_factory(min(4096, frames), fps=8.0)
            if a.camera == "synthetic"
            else EpisodeSource(a.episode)
        )
        camera = PacedFeeder(iter(source), tap, speed=a.speed)
        calibration = (
            {"status": "synthetic", "note": "deterministic 2D fixture"}
            if a.camera == "synthetic"
            else {"episode_manifest": str(a.episode), "sha256": source.episode.sha256}
        )
    lidar = recorder = None
    if a.lidar != "none":
        if a.lidar == "go2":
            command = [
                a.hardware_python,
                "-u",
                ROOT / "scripts/go2_lidar_bridge.py",
                "--iface",
                a.iface,
            ]
        else:
            command = [
                sys.executable,
                "-u",
                ROOT / "scripts/synthetic_lidar_bridge.py",
                *(["--recording", a.lidar_recording] if a.lidar == "recording" else []),
                *[x for opt in a.synthetic_lidar_option for x in opt.split()],
            ]
        sdk_path = (
            (os.pathsep + str(a.unitree_sdk.resolve()))
            if a.lidar == "go2" and a.unitree_sdk
            else ""
        )
        lidar = LidarSource(
            command,
            output / "lidar_bridge",
            env={"PYTHONPATH": str(ROOT / "src") + sdk_path},
        )
        recorder = LidarRecorder(output / "lidar", quota=lambda: True)
        lidar.recorder = recorder
        if a.record_lidar_continuously:
            recorder.start("between_trials")
    return Sensors(camera, tap, lidar, recorder, calibration, a.camera_silence_s)


def default_source_args(args, workspace):
    """Choose compatible RGB and LiDAR sources for the public observation mode."""
    if args.source == "live":
        return [
            "--camera",
            "go2",
            "--lidar",
            "go2",
            "--hardware-python",
            str(workspace / "envs/hw/bin/python"),
            "--unitree-sdk",
            str(workspace / "sdk/unitree_sdk2_python"),
        ]
    if args.source == "replay":
        options = ["--camera", "episode", "--episode", str(args.episode)]
        if args.lidar_recording:
            options += [
                "--lidar",
                "recording",
                "--lidar-recording",
                str(args.lidar_recording),
            ]
        return options
    return ["--camera", "synthetic", "--lidar", "synthetic"]
