"""Build ``Go2VLNObservation`` from a local ``go2-recording-v1`` directory (offline, read-only).

Reference time = midpoint of one ``cloud_base`` scan (robot clock). Geometry accumulates the scans
whose midpoints fall in (t_ref - window, t_ref], mapped through odometry into the reference scan's
base_link; ``cloud_deskewed`` is never used, so nothing is accumulated twice. The camera frame is
the latest one whose *estimated* exposure time is at or before t_ref (causal). RGB timing model:
t_robot = host_recv_first - (host - robot offset) - camera latency, with both terms measured, not
assumed (docs/GO2_SENSOR_TIMING.md); its uncertainty is carried in the stamp.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..adapters.go2.window import PoseBuffer
from . import geometry
from .calibration import load_calibration
from .observation import (HOST_CLOCK, ROBOT_CLOCK, GeometricObservation, Go2VLNObservation,
                          RobotState, Stamp, VisualObservation)

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "go2_vln_sensors.yaml"


def _jsonl(path):
    return [json.loads(line) for line in open(path)]


class Go2RecordingReplay:
    def __init__(self, root, config=DEFAULT_CONFIG, calibration=None, allow_synthetic=False):
        self.root = Path(root)
        cfg = yaml.safe_load(Path(config).read_text())
        self.acc, self.timing = cfg["lidar_accumulation"], cfg["timing"]
        self.geo = {**geometry.DEFAULTS, **cfg.get("geometry", {})}
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("schema") != "go2-recording-v1":
            raise ValueError(f"{root}: not a go2-recording-v1 recording")
        topic = self.acc["source_topic"].replace("rt/", "").replace("/", "_")
        self.lidar_dir = self.root / "lidar" / topic
        self.scans = _jsonl(self.lidar_dir / "index.jsonl")
        for s in self.scans:
            tpath = self.lidar_dir / s["file"].replace(".npy", "_time.npy")
            span = float(np.load(tpath).max()) if s.get("has_time") and tpath.exists() else 0.0
            s["t_mid"] = s["stamp"] + 0.5 * span
        self.odom = _jsonl(self.root / "state" / "utlidar_robot_odom.jsonl")
        self.poses = PoseBuffer(max_age_s=1e12, max_gap_s=0.1)
        for r in self.odom:
            self.poses.add(r["stamp"], r["position"], r["q_xyzw"])
        self._odom_t = np.array([r["stamp"] for r in self.odom])
        self.rgb = _jsonl(self.root / "rgb" / "index.jsonl") if (self.root / "rgb").exists() else []
        self.offset = float(min(r["host_recv"] - r["stamp"] for r in self.odom))
        lat = self.timing["camera_latency_s"]
        self.camera_latency, self.camera_latency_sigma = lat["value"], lat["sigma"]
        self.rgb_t_robot = np.array([r["host_recv_first"] - self.offset - self.camera_latency
                                     for r in self.rgb])
        self.intrinsics = self.extrinsics = None
        if calibration is not None:
            self.intrinsics, self.extrinsics = load_calibration(calibration)
            if not allow_synthetic and self.intrinsics.status != "measured":
                raise ValueError("Synthetic calibration on a real recording needs allow_synthetic=True")
        self.allow_synthetic = allow_synthetic

    def __len__(self):
        return len(self.scans)

    def reference_times(self):
        return [s["t_mid"] for s in self.scans]

    def _accumulate(self, t_ref, window_s):
        ref = self.poses.lookup(t_ref)
        if ref is None:
            return None, 0
        inv = np.linalg.inv(ref)
        parts = []
        for s in self.scans:
            if not (t_ref - window_s < s["t_mid"] <= t_ref + 1e-9):
                continue
            pose = self.poses.lookup(s["t_mid"])
            if pose is None:
                continue  # refuse uncompensated merging
            rel = inv @ pose
            xyz = np.load(self.lidar_dir / s["file"])[:, :3].astype(float)
            parts.append(xyz @ rel[:3, :3].T + rel[:3, 3])
        pts = np.concatenate(parts).astype(np.float32) if parts else np.empty((0, 3), np.float32)
        return pts, len(parts)

    def _visual(self, t_ref, history):
        if not self.rgb:
            return None, None
        idx = np.searchsorted(self.rgb_t_robot, t_ref, side="right") - 1
        if idx < 0:
            return None, None
        sigma = float(np.hypot(self.camera_latency_sigma, self.timing.get("offset_sigma_s", 0.002)))

        def frame(i):
            row = self.rgb[i]
            bgr = cv2.imread(str(self.root / "rgb" / row["file"]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Undecodable RGB frame {row['file']}")
            v = VisualObservation(
                image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), frame_id=row["frame_id"],
                stamp_host=Stamp(row["host_recv_first"], HOST_CLOCK, "host_receipt"),
                stamp_robot=Stamp(float(self.rgb_t_robot[i]), ROBOT_CLOCK, "exposure_estimated", sigma),
                rectified=False, source=f"rtp_h264_decoded:{Path(row['file']).suffix[1:]}")
            if self.intrinsics is not None:
                v.intrinsics_status, v.K, v.D = self.intrinsics.status, self.intrinsics.K, self.intrinsics.D
                v.distortion_model = self.intrinsics.model
                v.intrinsics_size = (self.intrinsics.width, self.intrinsics.height)
            return v

        current = frame(idx)
        current.history = [frame(i) for i in range(max(0, idx - history), idx)]
        return current, float(t_ref - self.rgb_t_robot[idx])

    def observation(self, k, history=0, instruction=None, include_display=True):
        scan = self.scans[k]
        t_ref = scan["t_mid"]
        warnings = []
        pts, used = self._accumulate(t_ref, self.acc["obstacle_window_s"])
        geo = None
        if pts is None:
            warnings.append("no odometry bracketing the reference scan: geometry unavailable")
        else:
            if used * 0.065 < 0.8 * self.acc["obstacle_window_s"]:
                warnings.append(f"obstacle window holds {used} scans (recording start or pose gaps)")
            grid, elev, origin, band = geometry.occupancy(pts, **self.geo)
            near, clear = geometry.sectors(pts, **self.geo)
            disp = self._accumulate(t_ref, self.acc["display_window_s"])[0] if include_display else None
            geo = GeometricObservation(
                frame_id="base_link", stamp=Stamp(t_ref, ROBOT_CLOCK, "scan_midpoint"),
                window_s=self.acc["obstacle_window_s"], frames_used=used, points=pts, grid=grid,
                cell_m=self.geo["cell_m"], origin_xy=origin, obstacle_z_band=band,
                elevation_max=elev, sector_deg=self.geo["sector_deg"], sector_range=near,
                sector_clear_to=clear, blind_radius_m=self.geo["blind_radius_m"],
                display_points=disp,
                display_window_s=self.acc["display_window_s"] if include_display else None,
                source_topic=self.acc["source_topic"],
                notes=["floor height assumed for a standing robot", "grid FREE = floor seen, not traversable"])
            geo.projected_depth = self._projected_depth(pts, warnings)
        visual, age = self._visual(t_ref, history)
        if visual is None:
            warnings.append("no camera frame at or before the reference time")
        elif age > 0.15:
            warnings.append(f"camera frame is {age:.3f} s older than geometry")
        robot = self._robot(t_ref, warnings)
        calib = {
            "intrinsics_status": "unknown" if self.intrinsics is None else self.intrinsics.status,
            "extrinsics_status": "unknown" if self.extrinsics is None else self.extrinsics.status,
            "allow_synthetic": self.allow_synthetic,
            "T_base_lidar_status": "firmware",
            "camera_latency_s": self.camera_latency, "camera_latency_sigma_s": self.camera_latency_sigma,
            "camera_latency_source": self.timing["camera_latency_s"].get("source"),
        }
        return Go2VLNObservation(
            visual=visual, geometric=geo, robot=robot, instruction=instruction,
            clock_epoch=f"go2_boot:host_minus_robot={self.offset:.1f}",
            host_minus_robot_s=self.offset, calibration=calib, warnings=warnings)

    def _projected_depth(self, pts, warnings):
        if self.intrinsics is None:
            warnings.append("projected depth unavailable: camera calibration missing")
            return None
        if self.extrinsics is None:
            warnings.append("projected depth unavailable: LiDAR-camera extrinsics (T_base_camera) missing")
            return None
        from .calibration import project_points
        cam = self.extrinsics.inverse().apply(pts.astype(float))
        uv, valid = project_points(cam, self.intrinsics)
        depth = np.full((self.intrinsics.height, self.intrinsics.width), np.nan, np.float32)
        u = np.round(uv[valid, 0]).astype(int)
        v = np.round(uv[valid, 1]).astype(int)
        z = cam[valid, 2].astype(np.float32)
        order = np.argsort(-z)  # write far first so the nearest return wins
        depth[v[order], u[order]] = z[order]
        return depth

    def _robot(self, t, warnings):
        T = self.poses.lookup(t)
        i = int(np.clip(np.searchsorted(self._odom_t, t), 0, len(self.odom) - 1))
        if T is None:
            warnings.append("robot pose unknown at reference time")
            return RobotState(Stamp(t, ROBOT_CLOCK, "message_stamp"), None)
        r = self.odom[i]
        return RobotState(
            stamp=Stamp(t, ROBOT_CLOCK, "message_stamp"), T_odom_base=T,
            linear_velocity=np.asarray(r.get("linear"), float) if r.get("linear") else None,
            angular_velocity=np.asarray(r.get("angular"), float) if r.get("angular") else None,
            velocity_frame="base_link (nav_msgs child-frame convention; unverified on Go2)")
