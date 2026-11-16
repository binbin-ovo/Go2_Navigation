"""Live read-only Go2 camera -> existing Go2VLNObservation -> ObservationInbox.

The camera receiver runs in the hardware environment as an owned child
(``scripts/go2_camera_bridge.py``); this module only parses its stdout. It opens no
socket, DDS participant or robot client, and sends nothing to the robot.

Clocks: the bridge reports host wall receipt times plus a same-host wall-minus-
monotonic sample. Visual stamps stay in the ``host_receipt`` clock (no robot clock is
subscribed, so ``stamp_robot`` is None and pose/LiDAR are None, never identity).
Estimated exposure exists only as an explicitly labelled host-monotonic mapping:
receipt minus the configured, once-measured camera latency. Without it, exposure age
is unknown. A wall-clock step or a new bridge process starts a new clock epoch.
"""

from collections import deque
import json
import math
import os
from pathlib import Path
from queue import Empty
import subprocess
import threading
import time

import numpy as np

from .episodes import file_sha256
from .observation import HOST_CLOCK, Go2VLNObservation, Stamp, VisualObservation
from .shadow_sources import ObservationInbox

LIVE_SOURCE_ID = "go2_front_camera_rtp_live"


class FrameError(ValueError):
    pass


def latency_model(config):
    """Configured camera latency from go2_vln_sensors.yaml, or None (exposure unknown)."""
    if config is None:
        return None
    value, sigma = float(config["value"]), float(config["sigma"])
    if not (math.isfinite(value) and math.isfinite(sigma) and value >= 0 and sigma >= 0):
        raise ValueError("Finite non-negative camera latency and sigma required")
    return {"value_s": value, "sigma_s": sigma, "source": config.get("source")}


def frame_observation(header, payload, intrinsics, calibration, epoch, latency):
    """Convert one bridge frame. Returns (observation, ObservationInbox.offer kwargs).

    Raw, distorted RGB with the measured fisheye calibration; rectification happens
    once, later, in the existing calibrated_view. No pose, LiDAR or T_base_camera.
    """
    shape = tuple(header.get("shape", ()))
    if (
        header.get("channel_order") != "RGB"
        or header.get("dtype") != "uint8"
        or shape != (intrinsics.height, intrinsics.width, 3)
        or len(payload) != header.get("nbytes")
        or len(payload) != int(np.prod(shape))
    ):
        raise FrameError(f"Frame {shape} does not match calibrated RGB {intrinsics.width}x{intrinsics.height}")
    first, last, decoded = (
        header["host_recv_first_wall_s"],
        header["host_recv_last_wall_s"],
        header["decoded_wall_s"],
    )
    offset, conversion = header["wall_minus_monotonic_s"], header["conversion_uncertainty_s"]
    if not all(math.isfinite(x) for x in (first, last, decoded, offset, conversion)) or not (
        first <= last <= decoded
    ):
        raise FrameError("Non-finite or non-causal receipt/decoding times")
    image = np.frombuffer(payload, np.uint8).reshape(shape).copy()
    visual = VisualObservation(
        image,
        intrinsics.camera_frame,
        Stamp(first, HOST_CLOCK, "host_receipt", None),
        None,  # no robot clock in an RGB-only bridge; never host time relabelled
        rectified=False,
        intrinsics_status=intrinsics.status,
        K=intrinsics.K,
        D=intrinsics.D,
        distortion_model=intrinsics.model,
        intrinsics_size=(intrinsics.width, intrinsics.height),
        source="go2_rtp_h264_live_decoded",
    )
    warnings = [
        "live RGB-only bridge: pose and LiDAR not subscribed (None, not identity)",
        "visual stamp is first-RTP-packet host receipt; exposure is not measured per frame",
    ]
    obs = Go2VLNObservation(
        visual,
        None,
        None,
        clock_epoch=epoch,
        host_minus_robot_s=None,
        calibration={**calibration, "camera_latency": latency},
        warnings=warnings,
    )
    received_mono = first - offset
    exposure_mono = mapping = None
    if latency is not None:
        exposure_mono = received_mono - latency["value_s"]
        mapping = (
            "host_monotonic = first_rtp_receipt_wall - bridge(wall-monotonic) "
            f"- configured camera latency {latency['value_s']:.3f} s "
            f"(sigma {latency['sigma_s']:.3f} s, {latency['source']}); single-boot "
            "measurement relative to odometry stamps, excludes unknown odometry "
            f"transport latency; conversion sample +/-{conversion:.2e} s"
        )
    frame_id = f"{header['bridge_id']}:{header['seq']:08d}:rtp{header['rtp_timestamp']}"
    provenance = {
        "kind": "go2_live_rtp_bridge",
        "bridge_id": header["bridge_id"],
        "seq": header["seq"],
        "decoded_index": header["decoded_index"],
        "rtp_timestamp": header["rtp_timestamp"],
        "complete_access_unit": header["complete"],
        "host_recv_first_wall_s": first,
        "host_recv_last_wall_s": last,
        "decoded_wall_s": decoded,
        "wall_minus_monotonic_s": offset,
        "conversion_uncertainty_s": conversion,
        "sensor_receipt_monotonic_s": received_mono,
        "estimated_exposure_monotonic_s": exposure_mono,
        "decoded_monotonic_s": header["decoded_monotonic_s"],
        "bridge_write_monotonic_s": header.get("write_monotonic_s"),
        "parent_read_monotonic_s": header.get("parent_read_monotonic_s"),
        "channel_order": "RGB",
        "rectified": False,
        "pose": None,
        "lidar": None,
        "T_base_camera": None,
    }
    offer = {
        "source_id": LIVE_SOURCE_ID,
        "frame_id": frame_id,
        "available_s": decoded,  # host_receipt wall domain, same as the visual stamp
        "provenance": provenance,
        "received_monotonic_s": received_mono,
        "exposure_monotonic_s": exposure_mono,
        "monotonic_mapping": mapping,
    }
    return obs, offer


class LiveCameraSource:
    """Owned read-only bridge child plus a bounded inbox. Never a robot client."""

    def __init__(
        self,
        *,
        python,
        intrinsics,
        calibration_path,
        output,
        host_ip="192.168.123.222",
        stride=2,
        capacity=2,
        latency=None,
        clock_step_tolerance_s=0.02,
        command=None,
        monotonic=time.monotonic,
    ):
        root = Path(__file__).resolve().parents[3]
        self.intrinsics, self.latency = intrinsics, latency
        self.calibration = {
            "intrinsics_status": intrinsics.status,
            "calibration_path": str(Path(calibration_path).resolve()),
            "calibration_sha256": file_sha256(calibration_path),
            "extrinsics_status": "unknown",
            "T_base_camera": None,
        }
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.inbox = ObservationInbox(capacity)
        self.tolerance = clock_step_tolerance_s
        self.monotonic = monotonic
        self.command = command or [
            str(python),
            "-u",
            str(root / "scripts/go2_camera_bridge.py"),
            "--host-ip",
            host_ip,
            "--stride",
            str(stride),
        ]
        self.root = root
        self.lock = threading.Lock()
        self.events = deque(maxlen=256)
        self.counts = {
            "frames_read": 0,
            "offered": 0,
            "inbox_overflow": 0,
            "bridge_seq_gaps": 0,
            "frames_missing_in_seq_gaps": 0,
            "rejected_frames": 0,
            "clock_epoch_changes": 0,
            "events_lost": 0,
        }
        self.hello, self.stats, self.bye = None, None, None
        self.base_offset, self.generation, self.last_seq = None, 0, None
        self.last_frame_monotonic = None
        self.process = self.reader = self.log = None
        self.error = None
        self.closed = False

    def epoch(self):
        return f"host_receipt_wall:bridge={self.hello['bridge_id']}:g{self.generation}"

    def _event(self, kind, **fields):
        with self.lock:
            if len(self.events) == self.events.maxlen:
                self.counts["events_lost"] += 1
            self.events.append({"event": "camera_" + kind, **fields})

    def drain_events(self):
        with self.lock:
            out = list(self.events)
            self.events.clear()
        return out

    def start(self):
        self.log = (self.output / "bridge.log").open("w")
        # Only the receiver package is needed; the Unitree SDK is not imported.
        env = {**os.environ, "PYTHONPATH": str(self.root / "src")}
        self.spawn_monotonic = self.monotonic()
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            env=env,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            stream = self.process.stdout
            while True:
                line = stream.readline()
                if not line:
                    break
                read_mono = self.monotonic()
                header = json.loads(line)
                kind = header.get("type")
                if kind == "frame":
                    payload = stream.read(header["nbytes"])
                    header["parent_read_monotonic_s"] = read_mono
                    self._frame(header, payload)
                elif kind == "hello":
                    # Same-host monotonic domain check: the child's hello time must lie
                    # between our spawn and our read. Otherwise refuse all frames.
                    if not self.spawn_monotonic <= header["monotonic_s"] <= read_mono:
                        raise RuntimeError("Bridge monotonic clock is not this host's domain")
                    self.hello = header
                    self._event("bridge_hello", hello=header)
                elif kind in ("stats", "bye"):
                    with self.lock:
                        self.stats = header
                        if kind == "bye":
                            self.bye = header
                    self._event("bridge_" + kind, stats=header)
        except Exception as error:  # surfaced to the event loop, never silent
            self.error = repr(error)
            self._event("bridge_error", error=self.error)

    def _frame(self, header, payload):
        if self.closed:
            return
        if self.hello is None:
            raise RuntimeError("Frame before bridge hello")
        self.counts["frames_read"] += 1
        seq = header["seq"]
        if self.last_seq is not None and seq != self.last_seq + 1:
            self.counts["bridge_seq_gaps"] += 1
            self.counts["frames_missing_in_seq_gaps"] += seq - self.last_seq - 1
            self._event("bridge_frames_dropped", after_seq=self.last_seq, seq=seq)
        self.last_seq = seq
        offset = header.get("wall_minus_monotonic_s")
        if self.base_offset is None:
            self.base_offset = offset
        elif isinstance(offset, (int, float)) and abs(offset - self.base_offset) > self.tolerance:
            # Host wall clock stepped/slewed: wall-receipt stamps are no longer comparable.
            self.generation += 1
            self.counts["clock_epoch_changes"] += 1
            self._event(
                "clock_epoch_changed",
                old_offset=self.base_offset,
                new_offset=offset,
                epoch=f"g{self.generation}",
            )
            self.base_offset = offset
        try:
            obs, offer = frame_observation(
                header, payload, self.intrinsics, self.calibration, self.epoch(), self.latency
            )
        except (FrameError, KeyError, TypeError) as error:
            self.counts["rejected_frames"] += 1
            self._event("frame_rejected", seq=seq, error=str(error))
            return
        offer["provenance"]["inbox_offer_monotonic_s"] = self.monotonic()
        if self.inbox.offer(obs, **offer):
            self.counts["offered"] += 1
            self.last_frame_monotonic = offer["received_monotonic_s"]
        else:
            self.counts["inbox_overflow"] += 1
            self._event("inbox_overflow", frame_id=offer["frame_id"])

    def get_nowait(self):
        return self.inbox.get_nowait()

    @property
    def alive(self):
        return self.process is not None and self.process.poll() is None

    def status(self, now):
        with self.lock:
            stats = dict(self.stats) if self.stats else None
        last = self.last_frame_monotonic
        return {
            "access": "read_only_multicast_bridge_no_robot_commands",
            "bridge_alive": self.alive,
            "bridge_error": self.error,
            "hello": self.hello,
            "clock_epoch": None if self.hello is None else self.epoch(),
            "counts": dict(self.counts),
            "bridge_stats": stats,
            "host_since_last_sensor_receipt_s": None if last is None else now - last,
            "camera_latency_model": self.latency,
        }

    def close(self, timeout=3.0):
        """Close the bridge: EOF on stdin, then terminate/kill. Sends nothing to the robot."""
        if self.closed:
            return
        self.closed = True
        if self.process is None:
            return
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=timeout)
        self.reader.join(timeout=timeout)
        self.process.stdout.close()
        self.log.close()
        while True:
            try:
                self.inbox.get_nowait()
            except Empty:
                break
