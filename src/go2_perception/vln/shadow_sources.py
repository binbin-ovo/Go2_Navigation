"""Local observation sources. The payload is the existing Go2VLNObservation.

Delivery adds application provenance/availability, never a new sensor schema.
No receiver, socket, SDK or robot client is constructed here.
"""

from dataclasses import dataclass
from queue import Full, Queue
import copy
import math
import time

import cv2
import numpy as np

from .calibration import load_calibration
from .episodes import file_sha256, load_episode, local_path
from .observation import Go2VLNObservation, Stamp, VisualObservation


@dataclass(frozen=True)
class Delivery:
    observation: Go2VLNObservation
    source_id: str
    frame_id: str
    available_s: float  # in the visual exposure stamp's domain; receipt if no exposure
    provenance: dict
    received_monotonic_s: float | None = None
    exposure_monotonic_s: float | None = None
    monotonic_mapping: str | None = None

    def __post_init__(self):
        if (
            not self.source_id
            or not self.frame_id
            or not math.isfinite(self.available_s)
        ):
            raise ValueError("Source/frame identity and finite availability required")
        if self.exposure_monotonic_s is not None and (
            not self.monotonic_mapping
            or self.received_monotonic_s is None
            or not math.isfinite(self.exposure_monotonic_s)
            or self.exposure_monotonic_s > self.received_monotonic_s
        ):
            raise ValueError(
                "Exposure monotonic time requires a documented causal mapping"
            )


class EpisodeSource:
    """Read the model_input manifest only; annotations and action refs stay separate."""

    def __init__(self, path):
        self.episode = load_episode(path)
        self.data = self.episode.data
        self.intr, _ = load_calibration(
            local_path(self.episode.root, self.data["calibration"]["path"])
        )

    def __iter__(self):
        d, intr = self.data, self.intr
        for frame in d["frames"]:
            visual = None
            if frame["image"] is not None:
                path = local_path(self.episode.root, frame["image"]["path"])
                if file_sha256(path) != frame["image"]["sha256"]:
                    raise ValueError("Episode image changed since validation")
                bgr = cv2.imread(str(path))
                if bgr is None or [bgr.shape[1], bgr.shape[0]] != frame["size_wh"]:
                    raise ValueError("Missing/undecodable or wrong-size episode image")
                visual = VisualObservation(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    intr.camera_frame,
                    Stamp(frame["host_receipt_s"], "host_receipt", "host_receipt"),
                    Stamp(
                        frame["observation_s"],
                        d["clock"]["domain"],
                        d["clock"]["observation_semantics"],
                        d["clock"]["uncertainty_s"],
                    ),
                    rectified=d["preprocessing"]["source_rectified"],
                    intrinsics_status=intr.status,
                    K=intr.K,
                    D=intr.D,
                    distortion_model=intr.model,
                    intrinsics_size=(intr.width, intr.height),
                    source=str(path),
                )
            obs = Go2VLNObservation(
                visual,
                None,
                None,
                instruction=d["instruction"],
                clock_epoch=d["clock"]["epoch"],
                host_minus_robot_s=d["clock"]["host_minus_robot_s"],
                calibration={"episode_calibration": d["calibration"]},
                warnings=[] if visual else [frame["missing_reason"]],
            )
            yield Delivery(
                obs,
                d["episode_id"],
                frame["frame_id"],
                frame["available_s"],
                {
                    "manifest": str(self.episode.path),
                    "manifest_sha256": self.episode.sha256,
                    "original_instruction": d["instruction"],
                    "source_frame": frame,
                    "clock": d["clock"],
                    "recording": d["recording"],
                    "declared_gaps": [
                        g for g in d["gaps"] if g["start_s"] <= frame["available_s"]
                    ],
                    "annotations_consumed": False,
                },
            )


class SyntheticSource:
    """Deterministic 2D camera fixture, not a navigation or model accuracy test."""

    def __init__(self, frames=96, fps=16.0):
        if not 1 <= frames <= 4096 or not math.isfinite(fps) or not 4 <= fps <= 60:
            raise ValueError("Synthetic frames 1..4096 and FPS 4..60 required")
        self.frames, self.fps = frames, fps

    def __iter__(self):
        for i in range(self.frames):
            t = 100.0 + i / self.fps
            image = np.full((256, 448, 3), (195, 207, 220), np.uint8)
            image[170:] = (132, 133, 133)
            shift = int(18 * np.sin(i / 20))
            cv2.rectangle(
                image, (160 + shift, 40), (280 + shift, 205), (104, 53, 40), 12
            )
            cv2.putText(
                image,
                f"SYNTHETIC {i:04d}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
            )
            v = VisualObservation(
                image,
                "synthetic_optical",
                Stamp(t + 1000.02, "host_receipt", "host_receipt", 0),
                Stamp(t, "synthetic_clock", "exposure", 0),
                rectified=True,
                intrinsics_status="synthetic",
                K=np.array([[300.0, 0, 224], [0, 300, 128], [0, 0, 1]]),
                D=np.zeros(5),
                distortion_model="pinhole_radtan",
                intrinsics_size=(448, 256),
                source="deterministic_2d_fixture",
            )
            yield Delivery(
                Go2VLNObservation(v, None, None, clock_epoch="synthetic_epoch_1"),
                "synthetic_shadow",
                f"f{i:05d}",
                t + 0.02,
                {"kind": "synthetic", "accuracy_claim": False},
            )


class ObservationInbox:
    """Future receiver-side consumer boundary; does not open a connection.

    offer() copies owned buffers, returns False on overflow (never silently drops).
    The caller must report loss via the session's invalidate_history(), or begin a
    new clock epoch/session. Optional pose/LiDAR stay in the existing observation.
    No host-wall/robot-clock conversion is invented here.
    """

    def __init__(self, capacity=2):
        if not 1 <= capacity <= 16:
            raise ValueError("Inbox capacity must be 1..16")
        self.queue = Queue(maxsize=capacity)

    def offer(
        self,
        observation,
        *,
        source_id,
        frame_id,
        available_s,
        provenance=None,
        received_monotonic_s=None,
        exposure_monotonic_s=None,
        monotonic_mapping=None,
    ):
        delivery = Delivery(
            copy.deepcopy(observation),
            source_id,
            frame_id,
            available_s,
            copy.deepcopy(provenance or {}),
            time.monotonic() if received_monotonic_s is None else received_monotonic_s,
            exposure_monotonic_s,
            monotonic_mapping,
        )
        try:
            self.queue.put_nowait(delivery)
            return True
        except Full:
            return False

    def get_nowait(self):
        return self.queue.get_nowait()
