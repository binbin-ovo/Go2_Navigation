from collections import deque
from copy import deepcopy
import numpy as np
from go2_perception.core.tracking_types import TrackOutput
from go2_perception.localization.ego_motion import normalize_detections
from .config import TrackerConfig
from .kalman_filter import ConstantVelocityFilter
from .association import associate, deduplicate, box_iou
from .motion_classifier import classify_motion
from .fragments import recover_fragments
from .trajectory import predict_trajectory


class _Track:
    def __init__(self, identifier, d, t, c):
        self.track_id = identifier
        self.filter = ConstantVelocityFilter(
            d.position[:2],
            d.measurement_covariance[:2, :2],
            c.initial_velocity_variance,
            c.process_noise,
            c.min_dt,
            c.max_prediction_interval,
        )
        self.created = self.last_observation = t
        self.hits = self.hit_streak = 1
        self.hit_times = deque([t])
        self.misses = 0
        self.confirmed = c.min_hits <= 1
        self.status = "Confirmed" if self.confirmed else "Tentative"
        self.motion = "UNCERTAIN"
        self.stable_motion = "UNCERTAIN"
        self.motion_candidate = "UNCERTAIN"
        self.motion_candidate_since = t
        self.history = deque(maxlen=c.history_size)
        self.observe_properties(d)

    def observe_properties(self, d):
        self.z = float(d.position[2])
        if d.class_name:
            self.class_name = d.class_name
        elif not hasattr(self, "class_name"):
            self.class_name = None
        self.confidence = d.confidence
        self.box_offset = (
            None
            if d.bbox_min is None
            else (d.bbox_min - d.position, d.bbox_max - d.position)
        )

    def box(self):
        if self.box_offset is None:
            return None, None
        p = np.r_[self.filter.state[:2], self.z]
        return p + self.box_offset[0], p + self.box_offset[1]


class MultiObjectTracker:
    def __init__(self, config=None):
        self.config = (
            config
            if isinstance(config, TrackerConfig)
            else TrackerConfig(**(config or {}))
        )
        self._tracks = []
        self._next_id = 1
        self._timestamp = None
        self._clock = self.config.clock_domain
        self.deleted_ids = []
        self.suppressed_duplicates = 0
        self.created_ids = []
        self.reactivated_ids = []
        self.fragment_groups = []

    def reset(self):
        """Drop every track before an unobservable interval. IDs keep increasing, never reused."""
        ended = [t.track_id for t in self._tracks]
        self._tracks = []  # the timestamp is kept so out-of-order updates still fail
        return ended

    def update(
        self,
        detections,
        timestamp,
        robot_pose=None,
        *,
        frame_id=None,
        clock_domain=None,
    ):
        """Reject invalid updates atomically. Empty frames are real misses, not missing poses."""
        c = self.config
        detections = list(detections)
        if not np.isfinite(timestamp):
            raise ValueError("Invalid timestamp")
        if self._timestamp is not None and timestamp - self._timestamp < c.min_dt:
            raise ValueError("Duplicate, out-of-order or too-close update timestamp")
        source = frame_id or (
            detections[0].frame_id
            if detections
            else (robot_pose.child_frame if robot_pose else c.sensor_frame)
        )
        domain = clock_domain or (
            detections[0].clock_domain
            if detections
            else (robot_pose.clock_domain if robot_pose else self._clock)
        )
        if not domain or (self._clock is not None and domain != self._clock):
            raise ValueError("Missing or changed clock domain")
        ids = [d.detection_id for d in detections if d.detection_id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate source detection ID within frame")
        normalized, target = normalize_detections(
            detections, timestamp, robot_pose, c, source, domain
        )
        observations = deduplicate(normalized, c.duplicate_distance)
        self.suppressed_duplicates = len(normalized) - len(observations)
        self.deleted_ids = []
        self.created_ids = []
        self.reactivated_ids = []
        active = []
        for t in self._tracks:
            if timestamp - t.last_observation > c.max_unobserved_s:
                t.status = "Deleted"
                self.deleted_ids.append(t.track_id)
            else:
                active.append(t)
        self._tracks = active
        if self._timestamp is not None:
            for t in self._tracks:
                t.filter.predict(timestamp - self._timestamp)
        observations, self.fragment_groups = recover_fragments(
            self._tracks, observations, c
        )
        matches, missed, new = associate(self._tracks, observations, c)
        for i, j in matches:
            t, d = self._tracks[i], observations[j]
            if t.status == "Lost" and t.confirmed:
                self.reactivated_ids.append(t.track_id)
            t.filter.update(d.position[:2], d.measurement_covariance[:2, :2])
            t.observe_properties(d)
            t.hits += 1
            t.hit_streak += 1
            t.misses = 0
            t.last_observation = timestamp
            t.hit_times.append(timestamp)
            while t.hit_times and timestamp - t.hit_times[0] > c.confirmation_window_s:
                t.hit_times.popleft()
            t.confirmed = t.confirmed or len(t.hit_times) >= c.min_hits
            t.status = "Confirmed" if t.confirmed else "Tentative"
        for i in missed:
            t = self._tracks[i]
            t.misses += 1
            t.hit_streak = 0
            if t.misses > c.max_misses and not (
                c.confirmed_time_expiry and t.confirmed
            ):
                t.status = "Deleted"
                self.deleted_ids.append(t.track_id)
            elif t.misses >= c.lost_after_misses:
                t.status = "Lost"
        self._tracks = [t for t in self._tracks if t.status != "Deleted"]
        for j in new:
            d = observations[j]
            # A rejected noisy measurement near an unobserved confirmed box must
            # not immediately create a second persistent ID for the same cluster.
            if any(
                t.confirmed
                and t.misses
                and (
                    not t.class_name or not d.class_name or t.class_name == d.class_name
                )
                and np.linalg.norm(np.r_[t.filter.state[:2], t.z] - d.position)
                < c.birth_suppression_distance
                and box_iou(*t.box(), d.bbox_min, d.bbox_max) > 0.3
                for t in self._tracks
            ):
                self.suppressed_duplicates += 1
                continue
            self._tracks.append(_Track(self._next_id, observations[j], timestamp, c))
            self.created_ids.append(self._next_id)
            self._next_id += 1
        self._timestamp, self._clock = timestamp, domain
        output = []
        for t in self._tracks:
            age = timestamp - t.created
            t.motion = (
                classify_motion(
                    t.filter.state[2:],
                    t.filter.P[2:, 2:],
                    t.hits,
                    age,
                    t.misses,
                    c,
                    t.stable_motion if c.motion_dwell_s else t.motion,
                )
                if t.confirmed
                else "UNCERTAIN"
            )
            if c.motion_dwell_s:
                candidate = t.motion
                if candidate != t.motion_candidate or t.misses:
                    t.motion_candidate_since = timestamp
                    t.motion_candidate = candidate
                if (
                    candidate == "UNCERTAIN"
                    or timestamp - t.motion_candidate_since < c.motion_dwell_s
                ):
                    t.motion = "UNCERTAIN"
                else:
                    t.stable_motion = candidate
            p = np.r_[t.filter.state[:2], t.z]
            t.history.append(
                dict(timestamp=timestamp, position=p.tolist(), observed=t.misses == 0)
            )
            trajectory = []
            if t.status == "Confirmed" and not t.misses and t.motion != "UNCERTAIN":
                trajectory = predict_trajectory(
                    t.filter.state,
                    t.filter.P,
                    t.z,
                    timestamp,
                    c.trajectory_horizon_s,
                    c.trajectory_step_s,
                    c.process_noise,
                )
            low, high = t.box()
            output.append(
                TrackOutput(
                    t.track_id,
                    timestamp,
                    target,
                    domain,
                    "odometry" if c.mode == "odom" else "fixed_sensor_relative",
                    p.copy(),
                    t.filter.state[2:].copy(),
                    t.filter.P.copy(),
                    low,
                    high,
                    t.class_name,
                    t.confidence,
                    t.status,
                    t.motion,
                    t.last_observation,
                    age,
                    t.hits,
                    t.misses,
                    deepcopy(list(t.history)),
                    trajectory,
                )
            )
        return output
