"""Geometry-supported object estimates; continuity IDs are metadata, never suppression masks."""

from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment


class SceneObjects:
    def __init__(
        self,
        ttl_s=0.6,
        association_m=0.6,
        position_alpha=0.35,
        dimension_alpha=0.2,
        motion_window_s=0.6,
        min_motion_span_s=0.4,
        moving_speed=0.25,
        release_speed=0.12,
        stationary_speed=0.16,
        min_displacement=0.18,
        activation_frames=3,
        release_frames=4,
        measurement_floor_m=0.025,
    ):
        if (
            min(
                ttl_s,
                association_m,
                motion_window_s,
                min_motion_span_s,
                moving_speed,
                release_speed,
                stationary_speed,
                min_displacement,
                activation_frames,
                release_frames,
                measurement_floor_m,
            )
            <= 0
            or not 0 < position_alpha <= 1
            or not 0 < dimension_alpha <= 1
        ):
            raise ValueError("Invalid object configuration")
        self.settings = locals().copy()
        self.settings.pop("self")
        self.objects = {}
        self.next_id = 1

    def clear(self):
        self.objects.clear()

    def update(self, proposals, t):
        s = self.settings
        for k, o in list(self.objects.items()):
            o["observed_box"] = None
            if t - o["last"] > s["ttl_s"]:
                del self.objects[k]
        ids = list(self.objects)
        matched = set()
        used = set()
        if ids and proposals:
            cost = np.full((len(ids), len(proposals)), 1e6)
            for i, identity in enumerate(ids):
                o = self.objects[identity]
                pred = (
                    o["center"] + o["velocity"] * min(t - o["last"], 0.2)
                    if o["state"] == "MOVING"
                    else o["center"]
                )
                for j, p in enumerate(proposals):
                    distance = np.linalg.norm(pred[:2] - p["center"][:2])
                    ratio = np.maximum(o["dimensions"], p["dimensions"]) / np.maximum(
                        np.minimum(o["dimensions"], p["dimensions"]), 0.1
                    )
                    if distance <= s["association_m"] and ratio[:2].max() < 3.5:
                        cost[i, j] = distance + 0.05 * np.linalg.norm(
                            o["dimensions"] - p["dimensions"]
                        )
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e5:
                    matched.add(ids[i])
                    used.add(j)
                    self._observe(self.objects[ids[i]], proposals[j], t)
        for j, p in enumerate(proposals):
            if j in used:
                continue
            o = dict(
                center=p["center"].copy(),
                dimensions=p["dimensions"].copy(),
                first=t,
                last=t,
                hits=0,
                state="UNCERTAIN",
                candidate=None,
                streak=0,
                velocity=np.zeros(3),
                cov=np.eye(3),
                history=deque(),
                observed_box=None,
            )
            self.objects[self.next_id] = o
            self.next_id += 1
            self._observe(o, p, t)
        for k, o in self.objects.items():
            if o["observed_box"] is None:
                # A non-repetitive scan may miss a visible object. Do not turn
                # consecutive *observations* into consecutive scan requirements.
                # snapshot() still reports UNCERTAIN for every stale box.
                if t - o["last"] > s["min_motion_span_s"]:
                    o["state"] = "UNCERTAIN"
                    o["candidate"] = None
                    o["streak"] = 0

    def _observe(self, o, p, t):
        s = self.settings
        prev = o["history"][-1][1] if o["history"] else p["center"]
        shift = np.linalg.norm(p["center"] - prev)
        shape_change = np.linalg.norm(p["dimensions"] - o["dimensions"])
        o["history"].append((t, p["center"].copy(), shape_change))
        o["last"] = t
        o["hits"] += 1
        while o["history"] and t - o["history"][0][0] > s["motion_window_s"]:
            o["history"].popleft()
        h = list(o["history"])
        times = np.array([a[0] for a in h])
        xyz = np.array([a[1] for a in h])
        span = times[-1] - times[0]
        state = "UNCERTAIN"
        velocity = np.zeros(3)
        cov = np.eye(3)
        if len(h) >= 4 and span >= s["min_motion_span_s"]:
            u = times - times.mean()
            den = float(u @ u)
            velocity = (u[:, None] * (xyz - xyz.mean(0))).sum(0) / den
            residual = xyz - xyz.mean(0) - u[:, None] * velocity
            variance = (residual**2).sum(0) / max(len(h) - 2, 1) + s[
                "measurement_floor_m"
            ] ** 2
            cov = np.diag(variance / den)
            sigma = np.sqrt(np.trace(cov[:2, :2]))
            speed = np.linalg.norm(velocity[:2])
            displacement = np.linalg.norm(xyz[-1, :2] - xyz[0, :2])
            threshold = (
                s["release_speed"] if o["state"] == "MOVING" else s["moving_speed"]
            )
            if (
                speed - 2 * sigma > threshold
                and displacement >= s["min_displacement"]
                and shift < 0.4
                and shape_change < 0.35
            ):
                state = "MOVING"
            elif speed + 2 * sigma < s["stationary_speed"] and displacement < 0.12:
                state = "STATIONARY"
        if state == o["candidate"]:
            o["streak"] += 1
        else:
            o["candidate"] = state
            o["streak"] = 1
        needed = s["activation_frames"] if state == "MOVING" else s["release_frames"]
        o["state"] = (
            state if state != "UNCERTAIN" and o["streak"] >= needed else "UNCERTAIN"
        )
        o["velocity"] = velocity
        o["cov"] = cov
        alpha = 0.7 if o["state"] == "MOVING" else s["position_alpha"]
        o["center"] = (1 - alpha) * o["center"] + alpha * p["center"]
        o["dimensions"] = (1 - s["dimension_alpha"]) * o["dimensions"] + s[
            "dimension_alpha"
        ] * p["dimensions"]
        o["observed_box"] = {
            k: p[k].tolist() if hasattr(p[k], "tolist") else p[k]
            for k in ["center", "dimensions", "point_count"]
        }
        if "observed_box" in p:
            o["observed_box"] = p["observed_box"]

    def snapshot(self, t):
        result = []
        s = self.settings
        for identifier, o in sorted(self.objects.items()):
            age = t - o["last"]
            if age > s["ttl_s"]:
                continue
            observed = o["observed_box"] is not None
            result.append(
                dict(
                    object_id=identifier,
                    track_id=None,
                    center=o["center"].tolist(),
                    dimensions=o["dimensions"].tolist(),
                    orientation_xyzw=None,
                    velocity=o["velocity"].tolist(),
                    velocity_covariance=o["cov"].tolist(),
                    motion_state=o["state"] if observed else "UNCERTAIN",
                    status="observed_filtered" if observed else "stale",
                    last_observed=o["last"],
                    observation_age_s=age,
                    age_s=t - o["first"],
                    observation_score=min(o["hits"] / 3, 1)
                    * max(0.0, 1 - age / s["ttl_s"]),
                    observed_box=o["observed_box"],
                    display_compact=observed
                    and o["hits"] >= 3
                    and t - o["first"] >= 0.2,
                )
            )
        return result
