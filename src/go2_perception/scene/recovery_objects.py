"""Direct-observation motion regression and ambiguity-gated gap continuity, opt-in only."""

from collections import deque
from scipy.optimize import linear_sum_assignment
import numpy as np
from .objects import SceneObjects


class RecoveryObjects(SceneObjects):
    def __init__(
        self,
        direct_window_s=1.0,
        direct_min_span_s=0.35,
        activation_speed=0.28,
        release_speed_direct=0.18,
        displacement_m=0.18,
        coherence_min=0.55,
        direct_activation_observations=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.direct = dict(
            window=direct_window_s,
            span=direct_min_span_s,
            speed=activation_speed,
            release=release_speed_direct,
            displacement=displacement_m,
            coherence=coherence_min,
            activation=direct_activation_observations,
        )
        if min(self.direct.values()) <= 0 or coherence_min > 1:
            raise ValueError("Invalid direct motion settings")

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
                    gap = t - o["last"]
                    compatible = (
                        ratio.max() < 2.5 and abs(o["center"][2] - p["center"][2]) < 0.4
                    )
                    # Reacquisition has a tighter prediction gate than ordinary association.
                    gate = 0.3 if gap > 0.2 else s["association_m"]
                    if distance <= gate and compatible:
                        cost[i, j] = distance + 0.05 * np.linalg.norm(
                            o["dimensions"] - p["dimensions"]
                        )
            # Near-tied assignments are not evidence of identity continuity.
            ambiguous = set()
            for i in range(len(ids)):
                ordered = np.sort(cost[i])
                if (
                    len(ordered) > 1
                    and ordered[0] < 1e5
                    and ordered[1] - ordered[0] < 0.15
                ):
                    ambiguous.add(i)
            for j in range(len(proposals)):
                ordered = np.argsort(cost[:, j])
                if (
                    len(ordered) > 1
                    and cost[ordered[0], j] < 1e5
                    and cost[ordered[1], j] - cost[ordered[0], j] < 0.15
                ):
                    ambiguous.update(ordered[:2])
            for i in ambiguous:
                self.objects[ids[i]]["direct_history"] = []
                self.objects[ids[i]]["recovery_streak"] = 0
                self.objects[ids[i]]["recovery_motion"] = dict(
                    state="UNCERTAIN", valid=False, reason="ambiguous_association"
                )
                self.objects[ids[i]]["velocity"] = np.zeros(3)
                self.objects[ids[i]]["cov"] = np.eye(3)
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
        previous_state = o.get("recovery_motion", {}).get("state", "UNCERTAIN")
        previous_velocity, previous_cov = o["velocity"].copy(), o["cov"].copy()
        super()._observe(o, p, t)
        if p.get("observed_box", p)["point_count"] < 3:
            # A single new return supports existence, not a new 3D center regression sample.
            o["velocity"], o["cov"] = previous_velocity, previous_cov
            evidence = o.get("recovery_motion", {})
            o["state"] = (
                previous_state
                if t - evidence.get("updated_at", -1e9) <= 0.2
                else "UNCERTAIN"
            )
            return
        history = o.setdefault("direct_history", [])
        # Current raw support is stored separately from window-filtered geometry.
        observed = p.get("observed_box", p)
        center = np.asarray(observed["center"], dtype=float)
        history.append((t, center.copy()))
        history[:] = [
            (stamp, point)
            for stamp, point in history
            if t - stamp <= self.direct["window"]
        ]
        evidence = dict(
            state="UNCERTAIN",
            valid=False,
            reason="insufficient_direct_history",
            observations=len(history),
        )
        if len(history) >= 4 and history[-1][0] - history[0][0] >= self.direct["span"]:
            times = np.array([x[0] for x in history])
            xyz = np.array([x[1] for x in history])
            # Median pairwise slopes tolerate a single badly sampled face without counting old points again.
            dt = times[:, None] - times[None, :]
            pairs = dt >= 0.2
            slopes = (xyz[:, None, :] - xyz[None, :, :])[pairs] / dt[pairs, None]
            velocity = np.median(slopes, axis=0)
            intercept = np.median(xyz - times[:, None] * velocity, axis=0)
            residual = xyz - intercept - times[:, None] * velocity
            variance = np.maximum(np.median(residual**2, axis=0), 0.025**2)
            cov = np.diag(variance / max(np.sum((times - times.mean()) ** 2), 1e-9))
            speed = float(np.linalg.norm(velocity[:2]))
            displacement = speed * (times[-1] - times[0])
            energy = float(np.sum((xyz[:, :2] - xyz[:, :2].mean(0)) ** 2))
            coherence = float(
                np.clip(1 - np.sum(residual[:, :2] ** 2) / max(energy, 1e-9), 0, 1)
            )
            state = "UNCERTAIN"
            threshold = (
                self.direct["release"]
                if previous_state == "MOVING"
                else self.direct["speed"]
            )
            if (
                speed >= threshold
                and displacement >= self.direct["displacement"]
                and coherence >= self.direct["coherence"]
            ):
                state = "MOVING"
            elif speed < 0.12 and np.linalg.norm(np.std(xyz[:, :2], axis=0)) < 0.08:
                state = "STATIONARY"
            evidence = dict(
                state=state,
                valid=True,
                reason="direct_center_regression",
                observations=len(history),
                span_s=float(times[-1] - times[0]),
                coherence=coherence,
                displacement_m=displacement,
                speed_mps=speed,
            )
            o["velocity"], o["cov"] = velocity, cov
        if not evidence["valid"]:
            o["velocity"], o["cov"] = np.zeros(3), np.eye(3)
        state = evidence["state"]
        o["recovery_streak"] = (
            o.get("recovery_streak", 0) + 1
            if o.get("recovery_candidate") == state
            else 1
        )
        o["recovery_candidate"] = state
        o["state"] = (
            state if o["recovery_streak"] >= self.direct["activation"] else "UNCERTAIN"
        )
        evidence["state"] = o["state"]
        evidence["updated_at"] = t
        o["recovery_motion"] = evidence

    def snapshot(self, t):
        # Keep history for up to 1 second for credible re-association, but never extend display TTL.
        return [o for o in super().snapshot(t) if o["observation_age_s"] <= 0.6]

    def motion_evidence(self, identity):
        return dict(
            self.objects[identity].get(
                "recovery_motion", dict(state="UNCERTAIN", valid=False)
            )
        )
