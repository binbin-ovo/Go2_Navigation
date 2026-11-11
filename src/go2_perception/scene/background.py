"""Local persistent-return voxels; absence is unknown, never certified free space."""

import numpy as np


class BackgroundMap:
    def __init__(
        self,
        voxel_size=0.1,
        ttl_s=2.0,
        min_observations=3,
        min_span_s=0.2,
        radius_m=8.0,
    ):
        if min(voxel_size, ttl_s, min_observations, radius_m) <= 0 or min_span_s < 0:
            raise ValueError("Invalid map settings")
        self.size = voxel_size
        self.ttl = ttl_s
        self.min_observations = min_observations
        self.min_span = min_span_s
        self.radius = radius_m
        self.cells = {}

    def clear(self):
        self.cells.clear()

    def update(self, points, t, scan_id, robot_position, moving_boxes=()):
        p = np.asarray(points)
        excluded = np.zeros(len(p), bool)
        for low, high in moving_boxes:
            excluded |= ((p >= low) & (p <= high)).all(1)
        p = p[~excluded]
        keys = np.unique(np.floor(p / self.size).astype(np.int64), axis=0)
        for key in keys:
            k = tuple(key)
            entry = self.cells.get(k)
            if entry is None:
                self.cells[k] = [1, t, t, str(scan_id)]
            elif entry[3] != str(scan_id):
                entry[0] += 1
                entry[2] = t
                entry[3] = str(scan_id)
        for key, entry in list(self.cells.items()):
            center = (np.array(key) + 0.5) * self.size
            in_motion = any(
                ((center >= lo) & (center <= hi)).all() for lo, hi in moving_boxes
            )
            if (
                t - entry[2] > self.ttl
                or np.linalg.norm(center - np.asarray(robot_position)) > self.radius
                or in_motion
            ):
                del self.cells[key]

    def stable_mask(self, points, t):
        mask = []
        for key in np.floor(np.asarray(points) / self.size).astype(np.int64):
            e = self.cells.get(tuple(key))
            mask.append(
                e is not None
                and e[0] >= self.min_observations
                and e[2] - e[1] >= self.min_span
                and t - e[2] <= self.ttl
            )
        return np.asarray(mask, bool)

    def snapshot(self, t):
        cells = []
        for key, e in sorted(self.cells.items()):
            age = t - e[2]
            if (
                age > self.ttl
                or e[0] < self.min_observations
                or e[2] - e[1] < self.min_span
            ):
                continue
            cells.append(
                dict(
                    center=((np.asarray(key) + 0.5) * self.size).tolist(),
                    observation_count=e[0],
                    last_observed=e[2],
                    age_s=age,
                    evidence_score=min(e[0] / self.min_observations, 1.0)
                    * max(0.0, 1 - age / self.ttl),
                )
            )
        return dict(
            voxel_size=self.size, free_space_policy="unobserved_is_unknown", cells=cells
        )
