"""Additional deterministic recovery experiments; truth is evaluation-only."""

import numpy as np
from .synthetic import sequence as original, SCENARIOS as ORIGINAL

SCENARIOS = (*ORIGINAL, "missing_sector", "stop_restart", "crossing", "near_wall")


def sequence(name, seed=101, frames=120):
    if name in ORIGINAL:
        yield from original(name, seed=17)
        return
    if name not in SCENARIOS:
        raise ValueError(name)
    rng = np.random.default_rng(seed)

    def pose(t):
        m = np.eye(4)
        m[2, 3] = 0.32
        return m

    for i in range(frames):
        t = i * 0.065
        y = -1 + (
            0.5 * t if name != "stop_restart" else 0.5 * min(t, 2) + 0.5 * max(0, t - 4)
        )
        v = 0.5 if name != "stop_restart" or t < 2 or t > 4 else 0
        centers = [(1, [2, y, 0.8], [0, v, 0])]
        if name == "crossing":
            centers = [
                (1, [1.7, -1 + 0.4 * t, 0.8], [0, 0.4, 0]),
                (2, [2.3, 1 - 0.4 * t, 0.8], [0, -0.4, 0]),
            ]
        if name == "near_wall":
            centers = [(1, [2.75, -1 + 0.3 * t, 0.8], [0, 0.3, 0])]
        points = []
        truth = []
        for identity, c, velocity in centers:
            size = [0.35, 0.35, 1.6]
            if name != "missing_sector" or i % 3 == 0:
                points.append(rng.uniform(-0.5, 0.5, (70, 3)) * size + c)
            truth.append(
                dict(
                    id=identity,
                    center=c,
                    dimensions=size,
                    velocity=velocity,
                    visible=True,
                )
            )
        points.append(
            np.column_stack(
                [rng.uniform(0, 4, 200), rng.uniform(-2, 2, 200), np.zeros(200)]
            )
        )
        if name == "near_wall":
            points.append(
                np.column_stack(
                    [np.full(300, 3), rng.uniform(-2, 2, 300), rng.uniform(0, 2, 300)]
                )
            )
        p = np.concatenate(points)
        yield dict(
            timestamp=t, points=p - pose(t)[:3, 3], pose_lookup=pose, truth=truth
        )
