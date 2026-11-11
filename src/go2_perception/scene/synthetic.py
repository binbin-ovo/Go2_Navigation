"""Deterministic physical-scene fixtures; truth never enters runtime perception."""

import numpy as np
from scipy.spatial.transform import Rotation

SCENARIOS = (
    "sampling",
    "rotation",
    "translation",
    "start_move",
    "near_furniture",
    "sparse_occlusion",
)


def sequence(name, seed=17, frames=90):
    if name not in SCENARIOS:
        raise ValueError(name)
    rng = np.random.default_rng(seed)

    def pose(t):
        m = np.eye(4)
        m[2, 3] = 0.32
        if name == "rotation":
            m[:3, :3] = Rotation.from_euler("z", 0.45 * np.sin(t)).as_matrix()
        if name == "translation":
            m[0, 3] = 0.18 * t
        return m

    for i in range(frames):
        t = i * 0.065
        world = []
        truth = []
        center = np.array([2.0, 0.0, 0.4])
        velocity = np.zeros(3)
        if name == "start_move":
            center[1] = max(0.0, t - 2.0) * 0.6
            velocity[1] = 0.6 if t > 2 else 0.0
        centers = [(1, center, np.array([0.5, 0.5, 0.8]), velocity)]
        if name == "near_furniture":
            centers.append(
                (
                    2,
                    np.array([1.6, -1.2 + 0.4 * t, 0.8]),
                    np.array([0.35, 0.35, 1.6]),
                    np.array([0, 0.4, 0]),
                )
            )
        for identity, c, size, v in centers:
            visible = not (name == "sparse_occlusion" and 30 <= i < 38)
            n = 18 if name == "sparse_occlusion" else 100
            p = rng.uniform(-0.5, 0.5, (n, 3)) * size + c
            # Alternate face-biased partial returns; same physical object, changing centroids.
            if name == "sampling":
                p = p[p[:, 1] < (c[1] + 0.12) if i % 2 else p[:, 1] > (c[1] - 0.12)]
            if visible:
                world.append(p + rng.normal(0, 0.005, p.shape))
            truth.append(
                dict(
                    id=identity,
                    center=c.tolist(),
                    velocity=v.tolist(),
                    dimensions=size.tolist(),
                    visible=visible,
                )
            )
        floor = np.column_stack(
            (rng.uniform(0, 4, 250), rng.uniform(-2, 2, 250), rng.normal(0, 0.003, 250))
        )
        wall = np.column_stack(
            (np.full(100, 3.0), rng.uniform(-2, 2, 100), rng.uniform(0, 2, 100))
        )
        world.extend([floor, wall])
        p = np.concatenate(world)
        m = pose(t)
        body = (p - m[:3, 3]) @ m[:3, :3]
        yield dict(timestamp=t, points=body, pose_lookup=pose, truth=truth)
