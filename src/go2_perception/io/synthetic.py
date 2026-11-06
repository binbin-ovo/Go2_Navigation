"""Deterministic fixtures, never an RGB accuracy benchmark."""

import numpy as np


def make_scene(seed=7):
    rng = np.random.default_rng(seed)
    ground = np.column_stack(
        (
            rng.uniform(0.5, 7, 1600),
            rng.uniform(-3, 3, 1600),
            rng.normal(-0.5, 0.008, 1600),
        )
    )
    a = rng.uniform([2, -1, -0.35], [2.5, -0.5, 0.6], (220, 3))
    b = rng.uniform([4, 1, -0.35], [4.5, 1.5, 0.8], (220, 3))
    return np.vstack((ground, a, b))


def cases():
    return {
        "two_obstacles": make_scene(),
        "sparse": np.array([[2, 0, 0], [3, 1, 0], [4, 0, 1.0]]),
        "empty": np.empty((0, 3)),
        "nan": np.vstack((make_scene(), [np.nan, 0, 0], [0, np.inf, 0])),
        "partial": make_scene()[make_scene()[:, 1] < 0],
        "nearby": np.vstack(
            (
                np.random.default_rng(7).uniform([2, 0, 0], [2.1, 0.1, 0.1], (40, 3)),
                np.random.default_rng(8).uniform([2, 0.5, 0], [2.1, 0.6, 0.1], (40, 3)),
            )
        ),
    }
