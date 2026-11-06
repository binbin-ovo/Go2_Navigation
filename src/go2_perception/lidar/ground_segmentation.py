"""Seeded, gravity-constrained RANSAC; a wall is never selected as ground."""

import numpy as np


def ground_mask(
    points,
    distance_threshold=0.06,
    iterations=200,
    max_tilt_deg=20,
    expected_ground_z=-0.5,
    height_tolerance=0.25,
    min_inliers=20,
    seed=7,
):
    if (
        distance_threshold <= 0
        or iterations < 1
        or not 0 <= max_tilt_deg < 90
        or height_tolerance < 0
        or min_inliers < 3
    ):
        raise ValueError("Invalid ground configuration")
    p = np.asarray(points)
    best = np.zeros(len(p), dtype=bool)
    if len(p) < min_inliers:
        return best
    # Require observed support near the expected ground height. Extrapolating a
    # tilted plane from an elevated object back to the origin is not evidence
    # of ground, even when that extrapolated intercept has the right height.
    candidates = np.flatnonzero(np.abs(p[:, 2] - expected_ground_z) <= height_tolerance)
    if len(candidates) < min_inliers:
        return best
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        a, b, c = p[rng.choice(candidates, 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            continue
        n /= norm
        if abs(n[2]) < np.cos(np.deg2rad(max_tilt_deg)):
            continue
        d = -n @ a
        if abs(-d / n[2] - expected_ground_z) > height_tolerance:
            continue
        mask = np.abs(p @ n + d) <= distance_threshold
        if mask.sum() > best.sum():
            best = mask
    return best if best.sum() >= min_inliers else np.zeros(len(p), dtype=bool)
