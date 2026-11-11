"""Keep raw measurements; separate extended vertical planes only for compact-object proposals."""

import numpy as np
from go2_perception.lidar.clustering import cluster_points


def vertical_plane_mask(
    points, threshold=0.06, min_points=40, min_extent=1.5, iterations=64
):
    p = np.asarray(points)
    best = np.zeros(len(p), bool)
    if len(p) < min_points:
        return best
    rng = np.random.default_rng(31)
    for _ in range(iterations):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n /= norm
        if abs(n[2]) > 0.3:
            continue
        mask = np.abs((p - a) @ n) <= threshold
        if mask.sum() < max(min_points, int(best.sum())):
            continue
        q = p[mask]
        horizontal = np.array([-n[1], n[0], 0.0])
        extent = np.ptp(q @ horizontal)
        if extent >= min_extent and np.ptp(q[:, 2]) >= 0.5:
            best = mask
    return best


def make_proposals(
    points,
    *,
    eps=0.3,
    min_points=8,
    max_xy=1.5,
    max_height=2.4,
    split_planes=True,
    background=None,
    timestamp=None,
):
    p = np.asarray(points)
    plane = vertical_plane_mask(p) if split_planes else np.zeros(len(p), bool)
    labels, clusters = cluster_points(p[~plane], eps=eps, min_points=min_points)
    proposals = []
    large = 0
    novel_split = 0
    for cluster in clusters:
        q = cluster.points
        dim = np.ptp(q, axis=0)
        if max(dim[:2]) <= max_xy and dim[2] <= max_height:
            proposals.append(q)
        else:
            large += 1
            # A foreground-only split is supplementary for extended mixed components.
            # Stable compact components remain eligible even after entering the map.
            if background is not None:
                novel = q[~background.stable_mask(q, timestamp)]
                _, parts = cluster_points(novel, eps=eps, min_points=min_points)
                for part in parts:
                    dims = np.ptp(part.points, axis=0)
                    if max(dims[:2]) <= max_xy and dims[2] <= max_height:
                        proposals.append(part.points)
                        novel_split += 1
    result = []
    for q in proposals:
        low, high = np.percentile(q, [2, 98], axis=0)
        result.append(
            dict(
                center=(low + high) / 2,
                dimensions=np.maximum(high - low, 0.03),
                point_count=len(q),
                low=low,
                high=high,
            )
        )
    return result, dict(
        non_ground_points=len(p),
        plane_points=int(plane.sum()),
        extended_components=large,
        novel_subcomponents=novel_split,
        proposals=len(result),
    )
