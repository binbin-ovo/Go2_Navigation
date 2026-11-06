import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou(a_min, a_max, b_min, b_max):
    if a_min is None or b_min is None:
        return 0.0
    intersection = np.prod(
        np.maximum(0, np.minimum(a_max, b_max) - np.maximum(a_min, b_min))
    )
    union = np.prod(a_max - a_min) + np.prod(b_max - b_min) - intersection
    return float(intersection / union) if union > 0 else 0.0


def associate(tracks, detections, config):
    """Gate first, then maximize match count and minimize normalized cost using dummies."""
    n, m = len(tracks), len(detections)
    if not n or not m:
        return [], list(range(n)), list(range(m))
    total_weight = (
        config.distance_weight + config.mahalanobis_weight + config.iou_weight
    )
    # Real costs <=1. Penalty larger than any possible sum makes valid cardinality primary.
    penalty = float(n + m + 1)
    costs = np.full((n, m + n), penalty)
    costs[:, :m] = penalty * 3
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            if t.class_name and d.class_name and t.class_name != d.class_name:
                continue
            if abs(t.z - d.position[2]) > config.max_vertical_distance:
                continue
            distance = float(np.linalg.norm(t.filter.state[:2] - d.position[:2]))
            mahal = t.filter.mahalanobis(
                d.position[:2], d.measurement_covariance[:2, :2]
            )
            if distance > config.max_distance or mahal > config.mahalanobis_gate:
                continue
            low, high = t.box()
            costs[i, j] = (
                config.distance_weight * distance / config.max_distance
                + config.mahalanobis_weight * mahal / config.mahalanobis_gate
                + config.iou_weight * (1 - box_iou(low, high, d.bbox_min, d.bbox_max))
            ) / total_weight
    rows, cols = linear_sum_assignment(costs)
    matches = [
        (int(i), int(j)) for i, j in zip(rows, cols) if j < m and costs[i, j] < penalty
    ]
    return (
        matches,
        [i for i in range(n) if i not in {a for a, _ in matches}],
        [j for j in range(m) if j not in {b for _, b in matches}],
    )


def deduplicate(detections, distance):
    """Suppress nearly identical centers with compatible class and highly overlapping boxes."""
    kept = []
    for d in sorted(
        detections, key=lambda d: -(d.confidence if d.confidence is not None else 0)
    ):
        duplicate = False
        for k in kept:
            compatible = (
                not d.class_name or not k.class_name or d.class_name == k.class_name
            )
            overlap = (
                d.bbox_min is None
                and k.bbox_min is None
                or box_iou(d.bbox_min, d.bbox_max, k.bbox_min, k.bbox_max) > 0.9
            )
            if (
                distance > 0
                and compatible
                and overlap
                and np.linalg.norm(d.position - k.position) < distance
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(d)
    return kept
