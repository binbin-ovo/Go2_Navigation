"""Conservative repeated-evidence suppression; never merge filter covariances."""

import numpy as np
from .association import box_iou


def suppress_track_duplicates(tracks, evidence, config):
    if not config.duplicate_track_hits:
        return tracks, [], {}
    counts, removed = {}, set()
    # Most reliable existing identity wins, but an unobserved winner cannot suppress evidence.
    ordered = sorted(tracks, key=lambda t: (-t.hits, t.created, t.track_id))
    for i, a in enumerate(ordered):
        if not a.confirmed or a.track_id in removed:
            continue
        for b in ordered[i + 1 :]:
            if not b.confirmed or b.track_id in removed or (a.misses and b.misses):
                continue
            if a.class_name and b.class_name and a.class_name != b.class_name:
                continue
            if (
                np.linalg.norm(a.filter.state[:2] - b.filter.state[:2])
                > config.duplicate_track_distance
            ):
                continue
            if (
                np.linalg.norm(a.filter.state[2:] - b.filter.state[2:])
                > config.duplicate_track_velocity
            ):
                continue
            if box_iou(*a.box(), *b.box()) < config.duplicate_track_iou:
                continue
            # A coasting prediction overlapping an observed object is ambiguous, not
            # a second confirmed observation. Retain its ID/filter until normal expiry.
            if a.misses:
                a.status = "Lost"
                continue
            if b.misses:
                b.status = "Lost"
            pair = tuple(sorted((a.track_id, b.track_id)))
            counts[pair] = evidence.get(pair, 0) + 1
            if counts[pair] >= config.duplicate_track_hits:
                removed.add(b.track_id)
    return (
        [t for t in tracks if t.track_id not in removed],
        sorted(removed),
        {
            pair: count
            for pair, count in counts.items()
            if not removed.intersection(pair)
        },
    )
