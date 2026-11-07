"""Recover a split of a previously observed compact object, never proximity-only leg merging."""

from dataclasses import replace
import numpy as np
from .association import box_iou


def recover_fragments(tracks, detections, config):
    if not config.fragment_recovery:
        return detections, []
    candidates = {}
    for i, track in enumerate(tracks):
        low, high = track.box()
        if (
            not track.confirmed
            or track.hits < config.fragment_min_hits
            or low is None
            or np.max(high - low) > config.fragment_max_extent
        ):
            continue
        indices = []
        for j, d in enumerate(detections):
            if d.bbox_min is None or (
                track.class_name and d.class_name and track.class_name != d.class_name
            ):
                continue
            if np.all(d.bbox_min >= low - config.fragment_margin) and np.all(
                d.bbox_max <= high + config.fragment_margin
            ):
                indices.append(j)
        if len(indices) >= 2:
            candidates[i] = indices
    ownership = {}
    for i, indices in candidates.items():
        for j in indices:
            ownership.setdefault(j, []).append(i)
    consumed = set()
    merged = []
    events = []
    for i, indices in candidates.items():
        if any(len(ownership[j]) != 1 for j in indices):
            continue
        track = tracks[i]
        low = np.min([detections[j].bbox_min for j in indices], axis=0)
        high = np.max([detections[j].bbox_max for j in indices], axis=0)
        if box_iou(*track.box(), low, high) < config.fragment_min_iou:
            continue
        # Another established identity inside the union makes this ambiguous.
        if any(
            k != i
            and other.confirmed
            and np.all(
                np.r_[other.filter.state[:2], other.z] >= low - config.fragment_margin
            )
            and np.all(
                np.r_[other.filter.state[:2], other.z] <= high + config.fragment_margin
            )
            for k, other in enumerate(tracks)
        ):
            continue
        classes = {
            detections[j].class_name for j in indices if detections[j].class_name
        }
        if len(classes) > 1:
            continue
        reference = detections[indices[0]]
        # Correlated fragments are not independent measurements: never shrink covariance.
        variance = max(
            np.linalg.eigvalsh(detections[j].measurement_covariance).max()
            for j in indices
        )
        merged.append(
            replace(
                reference,
                position=(low + high) / 2,
                bbox_min=low,
                bbox_max=high,
                measurement_covariance=np.eye(3) * variance,
            )
        )
        consumed.update(indices)
        events.append(
            dict(
                track_id=track.track_id,
                source_ids=[detections[j].detection_id for j in indices],
            )
        )
    return [d for j, d in enumerate(detections) if j not in consumed] + merged, events
