"""Return coverage is positive evidence only: absent rays never prove visibility/free space."""

import numpy as np


def body_exclusion(points, low, high):
    """Mask a configured conservative inner-body volume in raw base_link coordinates."""
    p = np.asarray(points)
    lo, hi = np.asarray(low), np.asarray(high)
    if (
        lo.shape != (3,)
        or hi.shape != (3,)
        or not np.isfinite([lo, hi]).all()
        or (lo >= hi).any()
    ):
        raise ValueError("Finite ordered body bounds required")
    return ((p >= lo) & (p <= hi)).all(1)


def return_coverage(points, center, dimensions, pose):
    """Angular return overlap, not a calibrated visibility/occlusion classifier."""
    body = (np.asarray(points) - pose[:3, 3]) @ pose[:3, :3]
    c = (np.asarray(center) - pose[:3, 3]) @ pose[:3, :3]
    r = np.linalg.norm(c)
    radius = np.linalg.norm(dimensions) / 2
    if not len(body) or r < 1e-6:
        return dict(state="no_sector_returns", returns=0, proves_visible=False)
    norms = np.linalg.norm(body, axis=1)
    cos = (body @ c) / np.maximum(norms * r, 1e-9)
    overlap = cos >= np.cos(min(np.arcsin(min(radius / r, 1.0)), np.pi / 2))
    ranges = norms[overlap]
    state = (
        "no_sector_returns"
        if not len(ranges)
        else "foreground_returns_occlusion_possible"
        if np.min(ranges) < r - radius
        else "sector_returns_without_object"
    )
    return dict(state=state, returns=int(overlap.sum()), proves_visible=False)
