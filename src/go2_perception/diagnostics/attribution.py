"""Attribute a missed or non-MOVING walking person to the first failing Phase 5A stage.

Evaluation-only: the annotated person position selects points *after* perception ran; nothing
here feeds back into runtime. Categories (first failing stage wins):

  absent_from_current_scan        fewer than ``support_min`` returns of the current scan near
                                  the person: 5A cannot refresh by design (reported separately)
  ground_or_background            most window points near the person were removed as ground
                                  or as a vertical plane, leaving fewer than ``min_points``
  short_window_support            <``min_points`` non-ground points in the 0.1 s object window,
                                  but >= ``min_points`` in a 0.3 s window
  min_support_thresholds          sparse even at 0.3 s (DBSCAN min_points binding), or a
                                  proposal existed but had <``support_min`` current-scan points
  foreground_clustering           enough points, but DBSCAN left them as noise (spread > eps)
                                  or merged them into a compact proposal centred > gate away
  aabb_proposal_rejection         points fell in a component rejected as wider than max_xy or
                                  taller than max_height, and no novel split rescued them
  lifetime_stale                  a supported proposal existed but was absorbed by an older
                                  object whose filtered centre lies outside the gate
For fresh-but-not-MOVING frames, ``motion_*`` reasons come from 5A's own gate quantities.
"""

from __future__ import annotations

import numpy as np

CATEGORIES = (
    "absent_from_current_scan",
    "ground_or_background",
    "short_window_support",
    "min_support_thresholds",
    "foreground_clustering",
    "aabb_proposal_rejection",
    "lifetime_stale",
)
MOTION_REASONS = ("motion_object_too_young", "motion_speed_gate", "motion_displacement_gate", "motion_jump_or_shape_gate", "motion_activation_streak")


def in_cylinder(points, xy, radius, z_lo, z_hi):
    p = np.asarray(points, float)
    if not len(p):
        return np.zeros(0, bool)
    return (np.hypot(p[:, 0] - xy[0], p[:, 1] - xy[1]) <= radius) & (p[:, 2] >= z_lo) & (p[:, 2] <= z_hi)


def attribute_miss(evidence, min_points=8, support_min=3, gate=0.5):
    """``evidence`` keys: scan_n, win_codes (array of stage codes for window points near the
    person), ng03_n (non-ground points near the person in a 0.3 s window), proposals (list of
    dict(dist, support)), fresh_near (bool). Returns (category, detail)."""
    if evidence["scan_n"] < support_min:
        return "absent_from_current_scan", f"{evidence['scan_n']} current-scan returns"
    codes = np.asarray(evidence["win_codes"])
    n_all = len(codes)
    n_ground = int((codes == 0).sum())
    n_plane = int((codes == 1).sum())
    n_compact = int((codes == 2).sum())
    n_reject = int((codes == 3).sum())
    n_noise = int((codes == 4).sum())
    rest = n_compact + n_reject + n_noise
    counts = f"window {n_all}: ground {n_ground}, plane {n_plane}, compact {n_compact}, rejected {n_reject}, noise {n_noise}; 0.3 s non-ground {evidence['ng03_n']}"
    if rest < min_points and (n_ground + n_plane) >= max(rest, 1) and n_all >= min_points:
        return "ground_or_background", counts
    if rest < min_points:
        if evidence["ng03_n"] >= min_points:
            return "short_window_support", counts
        return "min_support_thresholds", counts + " (sparse at 0.3 s too)"
    near = [p for p in evidence["proposals"] if p["dist"] <= gate]
    if not near:
        if n_reject >= max(n_compact, n_noise):
            return "aabb_proposal_rejection", counts
        if n_noise >= max(n_compact, n_reject):
            return "foreground_clustering", counts + " (left as DBSCAN noise)"
        return "foreground_clustering", counts + " (merged into an off-centre compact proposal)"
    if not any(p["support"] >= support_min for p in near):
        return "min_support_thresholds", counts + f" (proposal current support {max(p['support'] for p in near)} < {support_min})"
    return "lifetime_stale", counts + " (supported proposal absorbed by an older object outside the gate)"


def attribute_not_moving(internals, activation_frames=3):
    """Reason a fresh, matched walking person is not MOVING, from 5A's own gate quantities."""
    g = internals["gate"]
    if not g["enough_history"]:
        return "motion_object_too_young", f"history {internals['history']} obs over {internals['span_s']:.2f} s (needs 4 over 0.4 s); object age {internals['age_s']:.2f} s"
    if not g["speed_ok"]:
        return "motion_speed_gate", f"speed {internals['speed']:.2f} - 2x{internals['sigma']:.2f} sigma below threshold"
    if not g["displacement_ok"]:
        return "motion_displacement_gate", f"net displacement {internals['displacement']:.2f} m < 0.18 m"
    if not (g["shift_ok"] and g["shape_ok"]):
        return "motion_jump_or_shape_gate", f"centre jump {internals['shift']:.2f} m / shape change {internals['shape_change']:.2f} m"
    return "motion_activation_streak", f"candidate {internals['candidate']} streak {internals['streak']} < {activation_frames}"
