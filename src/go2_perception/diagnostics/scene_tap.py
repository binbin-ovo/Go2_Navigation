"""Instrumented replay of the unchanged Phase 5A ``StableScene``.

The real pipeline runs unmodified. While ``process`` executes, its module-level helpers
(ground mask, plane mask, DBSCAN, proposal builder) and the accumulator are wrapped so their
inputs and outputs are recorded. Outputs are therefore byte-for-byte the pipeline's own; the
recorded intermediates feed the viewer's Debug layers and the failure attribution.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import numpy as np

from go2_perception.scene import pipeline as pipeline_mod
from go2_perception.scene import proposals as proposals_mod
from go2_perception.scene.pipeline import StableScene


@dataclass
class TapRecord:
    """Intermediates of one ``process`` call. All points are odom metres."""

    scan_xyz: np.ndarray | None = None  # new incremental scan, transformed once by 5A
    scan_acquired: np.ndarray | None = None
    scan_nonground: np.ndarray | None = None  # bool over scan_xyz (range + ground filter)
    window_xyz: np.ndarray | None = None  # object window (range-filtered, voxelized)
    window_code: np.ndarray | None = None  # 0 ground, 1 vertical plane, 2 compact cluster, 3 rejected cluster, 4 DBSCAN noise
    clusters: list = field(default_factory=list)  # dict(lo, hi, centroid, n, accepted, reason, novel)
    proposals: list = field(default_factory=list)  # dict(low, high, n, support, supported)
    objects_internal: dict = field(default_factory=dict)  # id -> motion internals after update


class _Recorder:
    def __init__(self):
        self.ground_calls = []
        self.plane = None
        self.cluster_calls = []
        self.proposal_call = None


@contextlib.contextmanager
def _wrapped(rec: _Recorder):
    orig_ground = pipeline_mod.ground_mask
    orig_make = pipeline_mod.make_proposals
    orig_cluster = proposals_mod.cluster_points
    orig_plane = proposals_mod.vertical_plane_mask

    def ground(points, **kw):
        m = orig_ground(points, **kw)
        rec.ground_calls.append((np.asarray(points).copy(), np.asarray(m).copy()))
        return m

    def make(points, **kw):
        result, details = orig_make(points, **kw)
        rec.proposal_call = (np.asarray(points).copy(), result, details, kw)
        return result, details

    def cluster(points, **kw):
        labels, clusters = orig_cluster(points, **kw)
        rec.cluster_calls.append((np.asarray(points).copy(), np.asarray(labels).copy(), clusters))
        return labels, clusters

    def plane(points, *a, **kw):
        m = orig_plane(points, *a, **kw)
        rec.plane = np.asarray(m).copy()
        return m

    pipeline_mod.ground_mask = ground
    pipeline_mod.make_proposals = make
    proposals_mod.cluster_points = cluster
    proposals_mod.vertical_plane_mask = plane
    try:
        yield
    finally:
        pipeline_mod.ground_mask = orig_ground
        pipeline_mod.make_proposals = orig_make
        proposals_mod.cluster_points = orig_cluster
        proposals_mod.vertical_plane_mask = orig_plane


def object_internals(objects, t):
    """Motion-gate quantities of every live object, recomputed from 5A's own state."""
    s = objects.settings
    out = {}
    for k, o in objects.objects.items():
        h = list(o["history"])
        times = np.array([a[0] for a in h]) if h else np.empty(0)
        xyz = np.array([a[1] for a in h]) if h else np.empty((0, 3))
        span = float(times[-1] - times[0]) if len(h) else 0.0
        speed = float(np.linalg.norm(o["velocity"][:2]))
        sigma = float(np.sqrt(max(np.trace(o["cov"][:2, :2]), 0.0)))
        disp = float(np.linalg.norm(xyz[-1, :2] - xyz[0, :2])) if len(h) > 1 else 0.0
        shift = float(np.linalg.norm(h[-1][1] - h[-2][1])) if len(h) > 1 else 0.0
        out[k] = dict(
            hits=int(o["hits"]),
            age_s=float(t - o["first"]),
            history=len(h),
            span_s=span,
            speed=speed,
            sigma=sigma,
            displacement=disp,
            shift=shift,
            shape_change=float(h[-1][2]) if h else 0.0,
            candidate=o["candidate"],
            streak=int(o["streak"]),
            state=o["state"],
            observed=o["observed_box"] is not None,
            gate=dict(
                enough_history=len(h) >= 4 and span >= s["min_motion_span_s"],
                speed_ok=speed - 2 * sigma > (s["release_speed"] if o["state"] == "MOVING" else s["moving_speed"]),
                displacement_ok=disp >= s["min_displacement"],
                shift_ok=shift < 0.4,
                shape_ok=(float(h[-1][2]) if h else 0.0) < 0.35,
            ),
        )
    return out


class SceneTap:
    """Run ``StableScene.process`` and return ``(row, TapRecord)``."""

    def __init__(self, config):
        self.config = config
        self.pipeline = StableScene(config)
        acc = self.pipeline.accumulator
        self._orig_add = acc.add
        self._last_new = None

        def add(*a, **kw):
            new = self._orig_add(*a, **kw)
            self._last_new = new
            return new

        acc.add = add

    def process(self, xyz, stamp, scan_id, lookup, **kw):
        rec = _Recorder()
        self._last_new = None
        with _wrapped(rec):
            row = self.pipeline.process(xyz, stamp, scan_id, lookup, **kw)
        tap = TapRecord()
        new = self._last_new
        if new is None:
            return row, tap
        c = self.config
        pose = np.asarray(row["robot_pose"]["T_odom_base"])
        tap.scan_xyz = new.xyz
        tap.scan_acquired = new.acquired
        in_range = np.linalg.norm(new.xyz - pose[:3, 3], axis=1) <= c["max_range_m"]
        nonground = in_range.copy()
        win_pts, win_ground = rec.ground_calls[0]
        cur_pts, cur_ground = rec.ground_calls[1]
        if in_range.sum() != len(cur_pts):
            raise AssertionError("tap: current-scan range filter mismatch")
        nonground[in_range] = ~cur_ground
        tap.scan_nonground = nonground
        code = np.zeros(len(win_pts), np.int8)
        ng_idx = np.flatnonzero(~win_ground)
        prop_in, result, details, kw_p = rec.proposal_call
        plane = rec.plane if rec.plane is not None else np.zeros(len(prop_in), bool)
        code[ng_idx[plane]] = 1
        main_pts, labels, clusters = rec.cluster_calls[0]
        rest = ng_idx[~plane]
        pc = c["proposals"]
        for cl in clusters:
            q = cl.points
            dim = np.ptp(q, axis=0)
            accepted = max(dim[:2]) <= pc["max_xy"] and dim[2] <= pc["max_height"]
            reason = "" if accepted else (f"wide {max(dim[:2]):.1f} m" if max(dim[:2]) > pc["max_xy"] else f"tall {dim[2]:.1f} m")
            sel = labels == cl.cluster_id
            code[rest[sel]] = 2 if accepted else 3
            tap.clusters.append(dict(lo=q.min(0), hi=q.max(0), centroid=q.mean(0), n=len(q), accepted=bool(accepted), reason=reason, novel=False))
        code[rest[labels == -1]] = 4
        for pts, _, parts in rec.cluster_calls[1:]:
            for part in parts:
                q = part.points
                dims = np.ptp(q, axis=0)
                ok = max(dims[:2]) <= pc["max_xy"] and dims[2] <= pc["max_height"]
                tap.clusters.append(dict(lo=q.min(0), hi=q.max(0), centroid=q.mean(0), n=len(q), accepted=bool(ok), reason="novel split" if ok else "novel split rejected", novel=True))
        tap.window_xyz = win_pts
        tap.window_code = code
        current = cur_pts[~cur_ground]
        support_min = c.get("current_support_min", 3)
        for p in result:
            support = int(((current >= p["low"] - 0.05) & (current <= p["high"] + 0.05)).all(1).sum())
            tap.proposals.append(dict(low=p["low"], high=p["high"], n=int(p["point_count"]), support=support, supported=support >= support_min))
        tap.objects_internal = object_internals(self.pipeline.objects, row["timestamp"])
        return row, tap


def strip_volatile(row):
    """Row without wall-clock processing time, for determinism comparisons."""
    d = dict(row)
    d["diagnostics"] = {k: v for k, v in row["diagnostics"].items() if k != "processing_ms"}
    return d
