"""Ground-truth annotations for real tracking recordings, and evaluation against them.

Annotations are per-frame object states in the tracker's output frame (``odom``) and clock
(``go2_robot``). Each object state carries a visibility status, an optional planar point
(people) or polygon region (static obstacles), an optional motion label and a confidence.
Positions are absent when the object cannot be located reliably; nothing here invents them.

Evaluation matches tracker outputs to annotated objects frame by frame (Hungarian, gated by
distance to the point or to the region), then computes CLEAR-MOT-style identity counts, IDF1,
velocity error, motion-label agreement and recovery after occlusion. Tracks far from every
annotated object are ignored: the rest of the scene is not annotated, so they are neither
true nor false positives.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

SCHEMA = "go2-tracking-gt-v1"
OBJECT_TYPES = ("person", "static_obstacle", "other")
STATUSES = ("visible", "partial", "occluded", "out_of_view")
MOTIONS = ("moving", "stationary", None)
CONFIDENCES = ("high", "medium", "low")
# Statuses at which a tracker is expected to report the object.
DETECTABLE = ("visible", "partial")


def scorable(state):
    """Detectable, located and identifiable: the states a tracker is scored on. States marked
    ``identity_uncertain`` (unidentifiable in review) are don't-care, never guessed."""
    return (
        state["status"] in DETECTABLE
        and bool(state.get("xy") or state.get("region"))
        and not state.get("identity_uncertain")
    )


def validate(doc):
    """Raise ValueError on any schema violation; return the document."""
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported schema {doc.get('schema')!r}")
    for key in ("recording", "frame", "clock_domain", "t0", "objects", "frames", "intervals"):
        if key not in doc:
            raise ValueError(f"Missing {key}")
    for oid, obj in doc["objects"].items():
        if obj.get("type") not in OBJECT_TYPES:
            raise ValueError(f"{oid}: invalid type {obj.get('type')!r}")
    for interval in doc["intervals"]:
        if not interval["start"] < interval["end"]:
            raise ValueError(f"Empty interval {interval}")
    last = -np.inf
    for frame in doc["frames"]:
        t = frame["t"]
        if not np.isfinite(t) or t <= last:
            raise ValueError("Frame times must be finite and strictly increasing")
        last = t
        seen = set()
        for state in frame["objects"]:
            oid = state["id"]
            if oid not in doc["objects"] or oid in seen:
                raise ValueError(f"Unknown or repeated object {oid!r} at {t}")
            seen.add(oid)
            if state["status"] not in STATUSES:
                raise ValueError(f"{oid}: invalid status {state['status']!r}")
            if state.get("motion") not in MOTIONS:
                raise ValueError(f"{oid}: invalid motion {state.get('motion')!r}")
            if state.get("confidence") not in CONFIDENCES:
                raise ValueError(f"{oid}: invalid confidence {state.get('confidence')!r}")
            xy, region = state.get("xy"), state.get("region")
            if xy is not None and region is not None:
                raise ValueError(f"{oid}: give a point or a region, not both")
            if xy is not None and (len(xy) != 2 or not np.isfinite(xy).all()):
                raise ValueError(f"{oid}: invalid xy")
            if region is not None and (len(region) < 3 or not np.isfinite(region).all()):
                raise ValueError(f"{oid}: region needs at least three finite vertices")
            if state["status"] == "out_of_view" and (xy is not None or region is not None):
                raise ValueError(f"{oid}: out_of_view objects carry no position")
            if not isinstance(state.get("identity_uncertain", False), bool):
                raise ValueError(f"{oid}: identity_uncertain must be a boolean")
            if "velocity" in state and state["velocity"] is not None and xy is None:
                raise ValueError(f"{oid}: velocity requires a point position")
    return doc


def load(path):
    return validate(json.loads(Path(path).read_text()))


def save(doc, path):
    validate(doc)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")


def region_distance(p, polygon):
    """Planar distance from ``p`` to a polygon (0 inside)."""
    poly = np.asarray(polygon, float)
    x, y = p
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    if inside:
        return 0.0
    best = np.inf
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ab = b - a
        s = np.clip(np.dot(np.asarray(p) - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
        best = min(best, float(np.linalg.norm(np.asarray(p) - (a + s * ab))))
    return best


def distances(state, points):
    """Vectorised ``state_distance`` for an (N, 2) array (polygons must be convex)."""
    p = np.asarray(points, float)[:, :2]
    if state.get("xy") is not None:
        return np.linalg.norm(p - np.asarray(state["xy"]), axis=1)
    if state.get("region") is None:
        return np.full(len(p), np.inf)
    poly = np.asarray(state["region"], float)
    edges = np.roll(poly, -1, axis=0) - poly
    rel = p[:, None, :] - poly[None, :, :]
    cross = edges[None, :, 0] * rel[:, :, 1] - edges[None, :, 1] * rel[:, :, 0]
    inside = np.all(cross >= 0, axis=1) | np.all(cross <= 0, axis=1)
    s = np.clip((rel * edges[None]).sum(-1) / np.maximum((edges**2).sum(-1), 1e-12), 0, 1)
    d = np.linalg.norm(rel - s[..., None] * edges[None], axis=-1).min(axis=1)
    return np.where(inside, 0.0, d)


def state_distance(state, p):
    if state.get("xy") is not None:
        return float(np.hypot(p[0] - state["xy"][0], p[1] - state["xy"][1]))
    if state.get("region") is not None:
        return region_distance(p, state["region"])
    return np.inf


def track_rows(record, selection):
    """Tracker outputs from one compact ``tracks.jsonl`` record of the comparison harness.

    Row: [id, x, y, z, vx, vy, vstd, status, motion, consecutive_misses]. ``selection`` is
    ``confirmed`` (status C, the tracker's reported objects) or ``observed`` (any track updated
    by a detection in this frame, which removes differences in how coasting is labelled).
    """
    rows = []
    for r in record["tracks"]:
        if selection == "confirmed" and r[7] != "C":
            continue
        if selection == "observed" and (r[7] not in "CL" or r[9] != 0):
            continue
        rows.append(r)
    return rows


def match_frame(states, rows, gates, previous=None):
    """Match annotated states with a position to tracker rows.

    As in CLEAR-MOT, an object keeps the track it was matched to in its previous matched
    frame (``previous``: {object_id: track_id}) while that track is still inside the gate, even
    if another track is closer; the remaining objects and tracks are matched by Hungarian
    assignment on distance. Returns ``{object_id: row}`` and, per object, all rows inside its
    gate.
    """
    previous = previous or {}
    located = [s for s in states if scorable(s)]
    near = {s["id"]: [] for s in located}
    if not located or not rows:
        return {}, near
    cost = np.full((len(located), len(rows)), 1e6)
    for i, s in enumerate(located):
        gate = gates[s["type"]]
        for j, r in enumerate(rows):
            d = state_distance(s, r[1:3])
            if d <= gate:
                cost[i, j] = d
                near[s["id"]].append(r)
    matches, used_rows, used_states = {}, set(), set()
    for i, s in enumerate(located):
        for j, r in enumerate(rows):
            if r[0] == previous.get(s["id"]) and cost[i, j] < 1e6 and j not in used_rows:
                matches[s["id"]] = r
                used_rows.add(j)
                used_states.add(i)
    free_s = [i for i in range(len(located)) if i not in used_states]
    free_r = [j for j in range(len(rows)) if j not in used_rows]
    if free_s and free_r:
        sub = cost[np.ix_(free_s, free_r)]
        for a, b in zip(*linear_sum_assignment(sub)):
            if sub[a, b] < 1e6:
                matches[located[free_s[a]]["id"]] = rows[free_r[b]]
    return matches, near


def idf1(pairs, gt_frames, hyp_frames):
    """IDF1 (Ristani et al. 2016) from per-frame (gt_key, hyp_id) co-occurrences.

    ``pairs`` holds one entry per frame for every hypothesis inside a GT object's gate;
    ``gt_frames``: frames in which each GT key was detectable; ``hyp_frames``: frames each
    hypothesis was near annotated objects. The global one-to-one GT-hypothesis assignment
    maximises the total number of co-occurring frames.
    """
    gts, hyps = sorted(gt_frames), sorted(hyp_frames)
    if not gts:
        return dict(idf1=None, idp=None, idr=None, idtp=0)
    counts = collections.Counter(pairs)
    weight = np.zeros((len(gts), len(hyps)))
    for (g, h), n in counts.items():
        weight[gts.index(g), hyps.index(h)] = n
    idtp = 0
    if hyps:
        rows, cols = linear_sum_assignment(-weight)
        idtp = int(weight[rows, cols].sum())
    total_gt = sum(gt_frames.values())
    total_hyp = sum(hyp_frames.values())
    idfn, idfp = total_gt - idtp, total_hyp - idtp
    return dict(
        idf1=2 * idtp / max(2 * idtp + idfp + idfn, 1),
        idp=idtp / total_hyp if total_hyp else None,
        idr=idtp / total_gt if total_gt else None,
        idtp=idtp,
        idfp=idfp,
        idfn=idfn,
    )


def _ratio(a, b):
    return None if not b else a / b


def match_sequence(doc, records, selection="confirmed", gates=None):
    """Yield (frame, record, typed states, rows, matches, near) for every annotated frame with
    a tracker record (stamps equal within 1 ms), carrying matches forward as in CLEAR-MOT."""
    gates = gates or dict(person=0.5, static_obstacle=0.25, other=0.5)
    by_time = {round(r["t"], 3): r for r in records}
    types = {k: v["type"] for k, v in doc["objects"].items()}
    previous = {}
    for frame in doc["frames"]:
        record = by_time.get(round(frame["t"], 3))
        if record is None:
            continue
        states = [dict(s, type=types[s["id"]]) for s in frame["objects"]]
        rows = track_rows(record, selection)
        matches, near = match_frame(states, rows, gates, previous)
        previous.update({oid: row[0] for oid, row in matches.items()})
        yield frame, record, states, rows, matches, near


def evaluate(doc, records, selection="confirmed", gates=None, zone=None, recover_s=1.0):
    """Evaluate compact tracker records (absolute go2_robot stamps) against annotations.

    ``gates``: matching distance per object type (m, to point or region boundary).
    ``zone``: tracks within this distance of a detectable object are hypotheses; tracks
    further from every annotated object are ignored as unannotated scene.
    Only annotated frames are scored; a tracker record is paired with an annotated frame
    when their timestamps agree within 1 ms.
    """
    gates = gates or dict(person=0.5, static_obstacle=0.25, other=0.5)
    zone = zone or gates
    types = {k: v["type"] for k, v in doc["objects"].items()}
    per = collections.defaultdict(
        lambda: dict(
            frames=0,
            detectable=0,
            matched=0,
            switches=0,
            fragments=0,
            ids=collections.Counter(),
            duplicate_frames=0,
            errors=[],
            vel_errors=[],
            speeds_when_static=[],
            motion=collections.Counter(),
            last_id=None,
            gap=False,
        )
    )
    pairs, gt_frames, hyp_frames = [], collections.Counter(), collections.Counter()
    visit_pairs, visit_frames = [], collections.Counter()
    events = collections.defaultdict(list)  # object -> [(t, status, matched id or None)]
    scored = 0
    for frame, record, states, rows, matches, near in match_sequence(doc, records, selection, gates):
        scored += 1
        zone_ids = set()
        for s in states:
            if scorable(s):
                zone_ids |= {
                    r[0] for r in rows if state_distance(s, r[1:3]) <= zone[s["type"]]
                }
        for tid in zone_ids:
            hyp_frames[tid] += 1
        for s in states:
            m = per[s["id"]]
            m["frames"] += 1
            row = matches.get(s["id"])
            events[s["id"]].append((frame["t"], s["status"], None if row is None else row[0]))
            if not scorable(s):
                continue
            m["detectable"] += 1
            key = (s["id"], s.get("visit"))
            gt_frames[s["id"]] += 1
            visit_frames[key] += 1
            if len(near.get(s["id"], [])) > 1:
                m["duplicate_frames"] += 1
            for r in near[s["id"]]:
                pairs.append((s["id"], r[0]))
                visit_pairs.append((key, r[0]))
            if row is None:
                # CLEAR-MOT fragmentation: tracked -> untracked -> tracked while detectable.
                m["gap"] = bool(m["matched"])
                continue
            m["matched"] += 1
            if m.get("gap"):
                m["fragments"] += 1
                m["gap"] = False
            tid = row[0]
            m["ids"][tid] += 1
            if m["last_id"] is not None and tid != m["last_id"]:
                m["switches"] += 1
            m["last_id"] = tid
            m["errors"].append(state_distance(s, row[1:3]))
            v = np.asarray(row[4:6], float)
            if s.get("velocity") is not None:
                m["vel_errors"].append(float(np.linalg.norm(v - np.asarray(s["velocity"]))))
            if types[s["id"]] == "static_obstacle":
                m["speeds_when_static"].append(float(np.linalg.norm(v)))
            if s.get("motion") is not None:
                m["motion"][(s["motion"], row[8])] += 1

    result = dict(selection=selection, scored_frames=scored, objects={})
    result["missing_frames"] = len(doc["frames"]) - scored
    result["idf1"] = idf1(pairs, gt_frames, hyp_frames)
    result["idf1_per_visit"] = idf1(visit_pairs, visit_frames, hyp_frames)
    for oid, m in per.items():
        motion = m.pop("motion")
        mov = {k: motion[("moving", k)] for k in "MSU"}
        sta = {k: motion[("stationary", k)] for k in "MSU"}
        predicted_moving = mov["M"] + sta["M"]
        errors, vel = m.pop("errors"), m.pop("vel_errors")
        speeds = m.pop("speeds_when_static")
        ids = m.pop("ids")
        for key in ("last_id", "gap"):
            m.pop(key)
        result["objects"][oid] = dict(
            m,
            type=types[oid],
            coverage=_ratio(m["matched"], m["detectable"]),
            unique_ids=len(ids),
            dominant_share=_ratio(max(ids.values()), m["matched"]) if ids else None,
            position_error_median=float(np.median(errors)) if errors else None,
            velocity_error_median=float(np.median(vel)) if vel else None,
            velocity_error_frames=len(vel),
            static_speed_median=float(np.median(speeds)) if speeds else None,
            static_speed_p95=float(np.percentile(speeds, 95)) if speeds else None,
            motion_moving=mov,
            motion_stationary=sta,
            moving_recall=_ratio(mov["M"], sum(mov.values())),
            moving_precision=_ratio(mov["M"], predicted_moving),
            stationary_recall=_ratio(sta["S"], sum(sta.values())),
            false_moving_rate=_ratio(sta["M"], sum(sta.values())),
            recovery=recoveries(events[oid], recover_s),
        )
    return result


def recoveries(events, recover_s=1.0):
    """Identity kept across each interruption of detectability.

    An event starts when a detectable object becomes partial/occluded/out of view or loses
    its match, and ends when it is detectable and matched again. The identity is recovered if
    the first matched ID within ``recover_s`` after re-appearance equals the last ID before.
    Only interruptions caused by occlusion or partial view are counted (out-of-view exits
    need re-identification, which no tracker variant attempts); the status is reported.
    """
    out = []
    last_id, gap_start, gap_kind, reappeared = None, None, None, None
    for t, status, tid in events:
        if status in ("occluded", "partial") and last_id is not None and gap_start is None:
            gap_start, gap_kind = t, status
        if status == "out_of_view":
            last_id, gap_start, gap_kind, reappeared = None, None, None, None
            continue
        if gap_start is not None and status == "visible":
            reappeared = reappeared or t
            if tid is not None:
                out.append(
                    dict(
                        start=gap_start,
                        kind=gap_kind,
                        duration=reappeared - gap_start,
                        before=last_id,
                        after=tid,
                        delay=t - reappeared,
                        kept=tid == last_id and t - reappeared <= recover_s,
                    )
                )
                gap_start, gap_kind, reappeared = None, None, None
        if tid is not None and status == "visible" and gap_start is None:
            last_id = tid
    return out
