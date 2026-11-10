"""Phase 4.2 readiness: forecasting windows on identity-validated real pedestrian trajectories.

Uses the Phase 4 contract: 0.4 s grid, 8 observed and 12 future positions, odom metres,
go2_robot clock, bounded interpolation between real samples only (``resample_observed``,
max gap 0.15 s), no extrapolation, no window across an exit from view or an identity change.

Two history sources per eligible window, both scored against the annotated future:
- ``gt_history``: the annotated person positions themselves (isolates the forecaster);
- ``tracker_history``: the tracker's own filtered estimates of the track matched to the
  annotated person, accepted only if that same track ID was the person's match at every
  annotated frame of the history and is observed at the origin. The history goes through the
  Phase 4 adapter (``forecasting.adapter.track_history``) with ``assume_pedestrian=True``, which
  is justified here only by the ground-truth match, not by any tracker semantics.

Constant velocity (``forecasting.baseline.constant_velocity``) is always evaluated. Learned
checkpoints are evaluated only if given and present; ground truth never selects or alters a
prediction. Writes a local per-window JSON (private), an aggregate JSON, and an overlay file
for ``render_tracking_video.py --forecasts``.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2_perception.core.tracking_types import TrackOutput  # noqa: E402
from go2_perception.forecasting.adapter import track_history  # noqa: E402
from go2_perception.forecasting.baseline import constant_velocity  # noqa: E402
from go2_perception.forecasting.data import resample_observed  # noqa: E402
from go2_perception.forecasting.metrics import errors  # noqa: E402
from go2_perception.tracking import ground_truth as gt  # noqa: E402

DT, OBS, PRED, MAX_GAP = 0.4, 8, 12, 0.15
GATES = dict(person=0.5, static_obstacle=0.15, other=0.5)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def visits(doc, oid):
    """{visit: (times, xy)} of scorable annotated states; a visit ends at an exit from view."""
    out = collections.defaultdict(lambda: ([], []))
    for f in doc["frames"]:
        for s in f["objects"]:
            if s["id"] == oid and gt.scorable(s) and s.get("xy") is not None:
                out[s.get("visit", 0)][0].append(f["t"])
                out[s.get("visit", 0)][1].append(s["xy"])
    return {k: (np.array(t), np.array(p)) for k, (t, p) in out.items()}


def gt_window(times, xy, origin):
    hist_t = origin - np.arange(OBS - 1, -1, -1) * DT
    fut_t = origin + np.arange(1, PRED + 1) * DT
    hist, hist_i = resample_observed(times, xy, hist_t, max_gap=MAX_GAP)
    fut, fut_i = resample_observed(times, xy, fut_t, max_gap=MAX_GAP)
    return hist, fut, hist_i, fut_i


def person_matches(doc, records, oid):
    """{frame time: matched track row or None} where the annotated object is scorable."""
    out = {}
    for f, _, states, _, matches, _ in gt.match_sequence(doc, records, "confirmed", GATES):
        s = next((s for s in states if s["id"] == oid), None)
        if s is not None and gt.scorable(s):
            out[f["t"]] = matches.get(oid)
    return out


def tracker_history(records, matched, tid, origin):
    """Adapter-checked history of track ``tid`` ending at ``origin`` (raises with a reason)."""
    t_first = origin - (OBS - 1) * DT - MAX_GAP
    window = [row for t, row in matched.items() if t_first <= t <= origin + 1e-6]
    if not window or any(row is None or row[0] != tid for row in window):
        raise ValueError("identity not validated: another or no track matched the person in the history")
    history, current = [], None
    for r in records:
        if t_first <= r["t"] <= origin + 1e-6:
            row = next((x for x in r["tracks"] if x[0] == tid), None)
            if row is None:
                continue
            observed = row[9] == 0 and row[7] != "L"
            history.append(dict(timestamp=r["t"], position=np.array([row[1], row[2], row[3]]), observed=observed, track_id=tid))
            if abs(r["t"] - origin) < 1e-6:
                current = row
    if current is None:
        raise ValueError("track absent at origin")
    track = TrackOutput(
        track_id=tid, timestamp=origin, frame_id="odom", clock_domain="go2_robot", velocity_reference="odometry",
        position=np.array(current[1:4]), velocity=np.array(current[4:6]), state_covariance=np.eye(4),
        bbox_min=None, bbox_max=None, class_name=None, detection_confidence=None,
        status={"C": "Confirmed", "T": "Tentative", "L": "Lost"}[current[7]], motion_state=current[8],
        last_observation_timestamp=origin if current[9] == 0 else origin - 1, age_s=0.0, hits=0,
        consecutive_misses=current[9], history=history,
    )  # fmt: skip
    values, _, interpolated = track_history(track, OBS, DT, resample=True, max_gap_s=MAX_GAP, assume_pedestrian=True)
    return values, interpolated


def summarize(rows):
    if not rows:
        return dict(count=0)
    p = np.array([r["pred"] for r in rows])
    y = np.array([r["future"] for r in rows])
    ade, fde = errors(p, y)
    origins = sorted(r["origin_rel"] for r in rows)
    independent, last = 0, -1e9
    for o in origins:  # windows whose 7.6 s spans do not overlap
        if o - last >= (OBS - 1 + PRED) * DT:
            independent += 1
            last = o
    return dict(count=len(rows), independent_windows=independent, ade_m=float(ade.mean()), fde_m=float(fde.mean()),
                ade_median_m=float(np.median(ade)), fde_median_m=float(np.median(fde)))  # fmt: skip


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recordings", nargs="+", default=["test_b", "test_a2"])
    p.add_argument("--variants", nargs="+", default=["baseline", "improved", "refined"])
    p.add_argument("--annotations", default="data/annotations")
    p.add_argument("--tracks", default="data/outputs/mac/ground_truth_eval/tracks")
    p.add_argument("--checkpoints", nargs="*", default=["data/outputs/forecasting/eth_base/best.pt"],
                   help="single-agent Phase 4 checkpoints to evaluate if present")  # fmt: skip
    p.add_argument("--social-checkpoints", nargs="*", default=[])
    p.add_argument("--output", default="data/outputs/mac/forecasting_gt/windows.json")
    p.add_argument("--aggregate", default="data/outputs/mac/forecasting_gt/aggregate.json")
    p.add_argument("--overlay-every", type=float, default=2.0, help="seconds between overlay origins")
    args = p.parse_args()
    models = {}
    for path in args.checkpoints + args.social_checkpoints:
        models[path] = "not present on this machine" if not Path(path).exists() else "present (not evaluated by this script yet)"
    full, agg, overlay = {}, dict(contract=dict(dt_s=DT, observed=OBS, predicted=PRED, max_gap_s=MAX_GAP), checkpoints=models, recordings={}), {}
    for name in args.recordings:
        doc = gt.load(Path(args.annotations) / f"{name}.json")
        people = [k for k, v in doc["objects"].items() if v["type"] == "person"]
        records = {v: jsonl(Path(args.tracks) / v / name / "tracks.jsonl") for v in args.variants}
        frame_times = [f["t"] for f in doc["frames"]]
        rows, rejected = collections.defaultdict(list), collections.defaultdict(collections.Counter)
        visit_spans = {}
        for oid in people:
            matched = {v: person_matches(doc, records[v], oid) for v in args.variants}
            for visit, (times, xy) in visits(doc, oid).items():
                visit_spans[f"{oid}/{visit}"] = round(float(times[-1] - times[0]), 2)
                for origin in [t for t in frame_times if times[0] <= t <= times[-1]]:
                    try:
                        hist, fut, hist_i, fut_i = gt_window(times, xy, origin)
                    except ValueError:
                        rejected["gt_history"]["no complete 8+12 annotated window"] += 1
                        continue
                    base = dict(object=oid, visit=visit, origin=origin, origin_rel=round(origin - doc["t0"], 3), future=fut.tolist())
                    pred = constant_velocity(hist[None], PRED, DT)[0]
                    rows["gt_history"].append(dict(base, history=hist.tolist(), pred=pred.tolist(), interpolated=int(hist_i.sum() + fut_i.sum())))
                    for v in args.variants:
                        match = matched[v].get(origin)
                        if match is None:
                            rejected[v]["no Confirmed track matched at origin"] += 1
                            continue
                        try:
                            th, ti = tracker_history(records[v], matched[v], match[0], origin)
                        except ValueError as e:
                            rejected[v][str(e).split(":")[0][:60]] += 1
                            continue
                        tpred = constant_velocity(th[None], PRED, DT)[0]
                        rows[v].append(dict(base, track=match[0], history=th.tolist(), pred=tpred.tolist(), interpolated=int(ti.sum())))
        full[name] = dict(rows=rows, rejected={k: dict(v) for k, v in rejected.items()})
        agg["recordings"][name] = dict(
            visits_s=visit_spans,
            gt_history_cv=summarize(rows["gt_history"]),
            tracker_history_cv={v: summarize(rows[v]) for v in args.variants},
            paired_gt_vs_tracker={
                v: dict(gt=summarize([r for r in rows["gt_history"] if r["origin"] in {x["origin"] for x in rows[v]}]), tracker=summarize(rows[v]))
                for v in args.variants
            },
            rejected={k: dict(v) for k, v in rejected.items()},
        )
        # Overlay: one origin every --overlay-every seconds, held until the next.
        picks, last = [], -1e9
        for r in rows["gt_history"]:
            if r["origin"] - last >= args.overlay_every:
                tracker = next((x for x in rows["baseline"] if x["origin"] == r["origin"]), None) if "baseline" in rows else None
                picks.append(dict(origin=r["origin"], gt_history=r["history"], cv_gt=r["pred"], future=r["future"],
                                  cv_tracker=None if tracker is None else tracker["pred"], track=None if tracker is None else tracker["track"]))  # fmt: skip
                last = r["origin"]
        overlay[name] = picks
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(full) + "\n")
    Path(args.aggregate).write_text(json.dumps(agg, indent=1) + "\n")
    Path(args.output).with_name("overlay.json").write_text(json.dumps(overlay) + "\n")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
