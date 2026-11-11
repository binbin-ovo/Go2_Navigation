"""Motion-centric geometry metrics. Truth is used only here, after perception."""

from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment


def score_frames(frames, gate=0.5):
    counts = defaultdict(int)
    errors = []
    vel = []
    static_errors = []
    latency = []
    coverage = defaultdict(list)
    events = []
    pending = {}
    was_moving = {}
    dims = []
    for row, truth in frames:
        t = row["timestamp"]
        objects = row["objects"]
        visible = [g for g in truth if g["visible"]]
        obs = [o for o in objects if o["status"] == "observed_filtered"]
        counts["stale_boxes"] += sum(o["status"] == "stale" for o in objects)
        counts["all_boxes"] += len(objects)
        counts["detectable"] += len(visible)
        latency.append(row["diagnostics"]["processing_ms"])
        matched = {}
        if visible and obs:
            cost = np.array(
                [
                    [
                        np.linalg.norm(
                            np.array(g["center"])[:2] - np.array(o["center"])[:2]
                        )
                        for o in obs
                    ]
                    for g in visible
                ]
            )
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] <= gate:
                    matched[visible[i]["id"]] = obs[j]
            counts["duplicate_frames"] += sum((r <= gate).sum() > 1 for r in cost)
        for g in visible:
            identity = g["id"]
            moving = np.linalg.norm(np.asarray(g["velocity"])[:2]) > 0.2
            o = matched.get(identity)
            coverage[identity].append((t, o is not None))
            if moving and not was_moving.get(identity, False):
                pending[identity] = [t, None, None]
            was_moving[identity] = moving
            counts["moving_truth" if moving else "stationary_truth"] += 1
            if o is not None:
                counts["matched"] += 1
                error = np.asarray(o["center"]) - g["center"]
                errors.append(np.linalg.norm(error[:2]))
                dims.append(
                    np.linalg.norm(np.asarray(o["dimensions"]) - g["dimensions"])
                )
                vel.append(
                    np.linalg.norm(
                        np.asarray(o["velocity"])[:2] - np.asarray(g["velocity"])[:2]
                    )
                )
                if not moving:
                    static_errors.append(error[:2])
                    counts["stationary_matched"] += 1
                    counts["false_moving"] += o["motion_state"] == "MOVING"
                else:
                    counts["moving_recalled"] += o["motion_state"] == "MOVING"
                if identity in pending:
                    if pending[identity][1] is None:
                        pending[identity][1] = t - pending[identity][0]
                    if (
                        moving
                        and o["motion_state"] == "MOVING"
                        and pending[identity][2] is None
                    ):
                        pending[identity][2] = t - pending[identity][0]
            if not moving and identity in pending:
                events.append(pending.pop(identity))
        for o in obs:
            low = np.asarray(o["center"]) - np.asarray(o["dimensions"]) / 2
            high = np.asarray(o["center"]) + np.asarray(o["dimensions"]) / 2
            inside = sum(
                np.all(np.asarray(g["center"]) >= low - 0.05)
                and np.all(np.asarray(g["center"]) <= high + 0.05)
                for g in visible
            )
            counts["merged_boxes"] += inside > 1
    events.extend(pending.values())

    def ratio(a, b):
        return counts[a] / counts[b] if counts[b] else None

    longest = []
    for values in coverage.values():
        run = best = 0
        last = None
        for t, v in values:
            if last is not None and t - last > 0.2:
                run = 0
            run = run + 1 if v else 0
            best = max(best, run)
            last = t
        longest.append(best / len(values))
    return dict(
        counts={key: int(value) for key, value in counts.items()},
        coverage=ratio("matched", "detectable"),
        stationary_false_moving_rate=ratio("false_moving", "stationary_matched"),
        moving_recall=ratio("moving_recalled", "moving_truth"),
        stale_fraction=ratio("stale_boxes", "all_boxes"),
        position_error_mean_m=float(np.mean(errors)) if errors else None,
        stationary_jitter_xy_m=float(np.linalg.norm(np.std(static_errors, axis=0)))
        if len(static_errors) > 1
        else None,
        velocity_error_mean_mps=float(np.mean(vel)) if vel else None,
        dimension_error_mean_m=float(np.mean(dims)) if dims else None,
        longest_observed_fraction=float(np.mean(longest)) if longest else None,
        transition_events=[
            dict(onset=t, detection_delay_s=d, moving_delay_s=m) for t, d, m in events
        ],
        processing_ms=dict(
            mean=float(np.mean(latency)), p95=float(np.percentile(latency, 95))
        ),
    )
