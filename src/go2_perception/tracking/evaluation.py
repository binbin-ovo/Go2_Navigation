"""Geometric GT matching, explicit denominators, and global identity F1 (IDF1)."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate(frames, outputs, latencies_ms, match_distance=1.0):
    if len(frames) != len(outputs):
        raise ValueError("Frame/output length mismatch")
    if match_distance <= 0:
        raise ValueError("Invalid evaluation gate")
    report = {
        "frames": len(frames),
        "tracking_latency_ms_mean": float(np.mean(latencies_ms))
        if latencies_ms
        else None,
        "tracking_throughput_hz": len(latencies_ms) / (sum(latencies_ms) / 1000)
        if sum(latencies_ms) > 0
        else None,
    }
    if any(f.ground_truth is None for f in frames):
        report["ground_truth_metrics"] = "unavailable: sequence is not fully annotated"
        return report
    position_errors = []
    velocity_errors = []
    switches = 0
    matched = 0
    total_gt = 0
    total_predictions = 0
    previous = {}
    counts = {}
    first_seen = {}
    first_match = {}
    identity_counts = {}
    for frame, tracks in zip(frames, outputs):
        truth = frame.ground_truth
        tracks = [t for t in tracks if t.status == "Confirmed"]
        total_gt += len(truth)
        total_predictions += len(tracks)
        for g in truth:
            if g["frame_id"] != "odom":
                raise ValueError("Evaluation GT must be in odom")
            first_seen.setdefault(g["object_id"], frame.timestamp)
        if any(
            t.frame_id != "odom" or t.velocity_reference != "odometry" for t in tracks
        ):
            raise ValueError("Cannot evaluate world GT against fixed/local tracks")
        if not truth or not tracks:
            continue
        cost = np.array(
            [
                [
                    np.linalg.norm(np.asarray(g["position"])[:2] - t.position[:2])
                    for t in tracks
                ]
                for g in truth
            ]
        )
        # IDF1 uses globally consistent identity assignment over all eligible
        # GT/prediction pairs, not only framewise Hungarian matches.
        for gi, tj in zip(*np.where(cost <= match_distance)):
            key = (truth[gi]["object_id"], tracks[tj].track_id)
            identity_counts[key] = identity_counts.get(key, 0) + 1
        # Dummy assignments prefer all feasible matches over infeasible matches.
        penalty = len(truth) + len(tracks) + 1
        augmented = np.full((len(truth), len(tracks) + len(truth)), float(penalty))
        augmented[:, : len(tracks)] = np.where(
            cost <= match_distance, cost / match_distance, penalty * 3
        )
        rows, cols = linear_sum_assignment(augmented)
        for i, j in zip(rows, cols):
            if j >= len(tracks) or cost[i, j] > match_distance:
                continue
            g, t = truth[i], tracks[j]
            identifier = g["object_id"]
            matched += 1
            position_errors.append(np.abs(t.position[:2] - g["position"][:2]))
            velocity_errors.append(np.abs(t.velocity - g["velocity"]))
            if identifier in previous and previous[identifier] != t.track_id:
                switches += 1
            previous[identifier] = t.track_id
            counts[identifier] = counts.get(identifier, 0) + 1
            first_match.setdefault(identifier, frame.timestamp)
    gids = sorted(first_seen)
    tids = sorted(
        {t.track_id for tracks in outputs for t in tracks if t.status == "Confirmed"}
    )
    idtp = 0
    if gids and tids:
        matrix = np.array(
            [[identity_counts.get((g, t), 0) for t in tids] for g in gids]
        )
        rows, cols = linear_sum_assignment(-matrix)
        idtp = int(matrix[rows, cols].sum())
    delays = [first_match[g] - first_seen[g] for g in first_match]
    report.update(
        position_mae_m=float(np.mean(position_errors)) if position_errors else None,
        velocity_mae_mps=float(np.mean(velocity_errors)) if velocity_errors else None,
        track_continuity=matched / total_gt if total_gt else None,
        id_switches=switches,
        initialization_delay_s_mean=float(np.mean(delays)) if delays else None,
        initialized_objects=len(delays),
        ground_truth_objects=len(gids),
        idf1=2 * idtp / (total_gt + total_predictions)
        if total_gt + total_predictions
        else None,
        matched_gt_frames=matched,
        total_gt_frames=total_gt,
        confirmed_prediction_frames=total_predictions,
    )
    return report
