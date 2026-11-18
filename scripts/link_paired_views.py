"""Link two single-answer trials taken before and after a MANUAL viewpoint change.

Read-only over a saved run directory; appends one record to
``<run>/evaluation/paired_views.jsonl`` (operator-requested evaluation metadata). It checks,
from archived evidence only:

- each trial's model-input frame timestamps (host-receipt wall clock of this computer);
- odometry yaw recorded between and during the trials (``--record-lidar-continuously``),
  using its host-receipt wall time, to estimate when the operator's turn started and ended;
- whether every input frame of the second answer was received after the turn ended.

The two answers are independent observations: each Ask starts a fresh instruction history.
This is not continuous temporal navigation and not autonomous search; the operator chose the
direction and turned the robot with the handheld remote. Model outputs, operator verdicts
and operator words are copied from their own files and kept apart.
"""

import argparse
import json
from pathlib import Path

import numpy as np

SCHEMA = "go2-vln-paired-view-evaluation-v1"


def jsonl(path):
    path = Path(path)
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def yaw(q):
    x, y, z, w = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def trial_evidence(run, trial_id):
    folder = run / "trials" / trial_id
    trial = json.loads((folder / "trial.json").read_text())
    end = json.loads((folder / "trial_end.json").read_text()) if (folder / "trial_end.json").exists() else None
    preds = jsonl(folder / "predictions.jsonl")
    labels = [h for h in jsonl(folder / "human_labels.jsonl") if h.get("schema") == "go2-vln-guided-answer-review-v1"]
    if not preds:
        raise SystemExit(f"{trial_id}: no prediction recorded")
    pred = preds[0]
    snapshot = json.loads((run / pred["request_directory"] / "snapshot.json").read_text())
    # This process's host-monotonic receipt time of every archived model-input frame; odometry
    # receipt is converted to the same clock with its own wall-minus-monotonic sample.
    stamps = [h["source"]["received_monotonic_s"] for h in snapshot["history"]]
    return {
        "trial_id": trial_id, "instruction": trial["instruction"], "operator_text": trial["operator_text"],
        "instruction_revision": trial["instruction_revision"],
        "started_wall_utc": trial["started_wall_utc"], "end_reason": None if end is None else end["end_reason"],
        "model_output": {k: pred.get(k) for k in (
            "prediction_id", "native_output", "pointing", "target_fraction", "opos_not_visible",
            "label_at_receipt", "rejection_reason_at_receipt", "inference_duration_s", "source_image")},
        "receipt_freshness": {"label_at_receipt": pred["label_at_receipt"],
                              "display_timing_at_receipt": pred.get("display_timing_at_receipt")},
        "input_frames": {"count": len(stamps), "first_s": min(stamps), "last_s": max(stamps),
                         "clock": "host monotonic receipt (this process; not comparable across machines)",
                         "model_stamp_clock": pred["source_frame"]["clock"],
                         "clock_epoch": pred["source_frame"]["clock_epoch"]},
        "operator_verdicts": [{k: h.get(k) for k in (
            "choice", "choice_text", "verbatim_comment", "reviewer", "reviewed_wall_utc",
            "observation_age_at_review_s", "displayed_image_sha256")} for h in labels],
    }


def odometry(run, t0, t1):
    rows = [r for r in jsonl(run / "lidar/index.jsonl") if r.get("file") and r.get("host_receipt_interval_wall_s")]
    # Chunks are indexed in wall time; samples are filtered below after conversion to monotonic.
    times, quats, epochs, gaps = [], [], set(), [r for r in jsonl(run / "lidar/index.jsonl") if r.get("dropped_not_recorded")]
    for r in rows:
        data = np.load(run / "lidar" / r["file"])
        if "odom_wall_minus_monotonic" not in data:
            raise SystemExit("LiDAR chunks lack odometry wall-minus-monotonic samples (recorded by an older version)")
        times.append(data["odom_host_recv_wall"] - data["odom_wall_minus_monotonic"])
        quats.append(data["odom_q_xyzw"])
        epochs.add(r["robot_epoch"])
    if not times:
        return None
    t = np.concatenate(times)
    q = np.concatenate(quats)
    order = np.argsort(t)
    keep = (t[order] >= t0) & (t[order] <= t1)
    return {"t": t[order][keep], "yaw_deg": np.degrees(np.unwrap(yaw(q[order][keep]))), "epochs": sorted(epochs),
            "dropped_chunks": len(gaps)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--before", required=True, help="trial ID of the first (pre-turn) observation")
    p.add_argument("--after", required=True, help="trial ID of the second (post-turn) observation")
    p.add_argument("--operator-direction", required=True, help="direction the OPERATOR chose, e.g. 'left'")
    p.add_argument("--operator-report", default="", help="operator's own words about the turn (verbatim)")
    p.add_argument("--threshold-deg", type=float, default=2.0)
    a = p.parse_args(argv)
    run = a.run.resolve()
    before, after = trial_evidence(run, a.before), trial_evidence(run, a.after)
    if before["operator_text"] != after["operator_text"]:
        raise SystemExit("Target wording differs between the two trials")
    window = (before["input_frames"]["first_s"] - 2.0, after["input_frames"]["last_s"] + 2.0)
    odo = odometry(run, *window)
    turn = {"available": False, "reason": "no odometry recorded in the window"}
    post_turn = {"verified": False, "reason": "no turn interval"}
    if odo is not None and len(odo["t"]) > 10:
        t, y = odo["t"], odo["yaw_deg"]
        ref = float(np.median(y[t <= before["input_frames"]["last_s"]])) if np.any(t <= before["input_frames"]["last_s"]) else float(y[0])
        fin = float(np.median(y[t >= after["input_frames"]["first_s"]])) if np.any(t >= after["input_frames"]["first_s"]) else float(y[-1])
        moved_from_ref = np.abs(y - ref) > a.threshold_deg
        not_final = np.abs(y - fin) > a.threshold_deg
        if moved_from_ref.any():
            start = float(t[np.argmax(moved_from_ref)])
            last_off = np.nonzero(not_final)[0]
            end = float(t[min(last_off[-1] + 1, len(t) - 1)]) if len(last_off) else start
            after_win = (t >= after["input_frames"]["first_s"]) & (t <= after["input_frames"]["last_s"])
            sample = float(np.median(np.diff(t))) if len(t) > 1 else None
            turn = {
                "available": True, "yaw_change_deg": float(fin - ref),
                "turn_start_host_monotonic_s": start, "turn_end_host_monotonic_s": end, "duration_s": end - start,
                "yaw_range_during_second_input_deg": float(np.ptp(y[after_win])) if after_win.any() else None,
                "odometry_samples": int(len(t)), "median_sample_interval_s": sample,
                "robot_epochs": odo["epochs"], "dropped_lidar_chunks_in_run": odo["dropped_chunks"],
                "method": f"odometry yaw (host receipt) leaves the pre-turn median by >{a.threshold_deg} deg "
                          f"(start) and last differs from the post-turn median by >{a.threshold_deg} deg (end)",
                "uncertainty": "onset/offset limited by the threshold, odometry sampling and its unknown "
                               "transport latency (a few ms); camera receipt is about 48 ms after exposure (single-boot estimate); both are this "
                               "computer's host-monotonic receipt times",
            }
            margin = after["input_frames"]["first_s"] - end
            post_turn = {"verified": bool(margin > 0), "margin_s": margin,
                         "second_input_first_s": after["input_frames"]["first_s"], "turn_end_s": end,
                         "first_input_before_turn": bool(before["input_frames"]["last_s"] < start),
                         "first_input_margin_s": start - before["input_frames"]["last_s"]}
        else:
            turn = {"available": False, "reason": f"odometry yaw never changed by more than {a.threshold_deg} deg"}
    record = {
        "schema": SCHEMA, "run": str(run), "target_operator_text": before["operator_text"],
        "design": "two independent single-answer observations (each Ask starts a fresh instruction history); "
                  "not continuous temporal navigation; not autonomous search",
        "viewpoint_change": {"executed_by": "operator with the handheld remote; the app sent nothing",
                             "direction_chosen_by": "operator", "operator_direction": a.operator_direction,
                             "operator_report_verbatim": a.operator_report, "measured": turn},
        "second_answer_uses_post_turn_view": post_turn,
        "before": before, "after": after,
        "independent_histories": before["instruction_revision"] != after["instruction_revision"],
        "model_feedback": False,
    }
    out = run / "evaluation"
    out.mkdir(exist_ok=True)
    with (out / "paired_views.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    print(json.dumps({k: record[k] for k in ("target_operator_text", "viewpoint_change",
                                             "second_answer_uses_post_turn_view", "independent_histories")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
