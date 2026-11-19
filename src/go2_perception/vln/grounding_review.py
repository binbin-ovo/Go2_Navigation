"""Reconstruct five-move evidence from closed local files. No model or sensor imports."""

from pathlib import Path

from .bounded_context import validate_context
from .supervised import digest
from .prediction_archive import file_hash, load, prediction, require


def saved_input(session, prediction_id):
    """Check the recorded prompt, selected history, original inputs and displayed frame."""
    session = Path(session).resolve()
    pred, _ = prediction(session, prediction_id)
    request = session / pred["request_directory"]
    job, snapshot, native = [
        load(request / name) for name in ("job.json", "snapshot.json", "native/result.json")
    ]
    trial = load(session / "trials" / pred["trial_id"] / "trial.json")
    require(
        trial["instruction"] == job["instruction"]
        and trial["instruction_revision"]
        == pred["instruction_revision"]
        == snapshot["identity"]["instruction_revision"]
        and trial["session_id"] == snapshot["identity"]["session_id"],
        "instruction binding",
    )
    plan = snapshot["bounded_context"]
    validate_context(plan, job)
    ids = plan["source_indices"]
    require(
        native["native_selected_frame_indices"] == ids
        and native["context_policy"]["policy"] == plan["policy"]
        and snapshot["annotations_consumed"] is False
        and snapshot["previous_actions_supplied"] is False,
        "native history policy",
    )
    require(
        snapshot["source_cutoff_s"] == job["decision_time_s"]
        and len(job["frames"]) == len(snapshot["history"])
        and snapshot["current_source"] == snapshot["history"][-1]["source"],
        "causal history cutoff",
    )
    frames = []
    for i, frame, item in zip(ids, job["frames"], snapshot["history"], strict=True):
        path = Path(frame["path"]).resolve()
        src = item["source"]
        require(path.parent == request and file_hash(path) == frame["sha256"], "prepared frame")
        require(
            src["clock_epoch"] == job["clock_epoch"]
            and src["observation_s"]
            == frame["observation_s"]
            <= src["available_s"]
            == frame["available_s"]
            <= job["decision_time_s"],
            "frame identity and causal availability",
        )
        png = request / "native" / f"input_{i:04d}.png"
        frames.append(
            dict(
                index=i,
                frame_id=src["frame_id"],
                observation_s=frame["observation_s"],
                available_s=frame["available_s"],
                path=str(png),
                sha256=file_hash(png),
            )
        )
    image = pred["source_image"]
    require(
        Path(frames[-1]["path"]) == session / image["path"]
        and image["causal_grid_index"] == ids[-1]
        and image["job_frame_index"] == len(frames) - 1
        and pred["source_frame"]["frame_id"] == frames[-1]["frame_id"],
        "displayed current frame",
    )
    return dict(
        trial=trial,
        prediction=pred,
        request=request,
        job=job,
        snapshot=snapshot,
        native=native,
        frames=frames,
    )


def episode_decisions(session):
    """Join each recorded observation, review, selected action and send result by ID."""
    import json

    def rows(path):
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)

    session = Path(session)
    summary = load(session / "episode_summary.json")
    episode = summary["episode_id"]
    decisions, actions = {}, {}
    for event in rows(session / "executor/executor_events.jsonl"):
        if event.get("nonce") != episode:
            continue
        kind = event.get("type")
        if kind == "episode_rejected_observation":
            decisions[event["prediction_id"]] = dict(
                grounding=None, reviews=[], selection=None, result=None, rejection=event["reason"]
            )
        elif kind == "episode_observation":
            prop, ev = event["proposal"], event["evidence"]
            ident = prop["identity"]
            require(
                ident["prediction_id"] == ev["prediction_id"]
                and ident["source_image_sha256"] == ev["image_sha256"],
                "observation evidence",
            )
            decisions[ev["prediction_id"]] = dict(grounding=ev, reviews=[], selection=None, result=None)
        elif kind == "episode_dispatch":
            ticket, sel = event["ticket"], event["selection"]
            pid = ticket["proposal"]["identity"]["prediction_id"]
            require(
                pid in decisions
                and ticket["request_id"] not in actions
                and decisions[pid]["selection"] is None
                and sel["prediction_id"] == pid
                and sel["action"] == ticket["action"]
                and digest(event["binding"]) == ticket["request_id"]
                and event["binding"]["proposal"] == digest(ticket["proposal"])
                and event["binding"]["grounding"] == digest(decisions[pid]["grounding"])
                and event["binding"]["selection"] == digest(sel),
                "dispatch identity",
            )
            decisions[pid]["selection"] = sel
            actions[ticket["request_id"]] = pid
        elif kind == "result" and event["result"].get("request_id") in actions:
            res = event["result"]
            decisions[actions[res["request_id"]]]["result"] = {
                key: res.get(key)
                for key in (
                    "status",
                    "reason",
                    "command_send_statuses",
                    "command_robot_acknowledged",
                    "measured_progress",
                    "observed_stationary",
                    "ownership_released",
                    "post_release_stationary",
                )
            }
    for event in rows(session / "events.jsonl"):
        if event.get("event") != "episode_instance_review":
            continue
        row = decisions.get(event["prediction_id"])
        require(
            row
            and row["grounding"]
            and event["image_sha256"] == row["grounding"]["image_sha256"]
            and event["target"] == row["grounding"]["target"],
            "review bound to source and target",
        )
        row["reviews"].append({k: event.get(k) for k in ("decision", "roi", "image_sha256")})
    return summary, decisions
