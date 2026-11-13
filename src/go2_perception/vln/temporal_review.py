"""Offline panels/videos and reference-aware metrics, independent of model inputs."""

import subprocess

import cv2
import numpy as np


def _lines(text, width=85):
    import textwrap

    return "\n".join(textwrap.wrap(str(text), width, replace_whitespace=False))


def render_review(proposal, prepared, path, reference=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 9), layout="constrained")
    gs = fig.add_gridspec(3, 4, height_ratios=[3.2, 1.3, 2.0])
    current = fig.add_subplot(gs[0, :3])
    current.imshow(prepared.views[-1].image)
    current.axis("off")
    current.set_title(_lines(proposal["instruction"], 80), fontsize=12)
    info = fig.add_subplot(gs[0, 3])
    info.axis("off")
    t = proposal["timing"]
    s = prepared.selection
    info.text(
        0,
        1,
        _lines(
            f"{proposal['episode_id']} / {proposal['decision_id']}\nMode: {proposal['mode']}\n"
            f"Proposal: {proposal['proposal']}\nTask status: {proposal['task_status'] or 'UNKNOWN'}\n"
            f"Parse: {proposal['parse_status']}\nDecision: {s['decision_time_s']:.3f}s\n"
            f"Generate: {t['generation_s']:.2f}s\nReplay image age at output: {t['replay_observation_age_at_output_s']:.2f}s\n"
            f"History images encoded: {proposal['history_images_consumed']}\n"
            "Metric goal: unavailable\nCompletion: unverified",
            35,
        ),
        va="top",
        fontsize=10,
    )
    n = len(prepared.views)
    for i in range(4):
        ax = fig.add_subplot(gs[1, i])
        ax.axis("off")
        if i < n:
            # These are the actual normalized tensor views inverted for display.
            ax.imshow(prepared.previews[i])
            f = s["frames"][i]
            ax.set_title(
                f"{'current' if i == n - 1 else 'history'} {f['frame_id']}\n"
                f"{f['observation_s'] - s['decision_time_s']:+.3f}s",
                fontsize=8,
            )
        else:
            ax.text(0.5, 0.5, "not supplied", ha="center", va="center", fontsize=9)
    text = fig.add_subplot(gs[2, :])
    text.axis("off")
    native = _lines(proposal["native_model_output"], 130)
    if len(native) > 1800:
        native = native[:1800] + " [panel truncated; full text saved]"
    label = "Human reference: PENDING / unavailable"
    if reference:
        label = (
            f"EVALUATION ONLY — {reference['reviewer']['type']} / {reference['reviewer']['status']}: "
            + str(reference["decision"])
        )
    text.text(
        0,
        1,
        "Model-native response:\n"
        + native
        + "\n\n"
        + _lines(label, 130)
        + "\nNo camera-base registration or measured depth. No safe route or 3D goal inferred.\n"
        "Recorded images follow the operator route; proposed-action outcomes are unobserved.",
        va="top",
        fontsize=9,
    )
    fig.suptitle("OFFLINE PROPOSAL — NOT EXECUTED", fontsize=17, color="#9b251d")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_video(panel_paths, output, *, fps=1):
    """Encode the fixed-size review panels; replay timing is explicitly illustrative."""
    if not panel_paths or fps <= 0:
        raise ValueError("Need panels and positive FPS")
    first = cv2.imread(str(panel_paths[0]))
    if first is None:
        raise ValueError("Unreadable panel")
    h, w = first.shape[:2]
    w += w % 2
    h += h % 2
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for path in panel_paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Unreadable panel {path}")
            process.stdin.write(cv2.resize(image, (w, h)).tobytes())
        process.stdin.close()
        process.stdin = None
        _, err = process.communicate(timeout=60)
        if process.returncode:
            raise RuntimeError(err.decode())
    except BaseException:
        process.kill()
        process.wait()
        raise


def summarize(proposals, references=None):
    """Only reviewed human labels contribute to human agreement/false-completion metrics."""
    references = references or {}
    agreements = []
    false_complete = []
    completion_checks = []
    human_progress = []
    for p in proposals:
        ref = references.get((p["episode_id"], p["decision_id"]))
        if (
            not ref
            or ref["reviewer"]["type"] != "human"
            or ref["reviewer"]["status"] != "reviewed"
        ):
            continue
        d = ref["decision"]
        if d["acceptable_actions"]:
            agreements.append(
                p["parse_status"] != "rejected"
                and p["proposal"] in d["acceptable_actions"]
            )
        if d["completion_status"] != "unknown" and p["task_status"] == "COMPLETE":
            false_complete.append(d["completion_status"] != "complete")
        if (
            d["completion_status"] != "unknown"
            and p["parse_status"] != "rejected"
            and p["task_status"] is not None
        ):
            completion_checks.append(
                (p["task_status"] == "COMPLETE")
                == (d["completion_status"] == "complete")
            )
        if d["instruction_progress"] is not None:
            human_progress.append(
                {
                    "episode_id": p["episode_id"],
                    "decision_id": p["decision_id"],
                    "reference_progress": d["instruction_progress"],
                    "model_reason": p["reason"],
                    "consistency": "requires manual adjudication",
                }
            )
    repeated = opposite = 0
    groups = {}
    for p in proposals:
        groups.setdefault((p["episode_id"], p["mode"]), []).append(p)
    for rows in groups.values():
        rows.sort(key=lambda p: p["selected_observations"]["decision_time_s"])
        for a, b in zip(rows, rows[1:]):
            if a["parse_status"] == "rejected" or b["parse_status"] == "rejected":
                continue
            repeated += int(a["proposal"] == b["proposal"])
            opposite += int(
                {a["proposal"], b["proposal"]} == {"TURN_LEFT", "TURN_RIGHT"}
            )
    lat = [p["timing"]["generation_s"] for p in proposals]
    return {
        "inference_calls": len(proposals),
        "episodes": len({p["episode_id"] for p in proposals}),
        "unique_decisions": len(
            {(p["episode_id"], p["decision_id"]) for p in proposals}
        ),
        "accepted_outputs": sum(p["parse_status"] == "accepted" for p in proposals),
        "partial_outputs": sum(p["parse_status"] == "partial" for p in proposals),
        "rejected_outputs": sum(p["parse_status"] == "rejected" for p in proposals),
        "model_completion_claims": sum(
            p["task_status"] == "COMPLETE" for p in proposals
        ),
        "model_completion_claims_scope": "structured task_status only; claims in native prose require separate review",
        "human_action_comparisons": len(agreements),
        "human_action_agreement": None
        if not agreements
        else sum(agreements) / len(agreements),
        "human_checked_completion_claims": len(false_complete),
        "false_completion_declarations": None
        if not false_complete
        else sum(false_complete),
        "human_completion_consistency": None
        if not completion_checks
        else sum(completion_checks) / len(completion_checks),
        "human_completion_comparisons": len(completion_checks),
        "instruction_progress_checks": human_progress,
        "semantic_evaluation_status": "pending human references"
        if not agreements
        else "human references available; progress may need adjudication",
        "repeated_consecutive_actions": repeated,
        "opposite_consecutive_turn_pairs": opposite,
        "sequence_flags_note": "Review flags, not automatic errors; viewpoints can change.",
        "generation_latency_s": {
            "mean": float(np.mean(lat)),
            "median": float(np.median(lat)),
            "min": min(lat),
            "max": max(lat),
        }
        if lat
        else None,
        "peak_process_rss_bytes": max(
            (p["memory"]["process_lifetime_peak_rss_bytes"] for p in proposals),
            default=0,
        ),
        "closed_loop_navigation_success": None,
        "collision_avoidance_success": None,
        "execution_authorized": False,
    }
