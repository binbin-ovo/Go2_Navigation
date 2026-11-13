"""Evaluation-only comparison and local panels; never loads a learned model."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import textwrap

import numpy as np

from go2_perception.vln.episodes import file_sha256, load_episode, load_annotations
from go2_perception.vln.pretrained import comparison_action
from go2_perception.vln.temporal_review import make_video


def dump(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def check_baseline_binding(job, baseline, provenance, manifest_sha256, mode):
    if (
        baseline["episode_id"] != job["episode_id"]
        or baseline["decision_id"] != job["decision_id"]
        or baseline["mode"] != mode
        or baseline["execution_authorized"] is not False
        or provenance["episode_manifest_sha256"] != manifest_sha256
        or provenance["selection"]["decision_time_s"] != job["decision_time_s"]
        or provenance["annotations_consumed"] is not False
    ):
        raise ValueError(
            "Baseline is bound to another package/cutoff/mode or consumed labels"
        )


def render_panel(job, result, baselines, reference, inference_dir, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig = plt.figure(figsize=(16, 10), layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=[3, 1.7, 1.8])
    ax = fig.add_subplot(grid[0, :2])
    ax.imshow(Image.open(job["frames"][-1]["path"]))
    ax.axis("off")
    ax.set_title(textwrap.fill(job["instruction"], 85))
    ax = fig.add_subplot(grid[0, 2])
    wp = result.get("waypoints")
    if wp:
        wp = np.array(wp)
        ax.plot(-wp[:, 1], wp[:, 0], "o-")
        ax.scatter([0], [0], c="black")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("right = -native lateral (m)")
        ax.set_ylabel("native forward (m)")
        ax.set_xlim(
            min(-0.25, float(-wp[:, 1].max()) - 0.05),
            max(0.25, float(-wp[:, 1].min()) + 0.05),
        )
        ax.set_ylim(
            min(-0.25, float(wp[:, 0].min()) - 0.05),
            max(0.25, float(wp[:, 0].max()) + 0.05),
        )
        ax.set_aspect("equal", adjustable="box")
        ax.text(
            0.02,
            0.98,
            f"First / last yaw: {np.degrees(wp[0, 2]):.2f} / {np.degrees(wp[-1, 2]):.2f} deg",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
    ax.set_title("Native ego waypoints\nUnregistered to Go2; no time base")
    ax = fig.add_subplot(grid[1, :])
    ax.axis("off")
    timing = result.get("timing", {})
    summary = f"LightNav native: {result.get('native_output', result.get('error'))}\n"
    summary += f"RVQ validity: {result['status']} | comparison: {result.get('comparison_action')} | decoder stop: {result.get('decoder_stop')}\n"
    summary += f"Pointing: {result.get('pointing')}\nGeneration {timing.get('native_generate_s')} s; replay age {timing.get('replay_age_at_output_s')} s.\n"
    for mode, b in baselines.items():
        summary += f"SmolVLM {mode}: {b['proposal']} / {b['task_status']}; native: {b['native_model_output']}\n"
    ax.text(
        0,
        1,
        "\n".join(textwrap.fill(line, 140) for line in summary.splitlines()),
        va="top",
        fontsize=9,
    )
    ax = fig.add_subplot(grid[2, :])
    ax.axis("off")
    label = (
        "PENDING: no reviewed human reference"
        if reference is None
        else json.dumps(reference, ensure_ascii=False)
    )
    ax.text(
        0,
        1,
        "EVALUATION ONLY — HUMAN REFERENCE\n"
        + textwrap.fill(label, 145)
        + "\n\nActual native historical inputs: adjacent history.png (all selected indices; upstream duplicates/pooling in native_sample.json)."
        "\nOperator-driven replay: no navigation success, executed action, registered 3D goal or collision evidence."
        "\nE4 search-turn agreement is not absent-target recognition. STOP is not verified completion.",
        va="top",
        fontsize=9,
    )
    fig.suptitle("PHASE 6C — OFFLINE PROPOSALS, NOT EXECUTED", color="#982c27")
    fig.savefig(output / "review.png", dpi=100)
    plt.close(fig)
    images = sorted(inference_dir.glob("input_*.png"))
    if images:
        rows = (len(images) + 3) // 4
        fig, axes = plt.subplots(
            rows, 4, figsize=(16, rows * 2.5), squeeze=False, layout="constrained"
        )
        for ax in axes.flat:
            ax.axis("off")
        for ax, path in zip(axes.flat, images):
            index = int(path.stem.split("_")[1])
            frame = job["frames"][index]
            ax.imshow(Image.open(path))
            ax.set_title(
                f"input {index}: exposure {frame['observation_s']:.3f}s", fontsize=9
            )
        fig.suptitle(
            "Actual LightNav model-space history; native SlowFast pooling follows vision encoding"
        )
        fig.savefig(output / "history.png", dpi=100)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--annotations-name", default="../evaluation_only/annotations.json")
    ap.add_argument("--static-episode", action="append", default=[])
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    declaration = json.loads((a.run / "run_declaration.json").read_text())
    if file_sha256(a.run / "jobs.json") != declaration["jobs_sha256"]:
        raise ValueError("Changed prepared jobs")
    jobs = json.loads((a.run / "jobs.json").read_text())["jobs"]
    episodes, annotations, annotation_hashes = {}, {}, {}
    for info in declaration["episodes"]:
        # Scoring binds the frozen manifest; full source-package preservation is
        # audited separately. Panels use the exact hashed prepared model inputs.
        ep = load_episode(info["path"], verify_files=False)
        if ep.sha256 != info["sha256"]:
            raise ValueError("Changed episode package")
        eid = ep.data["episode_id"]
        episodes[eid] = ep
        path = ep.root / a.annotations_name
        annotations[eid] = load_annotations(path, ep) if path.is_file() else None
        annotation_hashes[eid] = file_sha256(path) if path.is_file() else None
    checked_frames = set()
    for job in jobs:
        for frame in job["frames"]:
            identity = (frame["path"], frame["sha256"])
            if identity not in checked_frames:
                if file_sha256(frame["path"]) != frame["sha256"]:
                    raise ValueError("Changed prepared review image")
                checked_frames.add(identity)
    a.output.mkdir(parents=True, exist_ok=False)
    rows, panels = [], []
    mapping = declaration["config"]["comparison"]
    for i, job in enumerate(jobs):
        eid, did = job["episode_id"], job["decision_id"]
        inf = a.run / "inference" / f"{i:03d}"
        result = (
            json.loads((inf / "result.json").read_text())
            if (inf / "result.json").is_file()
            else {"status": "not_run"}
        )
        if (
            result.get("episode_id", eid) != eid
            or result.get("decision_id", did) != did
            or result.get("clock_epoch", job["clock_epoch"]) != job["clock_epoch"]
        ):
            raise ValueError("Result identity mismatch")
        mapped = None
        if result["status"] == "decoded":
            mapped = comparison_action(
                result["waypoints"],
                native_stop=result["decoder_stop"],
                turn_rad=mapping["turn_threshold_rad"],
                advance_m=mapping["advance_threshold_m"],
                lateral_m=mapping["max_lateral_for_advance_m"],
            )
        result["comparison_action"] = mapped
        baselines = {}
        for mode in ("temporal", "current_only"):
            folder = a.baseline / eid / did / mode
            b = json.loads((folder / "proposal.json").read_text())
            prov = json.loads((folder / "input_provenance.json").read_text())
            check_baseline_binding(job, b, prov, episodes[eid].sha256, mode)
            baselines[mode] = b
        ann = annotations[eid]
        reference = None
        reviewed = bool(
            ann
            and ann["reviewer"]["type"] == "human"
            and ann["reviewer"]["status"] == "reviewed"
        )
        if ann:
            reference = {
                "reviewer": ann["reviewer"],
                "decision": next(
                    d for d in ann["decisions"] if d["decision_id"] == did
                ),
            }
        accepted = reference["decision"]["acceptable_actions"] if reviewed else []
        row = {
            "episode_id": eid,
            "decision_id": did,
            "recording_id": episodes[eid].data["recording"]["recording_id"],
            "static_absent_probe": eid in a.static_episode,
            "human_reviewed": reviewed,
            "native_status": result["status"],
            "comparison_action": mapped,
            "comparable": mapped is not None,
            "action_agrees": mapped in accepted if reviewed and mapped else None,
            "decoder_stop": result.get("decoder_stop"),
            "pointing": result.get("pointing"),
            "completion_verified": False,
            "timing": result.get("timing"),
            "wall_s_including_evidence": result.get("wall_s_including_evidence"),
            "replay_age_including_evidence_s": None
            if result.get("wall_s_including_evidence") is None
            else job["decision_time_s"]
            - job["frames"][-1]["observation_s"]
            + result["wall_s_including_evidence"],
            "memory": result.get("memory"),
            "baseline_agreement": {
                m: b["proposal"] in accepted if reviewed else None
                for m, b in baselines.items()
            },
            "evaluation_status": "human_reference_available" if reviewed else "pending",
        }
        rows.append(row)
        out = a.output / f"{i:03d}"
        out.mkdir()
        dump(out / "comparison.json", row)
        dump(out / "evaluation_reference.json", reference)
        render_panel(job, result, baselines, reference, inf, out)
        panels.append(out / "review.png")
    groups = defaultdict(list)
    for row in rows:
        groups[row["episode_id"]].append(row)
    summary = {
        "correlated_recordings": sorted({r["recording_id"] for r in rows}),
        "decisions": len(rows),
        "native_decoded": sum(r["native_status"] == "decoded" for r in rows),
        "human_reviewed": sum(r["human_reviewed"] for r in rows),
        "evaluation_pending": sum(not r["human_reviewed"] for r in rows),
        "comparable": sum(r["comparable"] for r in rows),
        "human_reviewed_comparable": sum(
            r["comparable"] and r["human_reviewed"] for r in rows
        ),
        "action_matches": sum(r["action_agrees"] is True for r in rows),
        "annotation_sha256": annotation_hashes,
        "by_episode": {
            eid: {
                "decisions": len(items),
                "comparable": sum(x["comparable"] for x in items),
                "matches": sum(x["action_agrees"] is True for x in items),
                "smol_temporal_matches": sum(
                    x["baseline_agreement"]["temporal"] is True for x in items
                ),
                "smol_current_matches": sum(
                    x["baseline_agreement"]["current_only"] is True for x in items
                ),
                "smol_temporal_on_comparable_cutoffs": sum(
                    x["comparable"] and x["baseline_agreement"]["temporal"] is True
                    for x in items
                ),
                "smol_current_on_comparable_cutoffs": sum(
                    x["comparable"] and x["baseline_agreement"]["current_only"] is True
                    for x in items
                ),
            }
            for eid, items in groups.items()
        },
        "limitations": [
            "One correlated test_c development recording; E1-E3 retrospective proposals, E4 static absent probe",
            "Native history and preprocessing differ; not a weights-only comparison",
            "No replay navigation-success or collision-avoidance claim",
            "Historical E4 UNRESOLVED-vs-search-turn prompt/reference mismatch preserved; turn agreement is not target recognition",
            "Full native chunks retained; unsupported mappings excluded only from comparable denominator, reported separately",
            "Human references are pre-existing; model-native semantic/evidence review is pending independent human review",
            "Manifest and prepared-image binding verified here; full source and historical artifact preservation audited separately",
        ],
    }
    dump(a.output / "summary.json", summary)
    dump(a.output / "decisions.json", rows)
    if a.video:
        make_video(panels, a.output / "review.mp4", fps=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
