"""Read-only rescore/audit of saved Phase 6D traces; never inference or execution.

Preserves raw metrics, corrects pre-goal vs post-goal outside STOP classification,
and verifies the actual causal input/history and checkpoint identity per run.
"""

import argparse
import json
from pathlib import Path

from go2_perception.vln.pretrained import CODE_REVISION, MODEL_REVISION
from go2_perception.vln.waypoint_declaration import digest
from go2_perception.vln.waypoint_evaluation import GoalRegion, summarize_run


def summarize(suite):
    original = json.loads((suite / "aggregate.json").read_text())
    frozen = json.loads((suite / "frozen_declaration.json").read_text())
    spec = frozen["spec"]
    expected = {
        (case["id"], policy) for case in spec["cases"] for policy in spec["policies"]
    }
    if {(x["case"], x["policy"]) for x in original["runs"]} != expected:
        raise ValueError("All declared conditions must be retained")
    output, epochs, decisions, native_stops = [], set(), 0, 0
    for old in original["runs"]:
        root = suite / f"{old['case']}__{old['policy']}"
        case = next(x for x in spec["cases"] if x["id"] == old["case"])
        rows = json.loads((root / "rollout/steps.json").read_text())
        model = json.loads((root / "model.json").read_text())
        if (
            model["model_revision"] != MODEL_REVISION
            or model["code_revision"] != CODE_REVISION
            or model["patch_sha256"] != spec["portability_patch_sha256"]
        ):
            raise ValueError("Runtime/model identity changed")
        epoch = rows[0]["proposal"]["clock_epoch"]
        if epoch in epochs:
            raise ValueError("Episode clock epoch reused")
        epochs.add(epoch)
        for i, row in enumerate(rows):
            job = json.loads((root / f"rollout/job_{i:03d}.json").read_text())
            if set(job) != {
                "episode_id",
                "clock_epoch",
                "decision_id",
                "decision_time_s",
                "instruction",
                "frames",
            }:
                raise ValueError("Unexpected model input context")
            if job["instruction"] != spec["instruction"] or len(job["frames"]) != i + 1:
                raise ValueError("Instruction or history changed")
            for k, frame in enumerate(job["frames"]):
                if (
                    set(frame) != {"path", "sha256", "observation_s", "available_s"}
                    or frame["observation_s"] != k * 0.25
                    or not frame["observation_s"]
                    <= frame["available_s"]
                    <= job["decision_time_s"]
                ):
                    raise ValueError("Noncausal source history")
            current = job["frames"][-1]
            if digest(current["path"]) != current["sha256"]:
                raise ValueError("Source image changed")
            proposal = row["proposal"]
            state = proposal["native_state"]
            if (
                state["reset_this_decision"] != (i == 0)
                or state["frames_appended"] != 1
                or state["total_frames"] != i + 1
            ):
                raise ValueError("Unintended native session reset/history duplication")
            if (
                proposal["status"] != "decoded"
                or i not in proposal["native_selected_frame_indices"]
            ):
                raise ValueError("Missing native decode or current observation")
            processed = root / f"rollout/inference/{i:03d}/input_{i:04d}.png"
            if digest(processed) != current["sha256"]:
                raise ValueError("Native-sized synthetic image unexpectedly changed")
            decisions += 1
            native_stops += int(proposal["decoder_stop"])
        rescored = summarize_run(
            case["pose"], rows, old["termination"], goal=GoalRegion(**spec["goal"])
        )
        changed = {
            k: {"original": old.get(k), "corrected": v}
            for k, v in rescored.items()
            if k in old and old[k] != v
        }
        output.append(
            old
            | rescored
            | {
                "changed_existing_metrics": changed,
                "raw_steps_sha256": digest(root / "rollout/steps.json"),
            }
        )
    return {
        "schema": "go2-waypoint-suite-rescored-v2",
        "model_inference_rerun": False,
        "original_aggregate_sha256": digest(suite / "aggregate.json"),
        "frozen_declaration_sha256": digest(suite / "frozen_declaration.json"),
        "metric_revision": "outside_stop_split_before_entry_vs_after_exit",
        "scorer_sha256": digest(
            Path(__file__).resolve().parents[1]
            / "src/go2_perception/vln/waypoint_evaluation.py"
        ),
        "actual_input_audit": {
            "decisions": decisions,
            "native_decodes": decisions,
            "native_stops": native_stops,
            "distinct_epochs": len(epochs),
            "current_processed_images_match_source": True,
            "history_append_and_causality_verified": True,
        },
        "declared_conditions": len(expected),
        "runs": output,
        "physical_execution_authorized": False,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    value = summarize(args.suite)
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(value["actual_input_audit"]))


if __name__ == "__main__":
    main()
