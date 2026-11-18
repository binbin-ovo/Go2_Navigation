#!/usr/bin/env python3
"""Create a small deterministic public archive, without a camera or model.

Images, clock values, native tokens, paths and human verdicts are ALL synthetic.
The file layout follows the published interactive recorder; only fixture files
are written. Existing destinations are refused.
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def write(path, data, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        data, sort_keys=True, indent=None if lines else 2, allow_nan=False
    )
    path.write_text(text + "\n")


def make_fixture(dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    root = dest / "run"
    root.mkdir()
    session = "synthetic-pair-session"
    instruction = "Go to the chair and stop."
    write(
        root / "run.json",
        {
            "schema": "go2-vln-interactive-run-v1",
            "backend": "synthetic_archive",
            "synthetic": True,
            "git_commit": "fixture-no-inference",
            "robot_commands_sent": 0,
            "physical_execution_authorized": False,
        },
    )
    write(
        root / "summary.json",
        {
            "termination": "synthetic_closed_archive",
            "trials_started": 2,
            "ended_wall_utc": "2026-01-01T00:03:00+00:00",
            "robot_commands_sent": 0,
        },
    )
    for sequence, role, time_s, choice in (
        (1, "before", 100.0, "wrong"),
        (2, "after", 118.0, "correct"),
    ):
        trial_id, pid = f"synthetic-{role}", f"synthetic-{role}-P{sequence:03d}"
        folder = root / "trials" / trial_id
        request = f"session/requests/{sequence:06d}"
        path = root / request
        (path / "native").mkdir(parents=True)
        trial = {
            "schema": "go2-vln-interactive-trial-v1",
            "trial_id": trial_id,
            "kind": "target",
            "operator_text": "chair",
            "instruction": instruction,
            "operator_text_is_verified_label": False,
            "single_answer_guided_test": True,
            "started_host_monotonic_s": time_s - 1,
            "session_id": session,
            "instruction_revision": sequence,
            "source_mode": "synthetic_archive",
        }
        write(folder / "trial.json", trial)
        write(
            folder / "trial_end.json",
            {
                "trial_id": trial_id,
                "end_reason": "guided_answer_saved",
                "ended_host_monotonic_s": time_s + 5,
                "inflight_at_end": False,
                "robot_commands_sent": 0,
            },
        )
        identity = {
            "session_id": session,
            "instruction_revision": sequence,
            "sequence": sequence,
            "history_generation": sequence,
            "clock_epoch": "synthetic-camera-epoch-0",
            "source_id": "synthetic-only",
        }
        frames, history = [], []
        point_token = 635 if role == "before" else 655
        fraction = [
            ((point_token - 1) % 48 + 0.5) / 48,
            ((point_token - 1) // 48 + 0.5) / 27,
        ]
        for i in range(3):
            image = Image.new("RGB", (480, 270), "#d9e6ed")
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 190, 480, 270), fill="#c4b39b")
            draw.text((12, 10), "SYNTHETIC ONLY - " + role.upper(), fill="#142b39")
            draw.text(
                (12, 30),
                f"Independent query {sequence}; input frame {i}",
                fill="#142b39",
            )
            if role == "before":
                draw.rectangle(
                    (80, 100, 190, 195), fill="#836348", outline="#142b39", width=4
                )
                for y in (120, 145, 170):
                    draw.line((80, y, 190, y), fill="#142b39", width=4)
                draw.text((75, 210), "shelf (wrong mark)", fill="#142b39")
            else:
                draw.rectangle(
                    (260, 75, 350, 151), fill="#2a778c", outline="#142b39", width=4
                )
                draw.rectangle(
                    (250, 140, 360, 165), fill="#2a778c", outline="#142b39", width=4
                )
                draw.line((260, 165, 260, 207), fill="#142b39", width=6)
                draw.line((350, 165, 350, 207), fill="#142b39", width=6)
                draw.text((260, 217), "chair", fill="#142b39")
            image.save(path / f"frame_{i:04d}.png")
            image.save(path / "native" / f"input_{i:04d}.png")
            sha = hashlib.sha256((path / f"frame_{i:04d}.png").read_bytes()).hexdigest()
            t = time_s + i * 0.25
            source = {
                "frame_id": f"{role}-frame-{i}",
                "source_id": identity["source_id"],
                "observation_s": t,
                "available_s": t + 0.01,
                "clock_epoch": identity["clock_epoch"],
                "stamp": {
                    "t": t,
                    "clock": "synthetic_camera",
                    "semantics": "exposure",
                    "uncertainty_s": 0.002,
                },
                "received_monotonic_s": t + 0.01,
                "exposure_monotonic_s": t,
                "monotonic_mapping": "Synthetic shared clock, offset 0, explicit error bound 0.002 seconds",
                "provenance": {
                    "synthetic": True,
                    "exposure_monotonic_error_bound_s": 0.002,
                },
            }
            frames.append(
                {
                    "path": f"{request}/frame_{i:04d}.png",
                    "sha256": sha,
                    "observation_s": t,
                    "available_s": t + 0.01,
                }
            )
            history.append(
                {
                    "source": source,
                    "rectification": {
                        "synthetic": True,
                        "note": "Synthetic pinhole fixture; no real fisheye calibration used",
                    },
                }
            )
        job = {
            "instruction": instruction,
            "episode_id": f"{session}.r{sequence}",
            "decision_id": str(sequence),
            "clock_epoch": identity["clock_epoch"],
            "decision_time_s": time_s + 0.51,
            "frames": frames,
        }
        snapshot = {
            "identity": identity,
            "history": history,
            "current_source": history[-1]["source"],
            "host_submit_monotonic_s": time_s + 0.6,
            "reset_policy": "reconstruct_complete_causal_instruction_history_per_request",
            "history_policy": "synthetic fixture of archived selected inputs",
            "annotations_consumed": False,
            "previous_actions_supplied": False,
        }
        native = {
            "episode_id": job["episode_id"],
            "decision_id": job["decision_id"],
            "clock_epoch": job["clock_epoch"],
            "execution_authorized": False,
            "status": "decoded",
            "native_selected_frame_indices": [0, 1, 2],
            "native_output": f"<opos_{point_token}><act_l0_201><act_l1_230><act_l2_226><|im_end|>",
            "pointing": {"opos": point_token, "posxy": {}, "sentinels": {}},
            "waypoints": [[round(i * 0.05, 3), 0.0, 0.0] for i in range(10)],
            "decoder_stop": False,
            "explicit_stop_code": False,
            "waypoint_convention": "absolute_ego_SE2_x_forward_y_left_yaw_CCW_metres_radians",
            "waypoint_time_base": None,
            "completion_verified": False,
            "fixture_note": "Canned synthetic output, NOT a model prediction",
        }
        receipt = {
            "receipt_monotonic_s": time_s + 2,
            "rejection_reason": None,
            "response": {
                "identity": identity,
                "result": native,
                "backend": "synthetic_archive",
                "inference_duration_s": None,
            },
        }
        image_ref = {
            "path": f"{request}/native/input_0002.png",
            "job_frame_index": 2,
            "file_sha256": frames[-1]["sha256"],
            "kind": "synthetic native-input fixture",
        }
        prediction = {
            "prediction_id": pid,
            "trial_id": trial_id,
            "request_sequence": sequence,
            "instruction_revision": sequence,
            "request_directory": request,
            "instruction": instruction,
            "received_host_monotonic_s": time_s + 2,
            "label_at_receipt": "CURRENT",
            "rejection_reason_at_receipt": None,
            "mock": False,
            "backend": "synthetic_archive",
            "native_output": native["native_output"],
            "pointing": native["pointing"],
            "target_fraction": fraction,
            "opos_not_visible": False,
            "source_image": image_ref,
            "waypoints": native["waypoints"],
            "waypoint_convention": native["waypoint_convention"],
            "waypoint_time_base": None,
            "native_stop": False,
            "explicit_stop_code": False,
            "completion_verified": False,
            "execution_status": "NOT EXECUTED - synthetic fixture",
            "history_frames_in_job": 3,
            "model_output_not_a_label": True,
        }
        review = {
            "schema": "go2-vln-guided-answer-review-v1",
            "trial_id": trial_id,
            "prediction_id": pid,
            "choice": choice,
            "choice_text": "The + is on the target"
            if choice == "correct"
            else "The + is not on the target",
            "source_image": image_ref,
            "target_operator_text": "chair",
            "verbatim_comment": "Synthetic verdict, not human-collected data.",
            "reviewer": "synthetic_fixture",
            "reviewed_host_monotonic_s": time_s + 4,
            "model_feedback": False,
            "confirmed_by_user": True,
            "source": "human_operator",
        }
        for name, data in (
            ("job.json", job),
            ("snapshot.json", snapshot),
            ("receipt.json", receipt),
            ("native/result.json", native),
        ):
            write(path / name, data)
        write(folder / "predictions.jsonl", prediction, lines=True)
        write(folder / "human_labels.jsonl", review, lines=True)
    pair = {
        "schema": "go2-vln-viewpoint-pair-v1",
        "episode_id": "synthetic-chair-pair",
        "target_wording": "chair",
        "before_trial_id": "synthetic-before",
        "after_trial_id": "synthetic-after",
        "turn": {
            "direction": "left",
            "provenance": {"source": "synthetic fixture, no robot movement"},
            "start": {
                "time_s": 110.0,
                "clock": "host_monotonic",
                "clock_epoch": session,
                "uncertainty_s": 0.01,
            },
            "end": {
                "time_s": 112.0,
                "clock": "host_monotonic",
                "clock_epoch": session,
                "uncertainty_s": 0.01,
            },
        },
    }
    write(dest / "pair.json", pair)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(make_fixture(parser.parse_args().output))
