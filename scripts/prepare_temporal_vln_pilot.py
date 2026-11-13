"""Copy bounded private recording snippets into v1 episodes. No labels enter manifests."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

from go2_perception.vln.episodes import file_sha256, load_episode

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(recordings, output, plan, calibration, timing):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Episode output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "declared_pilot.json",
        {"plan": plan, "calibration_sha256": file_sha256(calibration)},
    )
    for spec in plan["episodes"]:
        src = recordings / spec["recording"]
        manifest = json.loads((src / "manifest.json").read_text())
        if manifest.get("schema") != "go2-recording-v1":
            raise ValueError("Unexpected recording schema")
        rows = [
            json.loads(x) for x in (src / "rgb/index.jsonl").read_text().splitlines()
        ]
        odom = [
            json.loads(x)
            for x in (src / "state/utlidar_robot_odom.jsonl").read_text().splitlines()
        ]
        offset = min(r["host_recv"] - r["stamp"] for r in odom)
        latency = timing["camera_latency_s"]["value"]
        exposure = np.array([r["host_recv_first"] - offset - latency for r in rows])
        availability = np.array(
            [
                r["host_recv_last"] - offset + plan["availability_decode_margin_s"]
                for r in rows
            ]
        )
        if np.any(np.diff(exposure) <= 0) or np.any(np.diff(availability) < 0):
            raise ValueError("Nonchronological source recording")
        start, end = (
            exposure[0] + spec["start_offset_s"],
            exposure[0] + spec["end_offset_s"],
        )
        if end > exposure[-1]:
            raise ValueError("Episode exceeds source recording")
        dest = output / spec["episode_id"]
        (dest / "images").mkdir(parents=True)
        shutil.copyfile(calibration, dest / "calibration.yaml")
        epoch = f"go2_boot:host_minus_robot={offset:.1f}"
        # Package sampling is deterministic; it is distinct from detected sensor gaps.
        indices = []
        for t in np.arange(start, end + 1e-6, plan["package_sampling_period_s"]):
            i = int(np.searchsorted(exposure, t, side="right") - 1)
            if (
                i >= 0
                and start <= exposure[i] <= end
                and (not indices or i != indices[-1])
            ):
                indices.append(i)
        frames, source_rows = [], []
        for i in indices:
            row = rows[i]
            original = src / "rgb" / row["file"]
            path = dest / "images" / row["file"]
            missing = None
            if original.is_file() and row.get("complete", True):
                shutil.copyfile(original, path)
                image = {
                    "path": str(path.relative_to(dest)),
                    "sha256": file_sha256(path),
                }
            else:
                image = None
                missing = "Source image missing or marked incomplete"
            frames.append(
                {
                    "frame_id": f"rgb_{i:06d}",
                    "recording_id": spec["recording"],
                    "clock_epoch": epoch,
                    "observation_s": float(exposure[i]),
                    "available_s": float(availability[i]),
                    "host_receipt_s": row["host_recv_first"],
                    "image": image,
                    "size_wh": [row["width"], row["height"]],
                    "missing_reason": missing,
                    "robot_state_ref": None,
                }
            )
            source_rows.append(
                {
                    "frame_id": f"rgb_{i:06d}",
                    "original_file": str(original.resolve()),
                    "index_metadata": row,
                }
            )
        gaps = [
            {
                "start_s": float(max(start, exposure[i - 1])),
                "end_s": float(min(end, exposure[i])),
                "reason": "source RGB exposure interval > 0.25s",
            }
            for i in range(1, len(rows))
            if exposure[i] - exposure[i - 1] > 0.25
            and exposure[i] > start
            and exposure[i - 1] < end
        ]
        episode = {
            "schema_version": "go2-vln-episode-v1",
            "episode_id": spec["episode_id"],
            "instruction": spec["instruction"],
            "recording": {
                "recording_id": spec["recording"],
                "source": str(src.resolve()),
                "source_manifest_sha256": file_sha256(src / "manifest.json"),
                "kind": "go2-recording-v1",
            },
            "clock": {
                "domain": "go2_robot",
                "epoch": epoch,
                "units": "seconds",
                "observation_semantics": "exposure_estimated",
                "availability_semantics": f"last_RTP_receipt_minus_offset_plus_{plan['availability_decode_margin_s']}s_assumed_decode_margin_not_hardware_bound",
                "uncertainty_s": float(
                    np.hypot(
                        timing["camera_latency_s"]["sigma"],
                        timing.get("offset_sigma_s", 0.002),
                    )
                ),
                "host_minus_robot_s": offset,
            },
            "bounds": {
                "start_s": float(start),
                "end_s": float(end),
                "end_reason": "predeclared manual pilot cut, not task completion",
            },
            "calibration": {
                "path": "calibration.yaml",
                "sha256": file_sha256(dest / "calibration.yaml"),
            },
            "preprocessing": {
                "source_rectified": False,
                "policy": "phase6a_calibrated_view",
                "provenance": f"raw recording bytes; package samples latest exposure at {plan['package_sampling_period_s']}s grid; no model preprocessing yet",
            },
            "frames": frames,
            "decisions": [
                {"decision_id": f"d{i:02d}", "time_s": float(exposure[0] + t)}
                for i, t in enumerate(spec["decision_offsets_s"])
            ],
            "gaps": gaps,
        }
        write_json(dest / "episode.json", episode)
        write_json(
            dest / "source_index.json",
            {
                "selected_rows": source_rows,
                "offset_method": "minimum host_recv minus robot stamp over source odometry",
                "timing_config": timing,
            },
        )
        annotations = {
            "schema_version": "go2-vln-annotations-v1",
            "episode_id": spec["episode_id"],
            "manifest_sha256": file_sha256(dest / "episode.json"),
            "reviewer": {
                "name": "not yet assigned",
                "type": "human",
                "status": "pending",
            },
            "decisions": [
                {
                    "decision_id": d["decision_id"],
                    "target_visibility": "unknown",
                    "instruction_progress": None,
                    "acceptable_actions": [],
                    "completion_status": "unknown",
                    "ambiguity": False,
                    "insufficient_information": False,
                    "notes": "Pending human review; booleans are unreviewed defaults, not labels.",
                }
                for d in episode["decisions"]
            ],
        }
        write_json(dest / "annotations.pending.json", annotations)
        load_episode(dest / "episode.json")
        print(
            f"{spec['episode_id']}: {len(frames)} source frames, {len(gaps)} source gaps",
            flush=True,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--plan", type=Path, default=ROOT / "configs/temporal_vln_pilot.yaml"
    )
    ap.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "configs/go2_front_camera_calibration.yaml",
    )
    a = ap.parse_args()
    prepare(
        a.recordings,
        a.output,
        yaml.safe_load(a.plan.read_text()),
        a.calibration,
        yaml.safe_load((ROOT / "configs/go2_vln_sensors.yaml").read_text())["timing"],
    )
