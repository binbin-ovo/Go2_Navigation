"""Generate the deterministic synthetic episode contract fixture (no model required)."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from go2_perception.vln.episodes import file_sha256


def generate(root):
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    cal = {"schema": "go2-camera-calibration-v1", "status": "synthetic", "camera_frame": "synthetic_optical",
           "image_size": [160, 120], "model": "pinhole_radtan", "K": [[120, 0, 79.5], [0, 120, 59.5], [0, 0, 1]],
           "D": [0]*5, "T_base_camera": None, "provenance": {"source": "deterministic geometric fixture, not Go2"}}
    (root / "calibration.yaml").write_text(yaml.safe_dump(cal, sort_keys=True))
    frames = []
    for i in range(6):
        image = None
        if i != 2:
            img = np.full((120, 160, 3), 220, np.uint8)
            cv2.rectangle(img, (50 + i*4, 20), (85 + i*4, 100), (40, 80, 180), -1)
            path = root / "images" / f"{i:03d}.png"
            cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            image = {"path": f"images/{i:03d}.png", "sha256": file_sha256(path)}
        frames.append({"frame_id": f"f{i}", "recording_id": "synthetic_room", "clock_epoch": "synthetic_boot_1",
                       "observation_s": 100.0+i, "available_s": 100.1+i, "host_receipt_s": 1100.1+i,
                       "image": image, "size_wh": [160, 120], "missing_reason": "synthetic dropout" if i==2 else None,
                       "robot_state_ref": None})
    manifest = {"schema_version": "go2-vln-episode-v1", "episode_id": "synthetic_door", "instruction": "Approach the red doorway and stop before entering.",
                "recording": {"recording_id": "synthetic_room", "source": "generated fixture", "source_manifest_sha256": "0"*64, "kind": "synthetic"},
                "clock": {"domain": "go2_robot", "epoch": "synthetic_boot_1", "units": "seconds", "observation_semantics": "exposure",
                          "availability_semantics": "synthetic_known_available", "uncertainty_s": 0.0, "host_minus_robot_s": 1000.0},
                "bounds": {"start_s": 100.0, "end_s": 106.0, "end_reason": "synthetic manual cut"},
                "calibration": {"path": "calibration.yaml", "sha256": file_sha256(root/'calibration.yaml')},
                "preprocessing": {"source_rectified": False, "policy": "phase6a_calibrated_view", "provenance": "synthetic pinhole fixture"},
                "frames": frames, "decisions": [{"decision_id": "d0", "time_s": 103.2}, {"decision_id": "d1", "time_s": 105.2}],
                "gaps": [{"start_s": 101.5, "end_s": 102.5, "reason": "declared synthetic missing frame"}]}
    (root/'episode.json').write_text(json.dumps(manifest, indent=2)+'\n')
    annotations = {"schema_version": "go2-vln-annotations-v1", "episode_id": manifest["episode_id"], "manifest_sha256": file_sha256(root/'episode.json'),
                   "reviewer": {"name": "fixture generator", "type": "synthetic", "status": "reviewed"},
                   "decisions": [{"decision_id": d["decision_id"], "target_visibility": "visible", "instruction_progress": "synthetic illustration only",
                                  "acceptable_actions": ["UNRESOLVED"], "completion_status": "unknown", "ambiguity": False,
                                  "insufficient_information": True, "notes": "Contract example, not navigation ground truth."} for d in manifest['decisions']]}
    (root/'annotations.json').write_text(json.dumps(annotations, indent=2)+'\n')
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    generate(parser.parse_args().output)
