"""Model-free numeric paths, original-scene previews and official input images.

Geometry is used to diagnose the simulator/controller, never as learned navigation.
Reuses the existing isolated MuJoCo/model environment without loading weights.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import yaml

from go2_perception.vln.simulation import CameraRoom
from go2_perception.vln.waypoints import PrefixLimits, to_world, wrap_angle

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--assets", required=True, type=Path)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = a.assets / "source-local-v2"
    identity = json.loads((a.assets / "source_local_v2_identity.json").read_text())
    # Only native processor code is imported; weights and robot clients are not.
    for entry in identity["files"]:
        if (
            hashlib.sha256((source / entry["path"]).read_bytes()).hexdigest()
            != entry["sha256"]
        ):
            raise ValueError("Native source changed")
    sys.path.insert(0, str(source / "src"))
    from lightnav.inference.frame_preprocessing import rgb_frame_to_model_tensor

    config = yaml.safe_load((ROOT / "configs/waypoint_validation.yaml").read_text())
    scene = ROOT / config["scene"]
    records = []
    cases = {
        "straight": [[0.4, 0, 0]],
        "left_turn": [[0, 0, 1]],
        "right_turn": [[0, 0, -1]],
        "combined": [[0.3, 0.1, 0.5]],
        "curve": [[0.1, 0, 0.1], [0.2, 0.1, 0.3], [0.3, 0.15, 0.5]],
        "leading_zero": [[0, 0, 0], [0.2, 0, 0]],
        "native_stop": np.zeros((10, 3)).tolist(),
        "wrap": [[0, 0, 0.2]],
    }
    for name, chunk in cases.items():
        room = CameraRoom(scene)
        try:
            room.pose = np.array([3.2, -0.2, math.pi - 0.1 if name == "wrap" else -0.4])
            room._place()
            initial = room.pose.copy()
            Image.fromarray(room.render()).save(a.output / f"{name}_before.png")
            trace = room.execute_prefix(
                chunk,
                native_stop=name == "native_stop",
                limits=PrefixLimits(window_s=2),
            )
            expected = to_world(chunk, initial)[-1]
            error = room.pose - expected
            error[2] = wrap_angle(error[2])
            trace.update(
                case=name,
                expected_endpoint=expected.tolist(),
                endpoint_error=error.tolist(),
            )
            Image.fromarray(room.render()).save(a.output / f"{name}_after.png")
            records.append(trace)
        finally:
            room.close()
    for name, origin, chunk in (
        ("contact", [6.5, 1.5, 0], [[1, 0, 0]]),
        ("reachable_goal", [4.3, 0, 0], [[1.85, 0, 0]]),
    ):
        room = CameraRoom(scene)
        try:
            room.pose = np.array(origin, dtype=float)
            room._place()
            Image.fromarray(room.render()).save(a.output / f"{name}_before.png")
            trace = room.execute_prefix(chunk, limits=PrefixLimits(window_s=3))
            trace.update(case=name, evaluator=room.evaluation())
            Image.fromarray(room.render()).save(a.output / f"{name}_after.png")
            records.append(trace)
        finally:
            room.close()
    # Same camera/scene and proposed starts. These are previews, not declared runs.
    previews = []
    for case in config["cases"]:
        room = CameraRoom(scene)
        try:
            room.pose = np.asarray(case["pose"], dtype=float)
            room._place()
            rgb = room.render()
            raw = a.output / f"preview_{case['id']}_camera.png"
            processed = a.output / f"preview_{case['id']}_official_input.png"
            Image.fromarray(rgb).save(raw)
            tensor = rgb_frame_to_model_tensor(rgb, (256, 448))
            recovered = (
                np.rint((tensor.permute(1, 2, 0).numpy() + 1) * 127.5)
                .clip(0, 255)
                .astype(np.uint8)
            )
            Image.fromarray(recovered).save(processed)
            previews.append(
                {
                    **case,
                    "camera_image": str(raw),
                    "official_input_image": str(processed),
                    "pixel_max_difference": int(
                        np.abs(recovered.astype(int) - rgb.astype(int)).max()
                    ),
                    "camera_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                    "official_input_sha256": hashlib.sha256(
                        processed.read_bytes()
                    ).hexdigest(),
                    "scene_recognizable_human_review": "pending",
                }
            )
        finally:
            room.close()
    dump(
        a.output / "diagnostics.json",
        {
            "model_inference_executed": False,
            "learned_navigation_success": False,
            "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
            "official_processor_revision": identity["upstream_revision"],
            "real_go2_fisheye_rectification_applied": False,
            "numeric_cases": records,
            "scene_previews": previews,
            "known_path_geometry_is_controller_diagnostic_only": True,
        },
    )
    print(
        json.dumps(
            {
                "numeric_cases": len(records),
                "previews": len(previews),
                "model_inference_executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
