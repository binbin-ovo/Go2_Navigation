"""Read-only comparison of saved native tensors for repeated LightNav outputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from go2_perception.vln.waypoint_declaration import digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for folder in sorted((args.run / "inference").iterdir()):
        if not folder.is_dir() or not (folder / "result.json").exists():
            continue
        result = json.loads((folder / "result.json").read_text())
        sample = json.loads((folder / "native_sample.json").read_text())
        with np.load(folder / "model_inputs_and_generated_ids.npz") as tensors:
            arrays = {
                key: {
                    "shape": list(tensors[key].shape),
                    "dtype": str(tensors[key].dtype),
                    "sha256": hashlib.sha256(tensors[key].tobytes()).hexdigest(),
                }
                for key in tensors.files
                if key != "generated_ids"
            }
        frame_ids = sorted(
            {
                i
                for segment in sample["video_segments"]
                for i in segment["frame_indices"]
            }
        )
        rows.append(
            {
                "decision": folder.name,
                "native_output": result.get("native_output"),
                "source_result_sha256": digest(folder / "result.json"),
                "native_selected_frame_ids": frame_ids,
                "current_frame_sha256": digest(
                    folder / f"input_{max(frame_ids):04d}.png"
                ),
                "tensors": arrays,
            }
        )
    with args.output.open("x") as stream:
        json.dump(
            {
                "run": str(args.run.resolve()),
                "model_rerun": False,
                "decisions": rows,
                "interpretation_limit": "Distinct captured pixels rule out identical input tensors, not every possible model defect. Inspect the pinned HF path separately.",
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    print(
        json.dumps(
            {
                "decisions": len(rows),
                "distinct_current_images": len(
                    {r["current_frame_sha256"] for r in rows}
                ),
                "distinct_pixel_tensors": len(
                    {r["tensors"]["pixel_values_videos"]["sha256"] for r in rows}
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
