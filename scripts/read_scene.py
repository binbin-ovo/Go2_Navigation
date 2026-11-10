"""Minimal standard-library-only renderer consumer; no perception imports required."""

import argparse
import json


def summarize(path):
    with open(path) as source:
        for line in source:
            row = json.loads(line)
            if row["schema"] != "go2-scene-v1":
                raise ValueError("Unsupported scene version")
            # Render all coordinates directly in odom meters. Do not apply robot_pose again.
            # robot_pose places only the robot mesh, not the already transformed scene.
            observed = [o for o in row["objects"] if o["status"] == "observed_filtered"]
            stale = [o for o in row["objects"] if o["status"] == "stale"]
            yield dict(
                timestamp=row["timestamp"],
                clock=row["clock_domain"],
                epoch=row["epoch"],
                frame=row["frame_id"],
                background_cells=len(row["background"]["cells"]),
                observed_boxes=len(observed),
                stale_boxes=len(stale),
                motion=[o["motion_state"] for o in row["objects"]],
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    args = p.parse_args()
    for summary in summarize(args.input):
        print(json.dumps(summary))
