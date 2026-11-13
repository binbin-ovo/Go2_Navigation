"""Render evaluation-only panels for a saved synthetic step-and-wait rollout."""

import argparse
import json
from pathlib import Path
import textwrap

import numpy as np

from go2_perception.vln.temporal_review import make_video


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    summary = json.loads((a.run / "summary.json").read_text())
    steps = json.loads((a.run / "steps.json").read_text())
    if not summary["closed_loop_executed"] or summary["mode"] != "static_step_and_wait":
        raise ValueError("Expected an actual static simulation rollout")
    a.output.mkdir(parents=True, exist_ok=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    panels = []
    path = [steps[0]["before"]["pose_xy_yaw"]]
    for i, row in enumerate(steps):
        path.append(row["after"]["pose_xy_yaw"])
        fig = plt.figure(figsize=(16, 10), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=[3, 2])
        ax = fig.add_subplot(grid[0, 0])
        ax.imshow(Image.open(a.run / f"camera_{i:03d}.png"))
        ax.axis("off")
        ax.set_title(f"Model observation {i}; simulation paused for inference")
        ax = fig.add_subplot(grid[0, 1])
        xy = np.asarray(path)
        region = row["before"]["goal_region"]
        x0, x1 = region["x_range"]
        y0, y1 = region["y_range"]
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                alpha=0.2,
                color="green",
                label="stopping region",
            )
        )
        ax.plot(xy[:, 0], xy[:, 1], "o-", label="actual proxy path")
        x, y, yaw = row["before"]["pose_xy_yaw"]
        ax.arrow(
            x, y, 0.3 * np.cos(yaw), 0.3 * np.sin(yaw), head_width=0.07, color="black"
        )
        wp = row["proposal"].get("waypoints")
        world_x, world_y = xy[:, 0], xy[:, 1]
        if wp:
            wp = np.asarray(wp)
            world_x = x + np.cos(yaw) * wp[:, 0] - np.sin(yaw) * wp[:, 1]
            world_y = y + np.sin(yaw) * wp[:, 0] + np.cos(yaw) * wp[:, 1]
            ax.plot(
                world_x, world_y, ".--", label="native chunk (only first considered)"
            )
        ax.plot([7, 7], [-2.8, -0.7], color="gray", linewidth=4)
        ax.plot([7, 7], [0.7, 2.8], color="gray", linewidth=4)
        ax.set(
            xlim=(
                min(3.5, float(np.min(world_x)) - 0.2),
                max(7.5, float(np.max(world_x)) + 0.2),
            ),
            ylim=(
                min(-2, float(np.min(world_y)) - 0.2),
                max(2, float(np.max(world_y)) + 0.2),
            ),
            xlabel="simulation world x (m)",
            ylabel="world y (m)",
        )
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.set_title("EVALUATOR ONLY: path, doorway and stopping region")
        ax = fig.add_subplot(grid[1, :])
        ax.axis("off")
        p = row["proposal"]
        lines = [
            f"Instruction: {summary['instruction']}",
            f"Native proposal: {p.get('native_output', p.get('error'))}",
            f"Native first waypoint [forward m, left m, yaw rad]: {p.get('waypoints', [None])[0]}",
            f"Actual simulator action: {row['action']}",
            f"Decoder STOP: {p.get('decoder_stop')}; in stopping region: {row['after']['inside_stopping_region']}; contacts: {row['after']['contacts']}",
            f"Host observation age: {row['host_observation_age_at_response_s']:.3f} s; simulation time advanced during inference: False",
            f"Whole rollout outcome: {summary['termination']}; model stopped in region: {summary['model_stopped_in_region']}",
            "Kinematic camera proxy, synthetic geometry; no Go2 locomotion or dynamic collision-avoidance validation.",
        ]
        ax.text(
            0,
            1,
            "\n\n".join(textwrap.fill(s, 145) for s in lines),
            va="top",
            fontsize=10,
        )
        fig.suptitle(
            "PHASE 6C — STATIC SIMULATION ONLY; GEOMETRY NEVER SENT TO THE MODEL",
            color="#982c27",
        )
        output = a.output / f"step_{i:03d}.png"
        fig.savefig(output, dpi=100)
        plt.close(fig)
        panels.append(output)
    if a.video:
        make_video(panels, a.output / "rollout.mp4", fps=1)
    print(f"Saved {len(panels)} evaluation-only panels to {a.output}")


if __name__ == "__main__":
    main()
