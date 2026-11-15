"""Evaluation-only panels/videos for every Phase 6D run, including failures."""

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from go2_perception.vln.temporal_review import make_video  # noqa: E402
from go2_perception.vln.waypoints import to_world  # noqa: E402


def camera(ax, path, title):
    if path.exists():
        with Image.open(path) as img:
            ax.imshow(img)
    else:
        ax.text(0.5, 0.5, "No observation produced", ha="center")
    ax.axis("off")
    ax.set_title(title, fontsize=11)


def run_panels(folder, summary, output):
    output.mkdir(parents=True, exist_ok=False)
    steps_path = folder / "steps.json"
    rows = json.loads(steps_path.read_text()) if steps_path.exists() else []
    if not rows:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.axis("off")
        ax.text(
            0.05,
            0.85,
            f"{summary['case']} / {summary['policy']}\nNo completed model rollout\n{summary['termination']}\nSee the retained worker log.",
            fontsize=22,
            va="top",
        )
        p = output / "failure.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        make_video([p], output / "rollout.mp4", fps=1)
        return 1
    cumulative = [rows[0]["before"]["pose_xy_yaw"]]
    panels = []
    # Each decision panel shows its actual input, then one explicit terminal image.
    for i, row in enumerate(rows + [rows[-1]]):
        terminal = i == len(rows)
        proposal, action = row["proposal"], row.get("action") or {}
        if not terminal:
            cumulative.extend(action.get("actual_world_poses", [])[1:])
        fig = plt.figure(figsize=(19.2, 10.8), layout="constrained")
        grid = fig.add_gridspec(2, 3, height_ratios=[3, 2])
        ax = fig.add_subplot(grid[0, 0])
        current = (
            folder
            / "inference"
            / f"{row['index']:03d}"
            / f"input_{row['index']:04d}.png"
        )
        if not current.exists():
            current = folder / f"camera_{row['index']:03d}.png"
        camera(
            ax,
            folder / "camera_final.png" if terminal else current,
            "Final rendered view AFTER termination"
            if terminal
            else "Actual model input BEFORE this decision",
        )
        ax = fig.add_subplot(grid[0, 1])
        chunk = np.asarray(proposal.get("waypoints", []))
        if chunk.shape == (10, 3):
            ax.plot(chunk[:, 0], chunk[:, 1], "o--", label="full native chunk")
            ax.quiver(
                chunk[:, 0],
                chunk[:, 1],
                np.cos(chunk[:, 2]),
                np.sin(chunk[:, 2]),
                angles="xy",
                scale_units="xy",
                scale=8,
                width=0.006,
            )
            for k, point in enumerate(chunk):
                ax.annotate(
                    str(k),
                    point[:2],
                    xytext=(3, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
            origin = np.asarray(
                action.get("decision_pose", row["before"]["pose_xy_yaw"])
            )
            planned = np.asarray(action.get("planned_world_poses", [origin]))
            delta = planned[:, :2] - origin[:2]
            c, s = np.cos(origin[2]), np.sin(origin[2])
            local = delta @ np.array([[c, -s], [s, c]])
            ax.plot(
                local[:, 0],
                local[:, 1],
                "r-",
                linewidth=3,
                label="controller-selected motion",
            )
        ax.scatter([0], [0], c="black", marker="^")
        ax.set(
            xlabel="native x forward (m)",
            ylabel="native y left (m)",
            title="Predicted poses; arrows are native headings",
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        ax = fig.add_subplot(grid[0, 2])
        goal = row["before"]["goal_region"]
        x0, x1 = goal["x_range"]
        y0, y1 = goal["y_range"]
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                color="green",
                alpha=0.2,
                label="evaluator stop region",
            )
        )
        xy = np.asarray(cumulative)
        ax.plot(
            xy[:, 0], xy[:, 1], color="black", linewidth=2, label="actual executed path"
        )
        if chunk.shape == (10, 3):
            wp = to_world(chunk, row["before"]["pose_xy_yaw"])
            ax.plot(
                wp[:, 0],
                wp[:, 1],
                "b.--",
                alpha=0.6,
                label="native chunk in decision frame",
            )
        planned = np.asarray(action.get("planned_world_poses", []))
        if len(planned):
            ax.plot(
                planned[:, 0],
                planned[:, 1],
                "r-",
                linewidth=2,
                label="selected bounded motion",
            )
        last = np.asarray(row.get("after", row["before"])["pose_xy_yaw"])
        ax.arrow(
            last[0],
            last[1],
            0.3 * np.cos(last[2]),
            0.3 * np.sin(last[2]),
            head_width=0.08,
            color="black",
        )
        ax.plot([7, 7], [-2.8, -0.8], color="gray", linewidth=4)
        ax.plot([7, 7], [0.8, 2.8], color="gray", linewidth=4)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set(
            xlabel="simulation world x (m)",
            ylabel="world y (m)",
            title="EVALUATOR ONLY: goal, actual path, heading",
        )
        ax.legend(fontsize=7)
        ax = fig.add_subplot(grid[1, 0])
        ax.axis("off")
        table = "row   forward(m)   left(m)   yaw(deg)\n"
        if chunk.shape == (10, 3):
            table += "\n".join(
                f"{k:2}    {p[0]:+8.3f}   {p[1]:+8.3f}   {np.degrees(p[2]):+8.2f}"
                for k, p in enumerate(chunk)
            )
        else:
            table += "No valid native chunk"
        ax.text(0, 1, table, va="top", family="monospace", fontsize=10)
        ax = fig.add_subplot(grid[1, 1:])
        ax.axis("off")
        age = row.get("host_observation_age_at_response_s")
        latency = proposal.get("timing", {}).get("native_generate_s")
        lines = [
            f"Instruction: {summary.get('instruction', 'See declaration')}",
            f"Native tokens: {proposal.get('native_output', proposal.get('error', proposal.get('status')))}",
            f"Selected rows: {action.get('selected_indices')}; requested target: {action.get('requested_world_targets', [None])[-1:]}; actual pose: {last.round(3).tolist()}",
            f"Decoder STOP: {proposal.get('decoder_stop')}; explicit stop code: {proposal.get('explicit_stop_code')}; APOS: {proposal.get('pointing', {}).get('apos')}",
            f"In goal now: {row.get('after', {}).get('inside_stopping_region')}; ever entered: {summary['entered_goal_region']}; native STOP in goal: {summary['native_stop_inside_goal']}",
            f"Host generation: {latency}s; observation age at response: {None if age is None else round(age, 3)}s; sim t={row['simulated_observation_s']:.2f}s (paused during inference)",
            f"Constraint: {action.get('constraint_rejection')}; response accepted: {row.get('response_accepted')}; error: {row.get('error')}",
            f"Whole-run termination: {summary['termination']}; continued after goal entry: {summary['continued_motion_after_goal_entry']}",
            "Synthetic kinematic proxy. No physical Go2 execution, learned avoidance or live-latency validation.",
        ]
        ax.text(
            0,
            1,
            "\n\n".join(textwrap.fill(str(s), 135) for s in lines),
            va="top",
            fontsize=10,
        )
        fig.suptitle(
            f"Phase 6D | {summary['case']} | {summary['policy']} | {'TERMINAL' if terminal else f'decision {i}'} | static step-and-wait",
            fontsize=16,
        )
        panel = output / f"step_{i:03d}.png"
        fig.savefig(panel, dpi=100)
        plt.close(fig)
        panels.append(panel)
    make_video(panels, output / "rollout.mp4", fps=4)
    return len(panels)


def diagnostics(folder, output):
    data = json.loads((folder / "diagnostics.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    panels = []
    for trace in data["numeric_cases"]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), layout="constrained")
        name = trace["case"]
        camera(axes[0], folder / f"{name}_before.png", "Before known path")
        camera(axes[1], folder / f"{name}_after.png", "After bounded execution")
        planned = np.asarray(trace["planned_world_poses"])
        actual = np.asarray(trace["actual_world_poses"])
        axes[2].plot(planned[:, 0], planned[:, 1], "r--", label="requested prefix")
        axes[2].plot(actual[:, 0], actual[:, 1], "k-", label="executed safe path")
        axes[2].quiver(
            actual[0, 0],
            actual[0, 1],
            np.cos(actual[0, 2]),
            np.sin(actual[0, 2]),
            scale=4,
        )
        axes[2].quiver(
            actual[-1, 0],
            actual[-1, 1],
            np.cos(actual[-1, 2]),
            np.sin(actual[-1, 2]),
            scale=4,
            color="blue",
        )
        axes[2].legend(fontsize=8)
        axes[2].set_aspect("equal", adjustable="datalim")
        axes[2].set(
            xlabel="world x m",
            ylabel="world y m",
            title=f"{name}: applied={trace['applied']}\nsubstep time={trace['simulated_dt_s']:.2f}s",
        )
        fig.suptitle(
            "MODEL-FREE CONTROLLER DIAGNOSTIC — known paths, no learned navigation",
            fontsize=14,
        )
        p = output / f"{name}.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        panels.append(p)
    make_video(panels, output / "controller_diagnostics.mp4", fps=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--suite", type=Path)
    group.add_argument("--diagnostics", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.diagnostics:
        diagnostics(a.diagnostics, a.output)
        return
    aggregate = json.loads((a.suite / "aggregate.json").read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    index = []
    for summary in aggregate["runs"]:
        key = f"{summary['case']}__{summary['policy']}"
        count = run_panels(a.suite / key / "rollout", summary, a.output / key)
        index.append(
            {
                "condition": key,
                "panels": count,
                "video": str((a.output / key / "rollout.mp4").resolve()),
            }
        )
    (a.output / "index.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
