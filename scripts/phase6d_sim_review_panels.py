"""Read-only review panels for the saved Phase 6C rollout (no inference, no simulation).

Usage: python scripts/phase6d_sim_review_panels.py RUN_DIR OUT_DIR STEP [STEP ...]
Waypoints are absolute ego SE(2) relative to the pose *before* the step (x fwd, y left,
yaw CCW), transformed to world with that pose only. Pointing ids decode on the official
48x27 frame-relative grid; 1297/1298/1299 are apos rotate-L/rotate-R/stop, opos 0 = not visible.
"""
import hashlib, json, math, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

GW, GH = 48, 27


def decode(tok, chan):
    if tok is None:
        return "absent", None
    if chan == "apos" and tok in (1297, 1298, 1299):
        return {1297: "rotate-left directive", 1298: "rotate-right directive", 1299: "stop directive"}[tok], None
    if tok == 0:
        return ("not visible" if chan == "opos" else "no label"), None
    k = tok - 1
    col, row = k % GW, k // GW
    edge = col in (0, GW - 1) or row in (0, GH - 1)
    return f"grid cell col {col}, row {row}" + (" (edge cell, may be clamped)" if edge else ""), ((col + .5) / GW, (row + .5) / GH)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    run, out, steps = Path(sys.argv[1]), Path(sys.argv[2]), [int(s) for s in sys.argv[3:]]
    out.mkdir(parents=True, exist_ok=True)
    S = json.loads((run / "steps.json").read_text())
    summ = json.loads((run / "summary.json").read_text())
    goal = summ["final"]["goal_region"]
    path = [s["before"]["pose_xy_yaw"] for s in S] + [S[-1]["after"]["pose_xy_yaw"]]
    index = {}
    for n in steps:
        s = S[n]; p = s["proposal"]; b = s["before"]["pose_xy_yaw"]; a = s["after"]["pose_xy_yaw"]
        inp = sorted((run / "inference" / f"{n:03d}").glob("input_*.png"))
        cur = inp[-1]
        fig = plt.figure(figsize=(17, 10.5))
        gs = fig.add_gridspec(3, 3, height_ratios=[3.2, 1.0, 2.6], width_ratios=[1.25, 1, 1])
        ax = fig.add_subplot(gs[0, 0])
        img = Image.open(cur); W, H = img.size
        ax.imshow(img); ax.set_title(f"Current model input (step {n}, {cur.name}, {W}x{H})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ap, op = p["pointing"]["apos"], p["pointing"]["opos"]
        ad, axy = decode(ap, "apos"); od, oxy = decode(op, "opos")
        if axy:
            ax.plot(axy[0] * W, axy[1] * H, "o", ms=14, mfc="none", mec="cyan", mew=3)
            ax.annotate("apos (go-here)", (axy[0] * W, axy[1] * H), color="cyan", xytext=(-90, -20), textcoords="offset points", fontsize=9)
        if oxy:
            ax.plot(oxy[0] * W, oxy[1] * H, "x", ms=16, color="yellow", mew=3)
            ax.annotate("opos (target)", (oxy[0] * W, oxy[1] * H), color="yellow", xytext=(8, 8), textcoords="offset points", fontsize=9)
        # history strip
        hs = fig.add_subplot(gs[1, 0]); hs.axis("off")
        sel = p["native_selected_frame_indices"]
        thumbs = [Image.open(run / "inference" / f"{n:03d}" / f"input_{i:04d}.png") for i in sel][-6:]
        if thumbs:
            tw = 448 // 3; th = 256 // 3
            strip = Image.new("RGB", (tw * len(thumbs), th), "white")
            for j, t in enumerate(thumbs):
                strip.paste(t.resize((tw, th)), (j * tw, 0))
            hs.imshow(strip)
        hs.set_title(f"History given to model: {len(sel)} frame(s); last {len(thumbs)} shown, oldest -> current", fontsize=9)
        # local world plot
        ax2 = fig.add_subplot(gs[0:2, 1])
        ax2.add_patch(Rectangle((goal["x_range"][0], goal["y_range"][0]), goal["x_range"][1] - goal["x_range"][0],
                                goal["y_range"][1] - goal["y_range"][0], color="green", alpha=.18, label="stopping region (evaluator)"))
        ax2.plot([7, 7], [0.7, 2], color="gray", lw=4); ax2.plot([7, 7], [-2, -0.7], color="gray", lw=4, label="doorway wall (evaluator)")
        px, py = zip(*[(q[0], q[1]) for q in path[:n + 2]])
        ax2.plot(px, py, "-", color="tab:blue", lw=1, alpha=.6, label="actual path so far")
        c, sn = math.cos(b[2]), math.sin(b[2])
        wx = [b[0] + c * w[0] - sn * w[1] for w in p["waypoints"]]
        wy = [b[1] + sn * w[0] + c * w[1] for w in p["waypoints"]]
        wyaw = [b[2] + w[2] for w in p["waypoints"]]
        ax2.plot([b[0]] + wx, [b[1]] + wy, "--o", color="tab:orange", ms=4, label="native 10-waypoint chunk (world, via pose before)")
        for x, y, t in zip(wx, wy, wyaw):
            ax2.arrow(x, y, .12 * math.cos(t), .12 * math.sin(t), head_width=.03, color="tab:orange", alpha=.7)
        ax2.plot(wx[0], wy[0], "s", color="red", ms=8, label="waypoint 1 (the only one executed)")
        L = .35
        ax2.arrow(b[0], b[1], L * math.cos(b[2]), L * math.sin(b[2]), head_width=.05, color="black", label="heading before")
        ax2.arrow(a[0], a[1], L * math.cos(a[2]), L * math.sin(a[2]), head_width=.05, color="tab:blue", label="heading after")
        ax2.plot(6.15, 0, "+", color="green", ms=12)
        ax2.set_xlim(2.8, 7.4); ax2.set_ylim(-2.0, 1.6); ax2.set_aspect("equal"); ax2.grid(alpha=.3)
        ax2.set_xlabel("sim world x (m)"); ax2.set_ylabel("sim world y (m)")
        ax2.set_title("Evaluator-only top view (geometry never sent to model)", fontsize=10)
        ax2.legend(fontsize=7, loc="lower left")
        # yaw over steps
        ax3 = fig.add_subplot(gs[0:2, 2])
        yaws = [math.degrees(q[2]) for q in path]
        dist = [s2["before"]["distance_to_goal_center_m"] for s2 in S] + [S[-1]["after"]["distance_to_goal_center_m"]]
        ax3.plot(range(len(yaws)), yaws, "-o", ms=3, label="robot yaw (deg)")
        bear = [math.degrees(math.atan2(-q[1], 6.15 - q[0])) for q in path]
        ax3.plot(range(len(bear)), bear, ":", color="green", label="world bearing to stop-region centre (deg)")
        ax3.axvline(n, color="red", alpha=.4)
        ax3.set_xlabel("step (before)"); ax3.set_ylabel("deg"); ax3.grid(alpha=.3)
        a3b = ax3.twinx(); a3b.plot(range(len(dist)), dist, "-", color="purple", alpha=.6); a3b.set_ylabel("distance to region centre (m)", color="purple")
        ax3.legend(fontsize=7, loc="lower left"); ax3.set_title("Whole rollout; red line = this step", fontsize=10)
        # text
        tx = fig.add_subplot(gs[2, :]); tx.axis("off")
        wp = "  ".join(f"{i + 1}:({w[0]:+.3f},{w[1]:+.3f},{math.degrees(w[2]):+.1f}°)" for i, w in enumerate(p["waypoints"]))
        act = s["action"]["actual"]
        dx, dy = a[0] - b[0], a[1] - b[1]
        lines = [
            f"Instruction: {summ['instruction']}",
            f"Native output: {p['native_output']}",
            f"apos {ap} -> {ad};   opos {op} -> {od}   (frame-relative grid; no camera model)",
            "Native chunk, ego (forward m, left m, yaw deg), absolute from pose before, no time base:",
            "   " + wp[:len(wp) // 2], "   " + wp[len(wp) // 2:],
            f"Executed: waypoint 1 only, over {s['action']['simulated_dt_s']} simulated s: commanded ({act[0]*1000:+.1f} mm fwd, {act[1]*1000:+.1f} mm left, {math.degrees(act[2]):+.1f}°); applied={s['action']['applied']}",
            f"Actual sim motion: pose ({b[0]:.3f},{b[1]:.3f},{math.degrees(b[2]):+.1f}°) -> ({a[0]:.3f},{a[1]:.3f},{math.degrees(a[2]):+.1f}°): "
            f"translation {math.hypot(dx, dy)*1000:.1f} mm, yaw {math.degrees(a[2]-b[2]):+.1f}°; contacts {s['action']['contacts']}",
            f"MODEL stop output: decoder_stop={p['decoder_stop']}, explicit_stop_code={p['explicit_stop_code']}    |    "
            f"EVALUATOR: inside stopping region={s['after']['inside_stopping_region']}, distance to centre {s['before']['distance_to_goal_center_m']:.3f} -> {s['after']['distance_to_goal_center_m']:.3f} m",
        ]
        tx.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.6)
        fig.suptitle("PHASE 6D REVIEW — saved Phase 6C static rollout, read-only re-plot (no rerun)", color="darkred")
        fig.tight_layout()
        o = out / f"sim_step_{n:03d}.png"; fig.savefig(o, dpi=90); plt.close(fig)
        index[n] = {"panel": str(o), "current_input": str(cur), "current_input_sha256": sha(cur), "steps_json_sha256": sha(run / "steps.json")}
    (out / "panel_index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
