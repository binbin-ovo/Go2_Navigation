"""Read-only review panels for saved Phase 6C LightNav outputs on the recorded E1-E4 decisions.

Usage: python scripts/phase6d_recorded_review_panels.py PILOT_V3_DIR REVIEW_V2_DIR OUT_DIR INDEX [INDEX ...]
Nothing is executed: recorded frames follow the operator route. Pointing decodes on the
official 48x27 frame-relative grid; the plotted chunk is native ego SE(2), unregistered.
"""
import hashlib, json, math, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from phase6d_sim_review_panels import decode  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    v3, rv, out, idx = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), [int(i) for i in sys.argv[4:]]
    out.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((v3 / "jobs.json").read_text())["jobs"]
    index = json.loads((out / "rec_panel_index.json").read_text()) if (out / "rec_panel_index.json").exists() else {}
    for i in idx:
        d = v3 / "inference" / f"{i:03d}"
        r = json.loads((d / "result.json").read_text())
        ref = json.loads((rv / f"{i:03d}" / "evaluation_reference.json").read_text())["decision"]
        inp = sorted(d.glob("input_*.png")); cur = inp[-1]
        fig = plt.figure(figsize=(17, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[3.2, 1.0, 2.4], width_ratios=[1.5, 1])
        ax = fig.add_subplot(gs[0, 0]); img = Image.open(cur); W, H = img.size
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Current model input ({r['episode_id']} {r['decision_id']}, {cur.name}, {W}x{H})", fontsize=10)
        ap, op = r["pointing"]["apos"], r["pointing"]["opos"]
        ad, axy = decode(ap, "apos"); od, oxy = decode(op, "opos")
        if axy:
            ax.plot(axy[0] * W, axy[1] * H, "o", ms=16, mfc="none", mec="cyan", mew=3)
            ax.annotate("apos (go-here)", (axy[0] * W, axy[1] * H), color="cyan", xytext=(-100, -18), textcoords="offset points", fontsize=10, weight="bold")
        if oxy:
            ax.plot(oxy[0] * W, oxy[1] * H, "x", ms=18, color="yellow", mew=4)
            ax.annotate("opos (target)", (oxy[0] * W, oxy[1] * H), color="yellow", xytext=(10, 10), textcoords="offset points", fontsize=10, weight="bold")
        hs = fig.add_subplot(gs[1, 0]); hs.axis("off")
        th = [Image.open(p) for p in inp][-6:]
        tw, tt = 448 // 3, 256 // 3
        strip = Image.new("RGB", (tw * len(th), tt), "white")
        for j, t in enumerate(th):
            strip.paste(t.resize((tw, tt)), (j * tw, 0))
        hs.imshow(strip); hs.set_title(f"Inputs given to model: {len(inp)} image(s); last {len(th)} shown, oldest -> current", fontsize=9)
        ax2 = fig.add_subplot(gs[0:2, 1])
        w = r["waypoints"]
        xs = [-q[1] for q in w]; ys = [q[0] for q in w]
        ax2.plot([0] + xs, [0] + ys, "--o", color="tab:orange", ms=4, label="native 10-waypoint chunk")
        for x, y, q in zip(xs, ys, w):
            ax2.arrow(x, y, -.08 * math.sin(q[2]), .08 * math.cos(q[2]), head_width=.025, color="tab:orange", alpha=.7)
        ax2.plot(xs[0], ys[0], "s", color="red", ms=8, label="waypoint 1")
        ax2.arrow(0, 0, 0, .12, head_width=.03, color="black", label="current heading")
        span = max(0.4, max(abs(v) for v in xs + ys) * 1.15)
        ax2.set_xlim(-span, span); ax2.set_ylim(-span * .3, span * 1.2); ax2.set_aspect("equal"); ax2.grid(alpha=.3)
        ax2.set_xlabel("right (m) = -native lateral"); ax2.set_ylabel("native forward (m)")
        ax2.set_title("Native chunk, ego, unregistered to Go2, no time base\n(NOT executed: recorded operator route)", fontsize=10)
        ax2.legend(fontsize=8, loc="lower right")
        tx = fig.add_subplot(gs[2, :]); tx.axis("off")
        wp = "  ".join(f"{k + 1}:({q[0]:+.3f},{q[1]:+.3f},{math.degrees(q[2]):+.1f}°)" for k, q in enumerate(w))
        lines = [
            f"Instruction: {jobs[i]['instruction']}",
            f"Native output: {r['native_output']}",
            f"apos {ap} -> {ad};   opos {op} -> {od}   (frame-relative 48x27 grid; no camera model)",
            "Native chunk, ego (forward m, left m, yaw deg):",
            "   " + wp[:len(wp) // 2], "   " + wp[len(wp) // 2:],
            f"MODEL stop output: decoder_stop={r['decoder_stop']}, explicit_stop_code={r['explicit_stop_code']}",
            f"EXISTING HUMAN REFERENCE (evaluation only, unchanged): target {ref['target_visibility']}, acceptable {ref['acceptable_actions']}, "
            f"completion {ref['completion_status']}, ambiguous {ref['ambiguity']}",
        ]
        tx.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)
        fig.suptitle("PHASE 6D REVIEW — saved LightNav output on a recorded decision; nothing executed, nothing rerun", color="darkred")
        fig.tight_layout()
        o = out / f"rec_{i:03d}_{r['episode_id'][11:]}_{r['decision_id']}.png"; fig.savefig(o, dpi=90); plt.close(fig)
        index[str(i)] = {"panel": str(o), "current_input": str(cur), "current_input_sha256": sha(cur),
                         "result_sha256": sha(d / "result.json")}
    (out / "rec_panel_index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
