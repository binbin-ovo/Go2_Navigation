"""Compose private human-review panels from a finished bounded-context run.

Read-only over the run: no inference, no relabeling, no source mutation. Each panel
shows one full/bounded pair from repeat 0 (instruction, current model-input image with
native OPOS/APOS, selected history with pooling tier and actual relative time, all ten
native waypoints) and, in a separately boxed evaluation-only area, the existing human
reference. Timings are offline fixed-cutoff measurements. Outputs go to a new
directory that must not already exist.
"""
import argparse
import json
import math
import pathlib

from PIL import Image, ImageDraw, ImageFont

from go2_perception.vln.shadow_display import point_fraction

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
W = 1600
RED, BLUE, GREY = "#c34236", "#148cee", "#888888"


def font(size):
    return ImageFont.truetype(FONT, size)


def load_pair(report, case):
    rows = [r for r in report["requests"] if r["case"] == case and r["repeat"] == 0]
    pair = {r["condition"]: r for r in rows}
    if set(pair) != {"full", "bounded"}:
        raise SystemExit(f"{case}: expected one full and one bounded repeat-0 request")
    return pair


def native(row):
    return json.loads((pathlib.Path(row["folder"]) / "native/result.json").read_text())


def history(row):
    """Selected source indices, actual relative times and pooling factor per image."""
    folder = pathlib.Path(row["folder"])
    res = native(row)
    frames = json.loads((folder / "job.json").read_text())["frames"]
    selected = res.get("native_selected_frame_indices") or sorted(
        int(p.stem.split("_")[1]) for p in (folder / "native").glob("input_*.png"))
    pool = {}
    for seg in res["native_segments"]:
        src = seg.get("source_frame_indices") or seg["frame_indices"]
        for i in src:
            pool[i] = seg["pool_spatial"]
    t0 = frames[0]["observation_s"]
    by_name = {pathlib.Path(f["path"]).stem: f for f in frames}
    out = []
    for i in selected:
        f = by_name.get(f"frame_{i:04d}")
        rel = (f["observation_s"] - t0) if f else float("nan")
        out.append((i, rel, pool.get(i), folder / "native" / f"input_{i:04d}.png"))
    return out, res.get("context_policy", {}).get("source_count", len(frames))


def overlay(row, size):
    folder = pathlib.Path(row["folder"])
    idx = max(native(row)["native_selected_frame_indices"])
    im = Image.open(folder / "native" / f"input_{idx:04d}.png").convert("RGB")
    d = ImageDraw.Draw(im)
    for channel, color in (("opos", RED), ("apos", BLUE)):
        v = point_fraction(row["pointing"][channel])
        if v:
            x, y = v[0] * im.width, v[1] * im.height
            d.ellipse((x - 8, y - 8, x + 8, y + 8), outline=color, width=3)
            d.line((x - 13, y, x + 13, y), fill=color, width=2)
            d.line((x, y - 13, x, y + 13), fill=color, width=2)
    return im.resize(size), idx


def strip(draw, sheet, items, x, y, width, label):
    draw.text((x, y), label, font=font(15), fill="black")
    n = len(items)
    per_row = min(n, 13)
    tw = (width - (per_row - 1) * 4) // per_row
    th = int(tw * 256 / 448)
    for k, (i, rel, pool, path) in enumerate(items):
        cx = x + (k % per_row) * (tw + 4)
        cy = y + 22 + (k // per_row) * (th + 30)
        sheet.paste(Image.open(path).convert("RGB").resize((tw, th)), (cx, cy))
        draw.text((cx, cy + th + 1), f"#{i} t={rel:.2f}s", font=font(11), fill="black")
        draw.text((cx, cy + th + 14), f"pool x{pool}", font=font(11), fill=GREY)
    rows = math.ceil(n / per_row)
    return y + 22 + rows * (th + 30)


def path_plot(draw, x, y, w, h, pair):
    draw.rectangle((x, y, x + w, y + h), outline="#ccc")
    pts_all = [a for m in pair.values() for a in m["waypoints"]]
    span = max(0.5, max(max(abs(a[0]), abs(a[1])) for a in pts_all)) * 1.15
    scale = min(w / 2, h - 30) / span
    ox, oy = x + w // 2, y + h - 20
    draw.line((x + 5, oy, x + w - 5, oy), fill="#ddd")
    draw.line((ox, y + 5, ox, oy), fill="#ddd")
    draw.text((x + 6, y + 4), "All 10 native ego waypoints (x fwd, y left; m)", font=font(13), fill="black")
    draw.text((x + 6, y + 20), "red = full, blue = bounded; tick = heading", font=font(13), fill="black")
    draw.text((ox + 4, oy + 2), "robot @ decision", font=font(11), fill=GREY)
    for mode, color in (("full", RED), ("bounded", BLUE)):
        wps = pair[mode]["waypoints"]
        pts = [(ox, oy)] + [(ox - a[1] * scale, oy - a[0] * scale) for a in wps]
        draw.line(pts, fill=color, width=2)
        for k, ((px, py), a) in enumerate(zip(pts[1:], wps), 1):
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
            draw.line((px, py, px - 12 * math.sin(a[2]), py - 12 * math.cos(a[2])), fill=color)
            if k in (1, 5, 10):
                draw.text((px + (5 if mode == "full" else -18), py - 6), str(k), font=font(11), fill=color)
    draw.text((x + 6, y + h - 16), f"scale: {span:.2f} m = half width", font=font(11), fill=GREY)


def waypoint_table(draw, x, y, row, color, title):
    draw.text((x, y), title, font=font(14), fill=color)
    draw.text((x, y + 18), " k     x(m)    y(m)   yaw(deg)", font=font(12), fill="black")
    for k, a in enumerate(row["waypoints"], 1):
        draw.text((x, y + 34 + (k - 1) * 15),
                  f"{k:>2}  {a[0]:7.3f} {a[1]:7.3f}  {math.degrees(a[2]):7.1f}", font=font(12), fill="black")
    return y + 34 + 10 * 15


def compose(report, case, title, question, live_note, out):
    pair = load_pair(report, case)
    instruction = json.loads((pathlib.Path(pair["full"]["folder"]) / "job.json").read_text())["instruction"]
    hist = {m: history(pair[m]) for m in pair}
    sheet = Image.new("RGB", (W, 2400), "white")
    d = ImageDraw.Draw(sheet)
    d.text((16, 10), f"PRIVATE HUMAN REVIEW | {title}", font=font(24), fill="black")
    d.text((16, 42), f"Case {case} | Instruction: \"{instruction}\"", font=font(18), fill="black")
    d.text((16, 66), "Model outputs only above the grey box. No robot registration, no execution. "
           "red ring = OPOS target point, blue ring = APOS affordance. OPOS 0 = model says target not visible.",
           font=font(14), fill=GREY)
    y = 96
    for col, mode in enumerate(("full", "bounded")):
        r = pair[mode]
        x = 16 + col * 540
        im, idx = overlay(r, (520, 297))
        d.text((x, y), f"{mode.upper()}: current model-input image (source #{idx})", font=font(16),
               fill=RED if mode == "full" else BLUE)
        sheet.paste(im, (x, y + 22))
        lines = [
            f"native: {r['native_output']}",
            f"OPOS {r['pointing']['opos']}  APOS {r['pointing']['apos']}  | decoded STOP {r['native_stop']}  "
            f"completion claim {r['completion_claim']}",
            f"old 1st-waypoint mapping: {r['mapped_action'] or 'UNSUPPORTED (mixed motion)'}",
            f"images supplied {r['supplied_images']}, native unique {r['native_unique_images']}, "
            f"slots {r['native_slots']}",
            f"OFFLINE delivery {r['request_to_review_file_s']:.3f} s; reconstructed age "
            f"{r['reconstructed_age_s']:.3f} s (not live)" + ("; COLD after load" if r["cold_after_load"] else ""),
        ]
        for j, line in enumerate(lines):
            d.text((x, y + 324 + j * 17), line, font=font(13), fill="black")
    path_plot(d, 1100, y + 22, 484, 400, pair)
    y += 420
    for mode in ("full", "bounded"):
        items, total = hist[mode]
        label = (f"{mode.upper()} selected history: {len(items)} of {total} causal candidates "
                 f"(t = actual seconds since first candidate; pool = native spatial pooling factor)")
        y = strip(d, sheet, items, 16, y, W - 32, label) + 8
    y += 6
    waypoint_table(d, 16, y, pair["full"], RED, "FULL waypoints")
    end = waypoint_table(d, 300, y, pair["bounded"], BLUE, "BOUNDED waypoints")
    d.multiline_text((600, y), "Note: the old discrete mapping uses only waypoint 1 (5 deg turn, 2 cm\n"
                     "advance/lateral thresholds). Mixed turn+translation is 'unsupported' by\n"
                     "that mapping, which is not the same as 'wrong'. Judge the full path.\n\n"
                     "Bounded also changes timestamp encoding (actual ms ticks) and pooling\n"
                     "tiers (anchors get x4), not only the number of images.",
                     font=font(13), fill="black", spacing=4)
    y = end + 16
    box_top = y
    ref = pair["full"]["human_reference"]
    d.text((28, y + 10), "EVALUATION ONLY - existing human reference (never a model input)",
           font=font(17), fill="black")
    y += 36
    if ref:
        for k in ("target_visibility", "instruction_progress", "acceptable_actions", "completion_status",
                  "ambiguity", "insufficient_information", "notes"):
            d.text((28, y), f"{k}: {ref.get(k)}", font=font(14), fill="black")
            y += 19
    else:
        d.multiline_text((28, y), live_note or "No per-request human reference exists for this case.",
                         font=font(14), fill="black", spacing=4)
        y += 19 * (1 + (live_note or "").count("\n"))
    d.text((28, y + 4), f"Codex agreement flag (diagnostic, not human review): full {pair['full']['agreement']}, "
           f"bounded {pair['bounded']['agreement']}", font=font(14), fill=GREY)
    y += 30
    d.rectangle((16, box_top, W - 16, y), outline=GREY, width=2, fill=None)
    y += 14
    d.multiline_text((16, y), "QUESTION FOR REVIEWER:\n" + question, font=font(17), fill="black", spacing=5)
    y += 30 + 24 * question.count("\n") + 30
    sheet.crop((0, 0, W, y)).save(out)


PAIRS = [
    ("01_e2_d2", "p6b_test_c_e2_turn_left_d2", "E2 d2 - shoe rack visibility",
     "Is the shoe rack identifiable in the displayed current image?\n"
     "Is FULL's target point (OPOS) correct? Is BOUNDED's 'target not visible' (OPOS 0) correct?", None),
    ("02_e1_d2", "p6b_test_c_e1_turn_right_d2", "E1 d2 - extra initial right turn",
     "Both miss the STOP reference. Does BOUNDED's initial right turn (-6.1 deg) and its later,\n"
     "more rightward path suit the instruction and the progress visible so far? Compare with FULL.", None),
    ("03_e3_d2", "p6b_test_c_e3_forward_bottles_d2", "E3 d2 - forward vs mixed turn/translation",
     "Judge the complete paths, not the discrete labels. Is BOUNDED's short mixed right-turn +\n"
     "small-advance path acceptable for the instruction here? Is FULL's longer forward path?", None),
    ("04_live_r3", "live_r3", "Saved 39-frame live cutoff - doorway",
     "Are BOTH target points plausible doorway targets? Are both complete paths plausible?\n"
     "(Full saw 38 unique images incl. mid history; bounded saw 2 anchors + 6 most recent.)",
     "No per-request human label. Historical operator note (cached-live report, earlier session):\n"
     "full-history targets opos_501/502/504 were judged correct, all on the doorway. That note\n"
     "covers the FULL condition only; the bounded r3 output has not been human-reviewed."),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=pathlib.Path, help="finished experiment root (read-only)")
    ap.add_argument("--out", required=True, type=pathlib.Path, help="new private output directory")
    args = ap.parse_args()
    report = json.loads((args.run / "evaluation/comparison.json").read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    for stem, case, title, question, note in PAIRS:
        path = args.out / f"pair_{stem}.png"
        compose(report, case, title, question, note, path)
        print(path)


if __name__ == "__main__":
    main()
