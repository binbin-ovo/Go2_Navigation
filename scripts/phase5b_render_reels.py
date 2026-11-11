"""Phase 5B comparison videos from private recordings (1920x1080 H.264, git-ignored outputs).

  PYTHONPATH=src .venv/bin/python scripts/phase5b_render_reels.py --data ../Go2_Navigation/data \
      --out ../Go2_Navigation/data/outputs/phase5b/videos layers compare static montage highlights

Reels:
  layers      raw incremental scan | accumulated 1.0 s cloud (age) | 5A boxes, same instant
  compare     original 0.3 s clusters+tracks | 5A boxes (retrospective annotation ring, labelled)
  static      static-box stability examples (A1, C), original vs 5A
  montage     walking-person failures from the Test B diagnosis (one clip per category)
  highlights  successes, then failures, as separate captioned sections
Failure clips and success clips are written to separate folders (clips/failures, clips/successes).
"""

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import fsd_viewer  # noqa: E402
from go2_perception.viewer.layers import Layers, mode_layers  # noqa: E402
from go2_perception.viewer.privacy import require_private_dir  # noqa: E402
from go2_perception.visualization.diagnostic_video import VideoWriter, title_card  # noqa: E402

FPS = 15.4
NONE = Layers(**{k: False for k in Layers().__dict__})
CLEAN = mode_layers("clean")
P3_ONLY = replace(NONE, phase3_detections=True, phase3_tracks=True, motion=True, robot_path=True, occupancy_footprint=False)
GT = dict(retrospective_gt=True)

CATEGORY_TEXT = dict(
    absent_from_current_scan="person not swept by the latest ~62 ms scan: no current support, box goes stale",
    short_window_support="<8 non-ground person points in the 0.1 s object window (enough at 0.3 s)",
    foreground_clustering="person points left as DBSCAN noise / merged off-centre",
    ground_or_background="person points removed as ground or vertical plane",
    lifetime_stale="supported proposal absorbed by an older object outside the gate",
    motion_speed_gate="fresh box, but speed - 2 sigma < 0.25 m/s: stays UNCERTAIN (gray)",
    motion_object_too_young="fresh box re-created: < 4 observations over 0.4 s, no motion yet",
    motion_activation_streak="MOVING candidate needs 3 consecutive observations",
)


def session(a, recording, banner="", title=None):
    args = argparse.Namespace(
        scene=None, recording=recording, data=a.data, tap=None, mode="clean", layer=[], camera="chase", fps=FPS,
        start=0.0, end=None, interpolate=False, interpolation_delay_s=0.07, stale_max_s=0.6, ghost_hold_s=1.0,
        cv_horizon_s=1.5, retrospective_gt=False, no_rgb=False, title=title, banner=banner, output=None, still=None,
    )  # fmt: skip
    return fsd_viewer.build_session(args)


def write_clip(path, s, start, end, specs=None, status=(), card=None, card_s=1.2):
    require_private_dir(Path(path).parent)
    w = VideoWriter(path, FPS)
    if card:
        img = title_card(card)
        for _ in range(int(card_s * FPS)):
            w.write(img)
    c0 = s.t0 + start
    c1 = min(s.t0 + end, s.frames[-1].timestamp)
    for i in range(int((c1 - c0) * FPS) + 1):
        clock = c0 + i / FPS
        img = s.render(clock)[0] if specs is None else s.render_panels(clock, specs, status)
        w.write(img)
    w.close()
    print("wrote", path, w.frames, "frames", flush=True)
    return Path(path)


def concat(paths, out):
    lst = Path(out).with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in paths))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    lst.unlink()
    print("wrote", out, flush=True)


def layers_reel(a, out):
    specs = [
        ("RAW: current incremental scan only (~62 ms)", replace(NONE, raw_incremental=True, robot_path=True)),
        ("ACCUMULATED: 1.0 s in odom, colour = point age", replace(NONE, accumulated=True, point_age=True, robot_path=True)),
        ("5A BOXES: clean mode", CLEAN),
    ]
    clips = []
    for rec, start, end, note in [("test_b", 6.5, 19.0, "B pass 1: walking person"), ("test_c", 6.5, 20.0, "C: robot turned by hand")]:
        s = session(a, rec)
        clips.append(write_clip(out / "clips" / f"layers_{rec}.mp4", s, start, end, specs, [note + "; same instant in all panels; no future data"]))
    concat(clips, out / "raw_vs_accumulated_vs_boxes.mp4")


def compare_reel(a, out):
    specs = [
        ("ORIGINAL 0.3 s pipeline: clusters + tracks (filled = observed)", replace(P3_ONLY, **GT)),
        ("PHASE 5A: fresh (solid) / stale (dashed) boxes", replace(CLEAN, **GT)),
    ]
    status = ["magenta = RETROSPECTIVE annotation (LiDAR-derived, not human-verified): evaluation only"]
    clips = [write_clip(out / "clips" / f"compare_{rec}.mp4", session(a, rec), 0, 1e9, specs, status) for rec in ("test_b", "test_a2")]
    concat(clips, out / "original_vs_5a.mp4")


def static_reel(a, out, cmp):
    specs = [
        ("ORIGINAL 0.3 s: box track", replace(P3_ONLY, **GT)),
        ("PHASE 5A: box (stale = dashed)", replace(CLEAN, **GT)),
    ]
    clips = []
    for rec, start, end in [("test_a1", 20.0, 35.0), ("test_c", 6.5, 26.5)]:
        o, n = cmp[rec]["original"], cmp[rec]["5A"]
        so, sn = o["stability"]["static_box_step_m"], n["stability"]["static_box_step_m"]
        banner = f"STATIC BOX {rec[-2:].upper()}: displayed step p95 {so['p95'] * 1000:.0f} -> {sn['p95'] * 1000:.0f} mm | but 5A fresh coverage {o['perception']['box']['fresh_coverage']:.2f} -> {n['perception']['box']['fresh_coverage']:.2f}: steadiness is mostly held stale geometry"
        s = session(a, rec, banner=banner)
        clips.append(write_clip(out / "clips" / "successes" / f"static_{rec}.mp4", s, start, end, specs))
    concat(clips, out / "static_box_stability.mp4")


def montage_reel(a, out, diag):
    by_cat = {}
    for e in diag["episodes"]:
        by_cat.setdefault(e["category"], e)
    order = ["absent_from_current_scan", "motion_speed_gate", "motion_object_too_young", "motion_activation_streak", "short_window_support", "foreground_clustering", "ground_or_background", "lifetime_stale"]
    specs = [
        ("CURRENT SCAN (cyan) + 0.1 s clusters + 5A boxes", replace(CLEAN, raw_incremental=True, clusters=True, track_ids=True, **GT)),
        ("ACCUMULATED 1.0 s (age) + 5A boxes", replace(CLEAN, accumulated=True, point_age=True, **GT)),
    ]
    clips = []
    for cat in order:
        if cat not in by_cat:
            continue
        e = by_cat[cat]
        n = diag["all_frame_categories"].get(cat, {}).get("frames", 0)
        banner = f"FAIL {cat}: {CATEGORY_TEXT.get(cat, '')} | B {e['interval']} t={e['start']:.2f}-{e['end']:.2f} s; {n} of {diag['walking_frames']} walking frames in this category"
        s = session(a, "test_b", banner=banner)
        status = ["magenta ring = RETROSPECTIVE annotated person (evaluation only)", f"evidence: {e['detail']}"]
        clips.append(write_clip(out / "clips" / "failures" / f"montage_{cat}.mp4", s, max(e["start"] - 1.5, 0), e["end"] + 1.5, specs, status, card=f"FAILURE: {cat}|{CATEGORY_TEXT.get(cat, '')}"))
    concat(clips, out / "walking_person_failures.mp4")


def highlights_reel(a, out, cmp, diag):
    succ = out / "clips" / "successes"
    fail = out / "clips" / "failures"
    o, n = cmp["test_b"]["original"]["stability"], cmp["test_b"]["5A"]["stability"]
    parts = [("SUCCESSES|what Phase 5A + this viewer show well", None)]
    parts.append(("", write_clip(succ / "hl_c_static_in_odom.mp4", session(a, "test_c", banner="OK C: robot turned by hand, box stays put in odom | scene drawn in odom once; robot proxy alone uses T_odom_base"), 8.0, 16.0)))
    parts.append(("", write_clip(succ / "hl_b_moving_recalled.mp4", session(a, "test_b", banner="OK B: walking person marked MOVING (orange) with velocity +-2 sigma | recalled in only 14 of 240 walking frames (5.8%)"), 13.6, 15.6)))
    parts.append(("", write_clip(succ / "hl_b_no_false_moving.mp4", session(a, "test_b", banner=f"OK B: no MOVING on walls/clutter | unsupported MOVING frames/min: original {o['unsupported_moving_frames_per_min']:.0f}, 5A {n['unsupported_moving_frames_per_min']:.0f}"), 0.0, 6.0, [("ORIGINAL 0.3 s", P3_ONLY), ("PHASE 5A", CLEAN)])))
    parts.append(("FAILURES|kept separate; see walking_person_failures.mp4 for all categories", None))
    e = next(x for x in diag["episodes"] if x["category"] == "absent_from_current_scan")
    parts.append(("", write_clip(fail / "hl_b_absent_scan.mp4", session(a, "test_b", banner=f"FAIL B: person walking, latest scan has no returns on them | box held stale (dashed) - not fresh, not MOVING; {diag['all_frame_categories']['absent_from_current_scan']['frames']} of {diag['walking_frames']} walking frames"), e["start"] - 1.5, e["end"] + 1.5)))
    e = next(x for x in diag["episodes"] if x["category"] == "motion_speed_gate")
    parts.append(("", write_clip(fail / "hl_b_speed_gate.mp4", session(a, "test_b", banner="FAIL B: fresh box on a walking person stays UNCERTAIN (gray) | speed - 2 sigma below 0.25 m/s: heuristic sigma ~0.23 m/s"), e["start"] - 1.5, e["end"] + 1.5)))
    parts.append(("", write_clip(fail / "hl_a2_person_near_box.mp4", session(a, "test_a2", banner="FAIL A2: person beside the box | 5A moving recall 0.62 -> 0.12, fresh person coverage 0.74 -> 0.46"), 5.0, 12.0)))
    clips = []
    for i, (card, clip) in enumerate(parts):
        if clip is None:
            p = out / "clips" / f"_card_{i}.mp4"
            w = VideoWriter(p, FPS)
            img = title_card(card)
            for _ in range(int(1.6 * FPS)):
                w.write(img)
            w.close()
            clips.append(p)
        else:
            clips.append(clip)
    concat(clips, out / "highlights.mp4")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--comparison", help="phase5b_compare.py JSON (default <data>/outputs/phase5b/comparison_5A.json)")
    p.add_argument("--diagnosis", help="test_b diagnosis JSON (default <data>/outputs/phase5b/diagnosis/test_b.json)")
    p.add_argument("reels", nargs="+", choices=["layers", "compare", "static", "montage", "highlights"])
    a = p.parse_args()
    out = require_private_dir(Path(a.out))
    data = Path(a.data)
    cmp = json.loads(Path(a.comparison or data / "outputs/phase5b/comparison_5A.json").read_text())
    diag = json.loads(Path(a.diagnosis or data / "outputs/phase5b/diagnosis/test_b.json").read_text())
    np.seterr(all="ignore")
    for r in a.reels:
        dict(layers=lambda: layers_reel(a, out), compare=lambda: compare_reel(a, out), static=lambda: static_reel(a, out, cmp), montage=lambda: montage_reel(a, out, diag), highlights=lambda: highlights_reel(a, out, cmp, diag))[r]()


if __name__ == "__main__":
    main()
