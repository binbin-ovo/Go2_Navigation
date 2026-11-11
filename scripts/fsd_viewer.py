"""FSD-style 3D viewer for go2-scene-v1 (interactive window, MP4 export or PNG still).

Synthetic fixture (no private data needed):
  PYTHONPATH=src .venv/bin/python scripts/fsd_viewer.py --scene tests/fixtures/scene/renderer.jsonl
  PYTHONPATH=src .venv/bin/python scripts/fsd_viewer.py --scene tests/fixtures/scene/renderer.jsonl --still 0.8 --output /tmp/fx.png

Real recording (private inputs; outputs must be git-ignored):
  PYTHONPATH=src .venv/bin/python scripts/fsd_viewer.py --recording test_b --data ../Go2_Navigation/data
  PYTHONPATH=src .venv/bin/python scripts/fsd_viewer.py --recording test_b --data ../Go2_Navigation/data \
      --mode clean --output ../Go2_Navigation/data/outputs/phase5b/videos/test_b_clean.mp4

Interactive keys: space pause, a/d step one scan, A/D 2 s, m cycle mode, c camera mode,
[ ] orbit, - = zoom, f faster/slower toggle, layer keys as listed in Debug mode, q quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2_perception.viewer import sources as S  # noqa: E402
from go2_perception.viewer.layers import LAYER_INFO, MODES  # noqa: E402
from go2_perception.viewer.privacy import is_private_path  # noqa: E402
from go2_perception.viewer.session import Session, Sources  # noqa: E402
from go2_perception.viewer.state import DisplayConfig  # noqa: E402

TITLES = dict(test_a1="A1 static box", test_a2="A2 person near box", test_b="B walking person, three passes", test_c="C robot turned by hand")


def build_session(a):
    cfg = DisplayConfig(
        stale_max_s=a.stale_max_s,
        ghost_hold_s=a.ghost_hold_s,
        cv_horizon_s=a.cv_horizon_s,
        interpolate_pose=a.interpolate,
        interpolation_delay_s=a.interpolation_delay_s,
    )
    src = Sources(title=a.title or "", banner=a.banner)
    if a.recording:
        data = Path(a.data)
        out = data / "outputs"
        scene = Path(a.scene) if a.scene else out / "phase5b" / "tap" / a.recording / "scene.jsonl"
        frames = S.load_scene(scene)
        tap_dir = Path(a.tap) if a.tap else scene.parent
        if (tap_dir / "tap.npz").exists():
            src.tap = S.Tap(tap_dir)
        inputs = out / "mac" / "tracker_comparison" / "inputs" / a.recording
        tracks = out / "mac" / "ground_truth_eval" / "tracks" / "baseline" / a.recording / "tracks.jsonl"
        if inputs.exists() and tracks.exists():
            src.phase3 = S.Phase3(inputs, tracks)
        cache = data / "annotations" / "cache" / f"{a.recording}.npz"
        if not a.no_rgb and cache.exists():
            src.rgb = S.Rgb(data / "recordings" / a.recording, [f.source_time for f in frames], S.host_offset_from_cache(cache), out / "mac" / f"{a.recording}_replay" / "rgb_detections.json")
        ann = data / "annotations" / f"{a.recording}.json"
        if ann.exists():
            _, src.gt_frames = S.load_annotations(ann)
        src.title = a.title or f"{TITLES.get(a.recording, a.recording)} | Phase 5A scene"
    else:
        frames = S.load_scene(a.scene)
        src.title = a.title or f"{Path(a.scene).name} (synthetic fixture)"
    s = Session(frames, src, a.mode, cfg, a.camera, a.fps)
    for item in a.layer:
        name, on = item[1:], item[0] == "+"
        if name not in LAYER_INFO:
            sys.exit(f"Unknown layer {name}; choose from {', '.join(LAYER_INFO)}")
        setattr(s.layers, name, on)
    if a.retrospective_gt:
        s.layers.retrospective_gt = True
    return s


def clock_range(s, a):
    t0 = s.t0
    t_end = s.frames[-1].timestamp
    return t0 + max(a.start, 0.0), min(t0 + a.end, t_end) if a.end is not None else t_end


def export(s, a):
    from go2_perception.visualization.diagnostic_video import VideoWriter

    out = Path(a.output)
    if a.recording and not is_private_path(out):
        sys.exit(f"Refusing to write recording imagery to a path Git would track: {out}")
    c0, c1 = clock_range(s, a)
    if a.still is not None:
        img, _ = s.render(s.t0 + a.still)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
        print(f"Wrote {out}")
        return
    writer = VideoWriter(out, a.fps)
    n = int(np.floor((c1 - c0) * a.fps)) + 1
    start = time.perf_counter()
    for i in range(n):
        img, _ = s.render(c0 + i / a.fps)
        writer.write(img)
    writer.close()
    wall = time.perf_counter() - start
    print(f"Wrote {writer.frames} frames to {out}; render+encode {writer.frames / wall:.1f} fps; render only {1000 / np.mean(s.render_ms):.1f} fps")


def interactive(s, a):
    c0, c1 = clock_range(s, a)
    clock, paused, speed = c0, False, 1.0
    win = "Go2 FSD-style viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 900)
    last = time.perf_counter()
    period = np.median(np.diff(s.player.times)) if len(s.frames) > 1 else 0.1
    while True:
        img, _ = s.render(clock)
        cv2.imshow(win, img)
        key = cv2.waitKey(max(int(1000 / a.fps) - 5, 1)) & 0xFF
        now = time.perf_counter()
        if not paused:
            clock = min(clock + (now - last) * speed, c1)
        last = now
        ch = chr(key) if key != 255 else ""
        if ch in ("q", "\x1b"):
            break
        elif ch == " ":
            paused = not paused
        elif ch in ("a", "d"):
            clock = float(np.clip(clock + (period if ch == "d" else -period), c0, c1))
        elif ch in ("A", "D"):
            clock = float(np.clip(clock + (2.0 if ch == "D" else -2.0), c0, c1))
        elif ch == "m":
            s.set_mode(MODES[(MODES.index(s.mode) + 1) % len(MODES)])
        elif ch == "c":
            modes = ["chase", "fixed", "static"]
            s.camera.mode = modes[(modes.index(s.camera.mode) + 1) % 3]
        elif ch in ("[", "]"):
            s.camera.orbit_deg += 10 if ch == "]" else -10
        elif ch in ("-", "="):
            s.camera.zoom *= 1.1 if ch == "-" else 1 / 1.1
        elif ch == "f":
            speed = 0.25 if speed == 1.0 else 1.0
        elif ch:
            s.layers.toggle_key(ch)
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", help="go2-scene-v1 JSONL (default for --recording: phase5b tap output)")
    p.add_argument("--recording", help="real recording name, e.g. test_b (needs --data)")
    p.add_argument("--data", default=str(ROOT / "data"), help="data root with recordings/, annotations/, outputs/")
    p.add_argument("--tap", help="directory with tap.npz/tap.jsonl (Debug sources)")
    p.add_argument("--mode", default="clean", choices=MODES)
    p.add_argument("--layer", action="append", default=[], help="+name or -name, e.g. +raw_incremental")
    p.add_argument("--camera", default="chase", choices=["chase", "fixed", "static"])
    p.add_argument("--fps", type=float, default=15.4)
    p.add_argument("--start", type=float, default=0.0, help="seconds from first frame")
    p.add_argument("--end", type=float)
    p.add_argument("--interpolate", action="store_true", help="causal robot-pose interpolation (adds a display delay)")
    p.add_argument("--interpolation-delay-s", type=float, default=0.07)
    p.add_argument("--stale-max-s", type=float, default=0.6)
    p.add_argument("--ghost-hold-s", type=float, default=1.0)
    p.add_argument("--cv-horizon-s", type=float, default=1.5)
    p.add_argument("--retrospective-gt", action="store_true", help="draw annotations (evaluation only, labelled)")
    p.add_argument("--no-rgb", action="store_true")
    p.add_argument("--title")
    p.add_argument("--banner", default="")
    p.add_argument("--output", help=".mp4 to export a video, .png with --still")
    p.add_argument("--still", type=float, help="seconds from first frame")
    a = p.parse_args()
    if not a.scene and not a.recording:
        p.error("--scene or --recording required")
    s = build_session(a)
    if a.output:
        export(s, a)
    else:
        interactive(s, a)


if __name__ == "__main__":
    main()
