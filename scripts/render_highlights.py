"""Highlight montage of tracking successes and failures (private output, Git-ignored).

Each clip is rendered by ``render_tracking_video.py`` with a caption card, then all clips
are joined without re-encoding. Times are seconds since each recording's first LiDAR frame;
the captions state what the evaluation against ground truth found in that interval
(docs/TRACKING_GROUND_TRUTH.md). Run on the Mac that holds the recordings.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (recording, start, end, variants, caption[, extra renderer args]) — 'FAILURE' captions render
# red, others green.
CLIPS = [
    ("test_a1", 17.0, 23.0, ["baseline"],
     "SUCCESS (A1 17-23 s): static box keeps one ID|track #13 on the cases; jitter only, no ID change in this interval"),
    ("test_c", 6.5, 15.0, ["baseline"],
     "SUCCESS (C 6.5-15 s): robot turns -30 deg and back|box keeps track #79 and stays put in odom while it sweeps across the robot frame"),
    ("test_b", 21.8, 27.3, ["baseline"],
     "SUCCESS (B 21.8-27.3 s): walking person, one ID|track #235 from entry to the pause; velocity arrow and dashed constant-velocity forecast"),
    ("test_b", 7.8, 11.0, ["baseline"],
     "FAILURE (B 7.8-11 s): person merges into background clusters|#101 coasts, fragments spawn; at 10.3 s #101 jumps to the box (box #21->#101, person #101->#114 vs GT)"),
    ("test_b", 28.5, 32.5, ["baseline"],
     "FAILURE (B 28.5-32.5 s): person stands in front of the box|box occluded at 29.8 s: box #101->#312->#329->#337; #329 also takes the person at 31.4 s"),
    ("test_a2", 37.5, 40.8, ["baseline"],
     "FAILURE (A2 37.5-40.8 s): person beside the box merges with background|47 merged frames; person ID #304 deleted, new #362 at 38.9 s"),
    ("test_c", 34.5, 37.5, ["baseline"],
     "FAILURE (C 34.5-37.5 s): box merged into the counter/wall cluster|box #179 deleted, new #305 at 35.9 s while the robot walks forward"),
    ("test_b", 24.5, 31.0, ["baseline", "improved", "refined"],
     "COMPARISON (B 24.5-31 s): original / Phase 3.6 / Phase 3.7|3.6 keeps box #21 (one 0.3 s flicker) but #21 briefly takes the person at 27.4 s; all lose the box at 29.8 s"),
    ("test_b", 22.0, 30.0, ["baseline"],
     "FORECAST (B 22-30 s): constant velocity vs the annotated future|walking in, then a pause: the CV forecast from the annotated history overshoots the stop",
     ["--forecasts", "data/outputs/mac/forecasting_gt/overlay.json"]),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/outputs/mac/videos/highlights.mp4")
    p.add_argument("--card", type=float, default=2.5, help="caption card seconds")
    args = p.parse_args()
    out = Path(args.output)
    clips_dir = out.parent / (out.stem + "_clips")
    clips_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, (rec, a, b, variants, caption, *extra) in enumerate(CLIPS):
        path = clips_dir / f"{k:02d}_{rec}_{a:05.1f}-{b:05.1f}.mp4"
        cmd = [sys.executable, str(ROOT / "scripts/render_tracking_video.py"), "--recording", rec,
               "--start", str(a), "--end", str(b), "--banner", caption, "--title-card", str(args.card),
               "--output", str(path)]  # fmt: skip
        cmd += ["--compare", *variants] if len(variants) > 1 else ["--variant", variants[0]]
        cmd += extra[0] if extra else []
        subprocess.run(cmd, check=True, cwd=ROOT)
        paths.append(path.resolve())
    listing = clips_dir / "concat.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in paths))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c", "copy", "-movflags", "+faststart", str(out)], check=True)  # fmt: skip
    print("Wrote", out)


if __name__ == "__main__":
    main()
