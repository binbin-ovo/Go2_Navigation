"""Synthetic viewer demo: run 5A's deterministic synthetic scenarios through the instrumented
pipeline, then write scene/tap files and clean/hybrid/debug screenshots (safe to commit).

  PYTHONPATH=src .venv/bin/python scripts/phase5b_synthetic_demo.py --output data/outputs/phase5b_synthetic \
      --screenshots docs/media/fsd
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2_perception.diagnostics.scene_tap import SceneTap  # noqa: E402
from go2_perception.io.config import load_config  # noqa: E402
from go2_perception.scene.serialization import validate_scene  # noqa: E402
from go2_perception.scene.synthetic import sequence  # noqa: E402
from go2_perception.viewer import sources as S  # noqa: E402
from go2_perception.viewer.session import Session, Sources  # noqa: E402

SHOTS = dict(start_move=[3.2], near_furniture=[2.6])


def run(name, out):
    out.mkdir(parents=True, exist_ok=False)
    config = load_config(ROOT / "configs/stable_lidar.yaml")
    config["sensor_height_m"] = 0.32
    tap = SceneTap(config)
    scans, acq, ng, wins, codes, ts = [], [], [], [], [], []
    with open(out / "scene.jsonl", "w") as f, open(out / "tap.jsonl", "w") as tf:
        for i, fr in enumerate(sequence(name)):
            row, rec = tap.process(fr["points"], fr["timestamp"], str(i), fr["pose_lookup"], clock_domain="synthetic")
            f.write(json.dumps(validate_scene(row), allow_nan=False) + "\n")
            ts.append(row["timestamp"])
            scans.append(rec.scan_xyz.astype(np.float32))
            acq.append(rec.scan_acquired)
            ng.append(rec.scan_nonground)
            wins.append(rec.window_xyz.astype(np.float32))
            codes.append(rec.window_code)
            clusters = [dict({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in c.items()}) for c in rec.clusters]
            tf.write(json.dumps(dict(t=row["timestamp"], clusters=clusters, proposals=[], internals={})) + "\n")

    def off(parts):
        return np.r_[0, np.cumsum([len(x) for x in parts])]

    np.savez_compressed(out / "tap.npz", frame_t=np.array(ts), frame_src=np.array(ts), scan_off=off(scans), scan_xyz=np.concatenate(scans), scan_acq=np.concatenate(acq), scan_ng=np.concatenate(ng), win_off=off(wins), win_xyz=np.concatenate(wins), win_code=np.concatenate(codes))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True)
    p.add_argument("--screenshots")
    a = p.parse_args()
    for name, times in SHOTS.items():
        out = Path(a.output) / name
        run(name, out)
        if not a.screenshots:
            continue
        frames = S.load_scene(out / "scene.jsonl")
        for mode in ("clean", "hybrid", "debug"):
            s = Session(frames, Sources(tap=S.Tap(out), title=f"5A synthetic scenario '{name}' (seed 17)"), mode)
            s.layers.track_ids = mode == "debug"
            for t in times:
                path = Path(a.screenshots) / f"synthetic_{name}_{mode}_t{t:.1f}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), s.render(frames[0].timestamp + t)[0])
                print("wrote", path)


if __name__ == "__main__":
    main()
