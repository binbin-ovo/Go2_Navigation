"""Calibrate Go2 front-camera intrinsics from a recording of a ChArUco target (offline).

The square size must be the value measured with a ruler on the physical target, not the nominal one.
Writes a ``go2-camera-calibration-v1`` YAML (status: measured) only when held-out validation exists,
plus a JSON report of both models' fit and held-out errors and image coverage.

Usage:
  PYTHONPATH=src python scripts/phase5d_calibrate_intrinsics.py \
      --recording data/recordings/p5d_charuco_01 \
      --board data/outputs/phase5d/targets/charuco_letter_8x6_30mm.json --square-mm 29.9 \
      --output configs/go2_front_camera_calibration.yaml --report data/outputs/phase5d/intrinsics.json
"""

import argparse
import datetime
import json
from pathlib import Path

import cv2

from go2_perception.vln.calibration import save_calibration
from go2_perception.vln.intrinsics import calibrate, detect, make_board


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recording", type=Path, required=True)
    ap.add_argument("--board", type=Path, required=True, help="target spec JSON from phase5d_make_charuco.py")
    ap.add_argument("--square-mm", type=float, required=True, help="MEASURED square size in mm")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--step", type=int, default=1, help="use every n-th frame")
    a = ap.parse_args()

    spec = json.loads(a.board.read_text())
    board = make_board(spec, a.square_mm / 1000)
    rows = [json.loads(line) for line in open(a.recording / "rgb" / "index.jsonl")][::a.step]
    detections, size = [], None
    for row in rows:
        gray = cv2.imread(str(a.recording / "rgb" / row["file"]), cv2.IMREAD_GRAYSCALE)
        size = size or (gray.shape[1], gray.shape[0])
        if (gray.shape[1], gray.shape[0]) != size:
            raise SystemExit(f"Frame {row['file']} changes image size")
        d = detect(board, gray)
        if d is not None:
            detections.append({"file": row["file"], "obj": d[0], "img": d[1]})
    print(f"{len(detections)} of {len(rows)} frames with >= 12 ChArUco corners")
    result = calibrate(detections, size)

    report = {k: v for k, v in result.items() if k != "models"}
    report["models"] = {}
    for model, r in result["models"].items():
        entry = {k: v for k, v in r.items() if k != "intrinsics"}
        if "intrinsics" in r:
            entry["K"] = r["intrinsics"].K.tolist()
            entry["D"] = r["intrinsics"].D.tolist()
        report["models"][model] = entry
    report.update(recording=a.recording.name, board=spec, square_mm_measured=a.square_mm, image_size=size)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({m: {k: r.get(k) for k in ("fit_rms_px", "heldout", "error")}
                      for m, r in report["models"].items()}, indent=2))
    best = result["recommended"]
    if best is None:
        raise SystemExit("No model produced held-out results; nothing saved")
    intr = result["models"][best]["intrinsics"]
    held = result["models"][best]["heldout"]
    provenance = {
        "date": datetime.date.today().isoformat(), "recording": a.recording.name,
        "target": a.board.name, "square_mm_measured": a.square_mm,
        "views_fit": result["fit_views"], "views_heldout": result["heldout_views"],
        "heldout_rms_px": held["rms_px"], "heldout_centre_rms_px": held["centre_rms_px"],
        "heldout_border_rms_px": held["border_rms_px"],
        "model_selection": "lowest held-out RMS of fisheye_equidistant / pinhole_radtan",
        "tool": "scripts/phase5d_calibrate_intrinsics.py",
    }
    intr = type(intr)(**{**{k: getattr(intr, k) for k in intr.__dataclass_fields__},
                         "provenance": provenance})
    save_calibration(a.output, intr)
    print(f"recommended {best}: held-out RMS {held['rms_px']:.3f} px -> {a.output}")


if __name__ == "__main__":
    main()
