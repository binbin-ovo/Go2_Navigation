"""Write a printable ChArUco calibration target (PDF at exact scale) and its machine spec.

Usage:
  python scripts/phase5d_make_charuco.py --paper a3 --output data/outputs/phase5d/targets
Or, without a printer, a full-screen PNG for a flat display:
  python scripts/phase5d_make_charuco.py --screen 2560x1600 --output data/outputs/phase5d/targets
Print the PDF at 100% / "actual size" (never "fit to page"), then measure one full row of
squares with a ruler and pass the measured square size to calibration; the spec's nominal
size is only what was requested from the printer.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PAPERS_MM = {"a4": (297, 210), "a3": (420, 297), "letter": (279.4, 215.9)}
# squares_x, squares_y, square_mm, marker_mm per paper (landscape), leaving >= 10 mm margins.
LAYOUTS = {"a4": (9, 6, 30.0, 22.0), "a3": (11, 8, 35.0, 26.0), "letter": (8, 6, 30.0, 22.0)}
DICTIONARY = "DICT_5X5_100"
DPI = 600


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paper", choices=sorted(PAPERS_MM), default="a3")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--screen", help="WxH pixels: write a full-screen PNG instead of a PDF")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.screen:
        return write_screen(a.output, *map(int, a.screen.lower().split("x")))

    sx, sy, square_mm, marker_mm = LAYOUTS[a.paper]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    board = cv2.aruco.CharucoBoard((sx, sy), square_mm / 1000, marker_mm / 1000, dictionary)
    px_per_mm = DPI / 25.4
    board_px = (round(sx * square_mm * px_per_mm), round(sy * square_mm * px_per_mm))
    img = board.generateImage(board_px, marginSize=0, borderBits=1)

    paper_w, paper_h = PAPERS_MM[a.paper]
    page = np.full((round(paper_h * px_per_mm), round(paper_w * px_per_mm)), 255, np.uint8)
    oy, ox = (page.shape[0] - img.shape[0]) // 2, (page.shape[1] - img.shape[1]) // 2
    page[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
    label = (f"ChArUco {sx}x{sy} {DICTIONARY} square {square_mm} mm marker {marker_mm} mm  "
             f"PRINT AT 100% - measure 10 squares: expect {10 * square_mm:.0f} mm")
    cv2.putText(page, label, (ox, page.shape[0] - round(4 * px_per_mm)),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, 0, 4, cv2.LINE_AA)

    stem = f"charuco_{a.paper}_{sx}x{sy}_{square_mm:g}mm"
    Image.fromarray(page).save(a.output / f"{stem}.pdf", resolution=DPI)
    spec = {"type": "charuco", "dictionary": DICTIONARY, "squares_x": sx, "squares_y": sy,
            "square_size_m_nominal": square_mm / 1000, "marker_size_m_nominal": marker_mm / 1000,
            "paper": a.paper, "dpi": DPI, "legacy_pattern": False,
            "board_origin": "top-left inner corner of the printed board, x right, y down, z into board"}
    (a.output / f"{stem}.json").write_text(json.dumps(spec, indent=2))
    print(a.output / f"{stem}.pdf")


def write_screen(output, width, height):
    """Board with a one-square white quiet zone; square size in metres is unknown until measured."""
    sx, sy = (9, 6) if width / height >= 1.5 else (8, 6)
    square_px = min(width // (sx + 2), height // (sy + 2))
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    marker_ratio = 0.75
    board = cv2.aruco.CharucoBoard((sx, sy), 1.0, marker_ratio, dictionary)
    img = board.generateImage((sx * square_px, sy * square_px), marginSize=0, borderBits=1)
    page = np.full((height, width), 255, np.uint8)
    oy, ox = (height - img.shape[0]) // 2, (width - img.shape[1]) // 2
    page[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
    stem = f"charuco_screen_{width}x{height}_{sx}x{sy}"
    Image.fromarray(page).save(output / f"{stem}.png")
    spec = {"type": "charuco", "dictionary": DICTIONARY, "squares_x": sx, "squares_y": sy,
            "square_px": square_px, "marker_to_square_ratio": marker_ratio,
            "square_size_m_nominal": None, "display": f"{width}x{height}", "legacy_pattern": False,
            "note": "Show at native resolution, full screen, no scaling; measure squares on screen"}
    (output / f"{stem}.json").write_text(json.dumps(spec, indent=2))
    print(output / f"{stem}.png")


if __name__ == "__main__":
    main()
