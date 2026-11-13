"""Deterministic two-frame public synthetic input, not a navigation accuracy test."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    for i in range(2):
        im = Image.new("RGB", (448, 256), "#dfd9c8")
        d = ImageDraw.Draw(im)
        d.polygon([(0, 172), (448, 172), (448, 256), (0, 256)], fill="#877c66")
        d.rectangle(
            (273 - i * 2, 43, 362 - i * 2, 174),
            fill="#503c2b",
            outline="white",
            width=4,
        )
        d.ellipse((342 - i * 2, 110, 348 - i * 2, 116), fill="#d9bd38")
        d.rectangle((63, 116, 120, 177), fill="#b63132")
        d.line((150, 172, 100, 256), fill="#595449", width=2)
        d.line((246, 172, 270, 256), fill="#595449", width=2)
        im.save(a.output / f"{i:03d}.png")


if __name__ == "__main__":
    main()
