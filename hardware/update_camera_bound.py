"""Pin a new boot's camera capture bound in hardware/supervised_guards.py (file-only; no robot access).

Usage, after hardware/capture_latency.py printed its evidence path:
    python hardware/update_camera_bound.py evidence/hw_readiness/capture_latency_YYYYMMDD_HHMMSS.json
The previous boot offset moves to EARLIER_BOOT_OFFSETS_S so its bound is never reused. Commit the
result before preparing a capability (supervised startup requires a clean checkout).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GUARDS = ROOT / "hardware/supervised_guards.py"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("evidence", type=Path, help="capture_latency_*.json written by hardware/capture_latency.py")
    a = ap.parse_args(argv)
    path = a.evidence.resolve()
    record = json.loads(path.read_bytes())
    bound, offset, stats = record.get("bound_s"), record.get("boot_offset_s"), record.get("stats") or {}
    if not (isinstance(bound, (int, float)) and math.isfinite(bound) and 0 < bound < 0.5):
        sys.exit(f"unusable bound_s {bound!r}: re-run hardware/capture_latency.py with the screen in front of the camera")
    if not (isinstance(offset, (int, float)) and math.isfinite(offset)) or stats.get("n", 0) < 20:
        sys.exit("evidence lacks a boot offset or at least 20 measured switches")
    parts = path.parts
    if "evidence" not in parts:
        sys.exit("evidence file must live under $GO2_WORKSPACE/evidence/")
    rel = "/".join(parts[parts.index("evidence"):])
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    text = GUARDS.read_text()
    old = float(re.search(r'"boot_offset_s": (-?[0-9.]+),', text).group(1))
    if abs(old - offset) <= 5.0:
        sys.exit(f"this boot (offset {offset:.3f}) already has its bound pinned; nothing to change")
    earlier = re.search(r"EARLIER_BOOT_OFFSETS_S = \(([^)]*)\)", text)
    offsets = [f"{old:.3f}"] + [v.strip() for v in earlier.group(1).split(",") if v.strip()]
    stamp = time.strftime("%H:%M on %Y-%m-%d", time.localtime(record.get("created_wall_s", time.time())))
    comment = (f"    # Re-measured for the robot boot seen at {stamp} by hardware/capture_latency.py: "
               f"{stats['n']} switches,\n    # switch-to-receipt median {stats['median_s']:.3f} s, max {stats['max_s']:.3f} s; "
               f"bound = max - 0.048 configured + 0.020 margin.\n    # Earlier boots are listed in EARLIER_BOOT_OFFSETS_S "
               f"and are not reused.\n")
    text, n1 = re.subn(r'    # Re-measured for the robot boot.*?(?=    "bound_s")', comment, text, count=1, flags=re.S)
    text, n2 = re.subn(r'"bound_s": [0-9.]+,', f'"bound_s": {bound:.3f},', text, count=1)
    text, n3 = re.subn(r'"boot_offset_s": -?[0-9.]+,', f'"boot_offset_s": {offset:.3f},', text, count=1)
    text, n4 = re.subn(r'"basis": \("evidence/[^"]*"\n\s*"[0-9a-f]{64}"\)',
                       f'"basis": ("{rel} sha256 "\n              "{sha}")', text, count=1)
    text, n5 = re.subn(r"EARLIER_BOOT_OFFSETS_S = \([^)]*\)", f"EARLIER_BOOT_OFFSETS_S = ({', '.join(offsets)})",
                       text, count=1)
    if (n1, n2, n3, n4, n5) != (1, 1, 1, 1, 1):
        sys.exit(f"supervised_guards.py layout not recognised {(n1, n2, n3, n4, n5)}; edit CAMERA by hand")
    GUARDS.write_text(text)
    print(f"CAMERA bound {bound:.3f} s pinned for boot offset {offset:.3f}; previous boot {old:.3f} archived.")
    print("Next: run the tests, then commit hardware/supervised_guards.py before preparing a capability.")


if __name__ == "__main__":
    main()
