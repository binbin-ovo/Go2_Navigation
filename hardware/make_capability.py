"""Write the Go2 hardware capability record (supervised-navigation-v1) from current evidence.

Versions and hashes are computed, never typed: the control checkout revision and cleanliness, the guard
provider's sha256, and this branch's revision. Evidence statuses come from EVIDENCE below and must name a
reference; anything not physically verified stays "unknown", which refuses live arming. Run in the core
environment; it imports no SDK.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from go2_perception.vln.supervised import CONTROL_REVISION, Hold, capability_check  # noqa: E402

HOME = Path.home()
CONTROL_ROOT = HOME / "Desktop/Go2_Navigation-go2-control"
HARDWARE_PY = HOME / "Desktop/Go2_Navigation/.venv-go2/bin/python"
PROVIDER = ROOT / "hardware/supervised_guards.py"
HEADING = ROOT / "data/outputs/hw_readiness/camera_heading.json"
CENTRE_RAY_YAW_DEG = 0.85   # rectified image-centre column relative to the optical axis (+ = left)

# Updated only from recorded results; see docs/GO2_HARDWARE_READINESS.md.
EVIDENCE = {
    "frame_mapping": ("verified", "FORWARD ONLY: camera yaw +0.8 deg from static wall capture "
                                  "(hardware/camera_heading.py), +0.85 deg image-centre convention, declared "
                                  "+/-3 deg; model origin unknown, bounded 0.5 m (translation_m null); scale "
                                  "unverified, capped by forward_m and LiDAR clearance; turn not verified"),
    "intervention": ("unknown", "emergency button while API owns control not yet exercised"),
    "obstacle_handling": ("unknown", "provider LiDAR clearance not yet checked live against a placed box"),
    "stop_verification": ("verified", "canonical logs 2026-11-15: zero Move x3 still in 0.2-0.4 s from "
                                      "0.3 m/s (C4, X2, C5), released code 0, avoidance enabled after"),
    "motion_bounds": ("verified", "forward 0.30 m/s: realized ~70 % of command, target 0.5 m stopped at "
                                  "0.505 m LiDAR (X2); heading drift +1.7..+6.8 deg, lateral 3.5-12 cm"),
    "target_correspondence": ("unknown", "offline on live_09/10: stationary 21/24, moved 0/18, outside "
                                         "0/42; not yet observed on the trial target"),
}

LIMITS = {
    "forward_m": 0.40, "turn_rad": 0.52, "forward_mps": 0.30, "turn_radps": 1.0, "duration_s": 4.0,
    "resend_period_s": 0.05, "state_age_s": 0.25, "geometry_age_s": 0.5, "stationary_dwell_s": 1.0,
    "auth_ttl_s": 15, "pose_translation_m": 0.05, "pose_yaw_rad": 0.05,
    "lateral_m": 0.08, "straight_yaw_rad": 0.10, "turn_translation_m": 0.05,
}


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


def mapping():
    if not HEADING.exists():
        return None
    accepted = json.loads(HEADING.read_text())["accepted_yaw_deg"]
    if not accepted:
        return None
    camera_yaw = sum(accepted) / len(accepted)
    # Declared, not fitted: one accepted wall capture; Phase 5D board-yaw agreement within ~2 deg; plus
    # the model's image-centre convention (0.85 deg) not being independently confirmed.
    uncertainty_deg = 3.0
    return {"from_frame": "lightnav_decision", "to_frame": "base_link", "translation_m": None,
            "translation_bound_m": 0.5,
            "yaw_rad": round(math.radians(camera_yaw + CENTRE_RAY_YAW_DEG), 5),
            "yaw_uncertainty_rad": round(math.radians(uncertainty_deg), 5), "verified_actions": ["forward"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "data/outputs/hw_readiness/capability.json")
    ap.add_argument("--valid-hours", type=float, default=12.0)
    a = ap.parse_args()
    revision = git(CONTROL_ROOT, "rev-parse", "HEAD")
    if revision != CONTROL_REVISION or git(CONTROL_ROOT, "status", "--porcelain", "--untracked-files=no"):
        sys.exit(f"control checkout is not the clean pinned {CONTROL_REVISION}")
    cap = {
        "capability_id": f"go2-hw-{git(ROOT, 'rev-parse', '--short', 'HEAD')}-{int(time.time())}",
        "kind": "hardware",
        "configuration_id": "go2-pro-front-camera-f5dedd82-avoid-route-forward-pilot",
        "valid_until_wall_s": time.time() + 3600 * a.valid_hours,
        "control": {"revision": revision, "root": str(CONTROL_ROOT), "python": str(HARDWARE_PY),
                    "provider_path": str(PROVIDER),
                    "provider_sha256": hashlib.sha256(PROVIDER.read_bytes()).hexdigest(),
                    "shared_lock": str(CONTROL_ROOT / "data/outputs/go2_control/.command.lock")},
        "evidence": {k: {"status": s, "reference": r} for k, (s, r) in EVIDENCE.items()},
        "mapping": mapping(),
        "limits": LIMITS,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(cap, indent=1))
    try:
        capability_check(cap, live=True, wall_s=time.time())
        print("capability passes the live check")
    except Hold as hold:
        print(f"capability written; live arming refuses: {hold}")
    print(a.output)


if __name__ == "__main__":
    main()
