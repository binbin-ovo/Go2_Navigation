"""Write the Go2 hardware capability record (supervised-navigation-v1/v2) from current evidence.

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
from go2_perception.vln.motion_profile import INSPECTED, LIMITS as EFFECTIVE_LIMITS, PROFILE, SENSOR  # noqa: E402
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
                                  "unverified, capped by forward_m and LiDAR clearance; full turn mapping unverified"),
    "intervention": ("verified", "2026-11-18 go2_control/20261118_102052_ownership: API held control (claim 0, "
                                 "no Move); controller emergency button at 17.6 s -> error_code 100->1002 at "
                                 "18.9 s, operator saw it freeze standing; release 0, avoidance enabled; "
                                 "cleared by operator -> error_code 100 (20261118_102255_status)"),
    "obstacle_handling": ("verified", "FORWARD CORRIDOR ONLY, 2026-11-18 live provider (stationary): open floor "
                                      "nearest 2.88 m / clear 1.80 m; box centred ~1.1 m ahead of nose -> nearest "
                                      "1.53 m / clear 1.03 m; box 20 cm left -> nearest 1.58 m / clear 1.08 m; "
                                      "geometry age <= 0.11 s; rear/turn sweep lacks sensor coverage; "
                                      "onboard avoidance stays enabled as the independent layer"),
    "stop_verification": ("verified", "canonical logs 2026-11-15: zero Move x3 still in 0.2-0.4 s from "
                                      "0.3 m/s (C4, X2, C5), released code 0, avoidance enabled after"),
    "motion_bounds": ("verified", "forward 0.30 m/s: realized ~70 % of command, target 0.5 m stopped at "
                                  "0.505 m LiDAR (X2); heading drift +1.7..+6.8 deg, lateral 3.5-12 cm"),
    "target_correspondence": ("verified", "trial target (box), live_14 2026-11-18, robot stationary, operator ROI "
                                          "around the box: same view 0.3-1.1 s apart 20/24 valid (box at the "
                                          "left image edge refused 4/4, conservative); point outside ROI 0/6; "
                                          "box moved between requests 0/5 (changed_or_ambiguous_target_roi). "
                                          "Operator judged all 6 model points on the box. Earlier offline "
                                          "live_09/10: stationary 21/24, robot moved 0/18, outside 0/42"),
}

LIMITS = {
    "forward_m": 0.40, "turn_rad": 0.52, "forward_mps": 0.30, "turn_radps": 1.0, "duration_s": 4.0,
    "resend_period_s": 0.05, "state_age_s": 0.25, "geometry_age_s": 0.5, "stationary_dwell_s": 1.0,
    "auth_ttl_s": 15, "pose_translation_m": 0.05, "pose_yaw_rad": 0.05,
    "lateral_m": 0.08, "straight_yaw_rad": 0.10, "turn_translation_m": 0.05,
}


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


def mapping(heading_file=None):
    heading_file = HEADING if heading_file is None else heading_file
    if not heading_file.exists():
        return None
    accepted = json.loads(heading_file.read_text())["accepted_yaw_deg"]
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


HEADING_SHA256 = "c400119f8d95e42770f49c5565f1f73cebc96b7245159ba2d0526d68342b7bb8"
CONTROL_LOGS = {
    "forward": ("20261118_112542_move", "e8d6df2a4286d1a5c65f2493ab425dfc9c3eb42f8dc83b234866ceb6752c1dd1"),
    "right": ("20261118_112635_move", "321f2fe1d0c33a75bcabe9ed95270a93243a74be9858339539b03caae4ff6b6a"),
    "left": ("20261118_112705_move", "1c70ba76b1fcf02048d0223d60107c1a90dd922f49968c2ad369552f4575811e"),
}


def effective_profile(cap, scope, heading_file):
    if hashlib.sha256(heading_file.read_bytes()).hexdigest() != HEADING_SHA256:
        raise Hold("heading_evidence_differs_from_recorded_capture")
    for run, sha in CONTROL_LOGS.values():
        path = CONTROL_ROOT / "data/outputs/go2_control" / run / "events.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise Hold("changed_controller_evidence_"+run)
    cap.update(motion_profile=PROFILE, turn_scope=scope, configuration_id="go2-pro-front-camera-effective-v1")
    cap["limits"] = {**cap["limits"], **EFFECTIVE_LIMITS}
    cap["mapping"]["orientation_basis"] = "relative_planar_heading_ccw"
    evidence = cap["evidence"]
    evidence["motion_bounds"]["reference"] += " Individual fast trials " + str(CONTROL_LOGS)
    evidence["stop_verification"]["reference"] += " Fast trials settled forward 0.71 m odom, right 34.4 and left 40.8 degrees."
    evidence["frame_mapping"]["reference"] += " Relative planar yaw is invariant to the unknown origin. Yaw-only trial defers XY."
    evidence["obstacle_handling"]["reference"] += " Turn scope " + scope + ". Rear sweep requires explicit operator inspection."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "data/outputs/hw_readiness/capability.json")
    ap.add_argument("--valid-hours", type=float, default=12.0)
    ap.add_argument("--profile", choices=["legacy-first-row", PROFILE], default="legacy-first-row")
    ap.add_argument("--turn-scope", choices=[SENSOR, INSPECTED], default=SENSOR)
    ap.add_argument("--heading-file", type=Path, default=HEADING)
    a = ap.parse_args()
    if a.profile != PROFILE and a.turn_scope != SENSOR:
        ap.error("inspected turns require the effective profile")
    if a.output.exists():
        ap.error("choose a fresh capability output path")
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
        "mapping": mapping(a.heading_file),
        "limits": LIMITS,
    }
    if a.profile == PROFILE:
        effective_profile(cap, a.turn_scope, a.heading_file)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as stream:
        json.dump(cap, stream, indent=2)
    try:
        capability_check(cap, live=True, wall_s=time.time())
        print(f"{a.profile} passes static capability checks: {cap['limits']['forward_mps']} m/s, {cap['limits']['turn_radps']} rad/s")
    except Hold as hold:
        sys.exit(f"capability saved at {a.output}; static checks refused: {hold}")
    print(a.output)


if __name__ == "__main__":
    main()
