"""One supervised cycle; synthetic/mock by default. Hardware is separately gated.

Reuses run_vln_interactive sources, worker, calibrated preprocessing and recorder.
No hardware environment is changed. --help does not import any SDK or model.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import sys
import time

from go2_perception.vln.motion_profile import PROFILE
from go2_perception.vln.supervised import Hold, capability_check, contract_version
from go2_perception.vln.supervised_app import StationaryMockSource, SupervisedApp
from go2_perception.vln.supervised_executor import ExecutorClient
from go2_perception.vln import session_runner as interactive
from go2_perception.navigation import (DEFAULT_ALIGNMENT_DEG, DEFAULT_MOVES, DEFAULT_RADIUS_M, DEFAULT_SECONDS,
                                       MAX_EPISODE_SECONDS)

ROOT = Path(__file__).resolve().parents[3]


def add_episode_limits(p):
    p.add_argument("--move-budget", type=int, default=DEFAULT_MOVES)
    p.add_argument("--episode-seconds", type=float, default=DEFAULT_SECONDS)
    p.add_argument("--area-radius-m", type=float, default=DEFAULT_RADIUS_M)
    p.add_argument("--alignment-deg", type=float, default=DEFAULT_ALIGNMENT_DEG)


def valid_episode_limits(a):
    return (
        1 <= a.move_budget <= 5
        and 5 <= a.episode_seconds <= MAX_EPISODE_SECONDS
        and 1.2 <= a.area_radius_m <= 3
        and 8 <= a.alignment_deg <= 18
    )


def main(argv=None, *, website=None):
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument("--help", action="store_true")
    p.add_argument("--navigation-mode", choices=["mock", "live"], default="mock")
    p.add_argument("--profile", choices=["legacy-first-row", PROFILE], default="legacy-first-row")
    p.add_argument("--five-move-episode", action="store_true", help="One grant for a bounded search and approach episode")
    add_episode_limits(p)
    p.add_argument("--target-reference", type=Path, help="Optional private photograph of the requested instance")
    p.add_argument("--reference-target", help="Object name in the reference")
    p.add_argument("--reference-roi", type=float, nargs=4, metavar=("X0","Y0","X1","Y1"))
    p.add_argument("--capability", type=Path)
    p.add_argument("--control-root", type=Path)
    p.add_argument("--execute-live", action="store_true")
    p.add_argument("--exclusive-command-source", action="store_true")
    a, remaining = p.parse_known_args(argv)
    if a.help:
        p.print_help()
        interactive.parser().print_help()
        return 0
    if a.five_move_episode:
        if a.profile != PROFILE or not valid_episode_limits(a):
            p.error("episode requires the effective profile and supported movement, duration, area and alignment limits")
        reference = [a.target_reference, a.reference_target, a.reference_roi]
        if any(reference) and (not all(reference) or not a.target_reference.is_file()):
            p.error("Supply an existing reference image with its target name and normalized ROI")
        remaining = [*remaining, "--max-trials", "1", "--max-requests-per-trial", "64", "--trial-max-s", str(a.episode_seconds + 10),
                     "--history-seconds", str(a.episode_seconds + 10), "--history-frames", "1024"]
    args = interactive.parser().parse_args(remaining)
    if args.review_run:
        p.error("Use the existing read-only launcher to review saved runs")
    live = a.navigation_mode == "live"
    if live:
        if website is not None:
            p.error("The mode selector only hosts offline supervised episodes")
        if not a.capability or not a.control_root or not args.hardware_python or not args.unitree_sdk:
            p.error("Live needs capability, canonical control root, hardware Python and SDK paths")
        if not a.execute_live or not a.exclusive_command_source or os.environ.get("GO2_SUPERVISED_LIVE") != "1":
            p.error("Live disabled: explicit opt-in, exclusive source confirmation and GO2_SUPERVISED_LIVE=1 required")
        if args.camera != "go2" or args.backend != "lightnav" or args.lidar != "go2":
            p.error("Live needs existing Go2 camera/LiDAR bridges and pinned LightNav backend")
        capability = json.loads(a.capability.read_text())
        from .source_revision import clean_navigation
        clean_navigation(ROOT, capability.get("capability_id") or "")
    else:
        if args.camera != "synthetic" or args.backend != "mock" or args.lidar not in ("none", "synthetic"):
            p.error("This mock launcher accepts synthetic sources and mock worker only; no robot or model")
        capability = (json.loads(a.capability.read_text()) if a.capability else
                      json.loads((ROOT / "examples/supervised_navigation/synthetic_cycle.json").read_text())["capability"])
    scene = None
    if a.five_move_episode:
        from go2_perception.vln.episode_mock import EpisodeMockScene, effective_mock_capability
        if not live:
            scene = EpisodeMockScene(json.loads((ROOT / "examples/five_move_navigation/synthetic_episode.json").read_text()))
        from go2_perception.vln.motion_profile import INSPECTED
        if not live and not a.capability:
            capability = effective_mock_capability(capability)
        if capability.get("turn_scope") != INSPECTED or args.lidar == "none":
            p.error("Episode requires the inspected-area capability and LiDAR")
    if capability.get("motion_profile", "legacy-first-row") != a.profile:
        p.error("selected profile must match the generated capability")
    capability_check(capability, live=live, wall_s=time.time())
    print(f"Selected {a.profile} | forward {capability['limits']['forward_mps']} m/s | turn {capability['limits']['turn_radps']} rad/s", flush=True)
    nonce = secrets.token_hex(24)

    def app_factory(**kwargs):
        root = kwargs["store"].root
        cap_path = root / "capability.json"
        cap_path.write_text(json.dumps(capability, indent=2) + "\n")
        command = [str(args.hardware_python if live else sys.executable), "-u",
                   str(ROOT / "scripts/vln_supervised_executor.py"), "--mode", "live" if live else "mock",
                   "--capability", str(cap_path), "--nonce", nonce, "--output", str(root / "executor")]
        if a.five_move_episode:
            command.append("--five-move-episode")
        if live:
            command += ["--execute-live", "--exclusive-command-source", "--control-root", str(a.control_root),
                        "--unitree-sdk", str(args.unitree_sdk), "--iface", args.iface]
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "GO2_HARDWARE": "0", "MUJOCO_RENDER_TEST": "0"}
        executor = ExecutorClient(command, env=env, nonce=nonce)
        end = time.monotonic() + 5.
        while not executor.ready and executor.process.poll() is None and time.monotonic() < end:
            time.sleep(.02)
        if not executor.ready:
            executor.close()
            raise Hold("executor_not_ready; inspect private executor logs (no model worker loaded)")
        try:
            if a.five_move_episode:
                from go2_perception.vln.episode_app import EpisodeApp
                settings = dict(max_moves=a.move_budget, seconds=a.episode_seconds, area_radius_m=a.area_radius_m,
                                alignment_deg=a.alignment_deg, reference_path=a.target_reference,
                                reference_roi=a.reference_roi, reference_target=a.reference_target)
                return EpisodeApp(capability=capability, executor=executor, live=live, nonce=nonce,
                                  settings=settings, scene=scene, **kwargs)
            return SupervisedApp(capability=capability, executor=executor, live=live, nonce=nonce, **kwargs)
        except BaseException:
            executor.close()
            raise

    run_fields = {
        "integration": f"supervised-navigation-v{contract_version(capability)}",
        "profile": a.profile, "navigation_mode": a.navigation_mode,
        "physical_execution_authorized": False,
        "execution_note": "Permission is issued per fresh one-use cycle; see navigation_summary.json",
        "canonical_control_revision": capability["control"]["revision"] if live else None,
    }
    if a.five_move_episode:
        from go2_perception.vln.bounded_context import POLICY
        from go2_perception.vln.five_move import SCHEMA
        run_fields.update(integration=SCHEMA, temporal_policy=POLICY,
                          execution_note="See episode_summary.json and the executor log for this grant",
                          episode_limits={"max_moves":a.move_budget, "seconds":a.episode_seconds,
                                          "alignment_deg":a.alignment_deg, "area_radius_m":a.area_radius_m})
    return interactive.main(remaining, app_factory=app_factory, synthetic_source_factory=scene.source if scene else StationaryMockSource,
                            run_fields=run_fields, website=website)
