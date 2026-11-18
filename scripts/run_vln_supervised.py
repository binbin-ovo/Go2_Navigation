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
import run_vln_interactive as interactive

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument("--help", action="store_true")
    p.add_argument("--navigation-mode", choices=["mock", "live"], default="mock")
    p.add_argument("--profile", choices=["legacy-first-row", PROFILE], default="legacy-first-row")
    p.add_argument("--capability", type=Path)
    p.add_argument("--control-root", type=Path)
    p.add_argument("--execute-live", action="store_true")
    p.add_argument("--exclusive-command-source", action="store_true")
    a, remaining = p.parse_known_args(argv)
    if a.help:
        p.print_help()
        interactive.parser().print_help()
        return 0
    args = interactive.parser().parse_args(remaining)
    if args.review_run:
        p.error("Use the existing read-only launcher to review saved runs")
    live = a.navigation_mode == "live"
    if live:
        if not a.capability or not a.control_root or not args.hardware_python or not args.unitree_sdk:
            p.error("Live needs capability, canonical control root, hardware Python and SDK paths")
        if not a.execute_live or not a.exclusive_command_source or os.environ.get("GO2_SUPERVISED_LIVE") != "1":
            p.error("Live disabled: explicit opt-in, exclusive source confirmation and GO2_SUPERVISED_LIVE=1 required")
        if args.camera != "go2" or args.backend != "lightnav" or args.lidar != "go2":
            p.error("Live needs existing Go2 camera/LiDAR bridges and pinned LightNav backend")
        capability = json.loads(a.capability.read_text())
    else:
        if args.camera != "synthetic" or args.backend != "mock" or args.lidar not in ("none", "synthetic"):
            p.error("This mock launcher accepts synthetic sources and mock worker only; no robot or model")
        capability = (json.loads(a.capability.read_text()) if a.capability else
                      json.loads((ROOT / "examples/supervised_navigation/synthetic_cycle.json").read_text())["capability"])
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
        return SupervisedApp(capability=capability, executor=executor, live=live, nonce=nonce, **kwargs)

    return interactive.main(remaining, app_factory=app_factory, synthetic_source_factory=StationaryMockSource,
                            run_fields={"integration": f"supervised-navigation-v{contract_version(capability)}", "profile": a.profile, "navigation_mode": a.navigation_mode,
                                        "physical_execution_authorized": False,
                                        "execution_note": "Permission is issued per fresh one-use cycle; see navigation_summary.json",
                                        "canonical_control_revision": capability["control"]["revision"] if live else None})


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Hold as error:
        print("HOLD: " + str(error), file=sys.stderr)
        raise SystemExit(2)
