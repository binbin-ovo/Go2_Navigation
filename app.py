"""Start the local navigation website. The initial page is observation-only."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

# The shared core interpreter also knows about Claude's checkout. Pin imports here.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from go2_perception.vln import session_runner
from go2_perception.vln.web_modes import WebsiteMode


def observe(args, output, website):
    """Start independent sensor and advisory workers; a target starts a model request."""
    from go2_perception.sensors import default_source_args

    options = default_source_args(args, args.workspace)
    if args.model == "lightnav":
        options += [
            "--backend",
            "lightnav",
            "--model-python",
            str(args.workspace / "envs/model/bin/python"),
            "--assets",
            str(args.workspace / "weights/lightnav"),
            "--device",
            args.device,
        ]
    else:
        options += ["--mock-pointing", "alternate"]
    return session_runner.main([*options, "--output", str(output)], website=website)


def five_move(output, website):
    """Use the same bounded episode and independent executor with synthetic inputs."""
    from go2_perception.vln.motion_profile import PROFILE
    from go2_perception.vln.supervised_launch import main as supervised

    return supervised(
        [
            "--navigation-mode",
            "mock",
            "--five-move-episode",
            "--profile",
            PROFILE,
            "--lidar",
            "synthetic",
            "--output",
            str(output),
        ],
        website=website,
    )


def continuous(output, website):
    """Run the chair fixture in the offline proxy and show its actual control trace."""
    from go2_perception.vln.continuous_recording import TraceRecorder
    from go2_perception.vln.continuous_review import ContinuousReview
    from go2_perception.vln.continuous_simulation import simulate

    output.mkdir()
    case = {"scenario": "visible_after_search", "trace": "trace-01.jsonl"}
    review = {"schema": "go2-continuous-review-v1", "results": [case]}
    (output / "review.json").write_text(json.dumps(review))
    fixture = json.loads(
        (ROOT / "examples/supervised_navigation/synthetic_cycle.json").read_bytes()
    )
    server = website.server(ContinuousReview(output, review))
    server.start()
    print(f"Display: {server.url}\nPrivate simulation: {output}", flush=True)
    try:
        with TraceRecorder(output / case["trace"], paced=True) as record:
            result = simulate(
                fixture,
                case["scenario"],
                record,
                wall_clock=True,
                should_stop=website.stop.is_set,
            )
        summary = {
            "schema": "go2-continuous-offline-v1",
            "mode": "simulate",
            "physical_commands_sent": 0,
            "results": [
                dict(result, trace=case["trace"], recording_stats=record.summary())
            ],
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        while not website.changing.wait(0.25):
            pass  # Keep the finished trace available until the user changes mode or closes the server.
    finally:
        server.close()
    return 0


def supervised_operator(argv):
    """Enter the existing five-move operator procedure in a separate process."""
    parser = argparse.ArgumentParser(prog="app.py supervised", description=__doc__)
    flags = parser.add_mutually_exclusive_group(required=True)
    flags.add_argument(
        "--prepare-only", action="store_true", help="verify local evidence and exit"
    )
    flags.add_argument(
        "--live-exclusive",
        action="store_true",
        help="requires the existing live opt-in and a separate browser episode grant",
    )
    options = parser.parse_args(argv)
    action = "--prepare-only" if options.prepare_only else "--live-exclusive"
    launcher = ROOT / "hardware/launch_supervised.sh"
    os.execl("/bin/bash", "bash", str(launcher), "five-move", action)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["supervised"]:
        return supervised_operator(argv[1:])
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Separate physical operator procedure: app.py supervised --help",
    )
    parser.add_argument("--mode", choices=WebsiteMode.names, default="observe")
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="loopback port (0 picks an available port)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.environ.get("GO2_WORKSPACE", Path.home() / "Robotics/Go2")),
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "replay", "live"],
        default="synthetic",
        help="observation-only sensor source",
    )
    parser.add_argument(
        "--episode", type=Path, help="replay RGB model_input/episode.json"
    )
    parser.add_argument(
        "--lidar-recording",
        type=Path,
        help="optional go2-recording-v1 replay directory",
    )
    parser.add_argument(
        "--model",
        choices=["mock", "lightnav"],
        default="mock",
        help="observation-only backend",
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    args = parser.parse_args(argv)
    if args.source == "replay" and (not args.episode or not args.episode.is_file()):
        parser.error("--source replay requires an existing --episode")
    if args.source != "replay" and (args.episode or args.lidar_recording):
        parser.error("recorded inputs require --source replay")
    if args.lidar_recording and not args.lidar_recording.is_dir():
        parser.error("LiDAR recording directory does not exist")
    args.workspace = args.workspace.expanduser().resolve()
    runs = args.workspace / "runs/application"
    if not session_runner.git_ignored(runs):
        parser.error("choose a workspace with a private runs directory")
    runs.mkdir(mode=0o700, parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="site-", dir=runs))
    mode, port, sequence = args.mode, args.port, 0
    try:
        while mode:
            sequence += 1
            output = root / f"{sequence:03}-{mode}"
            website = WebsiteMode(mode, port)
            if mode == "observe":
                code = observe(args, output, website)
            elif mode == "five-move":
                code = five_move(output, website)
            else:
                code = continuous(output, website)
            if code:
                return code
            mode, port = website.next_mode, website.port
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
