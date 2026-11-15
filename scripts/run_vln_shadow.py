"""Terminal instruction -> local observations -> read-only LightNav advisory.

No live connection or motion-enabling option exists. Type `instruction TEXT`,
`cancel`, `status`, or `quit`. The local index.html refreshes without a server.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from queue import Empty, Queue
import sys
import threading
import time

from go2_perception.vln.shadow import BANNER, Limits, ShadowSession, write_json
from go2_perception.vln.episodes import file_sha256
from go2_perception.vln.shadow_display import render
from go2_perception.vln.shadow_sources import EpisodeSource, SyntheticSource
from go2_perception.vln.shadow_worker import MockWorker, ProcessWorker

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instruction", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--episode",
        type=Path,
        help="Existing model_input/episode.json; never annotations",
    )
    source.add_argument("--synthetic", action="store_true")
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New private directory (recommended data/outputs/)",
    )
    p.add_argument("--backend", choices=["mock", "lightnav"], default="mock")
    p.add_argument("--mode", choices=["paced", "fast"], default="paced")
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Paced replay clock multiplier, not live timing",
    )
    p.add_argument("--synthetic-frames", type=int, default=96)
    p.add_argument("--mock-delay-s", type=float, default=1.0)
    p.add_argument("--allow-synthetic-calibration", action="store_true")
    p.add_argument("--model-python", type=Path)
    p.add_argument("--assets", type=Path)
    p.add_argument(
        "--runtime-config", type=Path, default=ROOT / "configs/pretrained_vln.yaml"
    )
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    p.add_argument("--advisory-age-s", type=float, default=3.0)
    p.add_argument("--response-deadline-s", type=float, default=10.0)
    p.add_argument("--worker-timeout-s", type=float, default=30.0)
    p.add_argument("--history-frames", type=int, default=1024)
    p.add_argument("--history-seconds", type=float, default=60.0)
    p.add_argument("--request-interval-s", type=float, default=0.25)
    p.add_argument(
        "--warmup-seconds",
        type=float,
        default=1.0,
        help="Collect causal input before first request",
    )
    p.add_argument("--max-observations", type=int, default=4096)
    p.add_argument("--max-requests", type=int, default=64)
    p.add_argument(
        "--max-host-seconds",
        type=float,
        default=180.0,
        help="Whole application including model load",
    )
    p.add_argument(
        "--stay-open",
        action="store_true",
        help="Keep display after source EOF, within host limit",
    )
    return p


def handle_command(session, command, now):
    command = command.strip()
    if command.startswith("instruction "):
        session.instruction_changed(command[len("instruction ") :], now)
    elif command == "cancel":
        session.cancel(now)
        print("Advisory cancelled. No robot stop command was sent.", flush=True)
    elif command == "status":
        print_status(session, now)
    elif command in ("quit", "exit"):
        return False
    elif command:
        print("Commands: instruction TEXT | cancel | status | quit", flush=True)
    return True


def print_status(session, now):
    s = session.status(now)
    displayed = s["displayed"]
    print(
        json.dumps(
            {
                "banner": BANNER,
                "mode": s["source_mode"],
                "revision": s["instruction_revision"],
                "instruction": s["instruction"],
                "latest_observation_s": None
                if not s["latest"]
                else s["latest"]["observation_s"],
                "in_flight": s["inference_in_flight"],
                "pending": s["pending_requests"],
                "history_error": s["history_error"],
                "counts": s["counts"],
                "proposal_status": None if not displayed else displayed["label"],
                "native": None
                if not displayed
                else displayed["response"].get("result", {}).get("native_output"),
                "timing": None if not displayed else displayed["display_timing"],
            }
        ),
        flush=True,
    )


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if not (
        math.isfinite(a.speed)
        and 0 < a.speed <= 20
        and math.isfinite(a.max_host_seconds)
        and 0 < a.max_host_seconds <= 3600
        and math.isfinite(a.warmup_seconds)
        and 0 <= a.warmup_seconds <= 60
        and 1 <= a.max_observations <= 4096
        and 1 <= a.max_requests <= 256
    ):
        p.error("Invalid finite replay/run bounds")
    if a.backend == "lightnav" and (not a.assets or not a.model_python):
        p.error("LightNav needs the existing isolated --model-python and --assets")
    if a.output.exists():
        p.error(
            "Output must be a new directory; existing evidence is never overwritten"
        )
    source = (
        SyntheticSource(a.synthetic_frames) if a.synthetic else EpisodeSource(a.episode)
    )
    limits = Limits(
        a.history_frames,
        a.history_seconds,
        a.advisory_age_s,
        a.response_deadline_s,
        a.worker_timeout_s,
        a.request_interval_s,
    )
    # Worker directory is inside the new private run; Session creates the root first.
    worker = MockWorker(a.mock_delay_s)
    session = ShadowSession(
        a.output.resolve(),
        worker,
        mode=("synthetic_" if a.synthetic else "replay_") + a.mode,
        limits=limits,
        allow_synthetic=a.synthetic or a.allow_synthetic_calibration,
    )
    started = time.monotonic()
    write_json(
        session.output / "run.json",
        {
            "arguments": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
            },
            "limits": asdict(limits),
            "backend": a.backend,
            "session_id": session.session_id,
            "source_mode": session.mode,
            "physical_execution_authorized": False,
            "real_model_verification": "pending"
            if a.backend == "mock"
            else "attempted",
            "source_manifest_sha256": None if a.synthetic else source.episode.sha256,
            "implementation_sha256": {
                str(path.relative_to(ROOT)): file_sha256(path)
                for path in [
                    Path(__file__).resolve(),
                    ROOT / "scripts/vln_shadow_worker.py",
                    ROOT / "scripts/lightnav_worker.py",
                    *sorted((ROOT / "src/go2_perception/vln").glob("shadow*.py")),
                    ROOT / "src/go2_perception/vln/pretrained.py",
                    ROOT / "src/go2_perception/vln/model_preprocessing.py",
                ]
            },
            "runtime_config_sha256": file_sha256(a.runtime_config),
        },
    )
    commands = Queue(maxsize=16)

    def terminal():
        for line in sys.stdin:
            commands.put(line.rstrip("\n"))

    threading.Thread(target=terminal, daemon=True).start()
    termination, exit_code = "source_exhausted", 0
    try:
        if a.backend == "lightnav":
            print(
                "Loading fixed LightNav in its isolated process. Replay starts after readiness.",
                flush=True,
            )
            session.worker = ProcessWorker(
                python=a.model_python,
                assets=a.assets,
                config=a.runtime_config,
                device=a.device,
                output=session.output / "worker",
            )
        session.instruction_changed(a.instruction, time.monotonic())
        print(
            f"{BANNER}\nDisplay: {session.output / 'index.html'}\n"
            "Commands: instruction TEXT | cancel | status | quit",
            flush=True,
        )
        iterator, next_frame = iter(source), None
        source_done, playback_start, first_source_s = False, None, None
        last_render, last_counts = -float("inf"), None
        while time.monotonic() - started < a.max_host_seconds:
            now = time.monotonic()
            try:
                command = commands.get_nowait()
            except Empty:
                command = None
            if command is not None:
                try:
                    if not handle_command(session, command, now):
                        termination = "operator_quit"
                        break
                except ValueError as error:
                    print(f"Instruction rejected: {error}", flush=True)
            if not session.worker.ready:
                session.tick(now)
            else:
                if playback_start is None:
                    playback_start = now
                # Bounded work per tick. Replay timestamps keep advancing during
                # inference; delayed input decoding catches up without stopping time.
                for _ in range(8):
                    if source_done:
                        break
                    if next_frame is None:
                        next_frame = next(iterator, None)
                        if next_frame is None:
                            source_done = True
                            break
                    if first_source_s is None:
                        first_source_s = next_frame.available_s
                        session.set_replay_anchor(
                            first_source_s, playback_start, a.speed
                        )
                    due = (
                        playback_start
                        + (next_frame.available_s - first_source_s) / a.speed
                    )
                    if a.mode == "paced" and now < due:
                        break
                    session.receive(next_frame, time.monotonic())
                    next_frame = None
                    if session.observation_count >= a.max_observations:
                        source_done, termination = True, "observation_budget"
                        break
                now = time.monotonic()
                warm = session.latest is not None and (
                    source_done
                    or session.latest["available_s"] - first_source_s
                    >= a.warmup_seconds
                )
                if session.counts["submitted"] >= a.max_requests:
                    # Do not erase observations: suppress only the scheduling flag.
                    session.pending = False
                session.tick(now, allow_submit=warm)
                if (
                    source_done
                    and session.inflight is None
                    and not session.pending
                    and not a.stay_open
                ):
                    break
            now = time.monotonic()
            if now - last_render >= 0.5:
                render(session, now)
                last_render = now
            counts = (
                session.revision,
                session.counts["submitted"],
                session.counts["accepted"],
                session.counts["rejected"],
                session.history_error,
            )
            if counts != last_counts and (
                session.counts["accepted"] + session.counts["rejected"]
                or session.history_error
            ):
                print_status(session, now)
                last_counts = counts
            time.sleep(0.01)
        else:
            termination = "host_time_budget"
    except KeyboardInterrupt:
        termination = "keyboard_interrupt"
    except Exception as error:
        termination, exit_code = "application_error", 1
        session.event("application_error", time.monotonic(), error=repr(error))
        print(f"Advisory error: {error}", file=sys.stderr)
    finally:
        now = time.monotonic()
        # Preserve the last running state separately; final state marks the session
        # closed so a static page can never imply a fresh still-active proposal.
        write_json(session.output / "last_active_status.json", session.status(now))
        session.close(now)
        render(session, time.monotonic(), snapshot=True)
        write_json(
            session.output / "summary.json",
            {
                "termination": termination,
                "host_duration_s": time.monotonic() - started,
                "counts": session.counts,
                "backend": a.backend,
                "physical_execution_authorized": False,
                "live_sensor_validation": "pending",
                "completion_verified": False,
                "history_error": session.history_error,
            },
        )
        print_status(session, time.monotonic())
        print(f"{BANNER}\nDisplay: {session.output / 'index.html'}", flush=True)
    return exit_code or int(session.history_error is not None)


if __name__ == "__main__":
    raise SystemExit(main())
