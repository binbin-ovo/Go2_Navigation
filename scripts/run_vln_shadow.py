"""Terminal instruction -> local observations -> read-only LightNav advisory.

Sources: an existing episode, synthetic frames, or (``--go2-camera``) the read-only
Go2 front-camera bridge child. No motion-enabling option or robot client exists;
nothing is ever sent to the robot. Type `instruction TEXT`, `cancel` (advisory
only, never a robot stop), `status`, or `quit`. index.html refreshes without a server.
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

import yaml

from go2_perception.vln.calibration import load_calibration
from go2_perception.vln.shadow import Limits, ShadowSession, write_json
from go2_perception.vln.episodes import file_sha256
from go2_perception.vln.shadow_display import render
from go2_perception.vln.shadow_sources import EpisodeSource, SyntheticSource
from go2_perception.vln.shadow_live import LiveCameraSource, latency_model
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
    source.add_argument(
        "--go2-camera",
        action="store_true",
        help="Read-only front-camera bridge (multicast read socket in the hardware env)",
    )
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
    p.add_argument(
        "--hardware-python", type=Path, help="Hardware env Python for the camera bridge"
    )
    p.add_argument("--camera-host-ip", default="192.168.123.222")
    p.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "configs/go2_front_camera_calibration.yaml",
    )
    p.add_argument(
        "--camera-stride",
        type=int,
        default=2,
        help="Bridge forwards every Nth decoded frame (declared, counted)",
    )
    p.add_argument(
        "--camera-latency",
        choices=["configured", "unknown"],
        default="configured",
        help="Estimated exposure from configs/go2_vln_sensors.yaml, or leave unknown",
    )
    p.add_argument(
        "--camera-silence-s",
        type=float,
        default=1.0,
        help="No new frame for this long = interruption; inference pauses",
    )
    p.add_argument(
        "--camera-inbox",
        type=int,
        default=16,
        help="Bounded live inbox; absorbs main-loop input preparation without loss",
    )
    p.add_argument(
        "--paused-exit-s",
        type=float,
        default=15.0,
        help="Live: end the run if history stays paused this long without a new instruction",
    )
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
                "banner": session.banner,
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


def write_latency(session, displayed_s):
    """One row per answered request: where the time went, without cross-clock math."""
    request, record = session.displayed
    source = request.context["current_source"]
    prov = source["provenance"]
    response = record["response"]
    result = response.get("result", {})
    native = result.get("timing") if isinstance(result.get("timing"), dict) else {}
    submit = request.context["host_submit_monotonic_s"]
    start, end = response.get("host_inference_start_s"), response.get("host_inference_end_s")
    receipt = prov.get("sensor_receipt_monotonic_s")

    def span(a, b):
        return None if a is None or b is None else b - a

    row = {
        "sequence": request.context["identity"]["sequence"],
        "instruction_revision": request.context["identity"]["instruction_revision"],
        "frame_id": source["frame_id"],
        "rejection_reason": record["rejection_reason"],
        "label": "STALE"
        if (record["rejection_reason"] or "").startswith("stale")
        else "REJECTED"
        if record["rejection_reason"]
        else "CURRENT",
        "estimated_exposure_to_sensor_receipt_s": span(
            prov.get("estimated_exposure_monotonic_s"), receipt
        ),
        "sensor_receipt_to_decoded_s": span(receipt, prov.get("decoded_monotonic_s")),
        "decoded_to_parent_read_s": span(
            prov.get("decoded_monotonic_s"), prov.get("parent_read_monotonic_s")
        ),
        "parent_read_to_session_receive_s": span(
            prov.get("parent_read_monotonic_s"), prov.get("session_receive_monotonic_s")
        ),
        "session_receive_to_submit_s": span(prov.get("session_receive_monotonic_s"), submit),
        "input_preparation_and_dispatch_s": span(submit, start),
        "child_inference_s": span(start, end),
        "native_generate_s": native.get("native_generate_s"),
        "result_receipt_after_inference_s": span(end, record.get("receipt_monotonic_s")),
        "display_after_receipt_s": span(record.get("receipt_monotonic_s"), displayed_s),
        "sensor_receipt_to_display_s": span(receipt, displayed_s),
        "estimated_exposure_to_display_s": span(
            prov.get("estimated_exposure_monotonic_s"), displayed_s
        ),
        "display_timing_at_receipt": record["display_timing"],
        "source_age_at_cutoff_s": request.context["source_age_at_cutoff_s"],
        "history_frames_in_job": len(request.job["frames"]),
        "conversion_uncertainty_s": prov.get("conversion_uncertainty_s"),
        "camera_latency_model": session.source_status.get("camera_latency_model")
        if session.source_status
        else None,
    }
    with (session.output / "latency.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


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
    if a.go2_camera and (
        not a.hardware_python or not 1 <= a.camera_stride <= 8 or not 0.2 <= a.camera_silence_s <= 10
    ):
        p.error("--go2-camera needs --hardware-python, stride 1..8 and silence 0.2..10 s")
    if a.output.exists():
        p.error(
            "Output must be a new directory; existing evidence is never overwritten"
        )
    live = a.go2_camera
    if live:
        intrinsics, _ = load_calibration(a.calibration)
        if intrinsics.status != "measured":
            p.error("Live Go2 input requires the measured camera calibration")
        latency = None
        if a.camera_latency == "configured":
            sensors = yaml.safe_load((ROOT / "configs/go2_vln_sensors.yaml").read_text())
            latency = latency_model(sensors["timing"]["camera_latency_s"])
        source = None
    else:
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
        mode="live_unvalidated"
        if live
        else ("synthetic_" if a.synthetic else "replay_") + a.mode,
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
            "source_manifest_sha256": None
            if a.synthetic or live
            else source.episode.sha256,
            "robot_commands_sent": 0,
            "control_utility_imported": False,
            "implementation_sha256": {
                str(path.relative_to(ROOT)): file_sha256(path)
                for path in [
                    Path(__file__).resolve(),
                    ROOT / "scripts/vln_shadow_worker.py",
                    ROOT / "scripts/lightnav_worker.py",
                    ROOT / "scripts/go2_camera_bridge.py",
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
    camera, camera_down, latency_rows, paused_since = None, False, 0, None
    if live:
        camera = LiveCameraSource(
            python=a.hardware_python,
            intrinsics=intrinsics,
            calibration_path=a.calibration,
            output=session.output / "camera",
            host_ip=a.camera_host_ip,
            stride=a.camera_stride,
            capacity=a.camera_inbox,
            latency=latency,
        )
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
            f"{session.banner}\nDisplay: {session.output / 'index.html'}\n"
            "Commands: instruction TEXT | cancel | status | quit "
            "(cancel = advisory only; it never stops the robot)",
            flush=True,
        )
        iterator, next_frame = (None if live else iter(source)), None
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
                    if live:
                        # History starts from frames that arrive after model readiness.
                        camera.start()
                        session.event("camera_bridge_started", now, command=camera.command)
                if live:
                    for _ in range(8):
                        try:
                            delivery = camera.get_nowait()
                        except Empty:
                            break
                        received = time.monotonic()
                        delivery.provenance["session_receive_monotonic_s"] = received
                        session.receive(delivery, received)
                        if first_source_s is None:
                            first_source_s = delivery.available_s
                        if camera_down:
                            camera_down = False
                            session.event("camera_resumed", received)
                        if session.observation_count >= a.max_observations:
                            break
                    now = time.monotonic()
                    for item in camera.drain_events():
                        session.event(item.pop("event"), now, **item)
                    status = camera.status(now)
                    silent = status["host_since_last_sensor_receipt_s"]
                    if silent is not None and silent > a.camera_silence_s and not camera_down:
                        camera_down = True
                        session.event("camera_interrupted", now, silence_s=silent)
                        # Report; never re-feed or restamp the last frame.
                        session.invalidate_history("camera_interrupted", now)
                    status["camera_state"] = (
                        "interrupted"
                        if camera_down
                        else "waiting for first frame"
                        if silent is None
                        else "receiving"
                    )
                    session.source_status = status
                    if camera.error or not camera.alive:
                        termination, exit_code = "camera_bridge_exited", 1
                        break
                    if session.observation_count >= a.max_observations:
                        termination = "observation_budget"
                        break
                    # A paused live history keeps saving frames; end instead of filling disk.
                    if session.history_error and session.inflight is None:
                        paused_since = paused_since if paused_since is not None else now
                        if now - paused_since > a.paused_exit_s and not a.stay_open:
                            termination = "history_paused"
                            break
                    else:
                        paused_since = None
                # Bounded work per tick. Replay timestamps keep advancing during
                # inference; delayed input decoding catches up without stopping time.
                for _ in range(0 if live else 8):
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
                if live and session.counts["submitted"] >= a.max_requests:
                    # The live source never ends; the request budget does.
                    source_done, termination = True, "request_budget"
                if session.counts["submitted"] >= a.max_requests:
                    # Do not erase observations: suppress only the scheduling flag.
                    session.pending = False
                if live:
                    # Show a returned proposal before preparing the next request, so
                    # display latency is not inflated by the next job's input preparation.
                    answered = session.counts["accepted"] + session.counts["rejected"]
                    session.tick(now, allow_submit=False)
                    if session.counts["accepted"] + session.counts["rejected"] > answered:
                        render(session, time.monotonic())
                        last_render = time.monotonic()
                        latency_rows = answered + 1
                        write_latency(session, last_render)
                    now = time.monotonic()
                session.tick(now, allow_submit=warm)
                if (
                    source_done
                    and session.inflight is None
                    and not session.pending
                    and not a.stay_open
                ):
                    break
            now = time.monotonic()
            answered = session.counts["accepted"] + session.counts["rejected"]
            if now - last_render >= 0.5 or answered > latency_rows:
                render(session, now)
                last_render = now
            if live and answered > latency_rows and session.displayed:
                latency_rows = answered
                write_latency(session, time.monotonic())
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
        if camera is not None:
            camera.close()
            session.source_status = {
                **camera.status(time.monotonic()),
                "camera_state": "closed",
            }
            session.event("camera_bridge_closed", time.monotonic(), counts=camera.counts)
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
                "robot_commands_sent": 0,
                "camera": None if camera is None else session.source_status,
                "completion_verified": False,
                "history_error": session.history_error,
            },
        )
        print_status(session, time.monotonic())
        print(f"{session.banner}\nDisplay: {session.output / 'index.html'}", flush=True)
    return exit_code or int(session.history_error is not None)


if __name__ == "__main__":
    raise SystemExit(main())
