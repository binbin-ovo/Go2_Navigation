"""Interactive four-panel Go2 VLN review and data collection. ADVISORY ONLY — NO ROBOT COMMANDS.

Type a target in this terminal; a loopback-only browser page shows (A) the live camera,
(B) the model's native target point on the exact image it used, (C) all native waypoints
and headings in their own model-local frame, and (D) a LiDAR top-down view. Mark what
you see, end the trial, answer the review questions, and reopen saved trials later with
``--review-run``. Nothing in this program can move, stop or reposition the robot.

Sources: ``--camera go2`` (read-only RTP bridge child, hardware env), ``synthetic`` or
``episode``; ``--lidar go2`` (read-only DDS subscriber child, hardware env),
``synthetic``, ``recording`` or ``none``. The model is the unchanged LightNav worker with
the existing full-history policy (one process, loaded once, one request in flight).
"""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import subprocess
import sys
import threading
import time

import yaml

from go2_perception.vln.calibration import load_calibration
from go2_perception.vln.episodes import file_sha256
from go2_perception.vln.interactive_app import (HELP, InteractiveApp, PacedFeeder, PointingMockWorker,
                                                ResourceMonitor, TapInbox)
from go2_perception.vln.interactive_lidar import LidarRecorder, LidarSource
from go2_perception.vln.interactive_server import DisplayServer, _finite
from go2_perception.vln.interactive_trials import RunStore, utc_now
from go2_perception.vln.shadow import Limits, ShadowSession, write_json
from go2_perception.vln.shadow_live import LiveCameraSource, latency_model
from go2_perception.vln.shadow_sources import EpisodeSource, SyntheticSource
from go2_perception.vln.shadow_worker import MockWorker, ProcessWorker

ROOT = Path(__file__).resolve().parents[1]
CLOCKS = {
    "camera": "stamp_host = host wall receipt of the first RTP packet (clock host_receipt); "
              "available_s = decode completion, same clock; host monotonic via a per-frame "
              "same-host wall-minus-monotonic sample; estimated exposure = receipt - configured "
              "0.048 s (sigma 0.016 s, one earlier boot) or UNKNOWN; RGB has no robot-clock stamp",
    "lidar": "scan/odom stamps in the Go2 robot clock (resets per boot) within a robot epoch; "
             "host wall receipt + monotonic sample kept alongside; host-minus-robot is a per-epoch "
             "display estimate only, never reused across epochs and never used for fusion",
    "model": "host monotonic seconds of this computer (submit/start/end); not comparable across machines",
    "records": "wall UTC ISO strings for people; host monotonic seconds for ordering",
    "replay": "recorded/synthetic sources keep their original stamps; the receipt time is this "
              "process's own receipt; synthetic clocks are not Go2 clocks",
}


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=HELP)
    p.add_argument("--review-run", type=Path, help="Reopen a saved run for review only (no sensors/model)")
    p.add_argument("--output", type=Path, help="NEW private, Git-ignored run directory")
    p.add_argument("--camera", choices=["go2", "synthetic", "episode"], default="synthetic")
    p.add_argument("--episode", type=Path, help="model_input/episode.json for --camera episode")
    p.add_argument("--lidar", choices=["go2", "synthetic", "recording", "none"], default="none")
    p.add_argument("--lidar-recording", type=Path, help="go2-recording-v1 directory for --lidar recording")
    p.add_argument("--synthetic-lidar-option", action="append", default=[], help=argparse.SUPPRESS)
    p.add_argument("--backend", choices=["mock", "lightnav"], default="mock")
    p.add_argument("--mock-delay-s", type=float, default=1.2)
    p.add_argument("--mock-pointing", choices=["opos0", "alternate"], default="opos0",
                   help="Mock only: canned OPOS-0, or alternate a canned point and OPOS-0 (UI checks)")
    p.add_argument("--model-python", type=Path)
    p.add_argument("--assets", type=Path)
    p.add_argument("--runtime-config", type=Path, default=ROOT / "configs/pretrained_vln.yaml")
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    p.add_argument("--preparation-mode", choices=["baseline", "cached"], default="cached")
    p.add_argument("--tensor-evidence", choices=["skip", "record"], default="skip",
                   help="Opt in to archiving model-input tensors (cached mode writes them in the background)")
    p.add_argument("--hardware-python", type=Path, help="Hardware env Python for the Go2 bridges")
    p.add_argument("--camera-host-ip", default="192.168.123.222")
    p.add_argument("--iface", default="en9", help="DDS interface for --lidar go2")
    p.add_argument("--unitree-sdk", type=Path,
                   help="Unitree SDK checkout added to the LiDAR bridge's PYTHONPATH (macOS skips the "
                        "hardware venv's hidden editable .pth files; docs/MAC_CONNECTION.md)")
    p.add_argument("--calibration", type=Path, default=ROOT / "configs/go2_front_camera_calibration.yaml")
    p.add_argument("--camera-stride", type=int, default=2)
    p.add_argument("--camera-latency", choices=["configured", "unknown"], default="configured")
    p.add_argument("--camera-inbox", type=int, default=16)
    p.add_argument("--camera-silence-s", type=float, default=1.0)
    p.add_argument("--speed", type=float, default=1.0, help="Replay pacing multiplier (not live timing)")
    p.add_argument("--synthetic-seconds", type=float, default=600.0)
    p.add_argument("--history-seconds", type=float, default=20.0)
    p.add_argument("--history-frames", type=int, default=1024)
    p.add_argument("--request-interval-s", type=float, default=0.25)
    p.add_argument("--warmup-s", type=float, default=1.0)
    p.add_argument("--max-requests-per-trial", type=int, default=6)
    p.add_argument("--trial-max-s", type=float, default=45.0)
    p.add_argument("--max-trials", type=int, default=20)
    p.add_argument("--max-host-seconds", type=float, default=3600.0)
    p.add_argument("--run-quota-gb", type=float, default=6.0)
    p.add_argument("--min-free-disk-gb", type=float, default=20.0)
    p.add_argument("--min-available-memory-gb", type=float, default=2.0)
    p.add_argument("--max-app-rss-gb", type=float, default=4.0)
    p.add_argument("--record-lidar-continuously", action="store_true",
                   help="Also record LiDAR/odometry between trials (chunks tagged between_trials), e.g. to "
                        "measure a manual viewpoint change between two observations")
    p.add_argument("--port", type=int, default=8765, help="Loopback port (0 = any free port)")
    p.add_argument("--reviewer", help="Default reviewer name for review records")
    return p


def git_ignored(path):
    """True if path is outside this repository or ignored by it (private recordings only)."""
    path = path.resolve()
    if ROOT.resolve() not in path.parents:
        return True
    run = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", str(path)], check=False)
    return run.returncode == 0


def terminal_reader(commands, app, monotonic):
    for line in sys.stdin:
        item = (line.rstrip("\n"), monotonic(), app.viewer_snapshot())
        while True:
            try:
                commands.put(item, timeout=0.5)
                break
            except Full:
                continue
    commands.put(("quit", monotonic(), app.viewer_snapshot()))


def review_only(a):
    if not (a.review_run / "run.json").is_file():
        summary_path = a.review_run / "summary.json"
        if not summary_path.is_file():
            summary_path = a.review_run / "review.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_bytes())
            if summary.get("schema") in ("go2-continuous-offline-v1", "go2-continuous-review-v1"):
                from go2_perception.vln.continuous_review import review_loop
                return review_loop(a.review_run, summary, a.port)
    store = RunStore(a.review_run, create=False)
    run = json.loads((store.root / "run.json").read_text())
    app = InteractiveApp(store=store, session=None, camera=None, tap=None, mode="REVIEW",
                         config={"camera_silence_s": 1.0, "reviewer": a.reviewer, "max_trials": 0},
                         run_info={"clocks": run.get("clocks"), "backend": run.get("backend")})
    server = DisplayServer(app, a.port)
    server.start()
    store.event("review_session_opened", time.monotonic(), url=server.url)
    print(f"REVIEW ONLY — saved run {store.root}\nDisplay: {server.url}\n"
          "Commands: review [ID|close] | note TEXT | status | quit (no sensors, no model)", flush=True)
    commands = Queue(maxsize=64)
    threading.Thread(target=terminal_reader, args=(commands, app, time.monotonic), daemon=True).start()
    app.handle("review", time.monotonic(), {}, time.monotonic())
    try:
        while not app.quit:
            try:
                line, typed, viewer = commands.get(timeout=0.1)
                app.handle(line, typed, viewer, time.monotonic())
            except Empty:
                pass
            app.step(time.monotonic())
    except KeyboardInterrupt:
        pass
    finally:
        store.event("review_session_closed", time.monotonic())
        server.close()
    return 0


def main(argv=None, *, app_factory=InteractiveApp, run_fields=None, synthetic_source_factory=SyntheticSource):
    p = parser()
    a = p.parse_args(argv)
    if a.review_run:
        return review_only(a)
    if a.backend == "lightnav" and (not a.assets or not a.model_python):
        p.error("LightNav needs the existing isolated --model-python and --assets")
    if a.tensor_evidence == "skip" and a.preparation_mode != "cached":
        a.tensor_evidence = "record"  # baseline mode always archived tensors; unchanged
    if (a.camera == "go2" or a.lidar == "go2") and not a.hardware_python:
        p.error("Go2 sources need --hardware-python (hardware environment)")
    if a.camera != "go2" and a.lidar == "go2":
        p.error("Live LiDAR may only accompany the live camera (no live/replay mixing)")
    if a.camera == "go2" and a.lidar in ("synthetic", "recording"):
        p.error("Replay LiDAR may not accompany the live camera (no live/replay mixing)")
    if a.camera == "episode" and not a.episode:
        p.error("--camera episode needs --episode")
    if a.lidar == "recording" and not a.lidar_recording:
        p.error("--lidar recording needs --lidar-recording")
    if not (math.isfinite(a.max_host_seconds) and 0 < a.max_host_seconds <= 4 * 3600
            and 1 <= a.max_requests_per_trial <= 64 and 1 <= a.max_trials <= 200
            and 5 <= a.trial_max_s <= 600 and 1 <= a.camera_stride <= 8):
        p.error("Invalid bounds")
    output = a.output or ROOT / "data/outputs/vln_interactive" / time.strftime("run_%Y%m%d_%H%M%S")
    if output.exists():
        p.error("Output must be a new directory; existing evidence is never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not git_ignored(output):
        p.error(f"{output} is inside the repository but not Git-ignored; recordings must stay private")

    mode = {"go2": "LIVE", "synthetic": "SYNTHETIC", "episode": "REPLAY"}[a.camera]
    limits = Limits(a.history_frames, a.history_seconds, 3.0, 10.0, 30.0, a.request_interval_s)
    store = RunStore(output.resolve())
    started = time.monotonic()
    worker = (PointingMockWorker if a.mock_pointing == "alternate" else MockWorker)(a.mock_delay_s)
    session = ShadowSession(
        store.root / "session", worker,
        mode="live_unvalidated" if a.camera == "go2" else
        ("synthetic_paced" if a.camera == "synthetic" else "replay_paced"),
        limits=limits, allow_synthetic=a.camera == "synthetic", preparation_mode=a.preparation_mode)
    tap = TapInbox(a.camera_inbox)
    calibration = None
    if a.camera == "go2":
        intrinsics, _ = load_calibration(a.calibration)
        if intrinsics.status != "measured":
            p.error("Live Go2 input requires the measured camera calibration")
        latency = None
        if a.camera_latency == "configured":
            sensors = yaml.safe_load((ROOT / "configs/go2_vln_sensors.yaml").read_text())
            latency = latency_model(sensors["timing"]["camera_latency_s"])
        camera = LiveCameraSource(python=a.hardware_python, intrinsics=intrinsics,
                                  calibration_path=a.calibration, output=store.root / "camera",
                                  host_ip=a.camera_host_ip, stride=a.camera_stride,
                                  capacity=a.camera_inbox, latency=latency)
        camera.inbox = tap  # same bounded inbox contract + a display ring
        calibration = {"path": str(a.calibration.resolve()), "sha256": file_sha256(a.calibration),
                       "status": "measured", "T_base_camera": None, "camera_latency": latency}
    else:
        frames = int(a.synthetic_seconds * 8)
        source = (synthetic_source_factory(min(4096, frames), fps=8.0) if a.camera == "synthetic"
                  else EpisodeSource(a.episode))
        camera = PacedFeeder(iter(source), tap, speed=a.speed)
        calibration = ({"status": "synthetic", "note": "deterministic 2D fixture"} if a.camera == "synthetic"
                       else {"episode_manifest": str(a.episode), "sha256": source.episode.sha256})
    lidar = recorder = None
    if a.lidar != "none":
        if a.lidar == "go2":
            command = [a.hardware_python, "-u", ROOT / "scripts/go2_lidar_bridge.py", "--iface", a.iface]
        else:
            command = [sys.executable, "-u", ROOT / "scripts/synthetic_lidar_bridge.py",
                       *(["--recording", a.lidar_recording] if a.lidar == "recording" else []),
                       *[x for opt in a.synthetic_lidar_option for x in opt.split()]]
        sdk_path = (os.pathsep + str(a.unitree_sdk.resolve())) if a.lidar == "go2" and a.unitree_sdk else ""
        lidar = LidarSource(command, store.root / "lidar_bridge",
                            env={"PYTHONPATH": str(ROOT / "src") + sdk_path})
        recorder = LidarRecorder(store.root / "lidar", quota=lambda: True)
        lidar.recorder = recorder
        if a.record_lidar_continuously:
            recorder.start("between_trials")
    gib = 2**30
    monitor = ResourceMonitor(store.root, limits={
        "min_available_memory_bytes": a.min_available_memory_gb * gib,
        "max_parent_rss_bytes": a.max_app_rss_gb * gib,
        "min_free_disk_bytes": a.min_free_disk_gb * gib,
        "run_quota_bytes": a.run_quota_gb * gib,
    }, pids=lambda: {
        "model_worker": getattr(getattr(session.worker, "process", None), "pid", None),
        "camera_bridge": getattr(getattr(camera, "process", None), "pid", None),
        "lidar_bridge": None if lidar is None or lidar.process is None else lidar.process.pid,
    })
    if recorder is not None:
        recorder.quota = lambda: not monitor.snapshot().get("low")
    config = {"camera_silence_s": a.camera_silence_s, "warmup_s": a.warmup_s,
              "max_requests_per_trial": a.max_requests_per_trial, "trial_max_s": a.trial_max_s,
              "max_trials": a.max_trials, "reviewer": a.reviewer,
              "record_lidar_continuously": a.record_lidar_continuously}
    run_info = {
        "schema": "go2-vln-interactive-run-v1",
        "created_wall_utc": utc_now(),
        "mode": mode, "backend": a.backend,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "git_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                     text=True, check=False).stdout.strip(),
        "limits": asdict(limits), "advisory_age_s_unchanged": 3.0,
        "temporal_policy": "existing full-history reconstruction per request; bounded-context policy NOT enabled",
        "preparation_mode": a.preparation_mode, "tensor_evidence": a.tensor_evidence,
        "model_identity_file": "session/worker/model.json" if a.backend == "lightnav" else None,
        "runtime_config_sha256": file_sha256(a.runtime_config),
        "calibration": calibration, "clocks": CLOCKS,
        "lidar_source": a.lidar, "lidar_policy": "incremental cloud_base only; 0.5 s and 1.0 s odom-compensated "
                                                  "windows; cloud_deskewed never used; no RGB-LiDAR fusion",
        "physical_execution_authorized": False, "robot_commands_sent": 0,
        "control_utility_imported": False,
        "implementation_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in [
            Path(__file__).resolve(), ROOT / "scripts/go2_lidar_bridge.py", ROOT / "scripts/go2_camera_bridge.py",
            ROOT / "scripts/vln_shadow_worker.py", ROOT / "scripts/lightnav_worker.py",
            *sorted((ROOT / "src/go2_perception/vln").glob("interactive_*.py")),
            *sorted((ROOT / "src/go2_perception/vln").glob("shadow*.py"))]},
    }
    run_info.update(run_fields or {})
    write_json(store.root / "run.json", run_info)
    app = app_factory(store=store, session=session, camera=camera, tap=tap, mode=mode, lidar=lidar,
                         lidar_recorder=recorder, monitor=monitor, config=config, run_info=run_info)
    server = DisplayServer(app, a.port)
    termination, code = "host_time_budget", 0
    try:
        if a.backend == "lightnav":
            print("Loading LightNav once in its isolated process (about 45 s)…", flush=True)
            session.worker = ProcessWorker(python=a.model_python, assets=a.assets, config=a.runtime_config,
                                           device=a.device, output=session.output / "worker",
                                           preparation_mode=a.preparation_mode,
                                           tensor_evidence=a.tensor_evidence, episode_context=session.bounded_context)
        camera.start()
        if lidar is not None:
            lidar.start()
        monitor.start()
        server.start()
        store.event("run_started", time.monotonic(), url=server.url, mode=mode)
        print(f"{run_info['mode']} · {getattr(app, 'banner', 'ADVISORY ONLY — NO ROBOT COMMANDS')}\nDisplay: {server.url}\n"
              f"Run directory (private): {store.root}\n{getattr(app, 'terminal_help', HELP)}", flush=True)
        commands = Queue(maxsize=64)
        threading.Thread(target=terminal_reader, args=(commands, app, time.monotonic), daemon=True).start()
        sources_reported = set()
        while time.monotonic() - started < a.max_host_seconds:
            for _ in range(4):
                try:
                    line, typed, viewer = commands.get_nowait()
                except Empty:
                    break
                app.handle(line, typed, viewer, time.monotonic())
                if app.quit:
                    break
            if app.quit:
                termination = app.termination
                break
            if session.replay_anchor is None and getattr(camera, "anchor", None):
                session.set_replay_anchor(*camera.anchor)
            app.step(time.monotonic())
            for name, source in (("camera", camera), ("lidar", lidar)):
                if source is None or name in sources_reported:
                    continue
                failed = getattr(source, "error", None) or not source.alive
                if failed and getattr(source, "process", True) is not None:
                    sources_reported.add(name)
                    store.event(f"{name}_source_stopped", time.monotonic(), error=getattr(source, "error", None))
                    print(f"{name.upper()} SOURCE STOPPED ({getattr(source, 'error', None) or 'ended'}); "
                          "its panel shows this. Review stays available.", flush=True)
                    if name == "camera" and app.trial is not None:
                        app.end_trial(f"{name}_source_stopped", time.monotonic(), review=False)
            time.sleep(0.01)
    except KeyboardInterrupt:
        termination = "keyboard_interrupt"
    except Exception as error:
        termination, code = "application_error", 1
        store.event("application_error", time.monotonic(), error=repr(error))
        print(f"Application error: {error!r}", file=sys.stderr, flush=True)
    finally:
        now = time.monotonic()
        app.termination = app.termination or termination
        try:
            app.close(now)
        finally:
            session.close(time.monotonic())  # advisory cancel + owned worker shutdown only
            camera.close()
            if lidar is not None:
                lidar.close()
            if recorder is not None:
                recorder.close()
            monitor.close()
            server.close()
            summary = {
                "termination": app.termination, "host_duration_s": time.monotonic() - started,
                "trials_started": app.trials_started, "session_counts": session.counts,
                "history_error": session.history_error,
                "tensor_evidence_shutdown": getattr(session.worker, "tensor_evidence_summary", None),
                "camera": tap.status(time.monotonic(), a.camera_silence_s)["counts"],
                "camera_bridge": camera.status(time.monotonic()) if hasattr(camera, "status") and a.camera == "go2" else None,
                "lidar": None if lidar is None else lidar.snapshot(time.monotonic()),
                "lidar_recording": None if recorder is None else recorder.status(),
                "resources_final": monitor.snapshot(), "resource_history": list(monitor.history),
                "resource_stop": app.resource_stop,
                "robot_commands_sent": 0, "physical_execution_authorized": False,
                "ended_wall_utc": utc_now(),
            }
            if hasattr(app, "execution_summary"):
                summary.update(app.execution_summary())
            write_json(store.root / "summary.json", _finite(json.loads(json.dumps(summary, default=str))))
            message = "No robot command was sent."
            if session.bounded_context:
                message = "See episode_summary.json and executor events for episode status."
            elif hasattr(app, "navigation"):
                message = "See navigation_summary.json for execution/stop/release status."
            print(f"Closed ({app.termination}). {message}\n"
                  f"Reopen: run_vln_interactive.py --review-run {store.root}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
