"""Isolated one-shot executor. Default mock; never imported by read-only launchers."""
import argparse
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import signal
import subprocess
import sys
import threading
import time

from go2_perception.vln.supervised import Hold, capability_check, stop_and_release_verified
from go2_perception.vln.supervised_executor import (Executor, FakeGuards, HardwareMotion,
                                                   finite_json, live_preflight, load_module)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capability", type=Path, required=True)
    p.add_argument("--nonce", required=True)
    p.add_argument("--mode", choices=["mock", "live"], default="mock")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--control-root", type=Path)
    p.add_argument("--unitree-sdk", type=Path)
    p.add_argument("--iface")
    p.add_argument("--five-move-episode", action="store_true")
    p.add_argument("--execute-live", action="store_true")
    p.add_argument("--exclusive-command-source", action="store_true")
    a = p.parse_args(argv)
    # stdout is exclusively the protocol. SDK/provider chatter goes to stderr.
    protocol = sys.stdout
    sys.stdout = sys.stderr
    output_lock = threading.Lock()

    def emit(item):
        line = json.dumps(finite_json(item), allow_nan=False) + "\n"
        try:
            with output_lock:
                with (a.output / "executor_events.jsonl").open("a") as f:
                    f.write(line)
                protocol.write(line)
                protocol.flush()
        except (BrokenPipeError, OSError):
            cancelled.set()
            closing.set()

    a.output.mkdir(parents=True, exist_ok=False)
    capability = json.loads(a.capability.read_text())
    cancelled, closing = threading.Event(), threading.Event()
    def shutdown(signum, frame):
        cancelled.set()
        closing.set()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    queue, heartbeat = Queue(maxsize=8), [time.monotonic()]
    lock = executor = None
    try:
        live = a.mode == "live"
        capability_check(capability, live=live, wall_s=time.time())
        if live:
            if not a.control_root or not a.unitree_sdk or not a.iface:
                raise Hold("live_needs_canonical_control_sdk_and_interface")
            root, provider_path = live_preflight(capability, execute_live=a.execute_live,
                                                 exclusive_source=a.exclusive_command_source,
                                                 control_root=a.control_root)
            sdk_revision = subprocess.run(["git", "-C", str(a.unitree_sdk), "rev-parse", "HEAD"],
                                          capture_output=True, text=True, check=True).stdout.strip()
            if sdk_revision != "814556d15970dd2ecf1c9984e845ca02ab07e206":
                raise Hold("SDK_revision_differs_from_audited_route")
            if subprocess.run(["git", "-C", str(a.unitree_sdk), "diff", "--quiet", "HEAD", "--", "unitree_sdk2py"],
                              check=False).returncode != 0:
                raise Hold("audited_SDK_source_modified")
            sys.path.insert(0, str(root / "src"))
            utility = load_module(root / "scripts/go2_hl_control.py", "audited_go2_utility")
            candidate_lock = utility.CommandLock()
            candidate_lock.__enter__()  # never unlink somebody else's lock on failed acquisition
            lock = candidate_lock  # held for the WHOLE session
            sys.path.insert(0, str(a.unitree_sdk))
            # Sole SDK import/initialization path, after every static live gate and lock.
            from go2_control import safety, sdk_routes
            provider = load_module(provider_path, "verified_supervised_guards")
            sdk_routes.init_dds(a.iface)
            monitor = sdk_routes.StateMonitor().start()
            guards = provider.create_guards(monitor=monitor, capability=capability, monotonic=time.monotonic,
                                            observation_root=a.output.parent)
            # Telemetry must never block resends, zero, or cleanup on a UI pipe.
            feedback_queue = Queue(maxsize=4)

            def offer_feedback(value):
                try:
                    feedback_queue.put_nowait(value)
                except Full:
                    pass

            def deliver_feedback():
                while not closing.is_set():
                    try:
                        item = feedback_queue.get(timeout=.1)
                    except Empty:
                        continue
                    emit({"type": "feedback", "nonce": a.nonce, **item})

            threading.Thread(target=deliver_feedback, daemon=True, name="navigation-feedback").start()
            motion = HardwareMotion(safety, sdk_routes.AvoidRoute(), monitor, guards, capability,
                                    heartbeat=lambda: heartbeat[0], feedback=offer_feedback)
        else:
            guards, motion = FakeGuards(), None
        if a.five_move_episode:
            from go2_perception.vln.five_move import EpisodeExecutor, EpisodeFakeGuards
            if not live:
                guards = EpisodeFakeGuards()
            executor = EpisodeExecutor(capability, a.nonce, guards=guards, live=live, motion=motion,
                                       cancelled=cancelled, emit=emit)
        else:
            executor = Executor(capability, a.nonce, guards=guards, live=live, motion=motion, cancelled=cancelled)

        def read():
            for line in sys.stdin:
                try:
                    message = json.loads(line)
                    if message.get("type") == "heartbeat":
                        heartbeat[0] = time.monotonic()
                    elif message.get("type") in ("cancel", "close"):
                        cancelled.set()
                        if message["type"] == "close":
                            closing.set()
                    elif a.five_move_episode and message.get("type") == "episode_sensor_failure":
                        if message.get("episode_id") == a.nonce:
                            executor.interrupt_sensor(message["reason"])
                    elif (a.five_move_episode and message.get("type") in (
                            "episode_arm", "episode_observe", "episode_identified", "episode_goal")
                          or not a.five_move_episode and message.get("type") == "execute"):
                        queue.put_nowait(message)
                except Exception:
                    cancelled.set()  # malformed IPC cannot result in motion
            cancelled.set()
            closing.set()

        threading.Thread(target=read, daemon=True, name="executor-cancel-reader").start()
        emit({"type": "ready", "synthetic": not live, "nonce": a.nonce, "physical_commands_sent": 0})
        last = -1.
        while not closing.is_set():
            if time.monotonic()-heartbeat[0] > .5:
                if a.five_move_episode:
                    executor.interrupt_sensor("parent_heartbeat_lost")
                cancelled.set()
                closing.set()
                continue
            if a.five_move_episode:
                executor.tick()
            if time.monotonic()-last > .05:
                try:
                    emit({"type": "guards", "snapshot": guards.snapshot(), "nonce": a.nonce})
                except Exception as error:
                    emit({"type": "guard_error", "error": str(error)})
                if motion is not None:
                    motion.sample_idle()
                last = time.monotonic()
            try:
                message = queue.get(timeout=.01)
            except Empty:
                continue
            if a.five_move_episode:
                kind = message["type"]
                try:
                    if kind == "episode_arm":
                        executor.arm(message["grant"])
                    elif kind == "episode_observe":
                        executor.observe(message)
                    else:
                        executor.review(message)
                except (Hold, ValueError, KeyError, TypeError, RuntimeError) as error:
                    emit({"type":"episode_refusal", "nonce":a.nonce, "reason":str(error)})
                    if kind in ("episode_arm", "episode_observe"):
                        executor.finish("SENSOR_OR_CONTROL_FAILURE", str(error))
                continue
            result = executor.execute(message["ticket"])
            retain_lock = (live and not stop_and_release_verified(result)
                           and result["started_monotonic_s"] is not None)
            if retain_lock:
                # Retain canonical lock after unverified cleanup; no automatic unlock/rearm.
                lock = None
            emit({"type": "result", "result": result})
            if retain_lock:
                emit({"type": "intervention_required", "reason": "Stop/release unverified; canonical lock retained."})
        if a.five_move_episode:
            executor.tick()
        return 0
    except Exception as error:
        emit({"type": "fatal", "error": f"{type(error).__name__}: {error}"})
        return 2
    finally:
        if lock is not None and not getattr(executor, "retain_lock", False):
            lock.__exit__(None, None, None)


if __name__ == "__main__":
    code = main()
    sys.stderr.flush()
    os._exit(code)  # match the audited utility's DDS teardown behavior; cleanup already ran
