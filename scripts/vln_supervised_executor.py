"""Isolated one-shot executor. Default mock; never imported by read-only launchers."""
import argparse
import json
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import subprocess
import sys
import threading
import time

from go2_perception.vln.supervised import Hold, capability_check
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
    p.add_argument("--execute-live", action="store_true")
    p.add_argument("--exclusive-command-source", action="store_true")
    a = p.parse_args(argv)
    # stdout is exclusively the protocol. SDK/provider chatter goes to stderr.
    protocol = sys.stdout
    sys.stdout = sys.stderr
    output_lock = threading.Lock()

    def emit(item):
        line = json.dumps(finite_json(item), allow_nan=False) + "\n"
        with (a.output / "executor_events.jsonl").open("a") as f:
            f.write(line)
        try:
            with output_lock:
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
    lock = None
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
            motion = HardwareMotion(safety, sdk_routes.AvoidRoute(), monitor, guards, capability,
                                    heartbeat=lambda: heartbeat[0])
        else:
            guards, motion = FakeGuards(), None
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
                    elif message.get("type") == "execute":
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
                cancelled.set()
                closing.set()
                continue
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
            result = executor.execute(message["ticket"])
            retain_lock = live and (result["observed_stationary"] is not True or result["ownership_released"] is not True or
                                   result["post_release_stationary"] is not True) and result["started_monotonic_s"] is not None
            if retain_lock:
                # Retain canonical lock after unverified cleanup; no automatic unlock/rearm.
                lock = None
            emit({"type": "result", "result": result})
            if retain_lock:
                emit({"type": "intervention_required", "reason": "Stop/release unverified; canonical lock retained."})
        return 0
    except Exception as error:
        emit({"type": "fatal", "error": f"{type(error).__name__}: {error}"})
        return 2
    finally:
        if lock is not None:
            lock.__exit__(None, None, None)


if __name__ == "__main__":
    code = main()
    sys.stderr.flush()
    os._exit(code)  # match the audited utility's DDS teardown behavior; cleanup already ran
