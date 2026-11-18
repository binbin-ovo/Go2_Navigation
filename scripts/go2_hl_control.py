#!/usr/bin/env python
"""Guarded Go2 high-level command utility (obstacles_avoid route only).

Modes
  status    read-only: ping, sport state + remote input, read-only RPC queries. Default mode.
  watch     read-only: stream sport state + remote input at 2 Hz for --seconds. Subscribers only.
  range     read-only: median LiDAR range to the first surface ahead (cloud_base), a displacement
            reference that does not come from the robot's own odometry.
  preview   dry run: validate a request, print the exact confirmation phrase. No DDS, no RPC.
  ownership --execute: UseRemoteCommandFromApi(True), hold N s with no Move, then (False).
  move      --execute --confirm "<phrase>" --stop-verified: one bounded move, then stop + release.
  stop      --execute: zero velocity on the avoidance route (x3), then release API control.

No mode can switch avoidance off, change gait/mode, stand, sit, damp or call SportClient.Move.
Every session writes JSONL under data/outputs/go2_control/ (git-ignored).

  export PYTHONPATH=src:<main checkout>/third_party/unitree_sdk2_python
  .venv-go2/bin/python scripts/go2_hl_control.py status --iface en9
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from go2_control.safety import MoveRequest, Refused, confirmation_phrase, execute, validate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/outputs/go2_control"
ROBOT_IP = "192.168.123.161"


class Session:
    def __init__(self, mode):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = OUT / f"{stamp}_{mode}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._f = open(self.dir / "events.jsonl", "a")
        self._lock = threading.Lock()

    def log(self, event):
        event = {"wall": time.time(), "mono": time.monotonic(), **event}
        with self._lock:
            self._f.write(json.dumps(event, default=str) + "\n")
            self._f.flush()

    def save(self, name, obj):
        (self.dir / name).write_text(json.dumps(obj, indent=2, default=str))


class CommandLock:
    """One software command source among our tools; O_EXCL lock file."""

    path = OUT / ".command.lock"

    def __enter__(self):
        OUT.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise Refused(f"another command source holds {self.path}; remove it only if no tool is running")
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def ping():
    r = subprocess.run(["ping", "-c", "3", "-t", "5", ROBOT_IP], capture_output=True, text=True)
    return {"returncode": r.returncode, "summary": r.stdout.strip().splitlines()[-2:]}


def sample_state(session, monitor, stop_evt, hz=50):
    while not stop_evt.is_set():
        session.log({"event": "state", **monitor.snapshot()})
        time.sleep(1 / hz)


def start_monitor(iface, wait=3.0):
    from go2_control.sdk_routes import StateMonitor, init_dds

    init_dds(iface)
    mon = StateMonitor().start()
    end = time.monotonic() + wait
    while time.monotonic() < end and (mon.state_age() is None or mon.remote is None):
        time.sleep(0.05)
    return mon


def cmd_status(a):
    from go2_control.sdk_routes import read_only_queries

    s = Session("status")
    report = {"mode": "status", "iface": a.iface, "ping": ping()}
    mon = start_monitor(a.iface, a.seconds)
    report["state"] = mon.snapshot()
    report["operator_input"] = mon.operator_input()
    report["rpc"] = read_only_queries(timeout=a.timeout, attempts=a.attempts)
    s.save("status.json", report)
    print(json.dumps(report, indent=2, default=str))
    print(f"saved {s.dir}")


def cmd_watch(a):
    s = Session("watch")
    mon = start_monitor(a.iface)
    stop_evt = threading.Event()
    threading.Thread(target=sample_state, args=(s, mon, stop_evt), daemon=True).start()
    end = time.monotonic() + a.seconds
    while time.monotonic() < end:
        snap = mon.snapshot()
        r = snap["remote"] or {}
        print(f"err={snap.get('error_code')} h={snap.get('body_height', 0):.3f} "
              f"v=({snap.get('velocity', [0, 0])[0]:+.3f},{snap.get('velocity', [0, 0])[1]:+.3f}) "
              f"wz={snap.get('yaw_speed', 0):+.3f} yaw={snap.get('rpy', [0, 0, 0])[2]:+.3f} "
              f"sticks=({r.get('lx', 0):+.2f},{r.get('ly', 0):+.2f},{r.get('rx', 0):+.2f},{r.get('ry', 0):+.2f}) "
              f"buttons=0x{r.get('buttons', 0):04x}", flush=True)
        time.sleep(0.5)
    stop_evt.set()
    print(f"saved {s.dir}")


def cmd_range(a):
    import numpy as np
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

    from go2_control.sdk_routes import front_surface_x, init_dds
    from go2_perception.adapters.go2.pointcloud import frame_from_msg

    init_dds(a.iface)
    frames = []
    sub = ChannelSubscriber("rt/utlidar/cloud_base", PointCloud2_)
    sub.Init(lambda m: frames.append(frame_from_msg(m, "rt/utlidar/cloud_base", time.time()).xyz), 10)
    time.sleep(a.seconds)
    xyz = np.concatenate(frames)
    hist, edges = np.histogram(xyz[:, 2], bins=np.arange(-0.6, 0.8, 0.05))
    ground = edges[np.argmax(hist)] + 0.025
    result = {"frames": len(frames), "ground_z": float(ground),
              "front_surface_x_median": front_surface_x(xyz, ground)}
    Session("range").save("range.json", result)
    print(json.dumps(result))


def request_from(a):
    target = a.target or (abs(a.turn_deg) * 3.141592653589793 / 180 if a.turn_deg else 0.0)
    return validate(MoveRequest(a.vx, a.vyaw, a.duration, a.resend_period, round(target, 4)))


def cmd_preview(a):
    req = request_from(a)
    print("DRY RUN - nothing sent, no DDS opened")
    print(f"route: obstacles_avoid Move(mode=0), {req.describe()}")
    end = "target or deadline" if req.target else "deadline"
    print("sequence: SwitchGet must be true -> UseRemoteCommandFromApi(True) -> Move -> "
          f"{end} -> zero Move x3 -> observe still -> UseRemoteCommandFromApi(False)")
    if req.target and req.vyaw:
        print(f"progress = IMU heading change; stop at {req.target:.3f} rad = {req.target * 57.2958:.1f} deg")
    elif req.target:
        print("progress = LiDAR front-range change while it agrees with odometry (0.15 m + 30%), "
              "else odometry; aborts if both are missing; stall and deadline still apply")
    if req.target:
        print(f"stall check starts after {a.stall_grace:.1f}s (needs 0.06 m, or 0.06 rad for a turn, per 1.5 s)")
    print(f'confirm phrase: "{confirmation_phrase(req)}"')


def cmd_ownership(a):
    if not a.execute:
        raise Refused("dry run: ownership needs --execute")
    from go2_control.sdk_routes import AvoidRoute

    s = Session("ownership")
    with CommandLock():
        mon = start_monitor(a.iface)
        route = AvoidRoute(a.timeout)
        enabled = route.avoidance_enabled()
        s.log({"event": "switch_get", "enabled": enabled, **mon.snapshot()})
        if enabled is not True:
            raise Refused(f"avoidance state is {enabled}; not claiming control")
        stop_evt = threading.Event()
        threading.Thread(target=sample_state, args=(s, mon, stop_evt), daemon=True).start()
        try:
            code = route.claim_api_control()
            s.log({"event": "claim_api_control", "code": code})
            print(f"UseRemoteCommandFromApi(True) -> {code}; holding {a.seconds:.0f}s, no Move sent")
            end = time.monotonic() + a.seconds
            while time.monotonic() < end:
                time.sleep(0.5)
                snap = mon.snapshot()
                print(f"  remote={snap['remote']} v={snap.get('velocity')} wz={snap.get('yaw_speed')}")
        finally:
            code = route.release_api_control()
            s.log({"event": "release_api_control", "code": code})
            print(f"UseRemoteCommandFromApi(False) -> {code}")
            stop_evt.set()
            s.log({"event": "after", "enabled": route.avoidance_enabled(), **mon.snapshot()})
    print(f"saved {s.dir}")


def cmd_move(a):
    req = request_from(a)
    if not a.execute:
        cmd_preview(a)
        return
    from go2_control.sdk_routes import AvoidRoute, FrontRange, fuse_progress, odom_progress, yaw_progress

    s = Session("move")
    s.log({"event": "request", "request": req.__dict__, "confirm": a.confirm})
    with CommandLock():
        mon = start_monitor(a.iface)
        route = AvoidRoute(a.timeout)
        stop_evt = threading.Event()
        threading.Thread(target=sample_state, args=(s, mon, stop_evt), daemon=True).start()
        time.sleep(0.5)  # pre-move state samples
        progress = None
        if req.target and req.vyaw:
            turn = yaw_progress(mon, 1 if req.vyaw > 0 else -1)
            s.log({"event": "progress_start", "turn": True})

            def progress():
                done = turn() if turn else None
                s.log({"event": "progress", "yaw": done})
                return done

        elif req.target:
            direction = 1 if req.vx > 0 else -1
            front = FrontRange(direction=direction).start_sub()
            end = time.monotonic() + 2.0
            while front.current() is None and time.monotonic() < end:
                time.sleep(0.05)
            start_range = front.mark_start()
            odom = odom_progress(mon, direction)
            s.log({"event": "progress_start", "front_range": start_range})

            def progress():
                lidar = front.progress()
                odo = odom() if odom else None
                fused, lidar_used = fuse_progress(lidar, odo)
                s.log({"event": "progress", "lidar": lidar, "odom": odo, "fused": fused, "lidar_used": lidar_used})
                return fused

        s.log({"event": "before", **mon.snapshot()})
        try:
            out = execute(req, route, mon, confirm=a.confirm, execute_flag=True,
                          verified_stop=a.stop_verified, log=lambda e: s.log(e), progress=progress,
                          stall_grace=a.stall_grace)
        finally:
            time.sleep(3.0)  # post-move state samples; step C showed motion 0.8 s after release
            stop_evt.set()
        after = route.avoidance_enabled()
        s.log({"event": "after", "avoidance_enabled": after, **mon.snapshot()})
    summary = {k: v for k, v in out.__dict__.items() if k != "events"}
    summary["avoidance_enabled_after"] = after
    s.save("summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    if out.stop_confirmed is not True:
        print("!!! STOP NOT CONFIRMED BY STATE - operator: take over with the remote now !!!")
    print(f"saved {s.dir}")


def cmd_stop(a):
    if not a.execute:
        raise Refused("dry run: stop needs --execute")
    from go2_control.sdk_routes import AvoidRoute

    s = Session("stop")
    mon = start_monitor(a.iface)
    route = AvoidRoute(a.timeout)
    for _ in range(3):
        s.log({"event": "send_stop", "code": route.send_stop()})
        time.sleep(0.05)
    time.sleep(1.0)
    s.log({"event": "observed", "still": mon.is_still(), **mon.snapshot()})
    s.log({"event": "release_api_control", "code": route.release_api_control()})
    print(f"still={mon.is_still()} saved {s.dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", default="status", choices=["status", "watch", "range", "preview", "ownership", "move", "stop"])
    p.add_argument("--iface", default="en9")
    p.add_argument("--timeout", type=float, default=1.0, help="RPC reply timeout, s")
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--seconds", type=float, default=3.0, help="status wait / ownership hold, s")
    p.add_argument("--vx", type=float, default=0.0)
    p.add_argument("--vyaw", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=1.0)
    p.add_argument("--resend-period", type=float, default=0.0)
    p.add_argument("--target", type=float, default=0.0, help="distance goal in the commanded direction, m (LiDAR-measured)")
    p.add_argument("--turn-deg", type=float, default=0.0, help="heading goal for a turn, degrees (IMU)")
    p.add_argument("--stall-grace", type=float, default=2.0, help="s before the stall check starts (2-6)")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm", default="")
    p.add_argument("--stop-verified", action="store_true")
    a = p.parse_args(argv)
    try:
        {"status": cmd_status, "watch": cmd_watch, "range": cmd_range, "preview": cmd_preview, "ownership": cmd_ownership,
         "move": cmd_move, "stop": cmd_stop}[a.mode](a)
    except Refused as exc:
        print(f"REFUSED: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)  # avoid Cyclone DDS teardown hangs, as in the recorder scripts
