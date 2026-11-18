"""Summarise audited go2_hl_control move logs (read-only, no SDK).

For each `*_move/events.jsonl`: the request, how the move ended, IMU and odometry heading change at the
peak and after settling, translation of the body during the move and after release, and the time from
the first stop command until the state read still. Used to choose bounded-turn limits from measured
behaviour instead of requested values.
"""
import argparse
import json
import math
from pathlib import Path


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def odom_yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def summarise(path):
    events = [json.loads(line) for line in open(path)]
    req = next(e["request"] for e in events if e["event"] == "request")
    states = [e for e in events if e["event"] == "state" and e.get("lidar_odom")]
    first_send = next((e["mono"] for e in events if e["event"] == "send_velocity"), None)
    stop_t = next((e["mono"] for e in events if e["event"] == "send_stop"), None)
    if not states or first_send is None:
        return {"log": path.parent.name, "request": req, "started": False}
    before = [s for s in states if s["mono"] <= first_send] or states[:1]
    s0 = before[-1]
    imu0, od0 = s0["rpy"][2], odom_yaw(s0["lidar_odom"][3:7])
    x0, y0 = s0["lidar_odom"][0], s0["lidar_odom"][1]

    def rel(s):
        dx, dy = s["lidar_odom"][0] - x0, s["lidar_odom"][1] - y0
        c, si = math.cos(od0), math.sin(od0)
        return {"imu_deg": math.degrees(wrap(s["rpy"][2] - imu0)),
                "odom_deg": math.degrees(wrap(odom_yaw(s["lidar_odom"][3:7]) - od0)),
                "fwd_m": c * dx + si * dy, "left_m": -si * dx + c * dy}

    track = [(s["mono"], rel(s), s) for s in states if s["mono"] >= first_send]
    peak = max(track, key=lambda r: abs(r[1]["imu_deg"]))[1]
    final = track[-1][1]
    still_t = None
    if stop_t is not None:
        for t, _, s in track:
            v = s["velocity"]
            if t >= stop_t and math.hypot(v[0], v[1]) < 0.03 and abs(s["yaw_speed"]) < 0.05:
                still_t = t - stop_t
                break
    at_stop = min(track, key=lambda r: abs(r[0] - stop_t))[1] if stop_t else None
    summary = next((json.loads(Path(path.parent, "summary.json").read_text())
                    for _ in [0] if Path(path.parent, "summary.json").exists()), {})
    return {"log": path.parent.name, "request": req, "end_reason": summary.get("end_reason"),
            "released": summary.get("released"), "avoid_after": summary.get("avoidance_enabled_after"),
            "at_stop": at_stop, "peak": peak, "final": final,
            "springback_deg": abs(peak["imu_deg"]) - abs(final["imu_deg"]),
            "still_after_stop_s": still_t,
            "observed_after_stop_s": track[-1][0] - stop_t if stop_t else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    rows = [summarise(p) for p in sorted(args.root.glob("*_move/events.jsonl"))]
    for r in rows:
        q = r["request"]
        if not r.get("final"):
            print(f"{r['log']}: not started {q}")
            continue
        f, p = r["final"], r["peak"]
        print(f"{r['log']}: vx={q['vx']:.2f} vyaw={q['vyaw']:+.2f} dur={q['duration']} target={q.get('target')} "
              f"end={r['end_reason']} | imu peak {p['imu_deg']:+.1f} final {f['imu_deg']:+.1f} "
              f"odom {f['odom_deg']:+.1f} deg | fwd {f['fwd_m']:+.3f} left {f['left_m']:+.3f} m | "
              f"still {r['still_after_stop_s']} s, watched {r['observed_after_stop_s']}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
