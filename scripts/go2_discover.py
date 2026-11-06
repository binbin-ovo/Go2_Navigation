#!/usr/bin/env python
"""Passive DDS discovery of the Go2: lists every remote publication (topic, type, QoS).

Creates a DDS participant and built-in readers only. It creates no data writers, so it
cannot command the robot.
Usage: .venv/bin/python scripts/go2_discover.py [--iface eno2] [--seconds 8] [--json out.json]
"""
import argparse
import json
import os
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iface", default="eno2")
    p.add_argument("--domain", type=int, default=0)
    p.add_argument("--seconds", type=float, default=8)
    p.add_argument("--json")
    a = p.parse_args()
    os.environ["CYCLONEDDS_URI"] = (
        "<CycloneDDS><Domain><General><Interfaces>"
        f'<NetworkInterface name="{a.iface}" priority="default" multicast="default"/>'
        "</Interfaces></General></Domain></CycloneDDS>"
    )
    from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsPublication
    from cyclonedds.domain import DomainParticipant

    dp = DomainParticipant(a.domain)
    reader = BuiltinDataReader(dp, BuiltinTopicDcpsPublication)
    found, deadline = {}, time.time() + a.seconds
    while time.time() < deadline:
        for s in reader.take(N=256):
            if not getattr(s, "topic_name", None):
                continue
            rel = [str(q) for q in (s.qos or []) if "Reliability" in str(q)]
            found[s.topic_name] = {
                "type": s.type_name,
                "reliability": rel[0] if rel else None,
            }
        time.sleep(0.1)
    for t in sorted(found):
        print(f"{t:55s} {found[t]['type']:50s} {found[t]['reliability']}")
    print(f"\n{len(found)} remote publications")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(found, f, indent=2, sort_keys=True)
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
