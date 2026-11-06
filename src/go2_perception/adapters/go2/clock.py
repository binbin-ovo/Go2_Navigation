"""Robot/host clock offset from ICMP timestamps (read-only ping; integer ms, mod 24 h)."""

import re
import subprocess

_DAY_S = 86400.0


def parse_ping_timestamps(text):
    """Return host-minus-robot offsets (s, mod 24 h) from ``ping -T tsonly`` output."""
    return [
        -int(m) / 1000.0
        for m in re.findall(r"TS:\s+\d+ absolute\s+(-?\d+)\s", text)
    ]


def measure_offset_mod_day(robot_ip="192.168.123.161", iface="eno2", count=5):
    """Return the median host-minus-robot offset mod 24 h, or None if unavailable."""
    try:
        run = subprocess.run(
            ["ping", "-c", str(count), "-i", "0.2", "-T", "tsonly", "-I", iface, robot_ip],
            capture_output=True,
            text=True,
            timeout=count + 5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    values = sorted(parse_ping_timestamps(run.stdout))
    return values[len(values) // 2] if values else None


def resolve_full_offset(offset_mod_day, approximate_offset):
    """Recover whole days using a rough offset such as min(host_recv - stamp)."""
    days = round((approximate_offset - offset_mod_day) / _DAY_S)
    return offset_mod_day + days * _DAY_S
