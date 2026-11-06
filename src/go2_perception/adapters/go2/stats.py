"""Measured per-stream statistics. Only observed values are reported."""

import numpy as np


class StreamStats:
    """Receipt rate, stamp continuity and gaps. Stamps may be in any single clock."""

    def __init__(self, name, gap_factor=1.8):
        self.name = name
        self.gap_factor = gap_factor
        self.host_recv, self.stamps, self.sizes = [], [], []

    def add(self, host_recv, stamp=None, size=None):
        self.host_recv.append(host_recv)
        if stamp is not None:
            self.stamps.append(stamp)
        if size is not None:
            self.sizes.append(size)

    def summary(self):
        recv = np.asarray(self.host_recv, dtype=float)
        out = {"name": self.name, "count": len(recv)}
        if len(recv) < 2:
            return out
        d = np.diff(recv)
        out["duration_s"] = round(float(recv[-1] - recv[0]), 3)
        out["rate_hz"] = round(float(len(d) / max(recv[-1] - recv[0], 1e-9)), 3)
        out["recv_gap_ms"] = _quantiles(d * 1e3)
        if len(self.stamps) > 1:
            s = np.asarray(self.stamps, dtype=float)
            ds = np.diff(s)
            nominal = float(np.median(ds))
            out["stamp_dt_ms"] = _quantiles(ds * 1e3)
            out["nonincreasing_stamps"] = int((ds <= 0).sum())
            if nominal > 0:
                gaps = ds[ds > self.gap_factor * nominal]
                out["estimated_missed_messages"] = int(
                    np.round(gaps / nominal).sum() - len(gaps)
                )
                out["stamp_gaps_over_threshold"] = len(gaps)
        if self.sizes:
            z = np.asarray(self.sizes, dtype=float)
            out["size"] = {
                "mean": round(float(z.mean()), 1),
                "min": int(z.min()),
                "max": int(z.max()),
            }
        return out


def _quantiles(values):
    p = np.percentile(values, [50, 95, 99])
    return {
        "median": round(float(p[0]), 3),
        "p95": round(float(p[1]), 3),
        "p99": round(float(p[2]), 3),
        "min": round(float(np.min(values)), 3),
        "max": round(float(np.max(values)), 3),
    }


def latency_estimate(host_recv, stamps, offset_host_minus_robot):
    """One-way latency for a measured clock offset. Accuracy is limited to about 1 ms."""
    v = (np.asarray(host_recv) - np.asarray(stamps) - offset_host_minus_robot) * 1e3
    return _quantiles(v)
