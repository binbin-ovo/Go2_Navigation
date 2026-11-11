"""Measure Go2 stream timing from local recordings: rates, jitter, host-robot offset, drift and,
when the robot yaws, the camera's host-receipt latency relative to odometry stamps.

Offline and read-only. Clock model: every DDS stamp is the robot clock; the camera has only host
receipt. The host-robot offset is estimated as the lower envelope of (host_recv - stamp) on
robot_odom, so it includes odometry's minimum transport latency (about 1-2 ms on Linux, where
ICMP measured it). Camera latency comes from cross-correlating image yaw rate (phase correlation
of consecutive frames) with odometry yaw rate; it is only defined in recordings with rotation.

Usage:
  PYTHONPATH=src python scripts/phase5d_timing_report.py --data data/recordings \
      --recordings p5d_static_02 test_c --output data/outputs/phase5d/timing.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

LIDAR_TOPICS = ["utlidar_cloud", "utlidar_cloud_base", "utlidar_cloud_deskewed"]
STATE_TOPICS = ["utlidar_robot_odom", "utlidar_imu", "lf_sportmodestate"]
MIN_YAW_RATE = 0.15  # rad/s; below this the recording has too little rotation for a lag estimate


def jsonl(path):
    return [json.loads(line) for line in open(path)]


def pct(values, qs=(0, 50, 95, 100)):
    v = np.asarray(values, float)
    return dict(zip(["min", "median", "p95", "max"], [float(x) for x in np.percentile(v, qs)]))


def stream_timing(stamps, host):
    stamps, host = np.asarray(stamps, float), np.asarray(host, float)
    lag = host - stamps
    floor = lag.min()
    # Drift: slope of the per-second lower envelope of (host - stamp) against robot time.
    bins = np.floor(stamps - stamps[0]).astype(int)
    env_t = [stamps[bins == b].mean() for b in np.unique(bins)]
    env_v = [lag[bins == b].min() for b in np.unique(bins)]
    drift_ppm = float(np.polyfit(env_t, env_v, 1)[0] * 1e6) if len(env_t) >= 5 else None
    return {
        "count": len(stamps),
        "rate_hz_stamp": float((len(stamps) - 1) / (stamps[-1] - stamps[0])),
        "stamp_dt_ms": pct(np.diff(stamps) * 1e3),
        "recv_dt_ms": pct(np.diff(host) * 1e3),
        "host_minus_stamp_floor_s": float(floor),
        "host_minus_stamp_above_floor_ms": pct((lag - floor) * 1e3),
        "drift_ppm_host_vs_robot": drift_ppm,
        "nonincreasing_stamps": int(np.sum(np.diff(stamps) <= 0)),
    }


def image_yaw_rate(root, rows, scale=0.25):
    """Horizontal image motion (px/s at full resolution) between consecutive frames."""
    t, rate, prev, prev_t = [], [], None, None
    window = None
    for r in rows:
        img = cv2.imread(str(root / "rgb" / r["file"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        g = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
        if window is None:
            window = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        now = r["host_recv_first"]
        if prev is not None:
            (dx, _), _ = cv2.phaseCorrelate(prev, g, window)
            t.append(0.5 * (now + prev_t))
            rate.append(dx / scale / (now - prev_t))
        prev, prev_t = g, now
    return np.asarray(t), np.asarray(rate)


def camera_lag(root, rgb_rows, odom_rows, offset_floor):
    odom_t = np.array([r["stamp"] for r in odom_rows]) + offset_floor  # robot -> host clock
    wz = np.array([r["angular"][2] for r in odom_rows])
    if np.percentile(np.abs(wz), 95) < MIN_YAW_RATE:
        return {"status": "not_measurable", "reason": "no yaw above %.2f rad/s" % MIN_YAW_RATE,
                "odom_abs_yaw_rate_p95": float(np.percentile(np.abs(wz), 95))}
    t_img, px_rate = image_yaw_rate(root, rgb_rows)
    lags = np.arange(-0.300, 0.300001, 0.002)

    def scan(sel):
        # Hypothesis: image motion seen at host time t happened at odom (host-mapped) time t - lag.
        return np.abs([np.corrcoef(np.interp(t_img[sel] - lag, odom_t, wz), px_rate[sel])[0, 1]
                       for lag in lags])

    scores = scan(slice(None))
    i = int(np.argmax(scores))
    half = len(t_img) // 2
    halves = [float(lags[np.argmax(scan(sel))]) for sel in (slice(None, half), slice(half, None))]
    # Uncertainty: lags whose correlation is within 1% of the peak.
    near = lags[scores >= scores[i] - 0.01]
    w_best = np.interp(t_img - lags[i], odom_t, wz)
    focal_px_per_rad = float(np.polyfit(w_best, px_rate, 1)[0])
    return {
        "status": "measured",
        "method": "max |corr| of odom yaw rate vs image horizontal phase-correlation shift rate",
        "lag_s": float(lags[i]),
        "lag_band_within_1pct_s": [float(near.min()), float(near.max())],
        "peak_abs_corr": float(scores[i]),
        "lag_by_half_s": halves,
        "meaning": "camera host receipt minus odometry stamp (mapped by the odom latency floor) "
                   "for the same motion; equals camera exposure-to-receipt latency minus odom's "
                   "minimum latency, with frame-interval quantisation",
        "image_px_per_rad_center_proxy": abs(focal_px_per_rad),
        "frames": len(t_img),
    }


def recording_report(root):
    out = {"recording": root.name, "streams": {}}
    odom = jsonl(root / "state" / "utlidar_robot_odom.jsonl")
    for topic in LIDAR_TOPICS:
        rows = jsonl(root / "lidar" / topic / "index.jsonl")
        out["streams"][topic] = stream_timing([r["stamp"] for r in rows], [r["host_recv"] for r in rows])
    for topic in STATE_TOPICS:
        path = root / "state" / f"{topic}.jsonl"
        if path.exists():
            rows = jsonl(path)
            out["streams"][topic] = stream_timing([r["stamp"] for r in rows], [r["host_recv"] for r in rows])
    rgb = jsonl(root / "rgb" / "index.jsonl")
    first = np.array([r["host_recv_first"] for r in rgb])
    rtp = np.array([r["rtp_timestamp"] for r in rgb], float)
    out["streams"]["rgb"] = {
        "count": len(rgb), "rate_hz_host": float((len(rgb) - 1) / (first[-1] - first[0])),
        "recv_dt_ms": pct(np.diff(first) * 1e3),
        "assembly_ms": pct([(r["host_recv_last"] - r["host_recv_first"]) * 1e3 for r in rgb]),
        "rtp_rate_over_host_rate": float((rtp[-1] - rtp[0]) / 90000 / (first[-1] - first[0])),
        "timestamp_semantics": "host_recv_first: host time of first RTP packet; not exposure",
    }
    offset = out["streams"]["utlidar_robot_odom"]["host_minus_stamp_floor_s"]
    out["host_minus_robot_offset_estimate_s"] = offset
    out["offset_note"] = "lower envelope of odom host_recv - stamp; upper-biased by odom min latency"
    out["robot_stamp_range"] = [odom[0]["stamp"], odom[-1]["stamp"]]
    # Nearest-receipt pairing statistics (what earlier phases used), for comparison only.
    base = jsonl(root / "lidar" / "utlidar_cloud_base" / "index.jsonl")
    lidar_host = np.array([r["host_recv"] for r in base])
    idx = np.clip(np.searchsorted(lidar_host, first), 1, len(lidar_host) - 1)
    nearest = np.minimum(np.abs(lidar_host[idx] - first), np.abs(lidar_host[idx - 1] - first))
    out["rgb_to_nearest_cloud_base_receipt_ms"] = pct(nearest * 1e3)
    out["camera_latency_vs_odom"] = camera_lag(root, rgb, odom, offset)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--recordings", nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    reports = [recording_report(a.data / name) for name in a.recordings]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(reports, indent=2))
    for r in reports:
        s = r["streams"]
        print(r["recording"], "offset %.4f" % r["host_minus_robot_offset_estimate_s"],
              "robot stamps %.1f-%.1f" % tuple(r["robot_stamp_range"]),
              "cloud_base above-floor ms", {k: round(v, 1) for k, v in s["utlidar_cloud_base"]["host_minus_stamp_above_floor_ms"].items()},
              "drift ppm", s["utlidar_robot_odom"]["drift_ppm_host_vs_robot"],
              "camera", {k: v for k, v in r["camera_latency_vs_odom"].items() if k not in ("method", "meaning")})


if __name__ == "__main__":
    main()
