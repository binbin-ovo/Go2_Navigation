#!/usr/bin/env python
"""Record a short read-only Go2 sequence and measure the streams while doing so.

Usage: .venv/bin/python scripts/go2_record.py --seconds 20 --name sample
Writes to <data-dir>/<name> (default data/recordings, or $GO2_DATA_DIR). Sends no commands.
Exit status is nonzero when any required stream delivered no data.
"""
import argparse
import json
import os
import queue
import resource
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

from go2_perception.adapters.go2.clock import (
    measure_offset_mod_day,
    resolve_full_offset,
)
from go2_perception.adapters.go2.receivers import (
    LIDAR_TOPICS,
    Go2CameraReceiver,
    Go2DdsReceiver,
)
from go2_perception.adapters.go2.recording import RecordingWriter
from go2_perception.adapters.go2.rtp_h264 import RTP_CLOCK_HZ, unwrap_rtp
from go2_perception.adapters.go2.stats import StreamStats, latency_estimate
from go2_perception.io.config import load_config


def main():
    cfg = load_config("configs/go2_sensors.yaml")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--name", default=time.strftime("go2_%Y%m%d_%H%M%S"))
    p.add_argument("--data-dir", default=os.environ.get("GO2_DATA_DIR", "data/recordings"))
    p.add_argument("--iface", default=cfg["network"]["iface"])
    p.add_argument("--host-ip", default=cfg["network"]["host_ip"])
    p.add_argument("--robot-ip", default=cfg["network"]["robot_ip"])
    p.add_argument("--topics", nargs="+", default=list(LIDAR_TOPICS))
    p.add_argument("--rgb-format", choices=["jpg", "png"], default="jpg")
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--notes", default="")
    a = p.parse_args()

    root = Path(a.data_dir) / a.name
    writer = RecordingWriter(root, a.notes, a.rgb_format)
    jobs = queue.Queue(maxsize=512)
    dropped = {"writer_queue": 0}
    stats = {t: StreamStats(t) for t in a.topics}
    state_stats = {}
    rgb_stats = StreamStats("rgb")
    rtp = {"unwrapped": None}
    lock = threading.Lock()

    def enqueue(job):
        try:
            jobs.put_nowait(job)
        except queue.Full:
            dropped["writer_queue"] += 1

    def on_lidar(frame):
        with lock:
            stats[frame.topic].add(frame.host_recv, frame.stamp, len(frame.xyz))
        enqueue(("lidar", frame))

    def on_state(topic, record):
        with lock:
            state_stats.setdefault(topic, StreamStats(topic)).add(
                record["host_recv"], record["stamp"]
            )
        enqueue(("state", topic, record))

    def on_frame(frame):
        with lock:
            rtp["unwrapped"] = unwrap_rtp(rtp["unwrapped"], frame.rtp_timestamp)
            rgb_stats.add(frame.host_recv_last, rtp["unwrapped"] / RTP_CLOCK_HZ)
        enqueue(("rgb", frame))

    def write_loop():
        while True:
            job = jobs.get()
            if job is None:
                return
            if job[0] == "lidar":
                writer.write_lidar(job[1])
            elif job[0] == "state":
                writer.write_state(job[1], job[2])
            else:
                f = job[1]
                writer.write_rgb(
                    f.image, f.rtp_timestamp, f.host_recv_first, f.host_recv_last, f.complete
                )

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    offset_before = measure_offset_mod_day(a.robot_ip, a.iface)
    writer_thread = threading.Thread(target=write_loop, name="recorder")
    writer_thread.start()
    dds = Go2DdsReceiver(a.iface, a.topics, on_lidar, on_state)
    dds.start()
    camera = None
    if not a.no_camera:
        camera = Go2CameraReceiver(
            a.host_ip,
            cfg["camera"]["multicast_group"],
            cfg["camera"]["port"],
            on_frame,
            raw_path=str(root / "camera.h264"),
        )
        camera.start()

    wall0, cpu0 = time.time(), sum(os.times()[:2])
    deadline = wall0 + a.seconds
    while time.time() < deadline and not stop.is_set():
        time.sleep(0.2)
    wall, cpu = time.time() - wall0, sum(os.times()[:2]) - cpu0

    dds.stop()
    if camera:
        camera.stop()
        camera.join(timeout=3)
    jobs.put(None)
    writer_thread.join()
    offset_after = measure_offset_mod_day(a.robot_ip, a.iface)

    with lock:
        lidar_summary = {t: s.summary() for t, s in stats.items()}
        state_summary = {t: s.summary() for t, s in state_stats.items()}
        rgb_summary = rgb_stats.summary()

    # Full clock offset: ICMP gives ms precision mod 24 h; receipts give the whole days.
    report = {"clock": {"icmp_offset_mod_day_before_s": offset_before,
                        "icmp_offset_mod_day_after_s": offset_after}}
    reference = next((s for t, s in state_stats.items() if s.stamps), None)
    if reference and offset_before is not None:
        approx = min(r - s for r, s in zip(reference.host_recv, reference.stamps))
        full = resolve_full_offset(offset_before, approx)
        report["clock"]["host_minus_robot_s"] = full
        for name, source, summary in [
            *((t, stats[t], lidar_summary[t]) for t in stats),
            *((t, state_stats[t], state_summary[t]) for t in state_stats),
        ]:
            if len(source.stamps) == len(source.host_recv) and source.stamps:
                summary["latency_recv_minus_stamp_ms"] = latency_estimate(
                    source.host_recv, source.stamps, full
                )
    report.update(
        lidar=lidar_summary,
        state=state_summary,
        rgb=rgb_summary,
        recorder={
            "requested_s": a.seconds,
            "wall_s": round(wall, 3),
            "process_cpu_percent_of_one_core": round(100 * cpu / wall, 1),
            "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "dropped_by_writer_queue": dropped["writer_queue"],
            "lidar_decode_errors": dds.decode_errors,
        },
    )
    if camera:
        report["camera"] = {
            "senders": sorted(camera.senders),
            "rtp_packets": camera.assembler.packets,
            "rtp_lost_packets": camera.assembler.lost_packets,
            "rtp_reordered_packets": camera.assembler.reordered_packets,
            "incomplete_units": camera.incomplete_units,
            "frames_dropped_until_keyframe": camera.dropped_until_keyframe,
            "keyframes": len(camera.keyframe_rtp),
            "keyframe_interval_s": round(
                float(np.median(np.diff(camera.keyframe_rtp)) / RTP_CLOCK_HZ), 3
            ) if len(camera.keyframe_rtp) > 1 else None,
            "decoder_errors": camera.decoder_errors,
            "rejoins": camera.reconnects,
            "socket_rcvbuf_bytes": camera.rcvbuf_bytes,
            "thread_exception": repr(camera.exception) if camera.exception else None,
        }
    writer.close(report)
    print(json.dumps(report, indent=2))
    print(f"\nRecording: {root}")
    required = [lidar_summary[t]["count"] for t in a.topics]
    if camera:
        required.append(rgb_summary["count"])
    sys.stdout.flush()
    os._exit(0 if all(required) else 1)  # cyclonedds may hang interpreter teardown


if __name__ == "__main__":
    main()
