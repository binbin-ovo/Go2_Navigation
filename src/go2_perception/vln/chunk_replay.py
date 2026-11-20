"""Stream five-move recorder chunks in receipt-time order, retaining original clocks."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np


def chunk_directory(root):
    root = Path(root)
    return next(
        (p for p in (root, root / "lidar", root / "session/lidar") if (p / "index.jsonl").is_file()), None
    )


def chunk_events(root, no_odom=False):
    """Load one indexed NPZ at a time. Gaps/epochs invalidate accumulated geometry."""
    with (root / "index.jsonl").open() as stream:
        rows = sorted((json.loads(line) for line in stream if line.strip()), key=lambda r: r["chunk"])
    previous, epoch, origin, last_receipt = 0, None, None, None
    for row in rows:
        number = row["chunk"]
        if number <= previous:
            raise ValueError("duplicate or overlapping LiDAR chunk")
        missing = number != previous + 1
        previous = row.get("last_chunk", number)
        if row["dropped_not_recorded"]:
            yield (
                0 if last_receipt is None else last_receipt - origin,
                "replay_boundary",
                {"reason": row.get("reason", "dropped_chunk"), "recorded_chunk": number},
                None,
            )
            continue
        if missing or (epoch is not None and epoch != row["robot_epoch"]):
            yield (
                0 if last_receipt is None else last_receipt - origin,
                "replay_boundary",
                {
                    "reason": "missing_chunk" if missing else "recorded_epoch_changed",
                    "recorded_chunk": number,
                },
                None,
            )
        epoch = row["robot_epoch"]
        name = f"chunk_{number:06d}.npz"
        if row["file"] != name:
            raise ValueError("unexpected LiDAR chunk filename")
        content = (root / name).read_bytes()
        if len(content) != row["bytes"] or sha256(content).hexdigest() != row["sha256"]:
            raise ValueError("LiDAR chunk size or SHA-256 mismatch")
        with np.load(BytesIO(content), allow_pickle=False) as data:
            xyz, offsets = data["points_xyz_base_link"], data["scan_offsets"]
            if (
                xyz.shape != (row["n_points"], 3)
                or xyz.dtype.kind != "f"
                or offsets.shape != (row["n_scans"] + 1,)
                or offsets.dtype.kind not in "iu"
                or offsets[0] != 0
                or offsets[-1] != len(xyz)
                or np.any(offsets[1:] < offsets[:-1])
            ):
                raise ValueError("invalid LiDAR scan offsets or points")
            positions, quats, spans = data["odom_position"], data["odom_q_xyzw"], data["scan_time_span_s"]
            if (
                positions.shape != (row["n_odom"], 3)
                or quats.shape != (row["n_odom"], 4)
                or spans.shape != (row["n_scans"],)
                or not np.isfinite(positions).all()
                or not np.isfinite(quats).all()
                or np.isinf(spans).any()
                or (spans < 0).any()
            ):
                raise ValueError("invalid archived pose or scan span")
            events = []
            for kind, n in (("scan", row["n_scans"]), ("odom", row["n_odom"])):
                seq, stamp, wall, offset = [
                    data[f"{kind}_{key}"]
                    for key in ("seq", "stamp_robot", "host_recv_wall", "wall_minus_monotonic")
                ]
                if (
                    any(v.shape != (n,) or not np.isfinite(v).all() for v in (seq, stamp, wall, offset))
                    or seq.dtype.kind not in "iu"
                    or np.any(seq[1:] <= seq[:-1])
                    or row[kind + "_seq"] != ([int(seq[0]), int(seq[-1])] if n else None)
                ):
                    raise ValueError("invalid archived LiDAR sequence or clocks")
                mono = wall - offset
                recorded = f"{kind}_host_recv_monotonic_s"
                if recorded in data and not np.array_equal(mono, data[recorded]):
                    raise ValueError("inconsistent archived receipt clocks")
                if kind == "odom" and no_odom:
                    continue
                for i in range(n):
                    fields = dict(
                        stamp=float(stamp[i]),
                        seq=int(seq[i]),
                        recorded_robot_epoch=epoch,
                        recorded_chunk=number,
                        recorded_trial_id=row["trial_id"],
                        recorded_host_recv_wall_s=float(wall[i]),
                        recorded_host_recv_monotonic_s=float(mono[i]),
                    )
                    points = None
                    if kind == "scan":
                        points = xyz[offsets[i] : offsets[i + 1]].astype("<f4", copy=False)
                        fields.update(
                            frame_id="base_link", time_span_s=None if np.isnan(spans[i]) else float(spans[i])
                        )
                    else:
                        fields.update(position=positions[i].tolist(), q_xyzw=quats[i].tolist())
                    events.append((float(mono[i]), kind, fields, points))
        for receipt, kind, fields, points in sorted(events, key=lambda e: e[0]):
            if last_receipt is not None and receipt < last_receipt:
                raise ValueError("backward LiDAR chunk host-monotonic ordering")
            if origin is None:
                origin = receipt
            last_receipt = receipt
            yield receipt - origin, kind, fields, points
