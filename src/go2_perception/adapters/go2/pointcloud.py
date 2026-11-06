"""PointCloud2 decoding and conversion to the Phase 1 ``LidarMessage`` contract."""

from dataclasses import dataclass

import numpy as np

from go2_perception.io.sensor_interface import LidarMessage

_DATATYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
ROBOT_CLOCK = "go2_robot"  # header.stamp clock; NOT host time (see docs/SENSOR_VALIDATION.md)
BODY_FRAMES = {"base_link", "base_link_level"}  # x forward, y left, z up


@dataclass
class LidarFrame:
    """One decoded message. ``time`` is seconds since ``stamp`` per point, when published."""

    topic: str
    stamp: float
    frame_id: str
    host_recv: float
    xyz: np.ndarray  # (N, 3) float32, as published
    intensity: np.ndarray | None = None
    time: np.ndarray | None = None
    ring: np.ndarray | None = None

    @property
    def acquisition_midpoint(self):
        """Declared LiDAR timestamp: stamp + half the per-point time span, if available."""
        if self.time is None or not len(self.time):
            return self.stamp
        return self.stamp + 0.5 * float(np.max(self.time))


def decode_pointcloud2(data, fields, point_step, width, height, is_bigendian=False):
    """Decode by the message's own field table, returning {field name: array}."""
    n = int(width) * int(height)
    buffer = data if isinstance(data, (bytes, bytearray, memoryview)) else bytes(data)
    if point_step <= 0 or len(buffer) < n * point_step:
        raise ValueError(
            f"PointCloud2 holds {len(buffer)} bytes; {n} points x {point_step} required"
        )
    order = ">" if is_bigendian else "<"
    names, formats, offsets = [], [], []
    for f in fields:
        if f.datatype not in _DATATYPES:
            raise ValueError(f"Unsupported PointField datatype {f.datatype}")
        if f.count != 1:
            raise ValueError(f"Unsupported PointField count {f.count} for {f.name}")
        names.append(f.name)
        formats.append(order + _DATATYPES[f.datatype])
        offsets.append(f.offset)
    if not {"x", "y", "z"} <= set(names):
        raise ValueError(f"PointCloud2 lacks x/y/z fields: {names}")
    dtype = np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": point_step}
    )
    records = np.frombuffer(buffer, dtype=dtype, count=n)
    return {name: records[name].copy() for name in names}


def frame_from_msg(msg, topic, host_recv):
    """Convert an SDK ``PointCloud2_`` sample, keeping its stamp and frame unchanged."""
    d = decode_pointcloud2(
        msg.data, msg.fields, msg.point_step, msg.width, msg.height, msg.is_bigendian
    )
    return LidarFrame(
        topic,
        msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        msg.header.frame_id,
        host_recv,
        np.stack([d["x"], d["y"], d["z"]], axis=1).astype(np.float32),
        d.get("intensity"),
        d.get("time"),
        d.get("ring"),
    )


def quaternion_matrix(q_xyzw):
    x, y, z, w = np.asarray(q_xyzw, dtype=float) / np.linalg.norm(q_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def level_points(xyz, q_odom_base_xyzw):
    """Remove body roll and pitch using the odometry attitude, keeping heading."""
    r = quaternion_matrix(q_odom_base_xyzw)
    yaw = np.arctan2(r[1, 0], r[0, 0])
    c, s = np.cos(-yaw), np.sin(-yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.asarray(xyz, dtype=float) @ (rz @ r).T


def to_lidar_message(frame, q_odom_base_xyzw=None, clock_domain=ROBOT_CLOCK):
    """Build a ``LidarMessage``.

    The Phase 1 contract needs x forward, y left, z up, so only body-frame clouds qualify.
    ``rt/utlidar/cloud`` (tilted sensor frame) and ``rt/utlidar/cloud_deskewed`` (a rolling
    multi-frame buffer in ``odom``) are rejected rather than silently misinterpreted. When
    an attitude is supplied the result is levelled and labelled ``base_link_level``;
    otherwise the frame stays ``base_link`` and carries the body's tilt.
    """
    if frame.frame_id not in BODY_FRAMES:
        raise ValueError(
            f"{frame.topic} is in frame '{frame.frame_id}'; expected one of "
            f"{sorted(BODY_FRAMES)}. Use rt/utlidar/cloud_base."
        )
    xyz, frame_id = frame.xyz.astype(float), frame.frame_id
    if q_odom_base_xyzw is not None:
        xyz, frame_id = level_points(xyz, q_odom_base_xyzw), "base_link_level"
    if frame.intensity is None:
        points = xyz
    else:
        points = np.column_stack((xyz, frame.intensity.astype(float)))
    return LidarMessage(points, frame.acquisition_midpoint, frame_id, clock_domain)
