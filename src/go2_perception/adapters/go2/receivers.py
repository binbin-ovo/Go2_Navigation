"""Read-only live receivers. Requires eno2 link, the Unitree SDK (DDS) and PyAV (camera).

Only subscribers and a multicast read socket exist here. No DataWriter, RPC client or
outgoing datagram is created. Do not add one: this adapter must never command the robot.
"""

import socket
import threading
import time
from dataclasses import dataclass

import numpy as np

from .pointcloud import frame_from_msg
from .rtp_h264 import AccessUnitAssembler, H264Decoder, parse_rtp

LIDAR_TOPICS = (
    "rt/utlidar/cloud",
    "rt/utlidar/cloud_base",
    "rt/utlidar/cloud_deskewed",
)
ODOM_TOPIC = "rt/utlidar/robot_odom"
IMU_TOPIC = "rt/utlidar/imu"
SPORT_TOPIC = "rt/lf/sportmodestate"


def odom_record(msg, host_recv):
    h, pose, twist = msg.header, msg.pose.pose, msg.twist.twist
    return {
        "stamp": h.stamp.sec + h.stamp.nanosec * 1e-9,
        "host_recv": host_recv,
        "frame_id": h.frame_id,
        "child_frame_id": msg.child_frame_id,
        "position": [pose.position.x, pose.position.y, pose.position.z],
        "q_xyzw": [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        "linear": [twist.linear.x, twist.linear.y, twist.linear.z],
        "angular": [twist.angular.x, twist.angular.y, twist.angular.z],
    }


def imu_record(msg, host_recv):
    h = msg.header
    q, w, a = msg.orientation, msg.angular_velocity, msg.linear_acceleration
    return {
        "stamp": h.stamp.sec + h.stamp.nanosec * 1e-9,
        "host_recv": host_recv,
        "frame_id": h.frame_id,
        "q_xyzw": [q.x, q.y, q.z, q.w],
        "angular_velocity": [w.x, w.y, w.z],
        "linear_acceleration": [a.x, a.y, a.z],
    }


def sport_record(msg, host_recv):
    return {
        "stamp": msg.stamp.sec + msg.stamp.nanosec * 1e-9,
        "host_recv": host_recv,
        "mode": int(msg.mode),
        "error_code": int(msg.error_code),
        "body_height": float(msg.body_height),
        "position": [float(v) for v in msg.position],
        "velocity": [float(v) for v in msg.velocity],
        "rpy": [float(v) for v in msg.imu_state.rpy],
    }


class Go2DdsReceiver:
    """Subscribes to LiDAR and state topics; callbacks get decoded values on DDS threads."""

    def __init__(self, iface="eno2", lidar_topics=LIDAR_TOPICS, on_lidar=None, on_state=None):
        self.iface, self.lidar_topics = iface, tuple(lidar_topics)
        self.on_lidar, self.on_state = on_lidar, on_state
        self.decode_errors = 0
        self.last_recv = {}
        self._subs = []

    def start(self):
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

        from .dds_types import Imu_

        ChannelFactoryInitialize(0, self.iface)

        def lidar_cb(topic):
            def cb(msg):
                t = time.time()
                self.last_recv[topic] = t
                try:
                    frame = frame_from_msg(msg, topic, t)
                except ValueError:
                    self.decode_errors += 1
                    return
                if self.on_lidar:
                    self.on_lidar(frame)

            return cb

        def state_cb(topic, convert):
            def cb(msg):
                t = time.time()
                self.last_recv[topic] = t
                if self.on_state:
                    self.on_state(topic, convert(msg, t))

            return cb

        for topic in self.lidar_topics:
            sub = ChannelSubscriber(topic, PointCloud2_)
            sub.Init(lidar_cb(topic), 10)
            self._subs.append(sub)
        for topic, type_, convert in (
            (ODOM_TOPIC, Odometry_, odom_record),
            (IMU_TOPIC, Imu_, imu_record),
            (SPORT_TOPIC, SportModeState_, sport_record),
        ):
            sub = ChannelSubscriber(topic, type_)
            sub.Init(state_cb(topic, convert), 10)
            self._subs.append(sub)

    def stop(self):
        for sub in self._subs:
            sub.Close()
        self._subs = []


@dataclass
class CameraFrame:
    image: np.ndarray
    rtp_timestamp: int
    host_recv_first: float
    host_recv_last: float
    complete: bool


class Go2CameraReceiver(threading.Thread):
    """Listens to the front camera's H.264 RTP multicast and decodes it. Sends nothing."""

    def __init__(
        self,
        iface_ip="192.168.123.222",
        group="230.1.1.1",
        port=1720,
        on_frame=None,
        raw_path=None,
        stale_after_s=3.0,
    ):
        super().__init__(daemon=True, name="go2-camera")
        self.iface_ip, self.group, self.port = iface_ip, group, port
        self.on_frame = on_frame
        self.raw_path = raw_path
        self.stale_after_s = stale_after_s
        self.assembler = AccessUnitAssembler()
        self.senders = set()
        self.reconnects = 0
        self.frames = 0
        self.incomplete_units = 0
        self.dropped_until_keyframe = 0
        self.rcvbuf_bytes = None
        self.keyframe_rtp = []
        self._stop_event = threading.Event()
        self.exception = None
        self.decoder_errors = 0

    def _open(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self.rcvbuf_bytes = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        s.bind(("", self.port))
        s.setsockopt(
            socket.IPPROTO_IP,
            socket.IP_ADD_MEMBERSHIP,
            socket.inet_aton(self.group) + socket.inet_aton(self.iface_ip),
        )
        s.settimeout(0.5)
        return s

    def run(self):
        try:
            self._loop()
        except Exception as exc:  # surfaced to the caller; the thread must not die silently
            self.exception = exc

    def _loop(self):
        decoder = H264Decoder()
        raw = open(self.raw_path, "wb") if self.raw_path else None
        sock, last_rx, damaged = None, time.time(), True  # no reference frame until an IDR
        times = {}
        try:
            while not self._stop_event.is_set():
                if sock is None:
                    try:
                        sock = self._open()
                        last_rx = time.time()
                    except OSError:
                        time.sleep(0.5)  # interface down; keep retrying until stopped
                        continue
                try:
                    data, sender = sock.recvfrom(65535)
                except TimeoutError:
                    if time.time() - last_rx > self.stale_after_s:
                        sock.close()  # rejoin so a restarted link gets a fresh IGMP join
                        sock = None
                        damaged = True
                        self.reconnects += 1
                    continue
                except OSError:
                    sock.close()
                    sock = None
                    damaged = True
                    self.reconnects += 1
                    continue
                now = last_rx = time.time()
                packet = parse_rtp(data)
                if packet is None:
                    continue
                self.senders.add(sender[0])
                for unit in self.assembler.feed(packet, now):
                    if raw:
                        raw.write(unit.data)
                    if not unit.complete:
                        self.incomplete_units += 1
                        damaged = True
                    if unit.keyframe and unit.complete:
                        damaged = False
                        self.keyframe_rtp.append(unit.rtp_timestamp)
                    if damaged and not unit.parameter_sets_only:
                        # Later frames reference the damaged one; wait for the next IDR
                        # rather than handing corrupted pixels to the detector.
                        self.dropped_until_keyframe += 1
                        continue
                    times[unit.rtp_timestamp] = (unit.host_recv_first, unit.host_recv_last)
                    for image, rtp_timestamp in decoder.decode(unit):
                        first, last = times.pop(rtp_timestamp, (now, now))
                        self.frames += 1
                        if self.on_frame:
                            self.on_frame(
                                CameraFrame(image, rtp_timestamp, first, last, True)
                            )
                    if len(times) > 64:
                        times.clear()
        finally:
            if sock:
                sock.close()
            if raw:
                raw.close()
        self.decoder_errors = decoder.errors

    def stop(self):
        self._stop_event.set()
