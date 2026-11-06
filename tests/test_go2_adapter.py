"""Offline Go2 adapter tests. None need the robot, the Unitree SDK, or DDS."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from go2_perception.adapters.go2.clock import parse_ping_timestamps, resolve_full_offset
from go2_perception.adapters.go2.pointcloud import (
    LidarFrame,
    decode_pointcloud2,
    frame_from_msg,
    level_points,
    to_lidar_message,
)
from go2_perception.adapters.go2.recording import RecordingReader, RecordingWriter
from go2_perception.adapters.go2.rtp_h264 import (
    AccessUnitAssembler,
    H264Decoder,
    packetize_annexb,
    parse_rtp,
    unwrap_rtp,
)
from go2_perception.adapters.go2.stats import StreamStats
from go2_perception.adapters.go2.window import PoseBuffer, SlidingWindowAccumulator
from go2_perception.core.transforms import transform_points, validate_transform
from go2_perception.io.pointcloud_loader import load_pointcloud
from go2_perception.io.sensor_interface import process_lidar
from go2_perception.lidar.processor import LidarProcessor

ROOT = Path(__file__).resolve().parents[1]
REAL = np.load(ROOT / "tests/fixtures/go2/real_cloud_pair.npz")
# Field table measured on rt/utlidar/cloud; offset 12 and 28 are padding.
GO2_FIELDS = [
    SimpleNamespace(name="x", offset=0, datatype=7, count=1),
    SimpleNamespace(name="y", offset=4, datatype=7, count=1),
    SimpleNamespace(name="z", offset=8, datatype=7, count=1),
    SimpleNamespace(name="intensity", offset=16, datatype=7, count=1),
    SimpleNamespace(name="ring", offset=20, datatype=4, count=1),
    SimpleNamespace(name="time", offset=24, datatype=7, count=1),
]


def encode(xyzi, time, ring=1):
    out = bytearray()
    for (x, y, z, i), t in zip(xyzi, time):
        out += np.array([x, y, z, 0], "<f4").tobytes()
        out += np.array([i], "<f4").tobytes()
        out += np.array([ring, 0], "<u2").tobytes()
        out += np.array([t, 0], "<f4").tobytes()
    return bytes(out)


def sdk_message(xyzi, time, frame_id="utlidar_lidar", sec=1770427390, nanosec=500000000):
    return SimpleNamespace(
        data=list(encode(xyzi, time)),  # the SDK delivers sequence[uint8] as a list of ints
        fields=GO2_FIELDS,
        point_step=32,
        width=len(xyzi),
        height=1,
        is_bigendian=False,
        header=SimpleNamespace(
            frame_id=frame_id, stamp=SimpleNamespace(sec=sec, nanosec=nanosec)
        ),
    )


def make_frame(topic="rt/utlidar/cloud_base", frame_id="base_link", n=4):
    return LidarFrame(
        topic,
        100.0,
        frame_id,
        200.0,
        np.arange(n * 3, dtype=np.float32).reshape(-1, 3),
        np.arange(n, dtype=np.float32),
        np.linspace(0, 0.06, n, dtype=np.float32),
    )


def test_decode_real_layout_preserves_every_channel():
    xyzi, time = REAL["cloud_xyzi"], REAL["cloud_time"]
    d = decode_pointcloud2(encode(xyzi, time), GO2_FIELDS, 32, len(xyzi), 1)
    assert np.array_equal(d["x"], xyzi[:, 0])
    assert np.array_equal(d["intensity"], xyzi[:, 3])
    assert np.array_equal(d["time"], time)
    assert np.all(d["ring"] == 1)


def test_decode_big_endian_and_rejects_bad_input():
    fields = [SimpleNamespace(name=k, offset=4 * i, datatype=7, count=1) for i, k in enumerate("xyz")]
    data = np.array([1, 2, 3], ">f4").tobytes()
    assert decode_pointcloud2(data, fields, 12, 1, 1, True)["z"][0] == 3
    with pytest.raises(ValueError, match="bytes"):
        decode_pointcloud2(data[:-1], fields, 12, 1, 1)
    with pytest.raises(ValueError, match="x/y/z"):
        decode_pointcloud2(data, fields[:2], 12, 1, 1)
    with pytest.raises(ValueError, match="datatype"):
        decode_pointcloud2(data, [SimpleNamespace(name="x", offset=0, datatype=99, count=1)], 12, 1, 1)
    with pytest.raises(ValueError, match="count"):
        decode_pointcloud2(data, [SimpleNamespace(name="x", offset=0, datatype=7, count=3)], 12, 1, 1)


def test_empty_cloud_decodes():
    frame = frame_from_msg(sdk_message(np.empty((0, 4)), []), "t", 1.0)
    assert frame.xyz.shape == (0, 3)
    assert frame.acquisition_midpoint == frame.stamp


def test_frame_from_sdk_message_keeps_stamp_frame_and_midpoint():
    frame = frame_from_msg(sdk_message(REAL["cloud_xyzi"], REAL["cloud_time"]), "rt/utlidar/cloud", 5.0)
    assert frame.stamp == pytest.approx(1770427390.5)
    assert frame.frame_id == "utlidar_lidar"
    assert frame.host_recv == 5.0
    expected = 1770427390.5 + REAL["cloud_time"].max() / 2
    assert frame.acquisition_midpoint == pytest.approx(expected, abs=1e-6)
    assert frame.acquisition_midpoint - frame.stamp > 0.01  # float32 time must not collapse stamp


def test_recovered_extrinsic_reproduces_real_cloud_base():
    """Evidence for T_base_lidar: maps raw points exactly onto firmware's cloud_base."""
    t = validate_transform(
        np.array(yaml.safe_load((ROOT / "configs/go2_sensors.yaml").read_text())["T_base_lidar"]),
    )
    lookup = {
        (float(a), float(b)): i
        for i, (a, b) in enumerate(zip(REAL["cloud_time"], REAL["cloud_xyzi"][:, 3]))
    }
    pairs = [
        (lookup[(float(a), float(b))], j)
        for j, (a, b) in enumerate(zip(REAL["base_time"], REAL["base_xyzi"][:, 3]))
        if (float(a), float(b)) in lookup
    ]
    assert len(pairs) == len(REAL["base_xyzi"])
    raw, base = zip(*pairs)
    mapped = transform_points(REAL["cloud_xyzi"][list(raw), :3].astype(float), t)
    assert np.abs(mapped - REAL["base_xyzi"][list(base), :3]).max() < 1e-5


def test_lidar_message_rejects_non_body_frames():
    with pytest.raises(ValueError, match="cloud_base"):
        to_lidar_message(make_frame("rt/utlidar/cloud", "utlidar_lidar"))
    with pytest.raises(ValueError, match="cloud_base"):
        to_lidar_message(make_frame("rt/utlidar/cloud_deskewed", "odom"))


def test_lidar_message_contract_and_levelling():
    frame = make_frame()
    message = to_lidar_message(frame)
    assert message.points.shape == (4, 4)
    assert message.frame_id == "base_link"
    assert message.clock_domain == "go2_robot"
    assert message.timestamp == pytest.approx(100.03)
    roll = np.deg2rad(10)
    q = [np.sin(roll / 2), 0, 0, np.cos(roll / 2)]
    assert to_lidar_message(frame, q).frame_id == "base_link_level"
    # A horizontal world point seen by a rolled body returns to horizontal once levelled.
    body = np.array([[0, np.cos(roll), -np.sin(roll)]])
    np.testing.assert_allclose(level_points(body, q), [[0, 1, 0]], atol=1e-9)
    yaw = np.deg2rad(40)  # levelling must keep heading rather than rotating to odom
    np.testing.assert_allclose(
        level_points([[1, 0, 0]], [0, 0, np.sin(yaw / 2), np.cos(yaw / 2)]), [[1, 0, 0]], atol=1e-9
    )


def test_real_frame_runs_through_unchanged_processor():
    frame = LidarFrame(
        "rt/utlidar/cloud_base",
        float(REAL["stamp"]),
        "base_link",
        0.0,
        REAL["base_xyzi"][:, :3],
        REAL["base_xyzi"][:, 3],
        REAL["base_time"],
    )
    config = yaml.safe_load((ROOT / "configs/go2_lidar.yaml").read_text())
    result = process_lidar(LidarProcessor(**config), to_lidar_message(frame))
    assert result.frame_id == "base_link"
    assert result.clock_domain == "go2_robot"
    assert len(result.ground) > 50  # the floor was in view, at a standing height
    assert abs(np.median(result.ground[:, 2]) + 0.33) < 0.05


def test_rtp_parser_handles_extensions_padding_and_garbage():
    assert parse_rtp(b"") is None
    assert parse_rtp(b"\x00" * 20) is None
    header = bytes([0x80 | 0x20 | 0x10, 0xE0]) + (5).to_bytes(2, "big") + (7).to_bytes(4, "big")
    header += (9).to_bytes(4, "big") + b"\xbe\xde\x00\x01" + b"\0" * 4
    packet = parse_rtp(header + b"\x41\xAA" + b"\0\0\x03")
    assert (packet.seq, packet.timestamp, packet.marker, packet.payload_type) == (5, 7, True, 96)
    assert packet.payload == b"\x41\xAA"


def assemble(au, ts, seq=0, drop=()):
    assembler = AccessUnitAssembler()
    units = []
    for i, datagram in enumerate(packetize_annexb(au, ts, seq, mtu=50)):
        if i not in drop:
            units += assembler.feed(parse_rtp(datagram), 1.0 + i)
    return assembler, units


def test_fragmentation_round_trip_and_sequence_wrap():
    sps = b"\0\0\0\1" + bytes([0x67]) + b"\x42" * 10
    idr = b"\0\0\0\1" + bytes([0x65]) + bytes(range(1, 200))
    assembler, units = assemble(sps + idr, 1234, seq=65534)
    assert len(units) == 1 and units[0].data == sps + idr
    assert units[0].complete and units[0].keyframe and not units[0].parameter_sets_only
    assert assembler.lost_packets == 0
    assert units[0].host_recv_first == 1.0


def test_lost_packet_is_counted_and_marks_unit_incomplete():
    idr = b"\0\0\0\1" + bytes([0x65]) + bytes(range(1, 200))
    assembler, units = assemble(idr, 10, drop={2})
    assert assembler.lost_packets == 1
    assert len(units) == 1 and not units[0].complete


def test_loss_between_units_and_parameter_only_unit():
    a = AccessUnitAssembler()
    sps = b"\0\0\0\1" + bytes([0x67, 1, 2])
    p = b"\0\0\0\1" + bytes([0x41, 9, 9])
    assert a.feed(parse_rtp(packetize_annexb(sps, 1, 0)[0]), 1.0)[0].parameter_sets_only
    units = a.feed(parse_rtp(packetize_annexb(p, 2, 5)[0]), 2.0)
    assert a.lost_packets == 4
    assert not units[0].complete


def test_h264_decode_keeps_rtp_timestamps():
    av = pytest.importorskip("av")
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width, encoder.height, encoder.pix_fmt = 64, 48, "yuv420p"
    encoder.options = {"tune": "zerolatency", "bframes": "0", "repeat-headers": "1"}
    encoder.time_base = av.Fraction(1, 15) if hasattr(av, "Fraction") else __import__("fractions").Fraction(1, 15)
    decoder, assembler, decoded, seq = H264Decoder(), AccessUnitAssembler(), [], 0
    for i in range(5):
        frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 20 * i, np.uint8), format="bgr24")
        frame.pts = i
        for packet in encoder.encode(frame):
            datagrams = packetize_annexb(bytes(packet), i * 6000, seq, mtu=400)
            seq += len(datagrams)
            for datagram in datagrams:
                for unit in assembler.feed(parse_rtp(datagram), float(i)):
                    decoded += decoder.decode(unit)
    assert decoder.errors == 0 and decoded
    image, rtp = decoded[0]
    assert image.shape == (48, 64, 3) and image.dtype == np.uint8
    assert rtp % 6000 == 0


def test_rtp_unwrap():
    assert unwrap_rtp(None, 5) == 5
    assert unwrap_rtp(0xFFFFFFF0, 0x10) == 0x100000010
    assert unwrap_rtp(0x100000010, 0xFFFFFFF0) == 0xFFFFFFF0


def test_clock_parsing_uses_measured_sample_shape():
    text = (
        "64 bytes from 192.168.123.161: icmp_seq=1 ttl=64 time=0.153 ms\n"
        "TS: \t72843026 absolute\n\t-67593043\n\t0\n\t67593043\n\n"
    )
    assert parse_ping_timestamps(text) == [67593.043]
    assert resolve_full_offset(67593.043, 23568393.0425) == pytest.approx(23568393.043)


def test_stream_stats_reports_gaps_and_nonmonotonic_stamps():
    stats = StreamStats("s")
    for i, stamp in enumerate([0.0, 0.1, 0.2, 0.5, 0.6, 0.6]):
        stats.add(float(i), stamp, 100)
    summary = stats.summary()
    assert summary["estimated_missed_messages"] == 2
    assert summary["nonincreasing_stamps"] == 1
    assert summary["size"]["mean"] == 100
    assert StreamStats("empty").summary() == {"name": "empty", "count": 0}


def pose_buffer():
    poses = PoseBuffer(max_age_s=1e6, max_gap_s=0.1)
    for t in np.arange(0, 1.01, 0.05):
        poses.add(float(t), [float(t), 0, 0], [0, 0, 0, 1])  # moving at 1 m/s along +x
    return poses


def test_pose_lookup_refuses_extrapolation_and_large_gaps():
    poses = pose_buffer()
    assert poses.lookup(0.525)[0, 3] == pytest.approx(0.525)
    assert poses.lookup(1.5) is None and poses.lookup(-0.1) is None
    sparse = PoseBuffer(max_gap_s=0.1)
    sparse.add(0.0, [0, 0, 0], [0, 0, 0, 1])
    sparse.add(1.0, [1, 0, 0], [0, 0, 0, 1])
    assert sparse.lookup(0.5) is None


def test_window_compensates_ego_motion_and_ages_out():
    accumulator = SlidingWindowAccumulator(pose_buffer(), window_s=0.3)
    # A static world point at x=2 appears nearer in body coordinates as the robot advances.
    for t in (0.0, 0.1, 0.2, 0.3):
        assert accumulator.add([[2.0 - t, 0.0, 0.0]], t + 1e-9 if t == 0 else t)
    merged, _, used, span = accumulator.merged()
    assert used == 4 and span == pytest.approx(0.3)
    np.testing.assert_allclose(merged, np.tile([1.7, 0, 0], (4, 1)), atol=1e-6)
    assert accumulator.add([[1.0, 0, 0]], 0.7)
    assert accumulator.merged()[2] == 1


def test_window_refuses_unposed_frame_and_keeps_raw_intact():
    accumulator = SlidingWindowAccumulator(pose_buffer(), window_s=0.3)
    raw = np.array([[1.0, 2.0, 3.0]])
    assert not accumulator.add(raw, 99.0)
    assert accumulator.refused_no_pose == 1
    assert accumulator.merged()[2] == 0
    assert accumulator.add(raw, 0.5)
    accumulator.merged()
    np.testing.assert_array_equal(raw, [[1.0, 2.0, 3.0]])


def test_recording_round_trip(tmp_path):
    root = tmp_path / "rec"
    writer = RecordingWriter(root, notes="unit")
    image = np.zeros((48, 64, 3), np.uint8)
    image[:, 32:] = 255
    writer.write_rgb(image, 6000, 1.0, 1.1)
    frame = make_frame()
    frame.ring = np.ones(4, np.uint16)
    writer.write_lidar(frame)
    deskewed = LidarFrame("rt/utlidar/cloud_deskewed", 2.0, "odom", 3.0, frame.xyz)
    writer.write_lidar(deskewed)
    writer.write_state("rt/utlidar/robot_odom", {"stamp": 1.5, "q_xyzw": [0, 0, 0, 1]})
    writer.close({"ok": True})

    reader = RecordingReader(root)
    assert reader.manifest["rgb_frames"] == 1
    assert reader.manifest["lidar_frames"]["utlidar_cloud_base"] == 1
    record = reader.rgb_records()[0]
    assert record["rtp_timestamp"] == 6000 and record["width"] == 64
    assert reader.rgb_path(record).is_file()
    assert reader.state("rt/utlidar/robot_odom")[0]["stamp"] == 1.5
    replay = reader.lidar_frames("rt/utlidar/cloud_base")[0]
    loaded = replay.load()
    assert loaded.stamp == 100.0 and loaded.frame_id == "base_link"
    np.testing.assert_array_equal(loaded.xyz, frame.xyz)
    np.testing.assert_array_equal(loaded.intensity, frame.intensity)
    np.testing.assert_array_equal(loaded.time, frame.time)
    assert replay.record["ring_values"] == [1]
    assert load_pointcloud(replay.path).shape == (4, 4)  # readable by the Phase 1 loader
    bare = reader.lidar_frames("rt/utlidar/cloud_deskewed")[0].load()
    assert bare.intensity is None and bare.time is None
    with pytest.raises(FileExistsError):
        RecordingWriter(root)


def test_recording_rejects_unknown_schema(tmp_path):
    (tmp_path / "manifest.json").write_text('{"schema": "other"}')
    with pytest.raises(ValueError, match="schema"):
        RecordingReader(tmp_path)


def test_adapter_import_does_not_pull_in_transport():
    import subprocess
    import sys

    code = (
        "import sys, go2_perception.adapters.go2.pointcloud, "
        "go2_perception.adapters.go2.recording, go2_perception.adapters.go2.window; "
        "assert 'unitree_sdk2py' not in sys.modules and 'cyclonedds' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_adapter_source_cannot_publish_to_the_robot():
    folder = ROOT / "src/go2_perception/adapters/go2"
    source = "\n".join(p.read_text() for p in folder.glob("*.py"))
    code = [line for line in source.splitlines() if "import" in line or "(" in line]
    joined = "\n".join(code)
    for forbidden in ("ChannelPublisher", "SportClient", "DataWriter(", "VideoClient", ".sendto("):
        assert forbidden not in joined


def test_camera_receiver_rejoins_when_stream_is_silent_and_stops_cleanly():
    """Silence must trigger a rejoin, not a crash. A real cable pull is a separate test."""
    import time

    pytest.importorskip("av")
    from go2_perception.adapters.go2.receivers import Go2CameraReceiver

    receiver = Go2CameraReceiver(iface_ip="127.0.0.1", port=47999, stale_after_s=0.1)
    receiver.start()
    time.sleep(1.5)
    receiver.stop()
    receiver.join(timeout=3)
    assert not receiver.is_alive()
    assert receiver.exception is None
    assert receiver.frames == 0
