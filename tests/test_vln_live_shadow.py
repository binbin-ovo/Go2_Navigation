"""Live read-only camera bridge -> Go2VLNObservation -> ObservationInbox, mocked.

No socket to the robot, DDS, model or hardware: a fake bridge child speaks the same
stdout protocol. Covers conversion, clocks, gaps, overflow, interruption, shutdown.
"""

import ast
import io
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
import time

import cv2
import numpy as np
import pytest

from go2_perception.vln.calibration import load_calibration
from go2_perception.vln.model_preprocessing import calibrated_view
from go2_perception.vln.observation import load, validate
from go2_perception.vln.shadow import LIVE_BANNER, Limits, ShadowSession
from go2_perception.vln.shadow_live import (
    LIVE_SOURCE_ID,
    FrameError,
    LiveCameraSource,
    frame_observation,
    latency_model,
)
from go2_perception.vln.shadow_sources import Delivery
from go2_perception.vln.shadow_worker import MockWorker

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "configs/go2_front_camera_calibration.yaml"
FAKE = ROOT / "tests/fixtures/vln_live/fake_bridge.py"
LATENCY = latency_model({"value": 0.048, "sigma": 0.016, "source": "test"})


@pytest.fixture(scope="module")
def intr():
    return load_calibration(CALIBRATION)[0]


def header(seq, wall, mono, *, shape=(720, 1280, 3), bridge="b1"):
    return {
        "type": "frame",
        "bridge_id": bridge,
        "seq": seq,
        "decoded_index": 2 * seq,
        "rtp_timestamp": 9000 * seq,
        "complete": True,
        "host_recv_first_wall_s": wall - 0.012,
        "host_recv_last_wall_s": wall - 0.010,
        "decoded_wall_s": wall,
        "decoded_monotonic_s": mono,
        "wall_minus_monotonic_s": wall - mono,
        "conversion_uncertainty_s": 1e-6,
        "shape": list(shape),
        "dtype": "uint8",
        "channel_order": "RGB",
        "nbytes": int(np.prod(shape)),
    }


def rgb_payload(seq):
    image = np.zeros((720, 1280, 3), np.uint8)
    image[..., 0] = 200
    image[:, :640, 2] = seq % 256
    return image, image.tobytes()


def convert(intr, seq, wall, mono, latency=LATENCY, epoch="host_receipt_wall:bridge=b1:g0"):
    image, payload = rgb_payload(seq)
    obs, offer = frame_observation(
        header(seq, wall, mono), payload, intr, {"intrinsics_status": "measured"}, epoch, latency
    )
    return image, obs, offer


def test_conversion_preserves_rgb_size_calibration_and_unknowns(intr):
    image, obs, offer = convert(intr, 3, 1000.5, 50.5)
    v = obs.visual
    assert v.image.shape == (720, 1280, 3) and np.array_equal(v.image, image)
    assert v.image[0, 0, 0] == 200  # channel order kept RGB, no second swap
    assert not v.rectified and v.intrinsics_status == "measured"
    assert v.distortion_model == "fisheye_equidistant" and v.intrinsics_size == (1280, 720)
    np.testing.assert_array_equal(v.K, intr.K)
    np.testing.assert_array_equal(v.D, intr.D)
    assert v.frame_id == "go2_front_camera_optical" and v.history == []
    assert v.stamp_host.clock == "host_receipt" and v.stamp_host.semantics == "host_receipt"
    assert v.stamp_robot is None  # no robot clock; host time is never relabelled
    assert obs.robot is None and obs.geometric is None and obs.host_minus_robot_s is None
    assert validate(obs) == []
    p = offer["provenance"]
    assert p["pose"] is None and p["lidar"] is None and p["T_base_camera"] is None
    assert offer["available_s"] == 1000.5 >= v.stamp_host.t
    assert offer["received_monotonic_s"] == pytest.approx(50.5 - 0.012)
    assert offer["exposure_monotonic_s"] == pytest.approx(50.5 - 0.012 - 0.048)
    assert "0.048" in offer["monotonic_mapping"] and "sigma 0.016" in offer["monotonic_mapping"]
    assert offer["source_id"] == LIVE_SOURCE_ID and offer["frame_id"] == "b1:00000003:rtp27000"
    Delivery(obs, **offer)  # the existing boundary accepts it


def test_unknown_latency_leaves_exposure_unknown_not_zero(intr):
    _, _, offer = convert(intr, 1, 1000.0, 50.0, latency=None)
    assert offer["exposure_monotonic_s"] is None and offer["monotonic_mapping"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"shape": [360, 640, 3], "nbytes": 360 * 640 * 3},
        {"channel_order": "BGR"},
        {"decoded_wall_s": 0.0},
        {"host_recv_first_wall_s": float("nan")},
    ],
)
def test_bad_frames_rejected(intr, change):
    _, payload = rgb_payload(1)
    h = {**header(1, 1000.0, 50.0), **change}
    if "shape" in change:
        payload = np.zeros(change["shape"], np.uint8).tobytes()
    with pytest.raises(FrameError):
        frame_observation(h, payload, intr, {}, "e", LATENCY)


def live_session(tmp_path, delay=0.2, **limits):
    s = ShadowSession(
        tmp_path / "run", MockWorker(delay), mode="live_unvalidated",
        limits=Limits(**limits), session_id="live",
    )
    s.instruction_changed("Approach the doorway and stop before entering.", 0.0)
    return s


def deliveries(intr, n, *, start_wall=1000.0, start_mono=50.0, dt=1 / 7, epoch=None):
    out = []
    for i in range(n):
        _, obs, offer = convert(
            intr, i + 1, start_wall + i * dt, start_mono + i * dt,
            **({} if epoch is None else {"epoch": epoch}),
        )
        out.append(Delivery(obs, **offer))
    return out


def test_single_rectification_and_live_age_in_session(tmp_path, intr):
    s = live_session(tmp_path)
    items = deliveries(intr, 8)
    for i, d in enumerate(items):
        s.receive(d, 50.0 + i / 7 + 0.05)
    s.tick(51.2)
    request = s.inflight
    job = request.job
    raw = load(request.context["history"][-1]["source"]["observation_path"])
    assert not raw.visual.rectified  # stored raw; rectified exactly once for the job
    once = calibrated_view(raw).image
    prepared = cv2.cvtColor(cv2.imread(job["frames"][-1]["path"]), cv2.COLOR_BGR2RGB)
    assert np.array_equal(prepared, once)
    assert request.context["history"][-1]["rectification"]["view"]["method"].startswith("opencv_fisheye")
    assert request.context["current_source"]["T_base_camera"] is None
    assert set(job) == {"instruction", "episode_id", "clock_epoch", "decision_id", "decision_time_s", "frames"}
    s.tick(51.5)
    timing = s.displayed[1]["display_timing"]
    assert timing["age_basis"] == "caller_supplied_exposure_monotonic_mapping_unvalidated_live"
    exposure = items[-1].exposure_monotonic_s
    assert timing["observation_age_at_display_s"] == pytest.approx(51.5 - exposure)
    assert s.status(51.5)["banner"] == LIVE_BANNER


def test_clock_epoch_change_resets_history(tmp_path, intr):
    s = live_session(tmp_path)
    for i, d in enumerate(deliveries(intr, 4)):
        s.receive(d, 50.0 + i)
    gen = s.history_generation
    later = deliveries(intr, 2, start_wall=1004.0, start_mono=54.0, epoch="host_receipt_wall:bridge=b1:g1")
    s.receive(later[0], 55.0)
    assert s.history_generation == gen + 1 and len(s.history) == 1
    events = [json.loads(x) for x in (s.output / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "source_epoch_reset" for e in events)


def test_gap_pauses_without_padding(tmp_path, intr):
    s = live_session(tmp_path)
    items = deliveries(intr, 3) + deliveries(intr, 2, start_wall=1001.0, start_mono=51.0)
    for i, d in enumerate(items):
        s.receive(d, 52.0 + i * 0.01)
    assert s.history_error == "observation_gap_requires_new_instruction"
    assert len(s.history) == 3  # nothing filled in
    s.tick(53.0)
    assert s.inflight is None


def bridge_wrapper(tmp_path, plan):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    wrapper = tmp_path / "fakepython"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$(cat "{plan_path}")"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    return wrapper


def source(tmp_path, intr, plan, **kw):
    return LiveCameraSource(
        python=bridge_wrapper(tmp_path, plan), intrinsics=intr,
        calibration_path=CALIBRATION, output=tmp_path / "camera", latency=LATENCY, **kw,
    )


def wait_for(predicate, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_bridge_overflow_gaps_epoch_and_clean_shutdown(tmp_path, intr):
    frames = [{"dt": 0.02} for _ in range(6)]
    frames[3]["seq"] = 6  # bridge-side drop of seq 4 and 5
    frames[4] = {"dt": 0.02, "seq": 7, "offset_jump": 0.5}  # host wall clock step
    frames[5] = {"dt": 0.02, "seq": 8}
    cam = source(tmp_path, intr, {"frames": frames}, capacity=2)
    cam.start()
    assert wait_for(lambda: cam.counts["frames_read"] == 6)
    assert cam.counts["offered"] == 2 and cam.counts["inbox_overflow"] == 4  # never silent
    assert cam.counts["bridge_seq_gaps"] == 1 and cam.counts["frames_missing_in_seq_gaps"] == 2
    assert cam.counts["clock_epoch_changes"] == 1 and cam.epoch().endswith(":g1")
    kinds = [e["event"] for e in cam.drain_events()]
    assert {"camera_bridge_hello", "camera_inbox_overflow", "camera_bridge_frames_dropped",
            "camera_clock_epoch_changed"} <= set(kinds)
    first = cam.get_nowait()
    assert first.frame_id.startswith("fakebridge01:00000001") and first.observation.visual.image[1, 1, 0] == 200
    cam.close()
    assert cam.process.returncode == 0 and not cam.reader.is_alive()
    assert cam.bye is not None and cam.inbox.queue.empty()


def test_hung_bridge_is_terminated(tmp_path, intr):
    cam = source(tmp_path, intr, {"frames": [], "ignore_eof": True})
    cam.start()
    assert wait_for(lambda: cam.hello is not None)
    started = time.monotonic()
    cam.close(timeout=0.5)
    assert cam.process.returncode not in (None, 0) and time.monotonic() - started < 3
    assert not cam.reader.is_alive()


def test_bridge_stride_rgb_and_bounded_queue():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import go2_camera_bridge as bridge_module
    finally:
        sys.path.pop(0)
    out = io.BytesIO()
    bridge = bridge_module.Bridge(out, stride=2, capacity=1)
    bgr = np.zeros((4, 6, 3), np.uint8)
    bgr[..., 0] = 255  # blue in BGR
    for i in range(5):
        bridge.on_frame(SimpleNamespace(image=bgr, rtp_timestamp=i, host_recv_first=1.0 + i,
                                        host_recv_last=1.001 + i, complete=True))
    assert (bridge.decoded, bridge.forwarded, bridge.decimated, bridge.dropped) == (5, 3, 2, 2)
    header, image = bridge.queue.get_nowait()
    assert header["channel_order"] == "RGB" and image[0, 0].tolist() == [0, 0, 255]
    assert header["seq"] == 1 and header["decoded_index"] == 1


def test_live_entrypoint_interruption_latency_and_shutdown(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    frames = [{"dt": 0.1} for _ in range(30)]
    frames[14]["pause_after_s"] = 1.6  # camera interruption, then resumes
    wrapper = bridge_wrapper(tmp_path, {"frames": frames})
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import run_vln_shadow
    finally:
        sys.path.pop(0)
    out = tmp_path / "live_run"
    code = run_vln_shadow.main([
        "--go2-camera", "--hardware-python", str(wrapper),
        "--instruction", "Approach the doorway and stop before entering.",
        "--backend", "mock", "--mock-delay-s", "0.2", "--warmup-seconds", "0.5",
        "--max-requests", "64", "--max-host-seconds", "6", "--output", str(out),
    ])
    events = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert "camera_bridge_started" in kinds and "camera_interrupted" in kinds
    assert any(e["event"] == "history_paused" and e["reason"] == "camera_interrupted" for e in events)
    assert "camera_bridge_closed" in kinds and kinds[-1] == "session_closed"
    assert code == 1  # history error reported, not hidden
    summary = json.loads((out / "summary.json").read_text())
    assert summary["robot_commands_sent"] == 0 and summary["camera"]["camera_state"] == "closed"
    assert summary["camera"]["bridge_alive"] is False
    assert summary["counts"]["accepted"] >= 1
    rows = [json.loads(x) for x in (out / "latency.jsonl").read_text().splitlines()]
    assert rows and all(r["sensor_receipt_to_display_s"] > 0 for r in rows)
    assert all(r["estimated_exposure_to_sensor_receipt_s"] == pytest.approx(0.048) for r in rows)
    page = (out / "index.html").read_text()
    assert LIVE_BANNER in page and "does not stop the robot" in page
    # Observations after the interruption are not used for inference in this revision.
    assert summary["history_error"] == "camera_interrupted"


def test_live_path_has_no_control_route():
    paths = [ROOT / "src/go2_perception/vln/shadow_live.py", ROOT / "scripts/go2_camera_bridge.py",
             ROOT / "scripts/run_vln_shadow.py", ROOT / "src/go2_perception/adapters/go2/receivers.py"]
    forbidden_imports = ("unitree", "go2_control", "go2_hl_control", "cyclonedds", "rclpy")
    forbidden_names = {"UseRemoteCommandFromApi", "SportClient", "ObstaclesAvoidClient",
                       "SwitchSet", "StandDown", "StopMove", "Move", "ChannelPublisher",
                       "Publisher", "DataWriter", "sendto", "send", "Request"}
    for path in paths:
        names, identifiers = [], set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names += [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
                identifiers |= {x.name for x in node.names}
            elif isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
        if path.name != "receivers.py":  # its DDS reader is unused by the bridge
            assert not any(w in n for w in forbidden_imports for n in names), path
        assert not identifiers & forbidden_names, (path, identifiers & forbidden_names)
    bridge = ast.parse((ROOT / "scripts/go2_camera_bridge.py").read_text())
    imported = {a.name for n in ast.walk(bridge) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert imported == {"Go2CameraReceiver", "Empty", "Full", "Queue"}
    assert os.environ.get("GO2_HARDWARE", "0") in ("0", "")  # no physical opt-in here


def test_live_paused_history_and_observation_budget_end_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import run_vln_shadow
    finally:
        sys.path.pop(0)
    frames = [{"dt": 0.1} for _ in range(40)]
    frames[3]["pause_after_s"] = 1.3
    base = ["--go2-camera", "--instruction", "Approach the doorway.", "--backend", "mock",
            "--max-host-seconds", "8"]
    run_vln_shadow.main([*base, "--hardware-python", str(bridge_wrapper(tmp_path, {"frames": frames})),
                         "--paused-exit-s", "0.5", "--output", str(tmp_path / "paused")])
    summary = json.loads((tmp_path / "paused/summary.json").read_text())
    assert summary["termination"] == "history_paused"
    (tmp_path / "b").mkdir()
    run_vln_shadow.main([*base, "--hardware-python", str(bridge_wrapper(tmp_path / "b", {"frames": frames})),
                         "--max-observations", "5", "--output", str(tmp_path / "budget")])
    summary = json.loads((tmp_path / "budget/summary.json").read_text())
    assert summary["termination"] == "observation_budget" and summary["counts"]["received"] == 5


def test_slow_wall_clock_slew_keeps_the_epoch_until_its_limit(tmp_path, intr):
    # 2026-11-22: NTP slew of +20 ms over minutes ended an episode. Steps between frames still
    # start a new epoch, and so does cumulative slew beyond SLEW_LIMIT_S.
    from go2_perception.vln.shadow_live import SLEW_LIMIT_S
    frames = [{"dt": 0.02, "offset_jump": 0.015 if i else 0.0} for i in range(9)]
    cam = source(tmp_path, intr, {"frames": frames}, capacity=16)
    cam.start()
    assert wait_for(lambda: cam.counts["frames_read"] == 9)
    # 8 x 15 ms = 120 ms: each step is below the 20 ms tolerance; only the slew limit is crossed.
    assert 0.015 * 6 <= SLEW_LIMIT_S < 0.015 * 8
    assert cam.counts["clock_epoch_changes"] == 1 and cam.epoch().endswith(":g1")
    cam.close()
