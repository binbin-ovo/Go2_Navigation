"""Interactive four-panel review tool: synthetic fixtures, injected clocks, mock worker.

No model, GPU, hardware, socket to the robot or DDS. Covers trial/target changes,
cancellation and late results, stale labels, panel B image/point binding, click mapping,
missing sensors, clock epochs, bounded recording, limits, mark/note binding, the review
dialog, the loopback server and the absence of any motion-command route.
"""

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from go2_perception.vln.interactive_app import (InteractiveApp, PointingMockWorker, TapInbox,
                                                label_for)
from go2_perception.vln.interactive_lidar import (LidarRecorder, LidarSource, LidarState,
                                                  coverage)
from go2_perception.vln.interactive_server import DisplayServer
from go2_perception.vln.interactive_trials import (CHOICES, TARGET_TEMPLATE, RunStore,
                                                   as_instruction, correction_record, expand_target,
                                                   load_review, pixel_sha256, read_jsonl,
                                                   write_new_json)
from go2_perception.vln.shadow import Limits, ShadowSession
from go2_perception.vln.shadow_evidence import SkippedTensorEvidence
from go2_perception.vln.shadow_sources import SyntheticSource
from go2_perception.vln.shadow_worker import MockWorker, ProcessWorker

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SYNTH = ROOT / "scripts/synthetic_lidar_bridge.py"


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeCamera:
    def __init__(self, tap):
        self.tap = tap
        self.alive = True

    def get_nowait(self):
        return self.tap.get_nowait()

    def drain_events(self):
        return []


def make_app(tmp_path, *, delay=1.0, worker=None, history_s=20.0, **config):
    clock = Clock()
    store = RunStore(tmp_path / "run")
    session = ShadowSession(store.root / "session", worker or MockWorker(delay), mode="synthetic_fast",
                            limits=Limits(history_s=history_s), allow_synthetic=True,
                            preparation_mode="cached")
    tap = TapInbox(16, monotonic=clock)
    messages = []
    cfg = {"camera_silence_s": 1.0, "warmup_s": 0.5, "max_requests_per_trial": 6, "trial_max_s": 60,
           "max_trials": 20, "reviewer": "tester", **config}
    app = InteractiveApp(store=store, session=session, camera=FakeCamera(tap), tap=tap,
                         mode="SYNTHETIC", config=cfg, run_info={"backend": "mock"},
                         say=messages.append, monotonic=clock)
    frames = iter(SyntheticSource(4000, fps=8.0))

    def feed(n, dt=0.125, step=True):
        for _ in range(n):
            d = next(frames)
            tap.offer(d.observation, source_id=d.source_id, frame_id=d.frame_id,
                      available_s=d.available_s, provenance=d.provenance, received_monotonic_s=clock())
            clock.t += dt
            if step:
                app.step(clock())

    return app, clock, feed, messages


def predictions(app, trial_id):
    return read_jsonl(app.store.root / "trials" / trial_id / "predictions.jsonl")


# ---------- trials, template, target change, cancel ----------

def test_template_expansion_is_documented_and_saved_before_inference(tmp_path):
    spec = expand_target("  red   door ")
    assert spec["instruction"] == TARGET_TEMPLATE.format(target="red door") == "Go to the red door and stop."
    assert spec["operator_text"] == "red door" and spec["template_id"]
    assert as_instruction("Turn left.")["template_id"] is None
    with pytest.raises(ValueError):
        expand_target("   ")
    app, clock, feed, _ = make_app(tmp_path)
    seen = {}
    original = app.session.instruction_changed

    def check(instruction, now):
        folder = next((app.store.root / "trials").glob("T*"))
        seen["record"] = json.loads((folder / "trial.json").read_text())
        return original(instruction, now)

    app.session.instruction_changed = check
    trial = app.start_trial(spec, clock())
    assert seen["record"]["instruction"] == "Go to the red door and stop."
    assert seen["record"]["operator_text_is_verified_label"] is False
    assert seen["record"]["temporal_context"]["bounded_context_policy_enabled"] is False
    assert app.revision_trial[app.session.revision] == trial.id


def test_target_change_keeps_late_result_with_original_trial(tmp_path):
    app, clock, feed, messages = make_app(tmp_path, delay=2.0)
    first = app.start_trial(expand_target("door"), clock())
    feed(8)  # warm-up passes, a request goes in flight
    assert app.session.inflight is not None
    second = app.start_trial(expand_target("fridge"), clock())
    assert first.id != second.id
    feed(40)  # first trial's late result arrives, then the new trial's own results
    late = predictions(app, first.id)
    assert late and late[0]["label_at_receipt"] == "SUPERSEDED"
    assert late[0]["rejection_reason_at_receipt"] in ("superseded_instruction", "cancelled_or_closed")
    assert late[0]["trial_id"] == first.id and late[0]["prediction_id"].startswith(first.id)
    own = predictions(app, second.id)
    assert own and all(p["trial_id"] == second.id for p in own)
    assert any("Late result" in m for m in messages)
    assert json.loads((app.store.root / "trials" / first.id / "trial_end.json").read_text())[
        "end_reason"] == "superseded_by_new_trial"


def test_end_cancels_advisory_only_and_late_result_is_superseded(tmp_path):
    app, clock, feed, messages = make_app(tmp_path, delay=2.0)
    trial = app.start_trial(expand_target("chair"), clock())
    feed(8)
    assert app.session.inflight is not None
    app.handle("end", clock(), {}, clock())
    assert any("no robot command was sent" in m for m in messages)
    end = json.loads((app.store.root / "trials" / trial.id / "trial_end.json").read_text())
    assert end["robot_commands_sent"] == 0 and end["inflight_at_end"] is True
    assert app.dialog is not None  # review starts right after the trial
    feed(20)  # late result arrives while reviewing
    rows = predictions(app, trial.id)
    assert rows[-1]["label_at_receipt"] == "SUPERSEDED"
    assert rows[-1]["rejection_reason_at_receipt"] == "cancelled_or_closed"
    events = read_jsonl(app.session.events_path)
    assert any(e["event"] == "advisory_cancelled" and e["physical_robot_stop_sent"] is False for e in events)


def test_stale_label_and_point_bound_to_its_own_image(tmp_path):
    app, clock, feed, _ = make_app(tmp_path, worker=PointingMockWorker(0.5))
    app.start_trial(expand_target("box"), clock())
    feed(12)
    current = app._current_prediction(clock())
    assert current["label"] == "CURRENT"
    assert current["image_url"] == f"/api/pred_image?id={current['prediction_id']}"
    assert app.prediction_image_path(current["prediction_id"]).is_file()
    # The image served for this prediction is the one recorded with it (same hash).
    from go2_perception.vln.episodes import file_sha256
    assert file_sha256(app.prediction_image_path(current["prediction_id"])) == current["source_image"]["file_sha256"]
    clock.t += 3.5  # nothing new arrives: the same proposal ages past the unchanged 3 s limit
    aged = app._current_prediction(clock())
    assert aged["prediction_id"] == current["prediction_id"] and aged["label"] == "STALE"
    assert label_for("superseded_instruction") == "SUPERSEDED" and label_for(None) == "CURRENT"
    # Page contract: points are drawn only together with the image of the same record.
    from go2_perception.vln.interactive_page import PAGE
    assert "drawPred(snap,img)" in PAGE and "target_fraction" not in PAGE.split("function panelA")[1].split("function drawPred")[0]


def test_opos_zero_is_explicit_not_visible(tmp_path):
    app, clock, feed, _ = make_app(tmp_path, delay=0.5)  # canned MockWorker emits <opos_0>
    trial = app.start_trial(expand_target("cup"), clock())
    feed(12)
    rows = predictions(app, trial.id)
    assert rows and all(r["opos_not_visible"] and r["target_fraction"] is None for r in rows)
    assert all(r["execution_status"].startswith("NOT EXECUTED") and r["waypoint_time_base"] is None for r in rows)
    assert all(r["completion_verified"] is False and r["model_output_not_a_label"] for r in rows)


# ---------- limits ----------

def test_request_budget_trial_time_limit_and_trial_budget(tmp_path):
    app, clock, feed, messages = make_app(tmp_path, delay=0.2, max_requests_per_trial=2, trial_max_s=6,
                                          max_trials=2)
    trial = app.start_trial(expand_target("a"), clock())
    feed(40)  # 5 s
    assert app.session.counts["submitted"] == 2
    feed(16)
    assert app.trial is None
    assert json.loads((app.store.root / "trials" / trial.id / "trial_end.json").read_text())["end_reason"] == "trial_time_limit"
    app.dialog = None
    app.start_trial(expand_target("b"), clock())
    assert app.start_trial(expand_target("c"), clock()) is None
    assert any("Trial budget reached" in m for m in messages)


def test_resource_limit_stops_new_trials_and_keeps_records(tmp_path):
    class LowMonitor:
        def snapshot(self):
            return {"low": ["free disk 1.0 GiB below limit"]}

    app, clock, feed, messages = make_app(tmp_path, delay=0.2)
    trial = app.start_trial(expand_target("a"), clock())
    feed(10)
    app.monitor = LowMonitor()
    feed(1)
    assert app.trial is None and app.resource_stop
    assert (app.store.root / "trials" / trial.id / "trial.json").exists()
    assert (app.store.root / "trials" / trial.id / "trial_end.json").exists()
    app.dialog = None
    assert app.start_trial(expand_target("b"), clock()) is None
    assert any("RESOURCE LIMIT" in m for m in messages)


def test_camera_interruption_pauses_history_and_stops_archiving(tmp_path):
    app, clock, feed, messages = make_app(tmp_path, delay=0.5)
    app.start_trial(expand_target("a"), clock())
    feed(6)
    clock.t += 1.6  # silence > 1 s
    app.step(clock())
    assert app.tap.status(clock(), 1.0)["state"] == "INTERRUPTED"
    assert app.session.history_error == "camera_interrupted"
    received = app.session.counts["received"]
    feed(5)
    assert app.feeding_paused == "camera_interrupted"
    assert app.session.counts["received"] == received  # frames are not re-fed or archived


def test_history_limit_pauses_and_stops_frame_archiving(tmp_path):
    app, clock, feed, _ = make_app(tmp_path, delay=0.3, history_s=2.0, max_requests_per_trial=50)
    app.start_trial(expand_target("a"), clock())
    feed(30)
    assert app.feeding_paused == "history_limit_reached_start_new_instruction"
    received = app.session.counts["received"]
    feed(10)
    assert app.session.counts["received"] == received


# ---------- marks, notes, review ----------

def test_mark_binds_browser_view_and_falls_back_explicitly(tmp_path):
    app, clock, feed, _ = make_app(tmp_path, worker=PointingMockWorker(0.5))
    trial = app.start_trial(expand_target("box"), clock())
    feed(12)
    shown = app._current_prediction(clock())
    frame_id = app.tap.latest_id
    app.report_view({"prediction_id": shown["prediction_id"], "camera_frame_id": frame_id}, clock())
    app.handle("mark points at the box  ", clock(), app.viewer_snapshot(), clock())
    mark = read_jsonl(app.store.root / "trials" / trial.id / "marks.jsonl")[0]
    assert mark["binding"] == "browser_reported_view"
    assert mark["verbatim_note"] == "points at the box  "
    assert mark["prediction"]["prediction_id"] == shown["prediction_id"]
    assert mark["camera_frame"]["frame_id"] == frame_id
    ref = mark["camera_frame"]["image"]
    assert ref["store"] == "session_observation"  # stored once, by the session archive
    image = app.store.load_image(ref)
    assert pixel_sha256(image) == ref["pixel_sha256"]
    clock.t += 5  # the browser has not reported for a while
    app.handle("mark later", clock(), app.viewer_snapshot(), clock())
    later = read_jsonl(app.store.root / "trials" / trial.id / "marks.jsonl")[1]
    assert later["binding"] == "server_latest_browser_view_not_reported_recently"
    app.handle("note something odd on the left", clock(), app.viewer_snapshot(), clock())
    note = read_jsonl(app.store.root / "trials" / trial.id / "notes.jsonl")[0]
    assert note["verbatim_note"] == "something odd on the left" and note["context"] == "live_trial"
    assert note["model_feedback"] is False


def test_blob_store_stores_each_image_once_and_detects_changes(tmp_path):
    store = RunStore(tmp_path / "run")
    image = np.zeros((4, 5, 3), np.uint8)
    image[1, 2] = (10, 20, 30)
    a, b = store.store_image(image), store.store_image(image.copy())
    assert a == b and len(list((store.root / "blobs").rglob("*.png"))) == 1
    assert np.array_equal(store.load_image(a), image)  # lossless
    (store.root / a["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        store.load_image(a)
    with pytest.raises(FileExistsError):
        RunStore(tmp_path / "run")
    write_new_json(tmp_path / "x.json", {"a": 1})
    with pytest.raises(FileExistsError):
        write_new_json(tmp_path / "x.json", {"a": 2})


def test_review_dialog_requires_confirmation_and_keeps_comment_verbatim(tmp_path):
    app, clock, feed, messages = make_app(tmp_path, delay=0.5)
    trial = app.start_trial(expand_target("door"), clock())
    feed(10)
    app.handle("end", clock(), {}, clock())
    labels = app.store.root / "trials" / trial.id / "human_labels.jsonl"
    comment = "  I saw the door; model was LEFT of it.  "
    for line in (comment, "maybe", "present", "wrong", "uncertain", "not_applicable", ""):
        app.handle(line, clock(), {}, clock())
    assert any("Please answer one of" in m for m in messages)
    assert not labels.exists()  # nothing is saved before confirmation
    app.handle("no", clock(), {}, clock())
    assert not labels.exists() and app.dialog is None
    app.handle(f"review {trial.id}", clock(), {}, clock())
    for line in (comment, "absent", "ambiguous", "inappropriate", "uncertain", "AB", "yes"):
        app.handle(line, clock(), {}, clock())
    saved = read_jsonl(labels)
    assert len(saved) == 1
    r = saved[0]
    assert r["verbatim_comment"] == comment and r["reviewer"] == "AB" and r["confirmed_by_user"]
    assert (r["target_presence"], r["marking"], r["path"], r["stopping"]) == (
        "absent", "ambiguous", "inappropriate", "uncertain")
    assert r["model_feedback"] is False and set(CHOICES) <= set(r)
    # Model records are untouched by the review (separate files).
    assert all("verbatim_comment" not in p for p in predictions(app, trial.id))


def test_click_correction_maps_coordinates_and_is_stored_separately(tmp_path):
    app, clock, feed, _ = make_app(tmp_path, worker=PointingMockWorker(0.5))
    trial = app.start_trial(expand_target("box"), clock())
    feed(12)
    pid = predictions(app, trial.id)[0]["prediction_id"]
    with pytest.raises(ValueError):
        app.correction({"prediction_id": pid, "x_fraction": .5, "y_fraction": .5,
                        "natural_width": 448, "natural_height": 256}, clock())  # not in REVIEW
    app.end_trial("operator_end", clock(), review=False)
    record = app.correction({"prediction_id": pid, "x_fraction": 0.25, "y_fraction": 0.75,
                             "natural_width": 448, "natural_height": 256}, clock())
    assert (record["x_pixel"], record["y_pixel"]) == (112.0, 192.0)
    assert record["image"]["file_sha256"] and "origin top-left" in record["coordinate_frame"]
    with pytest.raises(ValueError):
        correction_record(app.store, trial.id, {"prediction_id": pid, "source_image": None},
                          {"x_fraction": 1.2, "y_fraction": 0.5, "natural_width": 448, "natural_height": 256},
                          clock(), "t")
    data = load_review(app.store, trial.id)
    assert len(data["corrections"]) == 1 and data["human_labels"] == []
    assert (app.store.root / "trials" / trial.id / "contact_sheet.png").is_file()


# ---------- LiDAR: epochs, missing pose, recording ----------

def scan_header(seq, stamp, host, offset=1000.0, n=10):
    return {"type": "scan", "seq": seq, "stamp": stamp, "time_span_s": 0.06,
            "host_recv_wall_s": host, "wall_minus_monotonic_s": offset, "n": n, "frame_id": "base_link"}


def odom_header(seq, stamp, host, offset=1000.0):
    return {"type": "odom", "seq": seq, "stamp": stamp, "host_recv_wall_s": host,
            "wall_minus_monotonic_s": offset, "position": [0, 0, 0.3], "q_xyzw": [0, 0, 0, 1]}


def feed_lidar(state, t0, seconds, robot0=50.0, host_offset=1000.0, odom=True):
    xyz = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.5], [0.1, 0.1, 0.0]], np.float32)
    s_seq = o_seq = 0
    for k in range(int(seconds * 50)):
        t = k / 50
        if odom:
            o_seq += 1
            state.odom(odom_header(o_seq, robot0 + t, t0 + t + 0.01, host_offset))
            state.after_odom()
        if k % 3 == 0:
            s_seq += 1
            state.scan(scan_header(s_seq, robot0 + t - 0.05, t0 + t + 0.02, host_offset, 3), xyz)


def test_lidar_epochs_offsets_and_windows():
    state = LidarState()
    state.hello({"bridge_id": "b1", "source": "go2_dds_read_only"})
    feed_lidar(state, 5000.0, 2.0)
    snap, views = state.snapshot(4000.0 + 2.0)
    assert snap["sync"]["status"] == "estimated"
    assert snap["windows"]["0.5"]["available"] and snap["windows"]["1.0"]["available"]
    assert snap["windows"]["1.0"]["span_robot_s"] <= 1.0 and snap["windows"]["0.5"]["span_robot_s"] <= 0.5
    assert snap["pose"]["valid"]
    first = snap["robot_epoch"]
    # Robot reboot: stamps jump backwards. Epoch changes; offset and windows reset.
    feed_lidar(state, 5002.0, 0.2, robot0=3.0)
    snap, _ = state.snapshot(4000.0 + 2.2)
    assert snap["robot_epoch"] != first and state.counts["robot_epoch_changes"] == 1
    assert snap["sync"]["status"] == "invalid" and snap["sync"]["host_minus_robot_s"] is None
    # Host wall clock step: host epoch changes and the offset estimate is discarded.
    feed_lidar(state, 5002.4, 1.5, robot0=3.5)
    assert state.sync()["status"] == "estimated"
    state.odom(odom_header(999, 5.1, 6000.0, offset=1000.5))
    assert state.counts["host_clock_steps"] == 1 and state.sync()["status"] == "invalid"
    # New bridge process: never continue the previous epoch.
    state.hello({"bridge_id": "b2", "source": "go2_dds_read_only"})
    assert state.counts["robot_epoch_changes"] == 2 and state.sync()["status"] == "invalid"


def test_lidar_without_pose_refuses_accumulation_and_shows_unknowns():
    state = LidarState()
    state.hello({"bridge_id": "b1", "source": "synthetic"})
    feed_lidar(state, 5000.0, 1.0, odom=False)
    snap, views = state.snapshot(4001.0)
    assert snap["pose"] == {"valid": False, "reason": "no odometry in this robot epoch"}
    assert not snap["windows"]["0.5"]["available"] and "pose" in snap["windows"]["0.5"]["reason"]
    assert state.counts["refused_no_pose"]["0.5"] > 0
    assert "latest" in views and "0.5" not in views  # the latest single scan is still shown
    assert snap["sync"]["status"] == "invalid"
    assert any("UNKNOWN" in u for u in snap["unknowns"])
    cov = coverage(np.array([[1.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    assert cov["sectors_with_returns"] == 1  # the blind-radius point does not count


def test_lidar_snapshot_keeps_ageing_after_the_stream_stops(tmp_path):
    src = LidarSource([PY, "-u", SYNTH, "--recording", ROOT / "tests/fixtures/vln/synthetic_recording"],
                      tmp_path / "replay")
    src.start()
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and not src.snapshot(time.monotonic())["ended"]:
            time.sleep(0.1)
        time.sleep(0.5)
        snap = src.snapshot(time.monotonic())
        assert snap["source"] == "recording_replay" and snap["ended"]
        st = snap["state"]
        assert st["counts"]["scans"] == 12 and st["counts"]["odom"] == 138  # every recorded message
        assert st["latest_scan"]["stamp"] == pytest.approx(1000.715)  # original robot stamps kept
        assert not st["pose"]["valid"] and "silent" in st["pose"]["reason"]
    finally:
        src.close()


def test_lidar_recorder_is_bounded_and_records_gaps_explicitly(tmp_path):
    import threading
    gate = threading.Event()
    recorder = LidarRecorder(tmp_path / "lidar", queue_size=1, chunk_s=0.1, quota=lambda: gate.wait(5) or True)
    recorder.start("T1")
    xyz = np.zeros((5, 3), np.float32)
    for i in range(12):
        h = scan_header(i + 1, 100 + i * 0.1, 2000 + i * 0.1)
        recorder.offer("scan", h, xyz, "e0", i * 0.1)
    recorder.stop()
    gate.set()
    recorder.close()
    rows = read_jsonl(tmp_path / "lidar/index.jsonl")
    written = [r for r in rows if not r["dropped_not_recorded"]]
    dropped = [r for r in rows if r["dropped_not_recorded"]]
    assert written and dropped and recorder.counts["chunks_dropped"] == len(dropped)
    assert all(r["source_interval_robot_s"] and r["host_receipt_interval_wall_s"] for r in rows)
    data = np.load(tmp_path / "lidar" / written[0]["file"])
    assert data["points_xyz_base_link"].dtype == np.float32 and "scan_stamp_robot" in data
    assert sorted(r["chunk"] for r in rows) == list(range(1, len(rows) + 1))


def test_lidar_source_with_synthetic_bridge_and_missing_odom(tmp_path):
    for name, extra, expect_window in (("with_odom", [], True), ("no_odom", ["--no-odom"], False)):
        src = LidarSource([PY, "-u", SYNTH, *extra], tmp_path / name)
        src.start()
        try:
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                snap = src.snapshot(time.monotonic())["state"]
                if snap and snap["latest_scan"] and snap["counts"]["scans"] > 20:
                    break
                time.sleep(0.1)
            assert snap["windows"]["1.0"]["available"] is expect_window
            assert snap["pose"]["valid"] is expect_window
            assert src.view("latest")
        finally:
            src.close()
        assert not src.alive


# ---------- server ----------

def test_server_is_loopback_token_protected_and_serves_snapshots(tmp_path):
    app, clock, feed, _ = make_app(tmp_path, worker=PointingMockWorker(0.5))
    app.start_trial(expand_target("box"), clock())
    feed(12)
    server = DisplayServer(app, 0)
    server.start()
    try:
        assert server.httpd.server_address[0] == "127.0.0.1"
        state = json.loads(urllib.request.urlopen(server.url + "api/state", timeout=5).read())
        assert state["banner"] == "ADVISORY ONLY — NO ROBOT COMMANDS" and state["robot_commands_sent"] == 0
        pid = state["prediction"]["prediction_id"]
        assert urllib.request.urlopen(server.url + f"api/pred_image?id={pid}", timeout=5).read()[:4] == b"\x89PNG"
        assert urllib.request.urlopen(server.url + "api/camera", timeout=5).read()[:2] == b"\xff\xd8"
        body = json.dumps({"prediction_id": pid, "camera_frame_id": "x"}).encode()
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(server.url + "api/viewing", data=body), timeout=5)
        assert error.value.code == 403
        page = urllib.request.urlopen(server.url, timeout=5).read().decode()
        token = re.search(r'const TOKEN="([^"]+)"', page).group(1)
        ok = urllib.request.Request(server.url + "api/viewing", data=body, headers={"X-Review-Token": token})
        assert json.loads(urllib.request.urlopen(ok, timeout=5).read())["ok"]
        assert app.viewer_snapshot()["prediction_id"] == pid
        bad_host = urllib.request.Request(server.url + "api/state", headers={"Host": "evil.example:80"})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(bad_host, timeout=5)
        assert error.value.code == 403
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(server.url + "api/pred_image?id=../../etc", timeout=5)
    finally:
        server.close()


# ---------- tensors opt-in ----------

def test_tensor_evidence_skip_is_explicit_and_cached_only(tmp_path):
    writer = SkippedTensorEvidence()
    writer.submit(tmp_path, {"x": np.zeros(4, np.float32)}, {})
    status = json.loads((tmp_path / "tensor_evidence.json").read_text())
    assert status["state"].startswith("not_recorded") and not (tmp_path / "model_inputs_and_generated_ids.npz").exists()
    with pytest.raises(ValueError):
        ProcessWorker(python=PY, assets=tmp_path, config=tmp_path, device="cpu", output=tmp_path / "w",
                      preparation_mode="baseline", tensor_evidence="skip")


# ---------- end to end through the CLI, then reopen ----------

def test_cli_trial_mark_end_review_and_reopen(tmp_path):
    out = tmp_path / "run"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    commands = "\n".join([
        "help", "target blue box", "__wait__", "mark looks right", "note box is centred", "end",
        "The model marked the box.", "present", "correct", "plausible", "not_applicable", "", "yes",
        "status", "quit"])
    proc = subprocess.Popen([PY, "-u", ROOT / "scripts/run_vln_interactive.py", "--camera", "synthetic",
                             "--lidar", "synthetic", "--backend", "mock", "--mock-pointing", "alternate",
                             "--mock-delay-s", "0.3", "--output", out, "--port", "0", "--reviewer", "cli"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env=env, cwd=ROOT)
    time.sleep(2.0)
    for line in commands.splitlines():
        if line == "__wait__":
            time.sleep(3.5)
            continue
        proc.stdin.write(line + "\n")
        proc.stdin.flush()
        time.sleep(0.3)
    output, _ = proc.communicate(timeout=30)
    assert proc.returncode == 0, output
    summary = json.loads((out / "summary.json").read_text())
    assert summary["robot_commands_sent"] == 0 and summary["termination"] == "operator_quit"
    run = json.loads((out / "run.json").read_text())
    assert run["advisory_age_s_unchanged"] == 3.0 and "NOT enabled" in run["temporal_policy"]
    trial = next((out / "trials").glob("T*"))
    assert read_jsonl(trial / "human_labels.jsonl")[0]["verbatim_comment"] == "The model marked the box."
    assert read_jsonl(trial / "marks.jsonl")[0]["verbatim_note"] == "looks right"
    assert read_jsonl(out / "lidar/index.jsonl"), "LiDAR chunks recorded during the trial"
    # Reopen read-only for review; existing records stay unchanged (append-only).
    before = (trial / "human_labels.jsonl").read_bytes()
    reopen = subprocess.run([PY, "-u", ROOT / "scripts/run_vln_interactive.py", "--review-run", out, "--port", "0"],
                            input=f"review {trial.name}\n!cancel\nquit\n", capture_output=True, text=True,
                            env=env, cwd=ROOT, timeout=30)
    assert reopen.returncode == 0, reopen.stdout + reopen.stderr
    assert "REVIEW ONLY" in reopen.stdout and trial.name in reopen.stdout
    assert (trial / "human_labels.jsonl").read_bytes() == before


# ---------- no motion commands anywhere ----------

def test_interactive_paths_have_no_control_route():
    paths = [*sorted((ROOT / "src/go2_perception/vln").glob("interactive_*.py")),
             ROOT / "scripts/run_vln_interactive.py", ROOT / "scripts/go2_lidar_bridge.py",
             ROOT / "scripts/synthetic_lidar_bridge.py", ROOT / "src/go2_perception/adapters/go2/receivers.py"]
    forbidden_imports = ("go2_control", "go2_hl_control", "rclpy", "unitree_sdk2py.go2", "cyclonedds")
    forbidden_names = {"UseRemoteCommandFromApi", "SportClient", "ObstaclesAvoidClient", "SwitchSet",
                       "StandDown", "StandUp", "StopMove", "Move", "Damp", "BalanceStand", "ChannelPublisher",
                       "Publisher", "DataWriter", "sendto", "Request", "Euler", "SpeedLevel"}
    for path in paths:
        tree = ast.parse(path.read_text())
        names, identifiers = [], set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
                identifiers |= {x.name for x in node.names}
            elif isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
        if path.name != "receivers.py":  # its subscribers import the SDK's channel/IDL modules
            assert not any(w in n for w in forbidden_imports for n in names), path
        assert not identifiers & forbidden_names, (path, identifiers & forbidden_names)
    bridge = ast.parse((ROOT / "scripts/go2_lidar_bridge.py").read_text())
    imported = {a.name for n in ast.walk(bridge) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert imported == {"ODOM_TOPIC", "Go2DdsReceiver", "Empty", "Full", "Queue"}
    source = (ROOT / "scripts/go2_lidar_bridge.py").read_text()
    assert 'SCAN_TOPIC = "rt/utlidar/cloud_base"' in source and "[SCAN_TOPIC]" in source
    assert "cloud_deskewed\"" not in source.split("SCAN_TOPIC =")[1]
    receivers = (ROOT / "src/go2_perception/adapters/go2/receivers.py").read_text()
    assert "ChannelSubscriber" in receivers and "ChannelPublisher" not in receivers
    assert os.environ.get("GO2_HARDWARE", "0") in ("0", "")
