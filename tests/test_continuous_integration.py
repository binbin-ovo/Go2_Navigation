"""Offline learned-worker protocol, observed handovers, review and hardware evidence boundary."""

from hashlib import sha256
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from go2_perception.vln.continuous_planner import LightNavPlanner, encode_view
from go2_perception.vln.continuous_recording import Feasibility, TraceRecorder
from go2_perception.vln.continuous_review import ContinuousReview
from go2_perception.vln.continuous_simulation import CAMERA, START, sensor, simulate
from go2_perception.vln.supervised import MODEL_REVISION


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture():
    return json.loads((ROOT / "examples/supervised_navigation/synthetic_cycle.json").read_bytes())


class DelayedWorker:
    def __init__(self, foreign=False):
        self.submitted = threading.Event()
        self.release = threading.Event()
        self.closed = False
        self.foreign = foreign

    def submit(self, request, now):
        self.request = request
        self.start = now
        self.submitted.set()

    def poll(self, now):
        if not self.release.is_set():
            return None
        j = self.request.job
        native = dict(
            status="decoded",
            episode_id=j["episode_id"],
            clock_epoch=j["clock_epoch"],
            decision_id=j["decision_id"],
            waypoint_convention="absolute_ego_SE2_x_forward_y_left_yaw_CCW_metres_radians",
            waypoint_time_base=None,
            waypoints=[[0.06 * i, 0, 0] for i in range(1, 11)],
            pointing={"opos": 0},
            decoder_stop=False,
            native_output="native fixture",
        )
        return dict(
            backend="lightnav",
            identity={} if self.foreign else self.request.context["identity"],
            result=native,
            inference_duration_s=now - self.start,
            host_inference_end_s=now,
        )

    def close(self):
        self.closed = True


@pytest.mark.parametrize("foreign", [False, True])
def test_asynchronous_native_worker_cannot_read_control_or_future_frames(tmp_path, fixture, foreign):
    worker = DelayedWorker(foreign)
    planner = LightNavPlanner(tmp_path, worker)
    image = np.zeros((101, 161), dtype=np.uint8)
    png, view = encode_view(image)
    t = START + 3
    for i in range(13):
        planner.capture(START + i * 0.25, png, view["png_sha256"], f"frame-{i}")
    guard = sensor(fixture["proposal"]["identity"]["pose"], t)
    origin = time.monotonic()
    try:
        planner.submit(fixture["proposal"], guard, "frame-12", CAMERA, (t, origin))
        assert worker.submitted.wait(2)
        assert planner.poll(t + 0.1) is None
        with pytest.raises(RuntimeError, match="already outstanding"):
            planner.submit(fixture["proposal"], guard, "another", CAMERA, (t, origin))
        planner.capture(t + 0.25, png, view["png_sha256"], "frame-13")
        req = worker.request
        assert set(req.job) == {
            "instruction",
            "episode_id",
            "clock_epoch",
            "decision_id",
            "decision_time_s",
            "frames",
        }
        assert all(set(f) == {"path", "observation_s", "available_s", "sha256"} for f in req.job["frames"])
        assert req.context["bounded_context"]["source_indices"] == [0, 1, 7, 8, 9, 10, 11, 12]
        assert max(f["available_s"] for f in req.job["frames"]) == t
        worker.release.set()
        if foreign:
            with pytest.raises(ValueError, match="foreign"):
                planner.future.result(2)
        else:
            delivery = planner.future.result(2)
            assert planner.poll(delivery["available_s"] - 0.0001) is None
            assert planner.poll(delivery["available_s"]) == delivery
            assert delivery["proposal"]["model_identity"]["revision"] == MODEL_REVISION
            assert delivery["proposal"]["identity"]["pose"] == guard["snapshot"]["pose"]
            assert delivery["proposal"]["identity"]["observation_id"] == "frame-12"
            assert delivery["grounding"]["status"] == "absent"
    finally:
        planner.close()
    assert worker.closed


def test_retained_pixel_history_is_bounded_through_a_full_minute(tmp_path):
    p = LightNavPlanner(tmp_path, DelayedWorker())
    try:
        for i in range(241):
            p.capture(i * 0.25, bytes([i]), sha256(bytes([i])).hexdigest(), str(i))
        assert p.max_images == len(p.pixels) == 8
        assert list(p.pixels) == [0, 1, 235, 236, 237, 238, 239, 240]
        with pytest.raises(ValueError, match="budget exhausted"):
            p.capture(60.25, b"", "", "too-many")
    finally:
        p.close()


@pytest.mark.parametrize("scenario", ["target_loss", "no_continuation", "visible_after_search"])
def test_policy_handover_frequency_and_loss_outcomes(fixture, scenario):
    rows = []
    simulate(fixture, scenario, rows.append)
    events = Feasibility()
    for row in rows:
        events.observe(row)
    ticks = [r for r in rows if r["kind"] == "tick"]
    metrics = events.summary()
    assert metrics["issued"] <= 5 and metrics["stops_without_continuation"] >= 1
    if scenario == "target_loss":
        change = next(
            i
            for i, r in enumerate(ticks)
            if r["packet"]["reason"] == "mode_or_direction_change_requires_settled_view"
        )
        assert ticks[change]["packet"]["velocity"] == [0.0, 0.0]
        assert any(
            r["packet"]["mode"] == "SEARCH" and r["t"] > ticks[change]["t"] + 1 for r in ticks[change:]
        )
        assert ticks[-1]["packet"]["reason"] == "plan_budget"
    elif scenario == "no_continuation":
        k = max(i for i, r in enumerate(ticks) if r["packet"]["handover"] == "stop")
        assert metrics["continuation_stop_reasons"]["selected_path_exhausted"] == 2
        assert all(r["packet"]["velocity"] == [0.0, 0.0] for r in ticks[k:])
    else:
        assert metrics["moving_replacements"] == 1 and metrics["offers_during_motion"] == 2
        p = next(r["packet"] for r in ticks if r["packet"]["handover"] == "replace")
        assert p["progress"] > 0 and p["remaining_path"] < p["action"]["magnitude"]
        assert p["path_horizon_s"] == pytest.approx(p["remaining_to_brake"] / p["action"]["speed"])


def test_fresh_geometry_always_wins_over_compatible_replacement(fixture):
    from test_continuous_navigation import Episode

    e = Episode(fixture)
    e.step(delivery=e.result(START))
    g = e.guard(START + 0.05)
    g["snapshot"]["geometry"]["monotonic_s"] -= 0.8
    b = e.nav.tick(START + 0.05, g, delivery=e.result(START + 0.05))
    assert b["reason"] == "stale_or_future_lidar" and b["velocity"] == [0.0, 0.0] and b["plans"] == 1


def test_trace_viewer_binds_latest_proposal_to_its_own_image(tmp_path, fixture):
    path = tmp_path / "trace.jsonl"
    with TraceRecorder(path) as recorder:
        simulate(fixture, "visible_after_search", recorder)
    assert recorder.peak <= 16 and not recorder.thread.is_alive()
    v = ContinuousReview(tmp_path, {"results": [{"scenario": "fixture", "trace": path.name}]})
    d = v.continuous_sample({"tick": 58})
    assert d["packet"]["handover"] == "replace" and d["incoming"]["id"] == d["packet"]["prediction_id"]
    assert d["camera"] and d["model_image"] and d["path"] and "svg" in d["native_svg"]
    assert v.continuous_sample({"tick": 0})["incoming"] is None
    with pytest.raises(ValueError, match="trace position"):
        v.continuous_sample({"tick": 999})
    bad = path.read_text().replace('"png_sha256":"', '"png_sha256":"bad', 1)
    path.write_text(bad)
    v = ContinuousReview(tmp_path, {"results": [{"scenario": "fixture", "trace": path.name}]})
    with pytest.raises(ValueError, match="image hash"):
        v.continuous_sample({"tick": 0})


def test_repeated_episodes_release_python_control_and_recording_memory(tmp_path, fixture):
    from check_continuous_memory import measure

    result = measure(fixture, tmp_path, 6)
    assert result["moving_replacements"] == 6 and result["packets"] > 300
    assert result["peak_trace_queue_rows"] <= result["queue_capacity"] == 16
    assert result["retained_python_delta_bytes"] < 2 * 1024 * 1024
    assert not list(tmp_path.iterdir())


def test_review_server_uses_existing_loopback_viewer_and_rejects_mutations(tmp_path, fixture):
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen
    from go2_perception.vln.interactive_server import DisplayServer

    with TraceRecorder(tmp_path / "trace") as rec:
        simulate(fixture, "visible_after_search", rec)
    v = ContinuousReview(tmp_path, {"results": [dict(trace="trace", scenario="example")]})
    server = DisplayServer(v, 0)
    server.start()
    try:
        assert "Incoming model proposal" in urlopen(server.url, timeout=3).read().decode()
        assert (
            json.load(urlopen(server.url + "api/continuous?tick=58", timeout=3))["packet"]["handover"]
            == "replace"
        )
        with pytest.raises(HTTPError) as err:
            urlopen(Request(server.url + "api/navigation", data=b"{}"), timeout=3)
        assert err.value.code == 405
    finally:
        server.close()


def test_physical_evidence_separates_pending_synthetic_and_closed_physical_manifests(tmp_path):
    from check_continuous_evidence import verify

    r = json.loads((ROOT / "examples/continuous_navigation/pending_evidence.json").read_bytes())
    assert verify(r, tmp_path)["disposition"] == "awaiting_claude"
    r.update(
        source_kind="synthetic",
        run_id="episode",
        reviewer="fixture",
        navigation_commit="a" * 40,
        controller_commit="b" * 40,
        sdk_commit="c" * 40,
    )
    manifest = dict(
        mode="SYNTHETIC",
        navigation_mode="mock",
        backend="mock",
        git_commit="a" * 40,
        canonical_control_revision="b" * 40,
        integration="go2-five-move-episode-v1",
    )

    def artifact(key, name, body):
        p = tmp_path / name
        p.write_text(json.dumps(body))
        return dict(id=key, path=name, sha256=sha256(p.read_bytes()).hexdigest())

    r["artifacts"] = [
        artifact("run_manifest", "run.json", manifest),
        artifact(
            "episode_summary",
            "episode_summary.json",
            dict(episode_id="episode", schema="go2-five-move-episode-v1", state="TERMINAL"),
        ),
        artifact("session_summary", "summary.json", dict(ended_wall_utc="2026-11-19T10:00:00Z")),
    ]
    r["topics"]["tested_motion"].update(
        status="observed", notes="unit fixture", evidence_ids=["run_manifest"], measurements={"attempts": 0}
    )
    assert verify(r, tmp_path)["disposition"] == "offline_reference"
    r["source_kind"] = "physical_five_move"
    with pytest.raises(ValueError, match="closed physical"):
        verify(r, tmp_path)
    manifest.update(mode="LIVE", navigation_mode="live", backend="lightnav")
    r["artifacts"][0] = artifact("run_manifest", "run.json", manifest)
    assert verify(r, tmp_path)["disposition"] == "operator_attested_for_human_review"
    assert verify(r, tmp_path)["physical_activation"] is False
    (tmp_path / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(r, tmp_path)


def test_control_receipt_cannot_refresh_an_expired_worker_result(fixture):
    from test_continuous_navigation import Episode

    e = Episode(fixture)
    q = e.result(START, latency=0.8)
    q["worker_finished_s"] = START + 0.1
    assert e.step(0.8, delivery=q)["reason"] == "stale_or_future_plan_result"


def test_worker_failure_latches_even_with_an_active_path(fixture):
    from test_continuous_navigation import Episode

    e = Episode(fixture)
    e.step(delivery=e.result(START))
    t = START + 0.05
    p = e.nav.tick(t, e.guard(t), worker_failed=True)
    assert p["reason"] == "model_worker_failed" and p["handover"] == "stop" and p["velocity"] == [0.0, 0.0]
    assert e.step(0.1, delivery=e.result(START + 0.1))["mode"] == "TERMINAL"


def test_following_growing_trace_ignores_partial_writer_lines(tmp_path, fixture):
    records = []
    simulate(fixture, "visible_after_search", records.append)
    root = tmp_path
    path = root / "trace"
    review = ContinuousReview(
        root, dict(schema="go2-continuous-review-v1", results=[dict(trace="trace", scenario="case")])
    )
    assert review.state(0)["following"] and review.continuous_sample({"tick": "latest"}) == {"waiting": True}
    with path.open("wb") as stream:

        def line(row):
            return (json.dumps(row) + "\n").encode()

        for r in records[:2]:
            stream.write(line(r))
        stream.flush()
        assert review.continuous_sample({"tick": "latest"})["index"] == 0
        data = line(records[2])
        stream.write(data[:12])
        stream.flush()
        assert review.continuous_sample({"tick": "latest"})["index"] == 0
        stream.write(data[12:])
        stream.flush()
        assert review.continuous_sample({"tick": "latest"})["index"] == 1


def test_guard_cannot_relabel_an_older_camera_capture(tmp_path, fixture):
    p = LightNavPlanner(tmp_path, DelayedWorker())
    image = np.zeros((101, 161), dtype=np.uint8)
    png, v = encode_view(image)
    try:
        with pytest.raises(ValueError, match="hash"):
            p.capture(START, png, "0" * 64, "f1")
        p.capture(START, png, v["png_sha256"], "f1")
        for dt, fid in ((0.1, "f1"), (0.0, "f2")):
            g = sensor(fixture["proposal"]["identity"]["pose"], START + dt)
            with pytest.raises(ValueError, match="same capture"):
                p.submit(fixture["proposal"], g, fid, CAMERA, (START, time.monotonic()))
    finally:
        p.close()


def test_invalid_capture_stops_and_records_failure_before_first_plan(fixture):
    class Failed:
        submitted = completed = max_images = 0

        def poll(self, now):
            return None

        def capture(self, *args):
            raise ValueError("unbound camera capture")

    rows = []
    result = simulate(fixture, "visible_after_search", rows.append, wall_clock=True, planner=Failed())
    assert result["end_reason"] == "model_worker_failed" and result["plans"] == 0
    assert (
        rows[-1]["packet"]["velocity"] == [0.0, 0.0] and rows[-1]["worker_error"] == "unbound camera capture"
    )
    assert result["control_gap_p95_s"] is None
