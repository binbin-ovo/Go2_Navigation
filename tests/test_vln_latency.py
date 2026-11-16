"""Lossless cache, causal reconstruction and bounded archival; no model or sensor."""

import ast
import json
from pathlib import Path
from queue import Queue
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from go2_perception.vln.episodes import file_sha256
from go2_perception.vln.observation import save
from go2_perception.vln.shadow import ShadowSession
from go2_perception.vln.shadow_cache import PreparedFrameCache
from go2_perception.vln.shadow_evidence import TensorEvidenceWriter
from go2_perception.vln.shadow_sources import SyntheticSource
from go2_perception.vln.shadow_worker import MockWorker, ProcessWorker


def stored(tmp_path, observation, name):
    path = tmp_path / f"{name}.npz"
    save(observation, path)
    return {"observation_path": str(path), "observation_sha256": file_sha256(path),
            "frame_id": name, "clock_epoch": observation.clock_epoch}


def test_cache_reuses_exact_pixels_and_preserves_source(tmp_path):
    obs = next(iter(SyntheticSource(1))).observation
    source = stored(tmp_path, obs, "frame")
    cache = PreparedFrameCache(tmp_path / "cache", allow_synthetic=True)
    first = cache.prepare(source, tmp_path / "a.png")
    second = cache.prepare(source, tmp_path / "b.png")
    assert first[0] == second[0] and first[1] == second[1]
    assert not first[2]["hit"] and second[2]["hit"]
    assert np.array_equal(cv2.imread(str(tmp_path / "a.png")), cv2.imread(str(tmp_path / "b.png")))
    assert file_sha256(source["observation_path"]) == source["observation_sha256"]
    assert second[1]["source"]["stamp_host"]["t"] == obs.visual.stamp_host.t
    assert cache.status()["hits"] == 1


@pytest.mark.parametrize("change", ["pixels", "K", "D", "epoch", "stamp", "version"])
def test_cache_key_binds_image_calibration_timestamp_epoch_and_version(tmp_path, change):
    obs = next(iter(SyntheticSource(1))).observation
    obs.visual.rectified = False
    cache = PreparedFrameCache(tmp_path / "cache", allow_synthetic=True)
    source = stored(tmp_path, obs, "frame")
    original = cache.prepare(source, tmp_path / "a.png")
    if change == "pixels":
        obs.visual.image[10, 10] ^= 255
    elif change == "K":
        obs.visual.K[0, 0] *= 1.1
    elif change == "D":
        obs.visual.D[0] = 0.05
    elif change == "epoch":
        obs.clock_epoch = "new_epoch"
    elif change == "stamp":
        obs.visual.stamp_host.t += 1.0
    else:
        cache.settings["preprocessor_sha256"] = "new-version"
    source = stored(tmp_path, obs, "frame")
    new = cache.prepare(source, tmp_path / "b.png")
    assert not new[2]["hit"] and new[2]["key"] != original[2]["key"]


def test_cache_rejects_changed_archive_or_prepared_bytes(tmp_path):
    obs = next(iter(SyntheticSource(1))).observation
    source = stored(tmp_path, obs, "frame")
    cache = PreparedFrameCache(tmp_path / "cache", allow_synthetic=True)
    cache.prepare(source, tmp_path / "a.png")
    Path(source["observation_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Stored observation changed"):
        cache.prepare(source, tmp_path / "b.png")
    source = stored(tmp_path, obs, "frame")
    (tmp_path / "a.png").write_bytes(b"tampered hard link")
    with pytest.raises(ValueError, match="cache evidence changed"):
        cache.prepare(source, tmp_path / "b.png")


def test_cache_eviction_and_reset_preserve_request_evidence(tmp_path):
    cache = PreparedFrameCache(tmp_path / "cache", max_entries=1, allow_synthetic=True)
    frames = list(SyntheticSource(3))
    for i, f in enumerate(frames):
        cache.prepare(stored(tmp_path, f.observation, str(i)), tmp_path / f"{i}.png")
        assert len(cache.entries) == 1
    assert cache.status()["evicted"] == 2
    cache.clear()
    assert not list((tmp_path / "cache").iterdir())
    assert all(cv2.imread(str(tmp_path / f"{i}.png")) is not None for i in range(3))


def test_cache_byte_bound_bypasses_without_losing_input(tmp_path):
    cache = PreparedFrameCache(tmp_path / "cache", max_bytes=1, allow_synthetic=True)
    source = stored(tmp_path, next(iter(SyntheticSource(1))).observation, "frame")
    cache.prepare(source, tmp_path / "a.png")
    assert cache.status()["bytes"] == 0 and cache.status()["bypassed"] == 1
    assert (tmp_path / "a.png").exists() and not list(cache.root.iterdir())


def new_session(path, mode):
    s = ShadowSession(path, MockWorker(0), mode="synthetic_fast", allow_synthetic=True,
                      session_id="same", preparation_mode=mode)
    s.instruction_changed("Approach doorway", 0)
    return s


def test_latest_anchored_inputs_and_native_anchor_metadata_are_equivalent(tmp_path):
    baseline, cached = [new_session(tmp_path / mode, mode) for mode in ("baseline", "cached")]
    frames = list(SyntheticSource(25, fps=7.1))
    last, previous_ids = 0, None
    for count in (9, 13, 25):
        for i in range(last, count):
            for s in (baseline, cached):
                s.receive(frames[i], i / 7.1)
        now = (count - 1) / 7.1
        for s in (baseline, cached):
            s.tick(now)
        a, b = baseline.inflight, cached.inflight
        assert a.context["grid_s"] == b.context["grid_s"]
        assert a.context["timestamp_quantization_error_s"] == b.context["timestamp_quantization_error_s"]
        ids = [h["source"]["frame_id"] for h in a.context["history"]]
        assert ids == [h["source"]["frame_id"] for h in b.context["history"]]
        assert ids[-1] == frames[count - 1].frame_id
        if previous_ids is not None:
            assert ids[:len(previous_ids)] != previous_ids  # append-only would be wrong
        previous_ids = ids
        for fa, fb, ha, hb in zip(a.job["frames"], b.job["frames"], a.context["history"], b.context["history"]):
            assert fa["observation_s"] == fb["observation_s"] <= fb["available_s"] <= b.job["decision_time_s"]
            assert np.array_equal(cv2.imread(fa["path"]), cv2.imread(fb["path"]))
            assert ha["rectification"] == hb["rectification"]
        for s in (baseline, cached):
            s.tick(now + 0.01, allow_submit=False)
        assert baseline.displayed[1]["response"]["result"] == cached.displayed[1]["response"]["result"]
        last = count


@pytest.mark.parametrize("change,reason", [("instruction", "superseded_instruction"),
    ("epoch", "superseded_source_or_epoch"), ("cancel", "cancelled_or_closed")])
def test_cached_session_resets_and_invalidates_late_result(tmp_path, change, reason):
    s = new_session(tmp_path / "run", "cached")
    frames = list(SyntheticSource(3))
    s.receive(frames[0], 0)
    s.tick(0)
    assert s.prepared_cache.status()["entries"] == 1
    if change == "instruction":
        s.instruction_changed("Turn left", 0.01)
    elif change == "epoch":
        frames[1].observation.clock_epoch = "new"
        s.receive(frames[1], 0.1)
    else:
        s.cancel(0.01)
    assert not s.prepared_cache.entries
    s.tick(0.2, allow_submit=False)
    assert s.displayed[1]["rejection_reason"] == reason


def test_cached_session_uses_unchanged_age_limit(tmp_path):
    s = new_session(tmp_path / "run", "cached")
    s.receive(next(iter(SyntheticSource(1))), 0)
    s.tick(0)
    s.tick(3.01, allow_submit=False)
    assert s.limits.advisory_age_s == 3
    assert s.displayed[1]["rejection_reason"] == "stale_advisory_age"


def test_writer_owns_arrays_is_bounded_and_drains(tmp_path, monkeypatch):
    writer = TensorEvidenceWriter()
    gate = threading.Event()
    original = writer._write
    monkeypatch.setattr(writer, "_write", lambda *args: (gate.wait(1), original(*args)))
    arrays = {"pixels": np.arange(12, dtype=np.uint16)}
    expected = arrays["pixels"].copy()
    try:
        writer.submit(tmp_path, arrays, {"pixels": {"dtype": "bfloat16"}})
        arrays["pixels"][:] = 99
        with pytest.raises(RuntimeError, match="backpressure"):
            writer.submit(tmp_path, arrays, {})
        with pytest.raises(TimeoutError):
            writer.flush(timeout=0.001)
        gate.set()
        writer.close()
        with np.load(tmp_path / "model_inputs_and_generated_ids.npz") as z:
            assert np.array_equal(z["pixels"], expected)
        assert json.loads((tmp_path / "tensor_evidence.json").read_text())["state"] == "complete"
        assert writer.status()["completed"] == 1 and writer.status()["outstanding"] == 0
        assert not writer.thread.is_alive()
    finally:
        gate.set()
        writer.close()


def test_writer_byte_bound_and_error_are_explicit(tmp_path, monkeypatch):
    writer = TensorEvidenceWriter(max_bytes=4)
    with pytest.raises(ValueError, match="byte bound"):
        writer.submit(tmp_path, {"x": np.arange(8)}, {})
    monkeypatch.setattr(writer, "_write", lambda *args: (_ for _ in ()).throw(OSError("disk unavailable")))
    writer.submit(tmp_path, {"x": np.zeros(1, dtype=np.uint8)}, {})
    with pytest.raises(RuntimeError, match="disk unavailable"):
        writer.close()
    assert json.loads((tmp_path / "tensor_evidence.json").read_text())["state"] == "error"
    assert not writer.thread.is_alive()


def test_cached_worker_shutdown_drains_via_eof_before_termination():
    calls = []

    class Pipe:
        def close(self):
            calls.append("eof")

    class Child:
        stdin, stdout = Pipe(), Pipe()
        done = False

        def poll(self):
            return 0 if self.done else None

        def wait(self, timeout):
            assert calls == ["eof"] and timeout == 5
            calls.append("archive_drained")
            self.done = True

        def terminate(self):
            pytest.fail("Completed archival must not be terminated")

    worker = ProcessWorker.__new__(ProcessWorker)
    worker.preparation_mode = "cached"
    worker.closed, worker.ready, worker.process = False, True, Child()
    worker.messages, worker.log = Queue(maxsize=2), Pipe()
    worker.reader = SimpleNamespace(join=lambda timeout: None)
    worker.close()
    assert calls[:2] == ["eof", "archive_drained"]


def test_archive_failure_remains_visible_when_last_request_ends(tmp_path):
    s = new_session(tmp_path / "run", "cached")
    s.worker.tensor_evidence_summary = {"submitted": 1, "completed": 0, "error": "disk full"}
    s.close(0.1)
    assert "disk full" in s.history_error
    assert s.status(0.1)["tensor_evidence_shutdown"]["completed"] == 0
    assert "tensor_evidence_shutdown" in s.events_path.read_text()


def test_latency_paths_import_no_robot_or_network_client():
    root = Path(__file__).resolve().parents[1]
    files = [root / "scripts/profile_vln_latency.py", root / "src/go2_perception/vln/shadow_cache.py",
             root / "src/go2_perception/vln/shadow_evidence.py"]
    forbidden = {"go2_control", "unitree_sdk2py", "cyclonedds", "socket", "rclpy"}
    for path in files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not {n.name.split('.')[0] for n in node.names} & forbidden
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split('.')[0] not in forbidden
