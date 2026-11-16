"""Cheap context-contract tests; no weights, GPU, network or physical adapter."""

import copy
import importlib.util
from pathlib import Path

import pytest

from go2_perception.vln.bounded_context import (
    MAX_IMAGES, POLICY, select_context, validate_context,
)


def frames(n, irregular=False):
    return [{"path": f"{i}.png", "sha256": "dummy", "observation_s": 100 + i*.25 - (.031 if irregular and i%2 else 0),
             "available_s": 100 + i*.25 + .05} for i in range(n)]


def job_and_context(n=39, irregular=False):
    source = frames(n, irregular)
    plan = select_context(source, source[-1]["available_s"])
    job = {"instruction":"Keep the original instruction", "episode_id":"one", "clock_epoch":"epoch0",
           "decision_id":"d0", "decision_time_s":source[-1]["available_s"],
           "frames":[source[i] for i in plan["source_indices"]]}
    return job, plan


@pytest.mark.parametrize("n", [1,2,3,4,5,6,7,8,9,19,39,129,241])
def test_size_current_anchors_and_no_source_duplication(n):
    job, plan = job_and_context(n)
    validate_context(plan, job)
    assert len(job["frames"]) <= MAX_IMAGES
    assert plan["source_indices"][-1] == n-1
    assert plan["source_indices"][:min(2,n)] == list(range(min(2,n)))
    assert plan["source_indices"] == sorted(set(plan["source_indices"]))
    assert set(plan["source_indices"]) | set(plan["omitted_source_indices"]) == set(range(n))


def test_irregular_actual_timestamps_are_not_relabelled_as_uniform():
    job, plan = job_and_context(irregular=True)
    assert plan["timestamp_ticks"] == [0,219,8219,8500,8719,9000,9219,9500]
    assert max(abs(e) for e in plan["timestamp_rounding_error_s"]) <= .000501
    assert job["frames"][-1]["observation_s"] - job["frames"][0]["observation_s"] == 9.5
    # The omitted eight-second interval still separates the anchor from recent history.
    assert plan["timestamp_ticks"][2] - plan["timestamp_ticks"][1] == 8000


@pytest.mark.parametrize("change", ["empty", "future", "duplicate", "nan", "missing_omitted", "gap", "duration", "count"])
def test_invalid_and_missing_history_fails_closed(change):
    fs = frames(39)
    cutoff = fs[-1]["available_s"]
    if change == "empty": fs = []
    if change == "future": fs[-1]["available_s"] = cutoff+1
    if change == "duplicate": fs[10]["observation_s"] = fs[9]["observation_s"]
    if change == "nan": fs[8]["observation_s"] = float("nan")
    if change == "missing_omitted": fs[10]["path"] = None
    if change == "gap": fs = fs[:9] + fs[12:]
    if change == "duration": fs = frames(242); cutoff = fs[-1]["available_s"]
    if change == "count": fs = frames(1025); cutoff = fs[-1]["available_s"]
    with pytest.raises(ValueError): select_context(fs, cutoff)


@pytest.mark.parametrize("change", ["label", "anchor", "current", "count", "tick", "stamp", "omitted"])
def test_worker_context_rejects_corruption_or_label_channel(change):
    job, plan = job_and_context()
    if change == "label": plan["human_reference"] = "STOP"
    if change == "anchor": plan["source_indices"][0] = 2
    if change == "current": plan["source_indices"][-1] -= 1
    if change == "count": plan["source_count"] = 38
    if change == "tick": plan["timestamp_ticks"][2] = 500
    if change == "stamp": job["frames"][2]["observation_s"] += .01
    if change == "omitted": plan["omitted_source_indices"] = []
    with pytest.raises(ValueError): validate_context(plan, job)


def test_selection_does_not_modify_baseline_or_accumulate_between_episodes():
    source = frames(39, irregular=True)
    before = copy.deepcopy(source)
    a = select_context(source, source[-1]["available_s"])
    assert source == before and a["policy"] == POLICY
    for _ in range(1000):
        job, b = job_and_context(5)
        job.update(instruction="Changed", episode_id="other", clock_epoch="new_epoch")
        validate_context(b, job)
        assert b["source_indices"] == list(range(5))
    assert a == select_context(source, source[-1]["available_s"])


def test_long_sessions_explicitly_stop_instead_of_silently_sliding_instruction():
    for n in (39,121,241):
        job, p = job_and_context(n)
        assert len(job["frames"]) <= 8
        assert p["origin_observation_s"] == 100
    with pytest.raises(ValueError, match="duration"):
        job_and_context(242)


def test_experiment_imports_no_hardware_or_action_client():
    import ast
    root = Path(__file__).resolve().parents[1]
    for name in ("scripts/run_vln_bounded_context.py", "scripts/vln_bounded_worker.py",
                 "src/go2_perception/vln/bounded_context.py"):
        tree = ast.parse((root/name).read_text())
        imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(any(s in m for s in ("unitree", "cyclonedds", "shadow_live", "go2_control", "sensor_bridge")) for m in imports)


def test_baseline_worker_invocation_keeps_default_protocol(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("bounded_worker_test", root/"scripts/vln_bounded_worker.py")
    import sys
    monkeypatch.syspath_prepend(str(root/"scripts"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import io, json
    from types import SimpleNamespace
    job, context = job_and_context()
    calls, sent = [], []
    def run(agent, job, dest, device, **kw):
        calls.append(kw); return {"status":"mocked"}
    for context in (None, context):
        message = {"identity":{"episode_id":"one"}, "job":job, "destination":str(tmp_path), "bounded_context":context}
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(message)+"\n"))
        module.serve(None, SimpleNamespace(device="cpu",profile_stages=True), sent.append, None, run, lambda data: None)
    assert calls[0]["sample_factory"] is None
    assert callable(calls[1]["sample_factory"])
    assert len(sent) == 2
