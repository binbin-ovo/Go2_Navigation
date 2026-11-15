"""Lightweight deterministic advisory tests: no torch, model, simulator or robot."""

import ast
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from go2_perception.vln.observation import load
from go2_perception.vln.shadow import Limits, ShadowSession, validate_proposal
from go2_perception.vln.shadow_display import point_fraction, target_fraction, render
from go2_perception.vln.shadow_sources import (
    EpisodeSource,
    ObservationInbox,
    SyntheticSource,
)
from go2_perception.vln.shadow_worker import MockWorker, ProcessWorker

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frames():
    return list(SyntheticSource(17, fps=16))


def session(tmp_path, *, delay=1.0, mode="synthetic_fast", **limits):
    s = ShadowSession(
        tmp_path / "run",
        MockWorker(delay),
        mode=mode,
        limits=Limits(**limits),
        allow_synthetic=True,
        session_id="test",
    )
    s.instruction_changed("Approach the doorway.", 0.0)
    return s


def receive(s, frames, first=0, last=None):
    for i in range(first, len(frames) if last is None else last):
        s.receive(frames[i], i / 16)


def completed(tmp_path, frames):
    s = session(tmp_path)
    receive(s, frames, last=5)
    s.tick(0.25)
    request = s.inflight
    s.tick(1.25)
    return s, request, s.displayed[1]["response"]["result"]


def test_one_inflight_coalesces_requests_but_preserves_real_history(tmp_path, frames):
    s = session(tmp_path)
    receive(s, frames, last=1)
    s.tick(0.0)
    first = s.inflight
    receive(s, frames, first=1)
    assert s.counts["submitted"] == 1
    assert s.pending is True and len(s.history) == 17
    assert s.counts["coalesced"] == 15
    s.tick(1.0)
    assert s.counts["accepted"] == 1 and s.counts["submitted"] == 2
    assert s.inflight.context["identity"]["sequence"] == 2
    # Latest-anchored real 4 Hz sampling, not the old one-request queue's frames.
    assert [x["observation_s"] for x in s.inflight.job["frames"]] == [
        100.0,
        100.25,
        100.5,
        100.75,
        101.0,
    ]
    assert len(first.job["frames"]) == 1
    assert s.inflight.job.keys() == {
        "instruction",
        "episode_id",
        "clock_epoch",
        "decision_id",
        "decision_time_s",
        "frames",
    }
    assert s.inflight.context["previous_actions_supplied"] is False


def test_snapshot_bytes_and_instruction_are_immutable(tmp_path, frames):
    s = session(tmp_path)
    s.receive(frames[0], 0.0)
    s.tick(0.0)
    request = s.inflight
    frames[0].observation.visual.image[:] = 0
    job = request.job
    job["instruction"] = "changed"
    job["frames"].clear()
    assert request.job["instruction"] == "Approach the doorway."
    assert len(request.job["frames"]) == 1
    assert load(s.latest["observation_path"]).visual.image.any()
    assert request.context["history"][0]["source"]["frame_id"] == "f00000"


def test_new_instruction_invalidates_old_result_and_resets_history(tmp_path, frames):
    s = session(tmp_path)
    receive(s, frames, last=1)
    s.tick(0.0)
    s.instruction_changed("Turn left.", 0.1)
    assert s.history == [] and not s.pending
    assert s.latest["frame_id"] == "f00000"  # historical view, not fed into revision 2
    s.receive(frames[4], 0.25)
    s.tick(1.0)
    assert s.displayed[1]["rejection_reason"] == "superseded_instruction"
    assert s.status(1.0)["displayed"]["label"] == "SUPERSEDED"
    assert s.inflight.job["instruction"] == "Turn left."
    assert len(s.inflight.job["frames"]) == 1
    assert s.inflight.context["identity"]["instruction_revision"] == 2


def test_cancel_during_generation_does_not_accept_or_send_stop(tmp_path, frames):
    s = session(tmp_path)
    s.receive(frames[0], 0.0)
    s.tick(0.0)
    s.cancel(0.1)
    s.tick(1.0)
    assert s.counts["accepted"] == 0 and s.counts["rejected"] == 1
    assert s.displayed[1]["rejection_reason"] == "cancelled_or_closed"
    assert not s.pending and s.inflight is None
    events = [json.loads(x) for x in s.events_path.read_text().splitlines()]
    assert (
        next(x for x in events if x["event"] == "advisory_cancelled")[
            "physical_robot_stop_sent"
        ]
        is False
    )


def test_epoch_reset_even_if_old_epoch_name_reappears(tmp_path, frames):
    s = session(tmp_path)
    s.receive(frames[0], 0.0)
    s.tick(0.0)
    frames[1].observation.clock_epoch = "boot2"
    s.receive(frames[1], 0.0625)
    s.receive(frames[2], 0.125)
    assert len(s.history) == 1
    s.tick(1.0)
    assert s.displayed[1]["rejection_reason"] == "superseded_source_or_epoch"


@pytest.mark.parametrize(
    "field,value", [("session_id", "wrong"), ("clock_epoch", "wrong"), ("sequence", 99)]
)
def test_wrong_response_identity(tmp_path, frames, field, value):
    s = session(tmp_path, delay=0)
    s.receive(frames[0], 0.0)
    s.tick(0.0)
    response = s.worker.poll(0.0)
    response["identity"][field] = value
    s.worker.poll = lambda _: response
    s.tick(0.1)
    assert s.displayed[1]["rejection_reason"] == "wrong_session_epoch_or_sequence"
    assert not s.counts["accepted"]


def test_stale_at_arrival_and_ages_while_displayed(tmp_path, frames):
    s = session(tmp_path, advisory_age_s=0.5)
    s.receive(frames[0], 0.0)
    s.tick(0.0)
    s.tick(1.0)
    assert s.status(1.0)["displayed"]["label"] == "STALE"
    assert s.counts["rejected"] == 1


def test_current_proposal_expires_without_new_response(tmp_path, frames):
    s, _, _ = completed(tmp_path, frames)
    assert s.status(1.25)["displayed"]["label"] == "CURRENT ADVISORY"
    assert s.status(5.0)["displayed"]["label"] == "STALE"


def test_response_deadline_independent_of_capture_age(tmp_path, frames):
    s = session(tmp_path, response_deadline_s=0.5, advisory_age_s=10)
    s.receive(frames[0], 0.0)
    s.tick(0.0)
    s.tick(1.0)
    assert s.displayed[1]["rejection_reason"] == "stale_response_deadline"


def test_paced_replay_age_uses_virtual_clock_not_robot_vs_host_wall(tmp_path, frames):
    s = session(tmp_path, mode="synthetic_paced")
    s.set_replay_anchor(100.02, 0.0, speed=2.0)
    s.receive(frames[0], 0.0)
    s.tick(0.0)
    s.tick(1.0)
    timing = s.status(1.0)["displayed"]["display_timing"]
    assert timing["observation_age_at_display_s"] == pytest.approx(2.02)
    assert timing["host_since_delivery_s"] == 1
    assert "not_live" in timing["age_basis"]


def test_fast_replay_does_not_claim_measured_capture_age(tmp_path, frames):
    s, request, _ = completed(tmp_path, frames)
    timing = s.timing(request, 1.25)
    assert timing["observation_age_at_display_s"] is None
    assert timing["source_age_at_cutoff_s"] == pytest.approx(0.02)
    assert timing["host_since_delivery_s"] == 1
    receipt_only = session(tmp_path / "receipt_only", mode="live_unvalidated")
    frames[0].observation.visual.stamp_robot = None
    receipt_only.receive(
        replace(frames[0], available_s=frames[0].observation.visual.stamp_host.t), 0.0
    )
    receipt_only.tick(0.0)
    timing = receipt_only.timing(receipt_only.inflight, 1.0)
    assert timing["observation_age_at_display_s"] is None
    assert timing["source_stamp_semantics"] == "host_receipt"


def test_unvalidated_live_mapping_requires_explicit_provenance(tmp_path, frames):
    s = session(tmp_path, mode="live_unvalidated")
    d = replace(
        frames[0],
        received_monotonic_s=0.0,
        exposure_monotonic_s=-0.1,
        monotonic_mapping="explicit test-only host calibration",
    )
    s.receive(d, 0.0)
    s.tick(0.0)
    assert s.timing(s.inflight, 1.0)["observation_age_at_display_s"] == 1.1
    with pytest.raises(ValueError, match="mapping"):
        replace(d, monotonic_mapping=None)


@pytest.mark.parametrize(
    "limits,expected",
    [({"history_frames": 2}, "history_limit"), ({"history_s": 0.07}, "history_limit")],
)
def test_capacity_pauses_instead_of_silently_losing_anchors(
    tmp_path, frames, limits, expected
):
    s = session(tmp_path, **limits)
    receive(s, frames, last=3)
    assert expected in s.history_error
    assert len(s.history) == 2 and not s.pending
    s.tick(0.5)
    assert s.counts["submitted"] == 0


def test_sparse_stream_is_not_padded_or_time_compressed(tmp_path, frames):
    s = session(tmp_path)
    s.receive(frames[0], 0.0)
    s.receive(frames[8], 0.5)
    assert "gap" in s.history_error and not s.pending
    assert len(s.history) == 1


def test_missing_image_and_unknown_calibration_fail_explicitly(tmp_path, frames):
    s = session(tmp_path)
    frames[0].observation.visual = None
    s.receive(frames[0], 0.0)
    assert s.history_error == "missing_image"
    s.instruction_changed("Retry with valid calibration", 0.1)
    frames[4].observation.visual.intrinsics_status = "unknown"
    s.receive(frames[4], 0.25)
    s.tick(0.25)
    assert "Measured image intrinsics" in s.history_error


def test_missing_pose_lidar_stay_none_and_out_of_model_input(tmp_path, frames):
    s, request, _ = completed(tmp_path, frames)
    assert not s.latest["pose_available"] and not s.latest["lidar_available"]
    obs = load(s.latest["observation_path"])
    assert obs.robot is None and obs.geometric is None
    assert "T_odom_base" not in request.job_json and "goal" not in request.job_json
    assert request.context["current_source"]["T_base_camera"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"native_output": "plausible nonsense"},
        {"waypoints": [[0, 0, float("nan")]] * 10},
        {"execution_authorized": True},
        {"decoder_stop": True},
        {"completion_verified": True},
        {"waypoint_time_base": 0.25},
        {"explicit_stop_code": True},
        {"waypoint_convention": "Go2 base"},
    ],
)
def test_malformed_output_never_current(tmp_path, frames, change):
    _, _, result = completed(tmp_path, frames)
    with pytest.raises((ValueError, KeyError, TypeError)):
        validate_proposal({**result, **change})


def test_stop_claim_and_completion_are_separate(tmp_path, frames):
    _, _, result = completed(tmp_path, frames)
    result.update(
        native_output="<apos_1299><opos_0><act_l0_6><act_l1_0><act_l2_0>",
        rvq_codes=[6, 0, 0],
        decoder_stop=True,
        explicit_stop_code=True,
        waypoints=np.zeros((10, 3)).tolist(),
        pointing={"apos": 1299},
    )
    semantics = validate_proposal(result)
    assert semantics["native_stop"] and semantics["model_completion_claim"]
    assert semantics["completion_verified"] is False
    assert semantics["physical_robot_stopped"] is None
    result["pointing"]["apos"] = 0
    assert validate_proposal(result)["model_completion_claim"] is False


def test_shutdown_and_hung_generation_are_bounded(tmp_path, frames):
    s = session(tmp_path, delay=100, worker_timeout_s=2, response_deadline_s=1)
    s.receive(frames[0], 0.0)
    s.tick(0.0)
    s.tick(3.0)
    assert not s.worker.ready and s.inflight is None
    assert s.history_error == "worker_timeout_restart_application"
    s.close(4.0)
    s.close(5.0)
    assert s.closed and not s.active


def test_close_invalidates_running_request(tmp_path, frames):
    s = session(tmp_path)
    s.receive(frames[0], 0.0)
    s.tick(0.0)
    s.close(0.2)
    assert s.inflight is None and s.worker.pending is None and s.closed
    assert "inflight_invalidated_on_shutdown" in s.events_path.read_text()


def test_inbox_ownership_overflow_and_optional_fields(frames):
    q = ObservationInbox(capacity=1)
    obs = frames[0].observation
    assert q.offer(
        obs,
        source_id="receiver",
        frame_id="a",
        available_s=100.02,
        received_monotonic_s=2.0,
    )
    assert not q.offer(obs, source_id="receiver", frame_id="b", available_s=100.02)
    obs.visual.image[:] = 0
    item = q.get_nowait()
    assert item.observation.visual.image.any() and item.observation.robot is None
    assert item.received_monotonic_s == 2.0


def test_episode_input_never_loads_annotations_and_preserves_files():
    source = EpisodeSource(ROOT / "tests/fixtures/vln/episode_v1/episode.json")
    before = source.episode.sha256
    values = list(source)
    assert values[2].observation.visual is None
    assert values[0].observation.robot is None
    assert values[0].provenance["annotations_consumed"] is False
    assert source.episode.sha256 == before
    assert values[0].observation.visual.intrinsics_status == "synthetic"


def test_display_has_exact_history_timestamps_and_no_latest_pose_overlay(
    tmp_path, frames
):
    s, request, _ = completed(tmp_path, frames)
    state = render(s, 1.25, snapshot=True)
    page = (s.output / "index.html").read_text()
    assert "ADVISORY ONLY — NO ROBOT COMMANDS" in page
    assert "CURRENT ADVISORY" in page and "MOCK — no model inference" in page
    assert "100.000000 s" in page and "100.250000 s" in page
    assert "No Go2 base registration" in page
    assert "Verified goal completion: UNKNOWN" in page
    assert (s.output / "advisory.png").is_file()
    assert state["physical_execution_authorized"] is False
    s.instruction_changed("<script>unsafe</script>", 1.3)
    render(s, 1.3)
    assert (
        "&lt;script&gt;unsafe&lt;/script&gt;" in (s.output / "index.html").read_text()
    )


@pytest.mark.parametrize(
    "token,point",
    [
        (0, None),
        (1299, None),
        (-1, None),
        (1, (0.5 / 48, 0.5 / 27)),
        (1296, (47.5 / 48, 26.5 / 27)),
    ],
)
def test_native_target_grid_pointing(token, point):
    assert point_fraction(token) == point


@pytest.mark.parametrize(
    "bad",
    [
        {"history_frames": 0},
        {"advisory_age_s": float("nan")},
        {"worker_timeout_s": 1},
        {"request_interval_s": 0},
    ],
)
def test_invalid_runtime_limits_rejected(bad):
    with pytest.raises(ValueError):
        Limits(**bad)


def test_no_control_imports_or_execution_options():
    paths = list((ROOT / "src/go2_perception/vln").glob("shadow*.py"))
    paths += [ROOT / "scripts/run_vln_shadow.py", ROOT / "scripts/vln_shadow_worker.py"]
    forbidden = (
        "unitree",
        "go2_control",
        "cyclonedds",
        "rclpy",
        "socket",
        "simulation",
        "adapters.go2",
    )
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(word in name for word in forbidden for name in names)
    spec = importlib.util.spec_from_file_location(
        "shadow_cli", ROOT / "scripts/run_vln_shadow.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    opts = module.parser()._option_string_actions
    assert not {"--execute", "--arm", "--live", "--robot", "--interface"}.intersection(
        opts
    )
    # This imports the actual entrypoint without importing Torch or any robot client.
    assert not any(
        name.startswith(("unitree_sdk2py", "go2_control"))
        for name in __import__("sys").modules
    )


def test_owned_process_shutdown_terminates_then_kills_if_needed():
    calls = []

    class Pipe:
        def close(self):
            calls.append("pipe_close")

    class Child:
        stdin, stdout = Pipe(), Pipe()

        def poll(self):
            return None

        def terminate(self):
            calls.append("terminate")

        def wait(self, timeout):
            calls.append(("wait", timeout))
            if "kill" not in calls:
                raise subprocess.TimeoutExpired("owned test child", timeout)

        def kill(self):
            calls.append("kill")

    from queue import Queue

    w = ProcessWorker.__new__(ProcessWorker)
    w.closed, w.ready, w.process = False, True, Child()
    w.messages, w.log = Queue(maxsize=2), Pipe()
    w.reader = SimpleNamespace(join=lambda timeout: calls.append(("join", timeout)))
    w.close()
    w.close()
    assert calls[:4] == ["terminate", ("wait", 3), "kill", ("wait", 3)]


@pytest.mark.parametrize(
    "bad_result",
    [None, [], {"waypoints": [[float("nan"), 0, 0]] * 10}, {"pointing": None}],
)
def test_malformed_worker_response_is_saved_and_rejected(tmp_path, frames, bad_result):
    s = session(tmp_path, delay=0)
    s.receive(frames[0], 0.0)
    s.tick(0.0)
    response = s.worker.poll(0.0)
    response["result"] = bad_result
    s.worker.poll = lambda _: response
    s.tick(0.1)
    assert s.counts["rejected"] == 1 and not s.counts["accepted"]
    render(s, 0.1)
    assert s.status(0.1)["displayed"]["label"] == "REJECTED"


def test_exposure_cannot_follow_declared_availability(tmp_path, frames):
    s = session(tmp_path)
    s.receive(replace(frames[0], available_s=99.0), 0.0)
    assert s.history_error == "invalid_observation_clock"
    assert not s.history


def test_duplicate_frame_does_not_become_new_history(tmp_path, frames):
    s = session(tmp_path)
    s.receive(frames[0], 0.0)
    s.receive(frames[0], 0.1)
    assert s.history_error == "duplicate_or_out_of_order_observation"
    assert len(s.history) == 1


def test_clock_regression_rejected(tmp_path, frames):
    s = session(tmp_path)
    s.receive(frames[0], 1.0)
    with pytest.raises(ValueError, match="monotonic"):
        s.tick(0.5)


def test_worker_cannot_claim_negative_duration(tmp_path, frames):
    s = session(tmp_path, delay=0)
    s.receive(frames[0], 0.0)
    s.tick(0.0)
    response = s.worker.poll(0.0)
    response["host_inference_end_s"] = -1
    s.worker.poll = lambda _: response
    s.tick(0.1)
    assert s.displayed[1]["rejection_reason"] == "invalid_inference_monotonic_clock"


def test_posxy_only_target_pointing_and_conflicting_formats():
    assert target_fraction({"posxy": {"opos": [0, 999]}}) == (0.0005, 0.9995)
    assert target_fraction({"posxy": {"apos": [0, 999]}}) is None
    assert target_fraction({"opos": 2, "posxy": {"opos": [0, 999]}}) is None
    assert target_fraction({"posxy": {"opos": [1000, 0]}}) is None


def test_external_history_not_silently_reobserved(tmp_path, frames):
    s = session(tmp_path)
    frames[1].observation.visual.history = [frames[0].observation.visual]
    s.receive(frames[1], 0.1)
    assert s.history_error == "deliver_current_frame_only_session_owns_history"


def test_terminal_command_revisions_and_cancel(tmp_path, frames):
    spec = importlib.util.spec_from_file_location(
        "commands", ROOT / "scripts/run_vln_shadow.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    s = session(tmp_path)
    s.receive(frames[0], 0.0)
    s.tick(0.0)
    assert module.handle_command(s, "instruction Turn left now", 0.1)
    assert s.revision == 2 and s.instruction == "Turn left now"
    assert module.handle_command(s, "cancel", 0.2)
    assert not s.active
    assert module.handle_command(s, "status", 0.2)
    assert not module.handle_command(s, "quit", 0.2)


def test_inference_duration_must_match_monotonic_bounds(tmp_path, frames):
    s = session(tmp_path, delay=0)
    s.receive(frames[0], 0.0)
    s.tick(0.0)
    response = s.worker.poll(0.0)
    response["inference_duration_s"] = 100
    s.worker.poll = lambda _: response
    s.tick(0.1)
    assert s.displayed[1]["rejection_reason"] == "invalid_inference_duration"


def test_real_terminal_entrypoint_with_mock_worker_and_new_instruction(tmp_path):
    import os
    import sys

    output = tmp_path / "cli"
    command = [
        sys.executable,
        str(ROOT / "scripts/run_vln_shadow.py"),
        "--synthetic",
        "--synthetic-frames",
        "8",
        "--instruction",
        "Original terminal instruction",
        "--mode",
        "fast",
        "--mock-delay-s",
        "0.01",
        "--output",
        str(output),
    ]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    run = subprocess.run(
        command,
        input="instruction Revised terminal instruction\n",
        text=True,
        capture_output=True,
        env=env,
        cwd=ROOT,
        timeout=10,
    )
    assert run.returncode == 0, run.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["counts"]["received"] == 8 and summary["counts"]["accepted"] >= 1
    assert not summary["physical_execution_authorized"]
    job = json.loads(next((output / "requests").glob("*/job.json")).read_text())
    assert job["instruction"] == "Revised terminal instruction"
    run_info = json.loads((output / "run.json").read_text())
    assert run_info["implementation_sha256"]["scripts/run_vln_shadow.py"]
    assert (output / "index.html").is_file() and (output / "advisory.png").is_file()
    old_summary = (output / "summary.json").read_bytes()
    again = subprocess.run(
        command, input="", text=True, capture_output=True, env=env, cwd=ROOT, timeout=10
    )
    assert again.returncode == 2 and "new directory" in again.stderr
    assert (output / "summary.json").read_bytes() == old_summary
