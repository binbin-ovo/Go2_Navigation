"""Offline episode behavior, using synthetic poses, images, clocks and controller routes."""

from copy import deepcopy
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from go2_perception.vln.bounded_context import select_context, validate_context
from go2_perception.vln.episode_app import target_spec
from go2_perception.vln.episode_grounding import ground, object_reference
from go2_perception.vln.episode_mock import effective_mock_capability
from go2_perception.vln.episode_policy import area_check, primitive, select_action
from go2_perception.vln.five_move import SCHEMA, EpisodeExecutor, EpisodeFakeGuards
from go2_perception.vln.motion_profile import INSPECTED, measure_outcome
from go2_perception.vln.supervised import Hold, MODEL_REVISION, digest
from test_effective_motion import configure
from test_supervised_executor import hardware_fakes

ROOT = Path(__file__).resolve().parents[1]
OLD = json.loads((ROOT / "examples/supervised_navigation/synthetic_cycle.json").read_text())
FIXTURE = json.loads((ROOT / "examples/five_move_navigation/synthetic_episode.json").read_text())


class Clock:
    def __init__(self):
        self.now = 101.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def grant_for(ex, count=5):
    return {
        "schema": SCHEMA,
        "episode_id": ex.nonce,
        "capability_sha256": digest(ex.cap),
        "operator": "pilot",
        "target": "chair",
        "instruction": "Go to the chair and stop.",
        "trial_id": "trial1",
        "session_id": "session1",
        "instruction_id": "instruction1",
        "clock_epoch": "camera_epoch1",
        "origin": ex.guards.snapshot()["pose"],
        "max_moves": count,
        "issued": ex.clock(),
        "deadline": ex.clock() + 60,
        "area_radius_m": 3.0,
        "alignment_deg": 15.0,
        "area_inspected": True,
        "exclusive_and_intervention_confirmed": True,
    }


def episode(count=5):
    clock = Clock()
    events = []
    ex = EpisodeExecutor(
        effective_mock_capability(deepcopy(OLD["capability"])),
        "nonce1",
        guards=EpisodeFakeGuards(clock),
        monotonic=clock,
        sleep=clock.sleep,
        wall=lambda: 1,
        emit=events.append,
    )
    ex.arm(grant_for(ex, count))
    return ex, events


def observation(ex, seq, *, angle=None, extent=0.48, yaw=0.0, stop=False, status=None):
    ex.sleep(0.10)
    snap = ex.guards.snapshot()
    p = deepcopy(OLD["proposal"])
    p["synthetic"] = not ex.live
    if ex.live:
        p["model_identity"].update(revision=MODEL_REVISION, processor_revision=MODEL_REVISION)
    i = p["identity"]
    i.update(
        {k: ex.grant[k] for k in ("session_id", "trial_id", "instruction_id", "instruction", "clock_epoch")}
    )
    i.update(
        prediction_id=f"P{seq}",
        observation_id=f"F{seq}",
        source_image_sha256=f"{seq:064x}",
        observed_monotonic_s=ex.clock(),
        received_monotonic_s=ex.clock(),
        uncertainty_bound_s=snap["capture_uncertainty_bound_s"],
        pose=snap["pose"],
        geometry=snap["geometry"],
    )
    p.update(
        waypoints=[[i * extent / 10.0, 0.0, i * math.radians(yaw) / 10.0] for i in range(1, 11)],
        native_stop=stop,
        point_fraction=None if angle is None else [0.5 - 300 / 448 * math.tan(math.radians(angle)), 0.5],
    )
    e = {
        "status": status or ("absent" if angle is None else "supported"),
        "target": "chair",
        "image_sha256": i["source_image_sha256"],
        "prediction_id": i["prediction_id"],
        "camera": {"width": 448, "fx": 300.0, "cx": 224.0},
        "method": "native_visibility",
    }
    if e["status"] == "supported":
        e.update(method="synthetic_fixture", roi=[0.0, 0.0, 1.0, 1.0])
    return dict(episode_id=ex.nonce, proposal=p, evidence=e)


def choice(ex, msg, aligned=False):
    return select_action(
        ex.grant, ex.cap, msg["proposal"], msg["evidence"], ex.views, len(ex.attempts), aligned
    )


def test_absent_search_then_new_grounded_view_approach_and_distinct_stop():
    ex, events = episode()
    ex.observe(observation(ex, 1, extent=0, yaw=-10))
    assert (ex.mode, ex.attempts[0]["source"], ex.attempts[0]["direction"]) == (
        "SEARCH",
        "search_policy",
        "right",
    )
    ex.observe(observation(ex, 2, angle=18, extent=0.04, yaw=11))
    assert ex.selection["action"]["magnitude"] == pytest.approx(math.radians(20))
    assert ex.selection["source"] == "approach_policy"
    ex.observe(observation(ex, 3, angle=9, yaw=11))
    assert ex.attempts[-1]["direction"] == "forward" and len(ex.attempts) == 3
    observed_target = ex.last_target
    ex.observe(observation(ex, 4, stop=True, extent=0))
    assert ex.outcome == "MODEL_STOPPED_ARRIVAL_UNVERIFIED" and ex.last_target == observed_target
    assert ex.found and len({a["action_id"] for a in ex.attempts}) == 3
    assert len([e for e in events if e["type"] == "episode_authorization"]) == 1
    with pytest.raises(Hold):
        ex.review(
            dict(
                type="episode_goal",
                episode_id=ex.nonce,
                operator="pilot",
                prediction_id="P3",
                image_sha256="x",
            )
        )
    ex.review(
        dict(
            type="episode_goal",
            episode_id=ex.nonce,
            operator="pilot",
            prediction_id="P4",
            image_sha256=ex.last["image_sha256"],
        )
    )
    assert ex.previous_outcome == "MODEL_STOPPED_ARRIVAL_UNVERIFIED" and ex.outcome == "GOAL_CONFIRMED"
    with pytest.raises(Hold):
        ex.arm(grant_for(ex))


@pytest.mark.parametrize(
    ("angle", "yaw", "previous", "direction"),
    [
        (6, 7, False, "forward"),
        (-14, -12, False, "forward"),
        (15, 12, False, "forward"),
        (16, 12, False, "left"),
        (-19, -12, False, "right"),
        (17, 12, True, "forward"),
        (-17, -12, True, "forward"),
        (19, 12, True, "left"),
    ],
)
def test_alignment_gap_and_hysteresis(angle, yaw, previous, direction):
    ex, _ = episode()
    s = choice(ex, observation(ex, 1, angle=angle, yaw=yaw), previous)
    assert s["action"]["direction"] == direction
    if direction != "forward":
        assert s["action"]["magnitude"] == pytest.approx(math.radians(20))
        assert s["expected_residual_rad"] + ex.cap["mapping"]["yaw_uncertainty_rad"] < math.radians(15)
    assert s["action"]["speed"] == (1.0 if direction == "forward" else 2.0)


@pytest.mark.parametrize("shape", ["curve", "reverse", "lateral"])
def test_incompatible_native_path_cannot_be_called_a_straight_approach(shape):
    ex, _ = episode()
    msg = observation(ex, 1, angle=7, yaw=10)
    for i, row in enumerate(msg["proposal"]["waypoints"], 1):
        if shape == "reverse":
            row[0] = -row[0]
        else:
            row[1] = i * (0.055 if shape == "curve" else 0.15)
    with pytest.raises(Hold, match="does_not_support_forward"):
        choice(ex, msg)


def test_area_rechecked_after_translation_and_no_floor_is_not_free_space():
    ex, _ = episode()
    g = dict(ex.grant, area_radius_m=2.0)
    action = primitive(ex.cap, "forward", 0.30, "test")
    area_check(g, g["origin"], action, attempts=1)
    ex.guards.x = 0.80
    with pytest.raises(Hold, match="outside_inspected_area"):
        area_check(g, ex.guards.snapshot()["pose"], action, attempts=2)
    with pytest.raises(Hold, match="epoch_or_frame_changed"):
        area_check(g, dict(g["origin"], robot_epoch="other"))
    msg = observation(ex, 1)
    geo = msg["proposal"]["identity"]["geometry"]
    geo.update(available=False, forward_clear_m=0.0, directions=[])
    with pytest.raises(Hold, match="no_supported_search_viewpoint"):
        choice(ex, msg)


def test_five_total_attempts_including_stall_then_observation_only():
    ex, events = episode()
    move = ex._fake_motion

    def simulate_first_stall(action, result, **kwargs):
        if len(ex.attempts) != 1:
            return move(action, result, **kwargs)
        result.update(
            started_monotonic_s=ex.clock(),
            status="ended",
            reason="stalled",
            command_send_statuses=[0],
            stop_send_statuses=[0, 0, 0],
            observed_stationary=True,
            ownership_released=True,
            release_rpc_code=0,
            post_release_stationary=True,
            measured_progress=0.0,
            motion_measurement=measure_outcome(
                action, 0.0, 0.0, 0.0, termination="blocked_stall", stop_verified=True
            ),
        )
        return result

    ex._fake_motion = simulate_first_stall
    ex.observe(observation(ex, 1))
    assert ex.attempts[0]["no_progress"] and ex.state == "OBSERVING"
    for seq in range(2, 6):
        ex.observe(observation(ex, seq, angle=0, extent=0.35))
    assert ex.state == "OBSERVING" and len(ex.attempts) == 5 and ex.found
    ex.observe(observation(ex, 6, angle=0))
    assert (
        ex.outcome == "MOVE_BUDGET_EXHAUSTED"
        and len([e for e in events if e["type"] == "episode_dispatch"]) == 5
    )
    with pytest.raises(Hold, match="inactive"):
        ex.observe(observation(ex, 7, angle=0))
    assert len(ex.used) == 5


def test_stale_duplicate_wrong_target_source_and_boot_cannot_dispatch():
    ex, _ = episode()
    msg = observation(ex, 1)
    msg["proposal"]["identity"]["observed_monotonic_s"] = ex.after
    ex.observe(msg)
    assert ex.state == "OBSERVING" and not ex.attempts
    with pytest.raises(Hold, match="already_seen"):
        ex.observe(msg)
    msg = observation(ex, 2)
    ex.sleep(4.0)
    ex.observe(msg)
    assert ex.reason == "stale_or_future_observation" and not ex.attempts
    msg = observation(ex, 3, angle=0)
    msg["evidence"]["target"] = "shoe rack"
    with pytest.raises(Hold, match="wrong_target"):
        ex.observe(msg)
    msg = observation(ex, 4)
    msg["proposal"]["identity"]["clock_epoch"] = "new epoch"
    with pytest.raises(Hold, match="epoch_changed"):
        ex.observe(msg)
    assert not ex.attempts


def test_mark_review_requires_new_observation_and_deadline_is_independent_of_inference():
    ex, _ = episode()
    msg = observation(ex, 1, angle=5, status="needs_identification")
    ex.observe(msg)
    assert ex.state == "IDENTIFYING" and not ex.found and not ex.attempts
    ex.review(
        dict(
            type="episode_identified",
            episode_id=ex.nonce,
            operator="pilot",
            prediction_id="P1",
            image_sha256=ex.last["image_sha256"],
        )
    )
    ex.sleep(61.0)
    ex.tick()
    assert ex.outcome == "EPISODE_DEADLINE"
    with pytest.raises(Hold):
        ex.observe(observation(ex, 2))


def test_operator_cancel_during_inference_and_execution_is_terminal():
    for during in (False, True):
        ex, _ = episode()
        msg = observation(ex, 1)
        if during:

            def cancel_after_send(dt):
                ex.clock.sleep(dt)
                ex.cancelled.set()

            ex.sleep = cancel_after_send
            ex.observe(msg)
            assert len(ex.attempts) == 1 and ex.attempts[0]["stop_verified"]
        else:
            ex.cancelled.set()
            ex.tick()
            with pytest.raises(Hold):
                ex.observe(msg)
            assert not ex.attempts
        assert ex.outcome == "OPERATOR_ABORT"
        with pytest.raises(Hold):
            ex.arm(grant_for(ex))


def test_real_per_action_code_with_fake_route_and_interrupted_geometry(tmp_path):
    motion, _, cancel, calls, clock = hardware_fakes(tmp_path)
    cap = configure(motion.cap, INSPECTED)
    base = motion.guards.snapshot

    def snapshot():
        snap = base()
        snap.update(
            capture_clock_epoch="camera_epoch1",
            capture_bound_basis="simulated",
            capture_uncertainty_bound_s=0.001,
        )
        snap["geometry"].update(
            forward_clear_m=3.0,
            observed_turn_disc={"radius_m": 0.65, "floor_returns": 10, "obstacle_returns": 0},
        )
        if calls.count("velocity") >= 2:
            snap["geometry"]["observed_turn_disc"]["obstacle_returns"] = 3
        return snap

    motion.guards.snapshot = snapshot
    events = []
    ex = EpisodeExecutor(
        cap,
        "nonce1",
        live=True,
        guards=motion.guards,
        motion=motion,
        monotonic=clock,
        sleep=clock.sleep,
        cancelled=cancel,
        wall=lambda: 1,
        emit=events.append,
    )
    ex.arm(grant_for(ex))
    ex.observe(observation(ex, 1, extent=0, yaw=12))
    assert (
        calls.count("claim") == 1
        and calls.count("velocity") == 2
        and calls[-4:] == ["stop", "stop", "stop", "release"]
    )
    result = next(e["result"] for e in events if e["type"] == "result")
    reject = next(e for e in result["events"] if e.get("event") == "guard_rejected")
    assert reject["guards"]["geometry"]["observed_turn_disc"]["obstacle_returns"] == 3
    assert reject["nonzero_sends"] == 2 and result["post_release_stationary"]
    assert ex.outcome == "BLOCKED_OR_NO_USEFUL_ACTION" and not ex.retain_lock


def test_grounding_requires_reidentification_in_the_current_image(tmp_path):
    rng = np.random.default_rng(167)
    patch = rng.integers(0, 256, (140, 120), dtype=np.uint8)
    old = np.full((256, 448), 50, np.uint8)
    old[45:185, 60:180] = patch
    new = np.full_like(old, 130)
    new[60:200, 260:380] = patch

    def save(name, img):
        f = tmp_path / name
        assert cv2.imwrite(str(f), img)
        return f

    ref = object_reference(save("ref.png", old), [60 / 448, 45 / 256, 180 / 448, 185 / 256], "chair", "pilot")
    dest = save("new.png", new)
    view = {
        "view": {
            "rectified": True,
            "distortion_model": "pinhole_radtan",
            "size_wh": [448, 256],
            "K": [[300, 0, 224], [0, 300, 128], [0, 0, 1]],
        }
    }
    pred = {
        "prediction_id": "p",
        "source_image": {"file_sha256": digest("new")},
        "opos_not_visible": False,
        "target_fraction": [0.71, 0.51],
    }
    e = ground(pred, dest, view, "chair", ref)
    assert e["status"] == "supported" and e["inliers"] >= 12
    assert ground(pred, dest, view, "shoe rack", ref)["status"] == "needs_identification"
    assert (
        ground(dict(pred, target_fraction=[0.1, 0.4]), dest, view, "chair", ref)["status"]
        == "needs_identification"
    )
    assert ground(pred, dest, view, "chair", None, rejected=ref)["status"] == "uncertain"
    assert (
        ground(dict(pred, target_fraction=None, opos_not_visible=True), dest, view, "chair", ref)["status"]
        == "absent"
    )


def test_bounded_selection_preserves_anchors_current_and_actual_source_time():
    frames = [{"observation_s": 10 + i * 0.25, "available_s": 10.01 + i * 0.25} for i in range(20)]
    plan = select_context(frames, frames[-1]["available_s"])
    assert plan["source_indices"] == [0, 1, 14, 15, 16, 17, 18, 19]
    job = {"frames": [frames[i] for i in plan["source_indices"]]}
    validate_context(plan, job)
    with pytest.raises(ValueError):
        select_context(frames[:4] + frames[10:], frames[-1]["available_s"])
    assert target_spec("Go to the chair.")["instruction"] == "Go to the chair and stop."


def test_archive_native_current_uses_grid_index_but_hashes_the_correct_job_frame(tmp_path):
    from types import SimpleNamespace
    from go2_perception.vln.interactive_app import prediction_source_image

    frames = [{"observation_s": i * 0.25, "available_s": i * 0.25} for i in range(20)]
    plan = select_context(frames, frames[-1]["available_s"])
    request = SimpleNamespace(
        context={"bounded_context": plan},
        job={"frames": [frames[i] for i in plan["source_indices"]]},
        folder=tmp_path,
    )
    (tmp_path / "native").mkdir()
    path = tmp_path / "native/input_0019.png"
    path.write_bytes(b"official archived input")
    result = {"native_selected_frame_indices": plan["source_indices"]}
    archived = prediction_source_image(request, result, tmp_path)
    assert (archived["path"], archived["job_frame_index"], archived["causal_grid_index"]) == (
        "native/input_0019.png",
        7,
        19,
    )
    assert prediction_source_image(request, {"native_selected_frame_indices": [0, 7]}, tmp_path) is None


@pytest.mark.parametrize("budget", [0, 6, False])
def test_child_rejects_grant_outside_dispatch_cap(budget):
    clock = Clock()
    ex = EpisodeExecutor(
        effective_mock_capability(deepcopy(OLD["capability"])),
        "test",
        guards=EpisodeFakeGuards(clock),
        monotonic=clock,
        sleep=clock.sleep,
        emit=lambda e: None,
    )
    with pytest.raises(Hold, match="move_budget_out_of_range"):
        ex.arm(grant_for(ex, budget))
    assert not ex.used and ex.state == "IDLE"


def test_sensor_error_during_motion_interrupts_immediately_and_records_cause():
    ex, _ = episode()
    msg = observation(ex, 1)

    def fail_recorder(dt):
        ex.clock.sleep(dt)
        ex.interrupt_sensor("lidar_recorder_source_stopped")

    ex.sleep = fail_recorder
    ex.observe(msg)
    assert ex.outcome == "SENSOR_OR_CONTROL_FAILURE" and ex.reason == "lidar_recorder_source_stopped"
    assert ex.attempts[0]["stop_verified"] and len(ex.attempts) == 1


def test_published_contract_accepts_actual_grant_active_terminal_and_fixture():
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(
        json.loads((ROOT / "src/go2_perception/vln/schemas/five-move-episode-v1.schema.json").read_text())
    )
    ex, events = episode(1)
    validator.validate(ex.grant)
    validator.validate(FIXTURE)
    ex.observe(observation(ex, 1))
    ex.observe(observation(ex, 2))
    for event in events:
        if event["type"] == "episode_state":
            validator.validate(event["episode"])
    assert not validator.is_valid(dict(ex.grant, max_moves=6))
    assert not validator.is_valid(dict(ex.public(), outcome="GOAL_VISIBLE"))


def test_time_budget_after_last_move_never_allows_arrival_on_its_pre_motion_image():
    ex, _ = episode(1)
    ex.observe(observation(ex, 1, angle=5))
    ex.sleep(61)
    ex.tick()
    assert ex.outcome == "MOVE_BUDGET_EXHAUSTED" and not ex.arrival_review_available()
    with pytest.raises(Hold, match="incompatible"):
        ex.review(
            dict(
                type="episode_goal",
                episode_id=ex.nonce,
                operator="pilot",
                prediction_id="P1",
                image_sha256=ex.last["image_sha256"],
            )
        )


def test_fixture_camera_changes_only_on_verified_progress_in_expected_direction():
    from go2_perception.vln.episode_mock import EpisodeMockScene

    scene = EpisodeMockScene(FIXTURE)
    camera = scene.source(6, 8)
    first = next(camera)
    result = {"motion_measurement": {"stop_verified": True, "outcome": "partial_progress"}}
    scene.settled("right", result)
    scene.settled(
        "left", {"motion_measurement": {"stop_verified": True, "outcome": "no_measurable_progress"}}
    )
    assert np.array_equal(first.observation.visual.image, next(camera).observation.visual.image)
    scene.settled("left", result)
    new = next(camera)
    assert new.provenance["synthetic_fixture_view"] == 1
    assert not np.array_equal(first.observation.visual.image, new.observation.visual.image)


def test_fresh_image_cannot_reuse_geometry_from_before_the_previous_settle():
    ex, _ = episode()
    ex.observe(observation(ex, 1))
    next_view = observation(ex, 2, angle=8)
    next_view["proposal"]["identity"]["geometry"]["monotonic_s"] = ex.after - 0.01
    ex.observe(next_view)
    assert ex.state == "OBSERVING" and ex.reason == "guard_observation_precedes_settling_or_grant"
    assert len(ex.attempts) == 1


@pytest.mark.parametrize("intervention", ["radio", "stale_remote"])
def test_idle_control_supervision_latches_before_an_inflight_model_can_dispatch(intervention):
    from types import SimpleNamespace

    ex, _ = episode()
    pending = observation(ex, 1)
    ex.live = True
    ex.motion = SimpleNamespace(
        monitor=SimpleNamespace(
            remote_t=ex.clock() if intervention == "radio" else ex.clock() - 1.0,
            operator_input=lambda: intervention == "radio",
        )
    )
    ex.tick()
    assert ex.outcome == ("OPERATOR_ABORT" if intervention == "radio" else "SENSOR_OR_CONTROL_FAILURE")
    assert ex.reason == (
        "operator_radio_input" if intervention == "radio" else "stale_or_future_remote_state"
    )
    ex.motion.monitor.operator_input = lambda: False
    ex.motion.monitor.remote_t = ex.clock()
    with pytest.raises(Hold, match="inactive"):
        ex.observe(pending)
    assert not ex.attempts
