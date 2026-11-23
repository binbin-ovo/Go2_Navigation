"""Offline episode behavior, using synthetic poses, images, clocks and controller routes."""

from copy import deepcopy
from hashlib import sha256
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
        "source_image": {"file_sha256": sha256(dest.read_bytes()).hexdigest()},
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


def test_independent_localization_separates_edge_mismark_partial_and_no_mark(tmp_path):
    # A textured visible surface is shifted and partly covered at the right edge.
    rng = np.random.default_rng(938)
    patch = rng.integers(0, 256, (128, 142), dtype=np.uint8)
    ref_im = np.full((256, 448), 32, np.uint8)
    ref_im[65:193, 160:302] = patch
    source, dest = tmp_path / "ref.png", tmp_path / "current.png"
    assert cv2.imwrite(str(source), ref_im)
    ref = object_reference(source, [160 / 448, 65 / 256, 302 / 448, 193 / 256], "box", "pilot")
    edge = np.full_like(ref_im, 60)
    edge[70:198, 340:] = patch[:, :108]
    edge[155:198, 340:378] = 100
    assert cv2.imwrite(str(dest), edge)
    rc = {
        "view": {
            "rectified": True,
            "distortion_model": "pinhole_radtan",
            "K": [[300, 0, 224], [0, 300, 128], [0, 0, 1]],
            "size_wh": [448, 256],
        }
    }

    def verdict(point, positive=ref, negative=None):
        pred = dict(
            prediction_id="p1",
            source_image={"file_sha256": sha256(dest.read_bytes()).hexdigest()},
            target_fraction=point,
            opos_not_visible=point is None,
        )
        return ground(pred, dest, rc, "box", positive, negative, seek_view=True)

    for pt in ([0.50, 0.5], None):
        e = verdict(pt)
        assert e["status"] == "uncertain" and e["method"] == "reference_visible_without_native_agreement"
        assert e["positive_reference"]["roi"][0] > 0.7
        assert e["positive_reference"]["mark_inside"] is (None if pt is None else False)
    assert verdict([0.9, 0.47])["status"] == "supported"
    assert verdict([0.9, 0.47], ref, ref)["method"] == "conflicting_instance_matches"
    assert verdict([0.9, 0.47], None, ref)["method"] == "previously_rejected_instance"
    assert cv2.imwrite(str(dest), np.full_like(edge, 100))
    e = verdict([0.5, 0.5])
    assert (e["status"], e["positive_reference"]["reason"]) == ("uncertain", "too_few_current_features")
    assert verdict(None)["native_visibility"] == "reported_absent"
    with pytest.raises(Hold, match="image_changed"):
        ground(
            dict(
                prediction_id="p",
                source_image={"file_sha256": "0" * 64},
                target_fraction=[0.5, 0.5],
                opos_not_visible=False,
            ),
            dest,
            rc,
            "box",
            ref,
            seek_view=True,
        )


def test_five_move_search_remembers_inspected_views_and_local_stalls():
    ex, _ = episode()
    msg = observation(ex, 1, extent=0)
    pose = msg["proposal"]["identity"]["pose"]

    def select(inspected, attempts):
        return select_action(
            ex.grant,
            ex.cap,
            msg["proposal"],
            msg["evidence"],
            attempts,
            len(attempts),
            False,
            inspected=inspected,
        )

    start = dict(pose, yaw_rad=math.pi - 0.05)
    pose.update(start)
    yaw = select([start], [])["action"]["magnitude"]
    inspected = [start, dict(start, yaw_rad=math.remainder(start["yaw_rad"] + yaw, 2 * math.pi))]
    skipped = select(inspected, [])
    assert (
        skipped["action"]["direction"] == "right"
        and "left_view_already_inspected" in skipped["skipped_viewpoints"]
    )
    failed = dict(start, attempt={"mode": "SEARCH", "direction": "right", "no_progress": True})
    later = dict(start, x_m=start["x_m"] + 1.0, attempt={"mode": "SEARCH", "direction": "left"})
    with pytest.raises(Hold, match="right_stalled_from_this_view"):
        select(inspected, [failed, later])
    # An identical turn in a different place, epoch or facing direction can be useful.
    for moved in (
        dict(failed, x_m=start["x_m"] + 0.7),
        dict(failed, robot_epoch="other"),
        dict(failed, yaw_rad=start["yaw_rad"] + 1),
    ):
        assert select(inspected, [moved, later])["action"]["direction"] == "right"


def test_reference_can_guide_search_but_never_authorize_approach_or_bypass_clearance():
    ex, _ = episode()
    msg = observation(ex, 1, angle=3, status="uncertain")
    msg["evidence"].update(
        method="reference_visible_without_native_agreement",
        positive_reference={
            "localized": True,
            "roi": [0.85, 0.3, 0.99, 0.7],
            "mark_inside": False,
            "operator": "pilot",
            "reference_sha256": "5" * 64,
            "inliers": 13,
            "inlier_fraction": 0.95,
            "feature_spread": 0.3,
            "median_residual_px": 0.5,
        },
    )

    def select():
        return select_action(ex.grant, ex.cap, msg["proposal"], msg["evidence"], [], 0, False, inspected=[])

    s = select()
    assert (s["mode"], s["action"]["direction"], s["source"], s["reason"]) == (
        "SEARCH",
        "right",
        "search_policy",
        "recenter_localized_reference",
    )
    msg["evidence"]["positive_reference"]["operator"] = "other"
    with pytest.raises(Hold, match="unbound_reference_viewpoint"):
        select()
    msg["evidence"]["positive_reference"]["operator"] = "pilot"
    msg["proposal"]["identity"]["geometry"]["observed_turn_disc"]["obstacle_returns"] = 1
    with pytest.raises(Hold, match="observed_turn_disc"):
        select()


def test_inconclusive_identity_gets_one_new_view_then_an_opportunity_to_reidentify():
    ex, _ = episode()

    def uncertain(seq):
        m = observation(ex, seq, angle=3, status="uncertain")
        m["evidence"].update(
            method="reference_inconclusive_seek_view", positive_reference={"reference_sha256": "f" * 64}
        )
        return m

    ex.observe(uncertain(1))
    assert len(ex.attempts) == 1 and ex.attempts[0]["mode"] == "SEARCH"
    ex.observe(uncertain(2))
    assert ex.state == "IDENTIFYING" and ex.reason == "reidentify_after_new_view" and len(ex.attempts) == 1
    ex.review(
        dict(
            type="episode_identified",
            episode_id=ex.nonce,
            operator="pilot",
            prediction_id="P2",
            image_sha256=ex.last["image_sha256"],
        )
    )
    m = uncertain(3)
    m["evidence"]["previous_image_review"] = {"prediction_id": "P2"}
    ex.observe(m)
    assert len(ex.attempts) == 2 and ex.attempts[-1]["mode"] == "SEARCH"


@pytest.mark.parametrize("side", [-1, 1])
def test_coarse_alignment_uses_heading_of_curved_path_only_with_clearance(side):
    ex, _ = episode()
    msg = observation(ex, 1, angle=side * 14)
    prop, ev = msg["proposal"], msg["evidence"]
    prop["waypoints"] = [
        [0.023, side * 0.007, side * 0.314],
        [0.165, side * 0.057, side * 0.364],
        [0.303, side * 0.115, side * 0.435],
    ]

    def choose():
        return select_action(ex.grant, ex.cap, prop, ev, [], 0, False, inspected=[])

    selected = choose()
    assert selected["aligned"] and selected["mode"] == "APPROACH"
    assert selected["action"]["direction"] == ("right" if side < 0 else "left")
    assert selected["action"]["speed"] == 2 and selected["action"]["magnitude"] == pytest.approx(
        math.radians(20)
    )
    assert selected["source"] == "approach_policy" and selected["expected_residual_rad"] < math.radians(8)
    disc = prop["identity"]["geometry"]["observed_turn_disc"]
    disc["obstacle_returns"] = 143
    with pytest.raises(Hold, match="obstacles=143"):
        choose()
    disc["obstacle_returns"] = 0
    for angle in (side * 5, -side * 14):
        prop["point_fraction"] = observation(ex, 2, angle=angle)["proposal"]["point_fraction"]
        with pytest.raises(Hold, match="native_path_does_not_support_forward"):
            choose()
    prop["point_fraction"] = observation(ex, 3, angle=-side * 22)["proposal"]["point_fraction"]
    with pytest.raises(Hold, match="native_heading_disagree"):
        choose()


def test_missing_target_curved_right_hint_and_no_forward_escape():
    ex, _ = episode()
    msg = observation(ex, 1, extent=0)
    prop = msg["proposal"]
    prop["waypoints"] = [[0.044, -0.008, -0.312], [0.16, -0.06, -0.37], [0.30, -0.12, -0.45]]

    def choose():
        return select_action(ex.grant, ex.cap, prop, msg["evidence"], [], 0, False, inspected=[])

    from go2_perception.navigation import initial_heading_sweep
    assert initial_heading_sweep([[0.15, 0, 0], [0.25, -0.02, -0.35]]) == 0
    turn = choose()
    assert turn["action"]["direction"] == "right" and turn["source"] == "search_policy"
    prop["identity"]["geometry"]["observed_turn_disc"].update(obstacle_returns=130)
    with pytest.raises(Hold, match=r"no_supported_search_viewpoint: right .*obstacles=130.*left"):
        choose()
    prop["identity"]["geometry"].pop("observed_turn_disc")
    with pytest.raises(Hold, match="floor=None, obstacles=None"):
        choose()
    ex.grant["area_inspected"] = False
    with pytest.raises(Hold, match="requires_current_episode_authorization"):
        choose()


def test_rotated_straight_native_path_supports_one_step_toward_a_verified_aligned_target():
    # November 22 hzx6aQ: a straight native line about 15 deg right, verified target a few deg right.
    ex, _ = episode()

    def rotated(degrees, curve=0.0):
        a = math.radians(degrees)
        rows, x, y, heading = [], 0.0, 0.0, a
        for _ in range(8):
            x, y = x + 0.145 * math.cos(heading), y + 0.145 * math.sin(heading)
            rows.append([x, y, heading])
            heading += curve
        return rows

    def choose(angle, rows, clear=None):
        msg = observation(ex, 1, angle=angle)
        prop = msg["proposal"]
        prop["waypoints"] = rows
        if clear is not None:
            prop["identity"]["geometry"]["forward_clear_m"] = clear
        return select_action(ex.grant, ex.cap, prop, msg["evidence"], [], 0, False, inspected=[])

    s = choose(-4, rotated(-15))
    assert s["mode"] == "APPROACH" and s["source"] == "approach_policy"
    assert s["reason"] == "verified_target_and_straight_native_heading_agree"
    assert s["action"]["direction"] == "forward" and s["action"]["magnitude"] == pytest.approx(0.30)
    assert s["action"]["speed"] == 1.0
    # The heading must agree with the verified target, not merely point forward-ish.
    with pytest.raises(Hold, match="native_path_does_not_support_forward"):
        choose(10, rotated(-15))
    # A curving path, a reversal and a sideways path keep holding.
    with pytest.raises(Hold, match="native_path_does_not_support_forward"):
        choose(-4, rotated(-15, curve=math.radians(-6)))
    with pytest.raises(Hold, match="native_path_does_not_support_forward"):
        choose(-4, [[-x, y, a] for x, y, a in rotated(-15)])
    with pytest.raises(Hold, match="native_path_does_not_support_forward"):
        choose(-4, rotated(-60))
    # The forward LiDAR corridor still decides whether the step may be selected.
    with pytest.raises(Hold, match="required_geometry_unavailable_or_insufficient"):
        choose(-4, rotated(-15), clear=0.2)
    # November 22 P005: a gentle right curve whose chord (-10 deg) agrees with a target at -11 deg.
    curve = [[0.15, -0.02, -0.16], [0.29, -0.05, -0.23], [0.44, -0.09, -0.31], [0.59, -0.13, -0.32]]
    s = choose(-11, curve)
    assert s["reason"] == "verified_target_and_native_chord_agree" and s["action"]["direction"] == "forward"
    # A curve must agree within half the band: the same curve with a target at +2 deg holds.
    with pytest.raises(Hold, match="native_path_does_not_support_forward"):
        choose(2, curve)
