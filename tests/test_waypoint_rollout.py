"""Loop logic is tested with an explicit fake predictor, never scored as LightNav."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from go2_perception.vln.waypoint_rollout import run_episode

ROOT = Path(__file__).resolve().parents[1]


def test_contradictory_stop_is_not_credited_as_success(tmp_path):
    room = Room()

    def predictor(job, output):
        return response(job) | {"decoder_stop": True}

    summary = run_episode(
        room,
        spec(),
        {"id": "case", "pose": [6, 0, 0]},
        "bounded_prefix",
        predictor,
        tmp_path / "bad_stop",
    )
    assert summary["termination"] == "response_or_execution_error"
    assert not summary["native_stop_inside_goal"] and room.clock == 0


def test_stale_response_is_rejected_before_any_simulated_motion(tmp_path, monkeypatch):
    import itertools
    from go2_perception.vln import waypoint_rollout

    ticks = itertools.count(step=0.1)
    monkeypatch.setattr(waypoint_rollout.time, "monotonic", lambda: next(ticks))
    declaration = spec()
    declaration["budget"]["response_timeout_s"] = 0.01
    room = Room()
    summary = run_episode(
        room,
        declaration,
        {"id": "case", "pose": [4, 0, 0]},
        "bounded_prefix",
        lambda j, d: response(j),
        tmp_path / "stale",
    )
    assert summary["termination"] == "response_or_execution_error"
    assert room.clock == 0 and room.pose[0] == 4


class Room:
    def __init__(self):
        self.pose = np.zeros(3)
        self.clock = 0.0

    def _place(self):
        pass

    def contacts(self):
        return []

    def render(self):
        return np.full((8, 8, 3), round(self.pose[0] * 10), dtype=np.uint8)

    def evaluation(self):
        return {"pose_xy_yaw": self.pose.tolist(), "contacts": []}

    def execute_waypoint_policy(self, chunk, policy, *, native_stop, limits):
        before = self.pose.copy()
        if not native_stop:
            self.pose[0] += 0.1
            self.clock += 0.25
        return {
            "actual_world_poses": [before.tolist(), self.pose.tolist()],
            "actual_times_s": [0, 0 if native_stop else 0.25],
            "simulated_dt_s": 0 if native_stop else 0.25,
            "constraint_rejection": None,
        }


def spec():
    s = yaml.safe_load((ROOT / "configs/waypoint_validation.yaml").read_text())
    s["budget"].update(max_decisions=3, max_simulated_s=0.75)
    return s


def response(job, stop=False):
    return {
        **{k: job[k] for k in ("episode_id", "clock_epoch", "decision_id")},
        "status": "decoded",
        "decoder_stop": stop,
        "waypoints": ([[0, 0, 0]] if stop else [[0.1, 0, 0]]) * 10,
        "execution_authorized": False,
    }


def test_loop_keeps_actual_history_and_never_stops_on_evaluator_goal(tmp_path):
    received, room = [], Room()

    def predictor(job, output):
        received.append(job)
        assert set(job) == {
            "episode_id",
            "clock_epoch",
            "decision_id",
            "decision_time_s",
            "instruction",
            "frames",
        }
        return response(job)

    summary = run_episode(
        room,
        spec(),
        {"id": "near", "pose": [6, 0, 0]},
        "bounded_prefix",
        predictor,
        tmp_path / "run",
    )
    assert summary["model_decisions"] == 3 and summary["entered_goal_region"]
    assert (
        summary["termination"] == "simulated_time_budget"
        and not summary["native_stop_inside_goal"]
    )
    assert summary["continued_motion_after_goal_entry"]
    assert [len(j["frames"]) for j in received] == [1, 2, 3]
    assert received[-1]["frames"][-1]["observation_s"] == 0.5
    assert len({f["sha256"] for f in received[-1]["frames"]}) == 3


def test_native_stop_and_wrong_episode_do_not_execute_motion(tmp_path):
    for wrong in (False, True):
        room = Room()

        def predictor(job, output):
            result = response(job, stop=True)
            if wrong:
                result["episode_id"] = "wrong"
            return result

        summary = run_episode(
            room,
            spec(),
            {"id": "case", "pose": [6, 0, 0]},
            "bounded_prefix",
            predictor,
            tmp_path / str(wrong),
        )
        assert room.clock == 0 and room.pose[0] == 6
        assert summary["native_stop_inside_goal"] != wrong
        rows = json.loads((tmp_path / str(wrong) / "steps.json").read_text())
        assert rows[0]["response_accepted"] != wrong


def test_invalid_output_is_retained_and_does_not_execute(tmp_path):
    room = Room()

    def predictor(job, output):
        return response(job) | {"waypoints": [[0, 0, 0]]}

    summary = run_episode(
        room,
        spec(),
        {"id": "case", "pose": [4, 0, 0]},
        "bounded_prefix",
        predictor,
        tmp_path / "run",
    )
    assert summary["termination"] == "response_or_execution_error" and room.clock == 0
    with pytest.raises(FileExistsError):
        run_episode(
            Room(),
            spec(),
            {"id": "case", "pose": [4, 0, 0]},
            "bounded_prefix",
            predictor,
            tmp_path / "run",
        )
