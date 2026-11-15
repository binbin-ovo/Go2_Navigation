import pytest

from go2_perception.vln.waypoint_evaluation import summarize_run


def step(start, end, *, stop=False, accepted=True):
    return {
        "proposal": {"decoder_stop": stop},
        "response_accepted": accepted,
        "action": {
            "actual_world_poses": [start, end],
            "actual_times_s": [0, 0.25],
            "simulated_dt_s": 0.25,
        },
    }


def test_entry_is_not_native_stop_or_success_and_budget_is_external():
    initial, inside = [5.7, 0, 0], [6, 0, 0]
    result = summarize_run(initial, [step(initial, inside)], "decision_budget")
    assert result["entered_goal_region"]
    assert not result["native_stop_received_and_accepted"]
    assert not result["learned_navigation_success"]
    assert result["externally_terminated"]
    assert result["path_length_m"] == pytest.approx(0.3)


def test_native_stop_inside_and_premature_are_separate():
    for pose, success in (([6, 0, 0], True), ([4.3, 0, 0], False)):
        result = summarize_run(pose, [step(pose, pose, stop=True)], "native_stop")
        assert result["native_stop_inside_goal"] == success
        assert result["premature_native_stop"] != success
        assert result["learned_navigation_success"] == success
        assert not result["externally_terminated"]


def test_enter_exit_and_movement_after_entry_are_retained():
    a, b, c = [5.7, 0, 0], [6.0, 0, 0], [6.7, 0, 0]
    result = summarize_run(a, [step(a, b), step(b, c)], "simulated_time_budget")
    assert result["entered_goal_region"] and result["continued_motion_after_goal_entry"]
    assert result["minimum_goal_distance_m"] == pytest.approx(0.15)
    assert result["final_goal_distance_m"] == pytest.approx(0.55)


def test_stale_stop_is_not_credited_and_initial_goal_entry_is_zero():
    pose = [6, 0, 0]
    result = summarize_run(
        pose, [step(pose, pose, stop=True, accepted=False)], "stale_response"
    )
    assert result["first_goal_entry_sim_s"] == 0
    assert not result["native_stop_inside_goal"]
    assert result["externally_terminated"]
