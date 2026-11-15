from copy import deepcopy

import pytest

from go2_perception.vln.native_history import NativeObservationSession


class Agent:
    def __init__(self):
        self.resets = 0
        self.observations = []

    def reset(self, instruction):
        self.resets += 1
        self.observations.clear()

    def observe(self, rgb):
        self.observations.append(rgb)


def job(n=1, epoch="one", episode="test"):
    return {
        "instruction": "Approach the doorway.",
        "episode_id": episode,
        "clock_epoch": epoch,
        "decision_id": str(n),
        "decision_time_s": (n - 1) * 0.25,
        "frames": [
            {
                "path": f"{i}.png",
                "sha256": str(i),
                "observation_s": i * 0.25,
                "available_s": i * 0.25,
            }
            for i in range(n)
        ],
    }


def test_native_buffer_appends_once_then_resets_at_boundaries():
    stream, agent = NativeObservationSession(), Agent()
    first = stream.observe(agent, job(), lambda f: f["path"])
    second = stream.observe(agent, job(2), lambda f: f["path"])
    assert first["reset_this_decision"] and not second["reset_this_decision"]
    assert second["frames_appended"] == 1
    assert agent.resets == 1 and agent.observations == ["0.png", "1.png"]
    stream.observe(agent, job(epoch="two"), lambda f: f["path"])
    stream.observe(agent, job(epoch="two", episode="other"), lambda f: f["path"])
    assert agent.resets == 3 and agent.observations == ["0.png"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda j: j.update(goal=[6, 0]),
        lambda j: j.update(previous_executed_action=[1, 0, 0]),
        lambda j: j.update(instruction="Changed"),
        lambda j: j["frames"][0].update(sha256="changed"),
        lambda j: j["frames"][1].update(observation_s=100),
        lambda j: j["frames"][1].update(available_s=100),
        lambda j: j["frames"][1].update(observation_s=0.20),
        lambda j: j["frames"][1].update(human_label="STOP"),
        lambda j: j.update(decision_id="1"),
    ],
)
def test_reject_noncausal_feedback_rewrites_and_duplicate_decisions(mutation):
    stream, agent = NativeObservationSession(), Agent()
    stream.observe(agent, job(), lambda f: f["path"])
    bad = deepcopy(job(2))
    mutation(bad)
    with pytest.raises(ValueError):
        stream.observe(agent, bad, lambda f: f["path"])
    assert agent.resets == 1 and agent.observations == ["0.png"]


def test_failed_append_requires_reset_and_does_not_silently_reuse_partial_state():
    stream, agent = NativeObservationSession(), Agent()
    stream.observe(agent, job(), lambda f: f["path"])

    def broken(frame):
        raise OSError("unavailable observation")

    with pytest.raises(OSError):
        stream.observe(agent, job(2), broken)
    with pytest.raises(ValueError, match="Failed native session"):
        stream.observe(agent, job(2), lambda f: f["path"])
    stream.observe(agent, job(epoch="new"), lambda f: f["path"])
    assert agent.resets == 2
