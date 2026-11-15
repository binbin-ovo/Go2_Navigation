"""Append-only native observation sessions for simulation; never action feedback."""

import copy
import math


class NativeObservationSession:
    """Keep native buffers across decisions; reset on episode/clock-epoch change.

    Jobs carry full source history for provenance. Only newly captured frames are
    appended to the policy. Rewriting history, changing instruction mid-episode,
    adding state/labels/actions, or repeating a decision is rejected.
    """

    def __init__(self):
        self.key = None
        self.instruction = None
        self.frames = []
        self.cutoff = None
        self.decisions = set()
        self.poisoned = False

    def observe(self, agent, job, load_rgb):
        if set(job) != {
            "instruction",
            "episode_id",
            "clock_epoch",
            "decision_id",
            "decision_time_s",
            "frames",
        }:
            raise ValueError(
                "Observation-only job whitelist excludes goal/state/labels/actions"
            )
        key = (job["episode_id"], job["clock_epoch"])
        if not all(
            isinstance(x, str) and x
            for x in (*key, job["instruction"], job["decision_id"])
        ):
            raise ValueError("Nonempty instruction and session identifiers required")
        frames = job["frames"]
        cutoff = job["decision_time_s"]
        if not frames or not math.isfinite(cutoff):
            raise ValueError("Finite cutoff and causal images required")
        for i, frame in enumerate(frames):
            if set(frame) != {"path", "sha256", "observation_s", "available_s"}:
                raise ValueError("Frame whitelist excludes evaluator context")
            obs, available = frame["observation_s"], frame["available_s"]
            if (
                not all(math.isfinite(x) for x in (obs, available))
                or not obs <= available <= cutoff
            ):
                raise ValueError("Noncausal frame")
            if i and not math.isclose(
                obs - frames[i - 1]["observation_s"], 0.25, abs_tol=1e-8
            ):
                raise ValueError("Expected actual 4 Hz simulated observations")
        resetting = key != self.key
        if not resetting:
            if self.poisoned:
                raise ValueError("Failed native session requires a new episode/epoch")
            if job["instruction"] != self.instruction:
                raise ValueError("Instruction changed within episode")
            if job["decision_id"] in self.decisions or cutoff <= self.cutoff:
                raise ValueError("Duplicate or out-of-order decision")
            if frames[: len(self.frames)] != self.frames or len(frames) <= len(
                self.frames
            ):
                raise ValueError(
                    "History changed, shrank, or contains no new observations"
                )
        else:
            agent.reset(instruction=job["instruction"])
            self.key = key
            self.instruction = job["instruction"]
            self.frames = []
            self.decisions = set()
            self.poisoned = False
        appended = len(frames) - len(self.frames)
        try:
            for frame in frames[len(self.frames) :]:
                agent.observe(load_rgb(frame))
        except Exception:
            self.poisoned = True
            raise
        self.frames = copy.deepcopy(frames)
        self.cutoff = cutoff
        self.decisions.add(job["decision_id"])
        return {
            "policy": "episode_epoch_reset_append_actual_observations",
            "reset_this_decision": resetting,
            "frames_appended": appended,
            "total_frames": len(self.frames),
            "previous_proposals_or_executed_actions_supplied": False,
        }
