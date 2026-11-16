"""Bounded read-only advisory sessions. Existing native jobs/results stay unchanged."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time
import uuid

import cv2
import numpy as np

from .episodes import file_sha256
from .model_preprocessing import calibrated_view
from .observation import TIME_SEMANTICS, load, save, validate
from .pretrained import causal_history, comparison_action, native_token_contract
from .shadow_cache import PreparedFrameCache

BANNER = "ADVISORY ONLY — NO ROBOT COMMANDS"
LIVE_BANNER = "LIVE INPUT — ADVISORY ONLY — NO ROBOT COMMANDS"


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class Limits:
    history_frames: int = 1024
    history_s: float = 60.0
    advisory_age_s: float = 3.0
    response_deadline_s: float = 10.0
    worker_timeout_s: float = 30.0
    request_interval_s: float = 0.25

    def __post_init__(self):
        if type(self.history_frames) is not int or not 1 <= self.history_frames <= 4096:
            raise ValueError("History frame capacity must be 1..4096")
        values = [v for k, v in asdict(self).items() if k != "history_frames"]
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Positive finite session limits required")
        if self.worker_timeout_s < self.response_deadline_s:
            raise ValueError("Worker timeout must not precede response deadline")


@dataclass(frozen=True)
class Request:
    """Immutable serialized snapshot; .job/.context return fresh owned objects."""

    job_json: str
    context_json: str
    folder: Path

    @property
    def job(self):
        return json.loads(self.job_json)

    @property
    def context(self):
        return json.loads(self.context_json)


def validate_proposal(result):
    """Validate the existing native result, retaining STOP and completion separately."""
    if not isinstance(result, dict) or not isinstance(result.get("pointing"), dict):
        raise ValueError("Native result and pointing must be objects")
    if (
        result.get("execution_authorized") is not False
        or result.get("status") != "decoded"
    ):
        raise ValueError(
            "Only decoded, read-only native proposals can be displayed as current"
        )
    codes = native_token_contract(result["native_output"])
    if codes != result["rvq_codes"]:
        raise ValueError("Native output/RVQ code mismatch")
    comparison_action(result["waypoints"], native_stop=result["decoder_stop"])
    if type(result.get("explicit_stop_code")) is not bool or result[
        "explicit_stop_code"
    ] != (codes[0] == 6):
        raise ValueError("Explicit native STOP code mismatch")
    if result["explicit_stop_code"] and not result["decoder_stop"]:
        raise ValueError("Explicit STOP disagrees with decoder")
    if (
        result.get("completion_verified") is not False
        or result.get("waypoint_time_base") is not None
    ):
        raise ValueError(
            "Completion verification / native waypoint timing cannot be invented"
        )
    if (
        result.get("waypoint_convention")
        != "absolute_ego_SE2_x_forward_y_left_yaw_CCW_metres_radians"
    ):
        raise ValueError("Unknown waypoint frame")
    return {
        "native_stop": result["decoder_stop"],
        "explicit_stop_code": result["explicit_stop_code"],
        "model_completion_claim": result.get("pointing", {}).get("apos") == 1299,
        "completion_verified": False,
        "physical_robot_stopped": None,
    }


class ShadowSession:
    """One in-flight request and one coalesced dirty flag, independent of history.

    History is disk-backed and bounded by count and duration. At capacity/gaps,
    inference pauses instead of deleting native anchors. New instructions clear
    history and invalidate pending responses. No action output is fed back.
    The caller owns monotonic time, source pacing and its existing sensor receiver.
    """

    def __init__(
        self,
        output,
        worker,
        *,
        mode="replay_paced",
        limits=None,
        allow_synthetic=False,
        session_id=None,
        preparation_mode="baseline",
    ):
        if preparation_mode not in ("baseline", "cached"):
            raise ValueError("Unknown input preparation mode")
        if mode not in (
            "replay_paced",
            "replay_fast",
            "synthetic_paced",
            "synthetic_fast",
            "live_unvalidated",
        ):
            raise ValueError("Explicit source/timing mode required")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "observations").mkdir()
        (self.output / "requests").mkdir()
        self.worker, self.mode = worker, mode
        self.limits, self.allow_synthetic = limits or Limits(), allow_synthetic
        self.preparation_mode = preparation_mode
        self.prepared_cache = (PreparedFrameCache(self.output / "prepared_cache",
            allow_synthetic=allow_synthetic) if preparation_mode == "cached" else None)
        self.session_id = session_id or uuid.uuid4().hex
        (
            self.revision,
            self.history_generation,
            self.sequence,
            self.observation_count,
        ) = 0, 0, 0, 0
        self.instruction, self.active, self.closed = None, False, False
        self.key, self.latest, self.inflight, self.displayed = None, None, None, None
        self.history, self.history_error = [], None
        self.pending, self.last_submit_s = False, -float("inf")
        self.counts = {
            "received": 0,
            "submitted": 0,
            "accepted": 0,
            "rejected": 0,
            "coalesced": 0,
        }
        self.events_path = self.output / "events.jsonl"
        self.replay_anchor = None
        self.last_now = -float("inf")
        self.banner = LIVE_BANNER if mode == "live_unvalidated" else BANNER
        self.source_status = None  # optional live receiver status, display only

    def event(self, kind, now, **fields):
        with self.events_path.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": kind,
                        "host_monotonic_s": now,
                        "session_id": self.session_id,
                        "instruction_revision": self.revision,
                        **fields,
                    },
                    allow_nan=False,
                )
                + "\n"
            )

    def _clock(self, now):
        if not math.isfinite(now) or now < self.last_now:
            raise ValueError("Finite nondecreasing host monotonic time required")
        self.last_now = now

    def set_replay_anchor(self, source_s, monotonic_s, speed=1.0):
        if (
            not all(math.isfinite(x) for x in (source_s, monotonic_s, speed))
            or speed <= 0
        ):
            raise ValueError("Invalid replay clock anchor")
        self.replay_anchor = (source_s, monotonic_s, speed)

    def instruction_changed(self, instruction, now):
        self._clock(now)
        if (
            self.closed
            or not isinstance(instruction, str)
            or not instruction.strip()
            or len(instruction) > 4096
        ):
            raise ValueError(
                "Open session and nonempty instruction <=4096 characters required"
            )
        self.revision += 1
        self.history_generation += 1
        self.instruction, self.active = instruction.strip(), True
        self.history, self.history_error, self.key = [], None, None
        if self.prepared_cache is not None:
            self.prepared_cache.clear()
        self.pending = False
        self.event("instruction_started", now, instruction=self.instruction)
        # The next genuinely arriving observation starts the new instruction history.
        # Retained latest view is display-only; it is never restamped/reobserved.

    def cancel(self, now):
        self._clock(now)
        self.active, self.pending = False, False
        self.history = []
        if self.prepared_cache is not None:
            self.prepared_cache.clear()
        self.event("advisory_cancelled", now, physical_robot_stop_sent=False)

    def invalidate_history(self, reason, now):
        self.history_error, self.pending = reason, False
        self.event("history_paused", now, reason=reason)

    def receive(self, delivery, now):
        receive_started = time.monotonic()
        self._clock(now)
        if self.closed:
            raise ValueError("Session is closed")
        obs = delivery.observation
        if not obs.clock_epoch:
            self.invalidate_history("missing_clock_epoch", now)
            return
        v = obs.visual
        if v is None:
            self.invalidate_history("missing_image", now)
            return
        if v.history:
            self.invalidate_history(
                "deliver_current_frame_only_session_owns_history", now
            )
            return
        issues = validate(obs)
        if (
            not isinstance(v.image, np.ndarray)
            or v.image.dtype != np.uint8
            or v.image.ndim != 3
            or v.image.shape[2] != 3
        ):
            self.invalidate_history("invalid_image", now)
            return
        if v.image.nbytes > 32 * 2**20:
            self.invalidate_history("image_exceeds_32MiB", now)
            return
        stamp = v.stamp_robot or v.stamp_host
        if (
            not math.isfinite(stamp.t)
            or not stamp.clock
            or stamp.semantics not in TIME_SEMANTICS
            or v.stamp_host.clock != "host_receipt"
            or v.stamp_host.semantics != "host_receipt"
            or not stamp.t <= delivery.available_s
            or not math.isfinite(v.stamp_host.t)
        ):
            self.invalidate_history("invalid_observation_clock", now)
            return
        uncertainty = stamp.uncertainty_s
        if uncertainty is not None and (
            not math.isfinite(uncertainty) or uncertainty < 0
        ):
            self.invalidate_history("invalid_timestamp_uncertainty", now)
            return
        if delivery.received_monotonic_s is not None and (
            not math.isfinite(delivery.received_monotonic_s)
            or delivery.received_monotonic_s > now
        ):
            raise ValueError(
                "Receipt must be in this process's causal monotonic domain"
            )
        key = (delivery.source_id, obs.clock_epoch, stamp.clock)
        if self.key is not None and key != self.key:
            self.history_generation += 1
            self.history, self.history_error, self.pending = [], None, False
            if self.prepared_cache is not None:
                self.prepared_cache.clear()
            self.event("source_epoch_reset", now, old_key=self.key, new_key=key)
        self.key = key
        self.observation_count += 1
        path = self.output / "observations" / f"{self.observation_count:06d}.vlnobs.npz"
        storage_started = time.monotonic()
        save(obs, path)
        storage_s = time.monotonic() - storage_started
        meta = {
            "frame_id": delivery.frame_id,
            "observation_s": stamp.t,
            "available_s": delivery.available_s,
            "image": {"path": str(path)},
            "source_id": delivery.source_id,
            "clock_epoch": obs.clock_epoch,
            "stamp": asdict(stamp),
            "stamp_host": asdict(v.stamp_host),
            "received_monotonic_s": now
            if delivery.received_monotonic_s is None
            else delivery.received_monotonic_s,
            "exposure_monotonic_s": delivery.exposure_monotonic_s,
            "monotonic_mapping": delivery.monotonic_mapping,
            "observation_path": str(path),
            "observation_sha256": file_sha256(path),
            "provenance": delivery.provenance,
            "validation_warnings": issues,
            "pose_available": obs.robot is not None
            and obs.robot.T_odom_base is not None,
            "lidar_available": obs.geometric is not None,
            "T_base_camera": None,
        }
        write_json(path.with_suffix(".json"), meta)
        self.latest = meta
        self.counts["received"] += 1
        if not cv2.imwrite(
            str(self.output / "latest.png"), cv2.cvtColor(v.image, cv2.COLOR_RGB2BGR)
        ):
            raise OSError("Cannot save latest view")
        self.last_receive_profile = {
            "receive_s": time.monotonic() - receive_started,
            "observation_npz_storage_s": storage_s,
        }
        if not self.active or self.history_error:
            return
        if self.history and (
            stamp.t <= self.history[-1]["observation_s"]
            or delivery.available_s < self.history[-1]["available_s"]
        ):
            self.invalidate_history("duplicate_or_out_of_order_observation", now)
            return
        if len(self.history) >= self.limits.history_frames or (
            self.history
            and stamp.t - self.history[0]["observation_s"] > self.limits.history_s
        ):
            self.invalidate_history("history_limit_reached_start_new_instruction", now)
            return
        if self.history and stamp.t - self.history[-1]["observation_s"] > 0.375:
            self.invalidate_history("observation_gap_requires_new_instruction", now)
            return
        if delivery.provenance.get("declared_gaps"):
            self.invalidate_history("declared_episode_gap", now)
            return
        self.history.append(meta)
        if self.pending:
            self.counts["coalesced"] += 1
        self.pending = True

    def _snapshot(self, now):
        snapshot_started = time.monotonic()
        stages = {"source_hash_and_load_s": 0.0, "rectification_s": 0.0,
                  "image_encoding_and_write_s": 0.0}
        cutoff = self.history[-1]["available_s"]
        episode = SimpleNamespace(
            data={
                "frames": self.history,
                "gaps": [],
                "clock": {"epoch": self.key[1]},
                "decisions": [{"decision_id": "current", "time_s": cutoff}],
            }
        )
        selection = causal_history(episode, "current")
        self.sequence += 1
        folder = self.output / "requests" / f"{self.sequence:06d}"
        folder.mkdir()
        frames, provenance = [], []
        for i, source in enumerate(selection["frames"]):
            dest = folder / f"frame_{i:04d}.png"
            if self.prepared_cache is not None:
                digest, metadata, cache, frame_stages = self.prepared_cache.prepare(source, dest)
                frames.append({"path": str(dest.resolve()), "sha256": digest,
                               "observation_s": source["observation_s"],
                               "available_s": source["available_s"]})
                provenance.append({"source": source, "rectification": metadata, "prepared_cache": cache})
                for name, duration in frame_stages.items():
                    stages[name] += duration
                continue
            stage_start = time.monotonic()
            path = Path(source["observation_path"])
            if file_sha256(path) != source["observation_sha256"]:
                raise ValueError("Stored observation changed")
            obs = load(path)
            stages["source_hash_and_load_s"] += time.monotonic() - stage_start
            stage_start = time.monotonic()
            view = calibrated_view(obs, allow_synthetic=self.allow_synthetic)
            stages["rectification_s"] += time.monotonic() - stage_start
            stage_start = time.monotonic()
            if not cv2.imwrite(str(dest), cv2.cvtColor(view.image, cv2.COLOR_RGB2BGR)):
                raise OSError("Cannot save prepared view")
            frames.append(
                {
                    "path": str(dest.resolve()),
                    "sha256": file_sha256(dest),
                    "observation_s": source["observation_s"],
                    "available_s": source["available_s"],
                }
            )
            provenance.append({"source": source, "rectification": view.metadata})
            stages["image_encoding_and_write_s"] += time.monotonic() - stage_start
        identity = {
            "session_id": self.session_id,
            "instruction_revision": self.revision,
            "history_generation": self.history_generation,
            "clock_epoch": self.key[1],
            "source_id": self.key[0],
            "sequence": self.sequence,
        }
        job = {
            "instruction": self.instruction,
            "episode_id": f"{self.session_id}.r{self.revision}",
            "clock_epoch": self.key[1],
            "decision_id": str(self.sequence),
            "decision_time_s": cutoff,
            "frames": frames,
        }
        context = {
            "identity": identity,
            "source_mode": self.mode,
            "host_submit_monotonic_s": now,
            "deadline_monotonic_s": now + self.limits.response_deadline_s,
            "current_source": selection["frames"][-1],
            "source_cutoff_s": cutoff,
            "source_age_at_cutoff_s": selection["current_observation_age_s"],
            "history_policy": selection["policy"],
            "grid_s": selection["grid_s"],
            "timestamp_quantization_error_s": selection[
                "timestamp_quantization_error_s"
            ],
            "history": provenance,
            "reset_policy": "reconstruct_complete_causal_instruction_history_per_request",
            "annotations_consumed": False,
            "previous_actions_supplied": False,
        }
        serialization_start = time.monotonic()
        write_json(folder / "job.json", job)
        write_json(folder / "snapshot.json", context)
        request = Request(
            json.dumps(job, allow_nan=False),
            json.dumps(context, allow_nan=False),
            folder,
        )
        write_json(folder / "preparation_profile.json", {
            **stages,
            "mode": self.preparation_mode,
            "cache": None if self.prepared_cache is None else self.prepared_cache.status(),
            "job_serialization_s": time.monotonic() - serialization_start,
            "snapshot_s": time.monotonic() - snapshot_started,
        })
        return request

    def timing(self, request, now):
        c = request.context
        source = c["current_source"]
        age, basis = None, "unavailable_without_capture_to_host_mapping"
        if self.mode.endswith("paced") and self.replay_anchor is not None:
            start, host, rate = self.replay_anchor
            age = start + (now - host) * rate - source["observation_s"]
            basis = "paced_replay_virtual_seconds_not_live_latency"
        elif (
            self.mode == "live_unvalidated"
            and source["exposure_monotonic_s"] is not None
        ):
            age = now - source["exposure_monotonic_s"]
            basis = "caller_supplied_exposure_monotonic_mapping_unvalidated_live"
        return {
            "observation_age_at_display_s": age,
            "age_basis": basis,
            "source_clock": source["stamp"]["clock"],
            "source_stamp_semantics": source["stamp"]["semantics"],
            "source_stamp_uncertainty_s": source["stamp"]["uncertainty_s"],
            "source_age_at_cutoff_s": c["source_age_at_cutoff_s"],
            "host_since_delivery_s": now - source["received_monotonic_s"],
            "latest_source_lag_s": None
            if self.latest is None
            or self.latest["clock_epoch"] != source["clock_epoch"]
            else self.latest["observation_s"] - source["observation_s"],
            "host_since_submit_s": now - c["host_submit_monotonic_s"],
        }

    def _rejection(self, request, now):
        identity = request.context["identity"]
        if self.closed or not self.active:
            return "cancelled_or_closed"
        if identity["instruction_revision"] != self.revision:
            return "superseded_instruction"
        if identity["history_generation"] != self.history_generation:
            return "superseded_source_or_epoch"
        if self.key != (
            identity["source_id"],
            identity["clock_epoch"],
            request.context["current_source"]["stamp"]["clock"],
        ):
            return "superseded_source_or_epoch"
        if self.history_error:
            return "invalid_history: " + self.history_error
        timing = self.timing(request, now)
        if now > request.context["deadline_monotonic_s"]:
            return "stale_response_deadline"
        ages = [
            timing["host_since_delivery_s"],
            timing["observation_age_at_display_s"],
            timing["latest_source_lag_s"],
        ]
        if any(v is not None and v > self.limits.advisory_age_s for v in ages):
            return "stale_advisory_age"
        return None

    def tick(self, now, *, allow_submit=True):
        self._clock(now)
        response = self.worker.poll(now)
        if response is not None:
            request = self.inflight
            self.inflight = None
            # Broken workers must not inject NaN into logs or crash the status page.
            # Retain the exact Python representation if strict JSON is impossible.
            try:
                json.dumps(response, allow_nan=False)
                if not isinstance(response, dict) or not isinstance(
                    response.get("result"), dict
                ):
                    raise ValueError("Response/result must be objects")
            except (ValueError, TypeError):
                if request is not None:
                    (request.folder / "malformed_response.txt").write_text(
                        repr(response)
                    )
                response = {
                    "result": {
                        "status": "error",
                        "native_output": "Malformed response (see retained artifact)",
                    },
                    "malformed_response": True,
                }
            reason = (
                "unsolicited_response"
                if request is None
                else self._rejection(request, now)
            )
            if request is not None:
                if response.get("identity") != request.context["identity"]:
                    reason = "wrong_session_epoch_or_sequence"
                result = response.get("result", {})
                job = request.job
                if any(
                    result.get(k) != job[k]
                    for k in ("episode_id", "clock_epoch", "decision_id")
                ):
                    reason = "wrong_native_result_identity"
                start, end = (
                    response.get("host_inference_start_s"),
                    response.get("host_inference_end_s"),
                )
                if not (
                    isinstance(start, (int, float))
                    and isinstance(end, (int, float))
                    and math.isfinite(start)
                    and math.isfinite(end)
                    and request.context["host_submit_monotonic_s"]
                    <= start
                    <= end
                    <= now
                ):
                    reason = "invalid_inference_monotonic_clock"
                elif not isinstance(
                    response.get("inference_duration_s"), (int, float)
                ) or not math.isclose(
                    response["inference_duration_s"], end - start, abs_tol=1e-6
                ):
                    reason = "invalid_inference_duration"
                try:
                    semantics = validate_proposal(result)
                except (KeyError, TypeError, ValueError) as error:
                    semantics = None
                    reason = reason or "malformed_output: " + str(error)
                if response.get("malformed_response"):
                    reason = (
                        "malformed_output: invalid response structure or nonfinite JSON"
                    )
                record = {
                    "receipt_monotonic_s": now,
                    "response": response,
                    "rejection_reason": reason,
                    "display_timing": self.timing(request, now),
                    "semantics": semantics,
                }
                write_json(request.folder / "receipt.json", record)
                self.displayed = (request, record)
            self.counts["rejected" if reason else "accepted"] += 1
            self.event(
                "response_rejected" if reason else "response_accepted",
                now,
                reason=reason,
                identity=response.get("identity"),
            )
        if (
            self.inflight is not None
            and now - self.inflight.context["host_submit_monotonic_s"]
            > self.limits.worker_timeout_s
        ):
            self.worker.close()
            self.event(
                "worker_timeout", now, identity=self.inflight.context["identity"]
            )
            self.inflight = None
            self.invalidate_history("worker_timeout_restart_application", now)
        if (
            allow_submit
            and not self.closed
            and self.active
            and self.pending
            and self.inflight is None
            and not self.history_error
            and self.worker.ready
            and now - self.last_submit_s >= self.limits.request_interval_s
        ):
            try:
                request = self._snapshot(now)
                self.worker.submit(request, now)
                self.inflight = request
                self.pending, self.last_submit_s = False, now
                self.counts["submitted"] += 1
                self.event(
                    "request_submitted",
                    now,
                    identity=request.context["identity"],
                    snapshot=str(request.folder / "snapshot.json"),
                )
            except (ValueError, OSError) as error:
                self.invalidate_history("input_or_worker_error: " + str(error), now)

    def status(self, now):
        shown = None
        if self.displayed:
            request, record = self.displayed
            reason = record["rejection_reason"] or self._rejection(request, now)
            shown = {
                "request_directory": str(request.folder),
                "snapshot": request.context,
                **record,
                "display_timing": self.timing(request, now),
                "current_rejection_reason": reason,
                "label": "STALE"
                if reason and reason.startswith("stale")
                else "REJECTED"
                if reason
                and reason.startswith(("malformed", "wrong", "invalid_inference"))
                else "SUPERSEDED"
                if reason
                else "CURRENT ADVISORY",
            }
        return {
            "banner": self.banner,
            "source_mode": self.mode,
            "source_status": self.source_status,
            "session_id": self.session_id,
            "instruction_revision": self.revision,
            "instruction": self.instruction,
            "active": self.active,
            "closed": self.closed,
            "latest": self.latest,
            "inference_in_flight": self.inflight is not None,
            "worker_ready": self.worker.ready,
            "pending_requests": int(self.pending),
            "history_frames": len(self.history),
            "history_error": self.history_error,
            "counts": self.counts.copy(),
            "limits": asdict(self.limits),
            "displayed": shown,
            "completion_verified": False,
            "physical_execution_authorized": False,
            "preparation_mode": self.preparation_mode,
            "prepared_cache": None if self.prepared_cache is None else self.prepared_cache.status(),
            "tensor_evidence_shutdown": getattr(self.worker, "tensor_evidence_summary", None),
        }

    def close(self, now):
        if self.closed:
            return
        self.cancel(now)
        self.closed = True
        if self.inflight:
            self.event(
                "inflight_invalidated_on_shutdown",
                now,
                identity=self.inflight.context["identity"],
            )
        self.worker.close()
        evidence = getattr(self.worker, "tensor_evidence_summary", None)
        if evidence is not None:
            self.event("tensor_evidence_shutdown", now, evidence=evidence)
            if evidence.get("error"):
                self.history_error = "tensor_evidence_incomplete: " + evidence["error"]
        self.inflight = None
        self.event("session_closed", now, counts=self.counts)
