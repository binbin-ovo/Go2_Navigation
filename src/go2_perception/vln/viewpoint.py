"""Read-only paired-view evidence analysis. No inference, receiver, or control path.

The two trials are independent queries. Human annotations are consumed only here,
after inference. All source files are read through Archive and checked for changes.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

from PIL import Image

from .shadow_display import target_fraction

PAIR_SCHEMA = "go2-vln-viewpoint-pair-v1"
SUMMARY_SCHEMA = "go2-vln-viewpoint-analysis-v1"


class EvidenceError(ValueError):
    """Archive is incomplete, changing, malformed, or references unrelated evidence."""


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Archive:
    """Only closed runs; never opens receivers or writes to an input archive.

    The manifest covers consumed metadata and images, not large raw NPZ/LiDAR data.
    A final re-read detects concurrent edits/deletion/creation of consumed files.
    A closed-run marker is necessary: trial_end alone can precede a late response.
    """

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = {}
        self.summary = self.json("summary.json", required=True)
        if not self.summary.get("termination") or not self.summary.get(
            "ended_wall_utc"
        ):
            raise EvidenceError("No final run summary; do not analyze an active run")
        self.run = self.json("run.json", required=True)
        if self.run.get("schema") != "go2-vln-interactive-run-v1":
            raise EvidenceError("Unsupported interactive run schema")

    def path(self, name):
        p = Path(name)
        if p.is_absolute() or ".." in p.parts:
            raise EvidenceError(f"Archive reference must be relative: {name}")
        resolved = (self.root / p).resolve()
        if not resolved.is_relative_to(self.root):
            raise EvidenceError(f"Archive reference escapes its run: {name}")
        return resolved

    def read(self, name, *, required=False):
        p = self.path(name)
        try:
            data = p.read_bytes()
        except FileNotFoundError:
            data = None
        sha = None if data is None else digest(data)
        key = str(Path(name))
        if key in self.manifest and self.manifest[key] != sha:
            raise EvidenceError(f"Archive changed while reading: {name}")
        self.manifest[key] = sha
        if data is None and required:
            raise EvidenceError(f"Missing required completed-archive artifact: {name}")
        return data

    def json(self, name, *, required=False):
        data = self.read(name, required=required)
        if data is None:
            return None
        try:
            result = json.loads(
                data, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s))
            )
        except (ValueError, UnicodeError) as error:
            raise EvidenceError(f"Invalid JSON in {name}: {error}") from error
        if not isinstance(result, dict):
            raise EvidenceError(f"Expected a JSON object: {name}")
        return result

    def rows(self, name):
        data = self.read(name)
        if data is None:
            return []
        rows = []
        for line, raw in enumerate(data.splitlines(), 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(
                    raw, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s))
                )
            except ValueError as error:
                raise EvidenceError(f"Invalid JSONL {name}:{line}") from error
            if not isinstance(record, dict):
                raise EvidenceError(f"Expected object in {name}:{line}")
            rows.append(
                {
                    "record": record,
                    "provenance": {
                        "path": name,
                        "line": line,
                        "record_sha256": digest(raw),
                    },
                }
            )
        return rows

    def image(self, name, expected_sha=None):
        data = self.read(name)
        if data is None:
            return {"path": name, "status": "missing_or_deleted", "sha256": None}
        sha = digest(data)
        if expected_sha is not None and sha != expected_sha:
            return {"path": name, "status": "hash_mismatch", "sha256": sha}
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
                size = list(im.size)
        except (OSError, ValueError) as error:
            return {
                "path": name,
                "status": "undecodable",
                "error": str(error),
                "sha256": sha,
            }
        return {
            "path": name,
            "status": "available",
            "sha256": sha,
            "size_wh": size,
            "hash_checked_against_archive": expected_sha is not None,
        }

    def assert_unchanged(self):
        for name, sha in list(self.manifest.items()):
            data = self.read(name)
            if (None if data is None else digest(data)) != sha:
                raise EvidenceError(f"Archive changed: {name}")


def stamp(time_s, clock, epoch, uncertainty_s=None, **extra):
    return {
        "time_s": time_s if finite(time_s) else None,
        "clock": clock,
        "clock_epoch": epoch,
        "uncertainty_s": uncertainty_s,
        **extra,
    }


def host_stamp(time_s, trial):
    # Session scopes the host clock; never compare monotonic values across runs/hosts.
    return stamp(time_s, "host_monotonic", trial.get("session_id"))


def source_time(source, trial):
    """Expose point estimates without inventing bounds for the live latency estimate."""
    s = source.get("stamp", {})
    original = stamp(
        s.get("t"),
        s.get("clock"),
        source.get("clock_epoch"),
        None,
        semantics=s.get("semantics"),
        source_stamp=s,
        source_uncertainty_sigma_s=s.get("uncertainty_s"),
    )
    mapped = source.get("exposure_monotonic_s")
    mapping = source.get("monotonic_mapping")
    if finite(mapped) and mapping:
        provenance = source.get("provenance", {})
        # Only an explicit bound counts. A calibration sigma is NOT an error bound.
        bound = provenance.get("exposure_monotonic_error_bound_s")
        return stamp(
            mapped,
            "host_monotonic",
            trial.get("session_id"),
            bound,
            basis="archived_exposure_mapping",
            mapping=mapping,
            original=original,
            estimated=True,
        )
    if s.get("semantics") == "exposure":
        return {
            **original,
            "uncertainty_s": source.get("provenance", {}).get(
                "exposure_source_error_bound_s"
            ),
            "basis": "source_exposure_stamp",
            "estimated": False,
        }
    return {
        **original,
        "time_s": None,
        "basis": "capture_time_unknown_receipt_is_not_exposure",
        "original": original,
        "estimated": False,
    }


def bounded(t):
    return (
        finite(t.get("time_s"))
        and finite(t.get("uncertainty_s"))
        and t["uncertainty_s"] >= 0
        and bool(t.get("clock"))
        and bool(t.get("clock_epoch"))
    )


def same_clock(a, b):
    return (
        bool(a.get("clock"))
        and bool(a.get("clock_epoch"))
        and (a["clock"], a["clock_epoch"]) == (b.get("clock"), b.get("clock_epoch"))
    )


def mapped_time(t, mappings):
    for m in mappings:
        if (t.get("clock"), t.get("clock_epoch")) == (
            m["source_clock"],
            m["source_epoch"],
        ):
            if not finite(t.get("time_s")):
                return t
            uncertainty = t.get("uncertainty_s")
            return {
                **t,
                "time_s": t["time_s"] + m["offset_s"],
                "clock": m["target_clock"],
                "clock_epoch": m["target_epoch"],
                "uncertainty_s": None
                if uncertainty is None
                else uncertainty + m["uncertainty_s"],
                "evaluation_only_mapping": m,
                "unmapped": t,
            }
    return t


def relation(t, turn, mappings=()):
    if not turn.get("verified_interval"):
        return "unknown"
    t = mapped_time(t, mappings)
    start, end = turn["start"], turn["end"]
    if not bounded(t) or not same_clock(t, start):
        return "unknown"
    lo, hi = t["time_s"] - t["uncertainty_s"], t["time_s"] + t["uncertainty_s"]
    if hi < start["time_s"] - start["uncertainty_s"]:
        return "before"
    if lo > end["time_s"] + end["uncertainty_s"]:
        return "after"
    if (
        lo >= start["time_s"] + start["uncertainty_s"]
        and hi <= end["time_s"] - end["uncertainty_s"]
    ):
        return "during"
    return "overlaps_uncertain_turn_boundary"


def age(event_time, exposure, trial, mappings=()):
    t = mapped_time(exposure, mappings)
    event = host_stamp(event_time, trial)
    if (
        not same_clock(event, t)
        or not finite(event["time_s"])
        or not finite(t.get("time_s"))
    ):
        return {
            "seconds": None,
            "error_bound_s": None,
            "basis": "unmapped_or_unknown_capture_time",
        }
    value = event["time_s"] - t["time_s"]
    return {
        "seconds": value if value >= 0 else None,
        "error_bound_s": t.get("uncertainty_s"),
        "basis": t.get("basis"),
        "estimated": t.get("estimated", False),
        "error": "event_precedes_capture" if value < 0 else None,
    }


def read_turn(archive, sidecar):
    raw_turn = (sidecar or {}).get("turn") or {}
    if not isinstance(raw_turn, dict):
        raise EvidenceError("Turn annotation must be an object")
    turn = dict(raw_turn)
    mappings = (sidecar or {}).get("clock_mappings", [])
    if not isinstance(mappings, list) or any(not isinstance(m, dict) for m in mappings):
        raise EvidenceError("Clock mappings must be a list of objects")
    seen = set()
    for m in mappings:
        keys = (
            "source_clock",
            "source_epoch",
            "target_clock",
            "target_epoch",
            "provenance",
        )
        if (
            not all(m.get(k) for k in keys)
            or not finite(m.get("offset_s"))
            or not finite(m.get("uncertainty_s"))
            or m["uncertainty_s"] < 0
        ):
            raise EvidenceError(
                "Clock mapping needs domains, epochs, offset, error bound and provenance"
            )
        source = m["source_clock"], m["source_epoch"]
        if source in seen:
            raise EvidenceError("Ambiguous repeated clock mapping")
        seen.add(source)
    for key in ("start", "end"):
        t = turn.get(key)
        if t is None:
            continue
        if not isinstance(t, dict):
            raise EvidenceError("Turn stamps must be objects")
        for name in ("time_s", "uncertainty_s"):
            if t.get(name) is not None and (
                not finite(t[name]) or (name == "uncertainty_s" and t[name] < 0)
            ):
                raise EvidenceError(
                    "Turn times/bounds must be finite seconds or null; bounds must be nonnegative"
                )
        if "event_ref" in t:
            ref = t["event_ref"]
            rows = archive.rows(ref["path"])
            matches = [r for r in rows if r["provenance"]["line"] == ref["line"]]
            if len(matches) != 1 or matches[0]["record"].get("event") != ref.get(
                "event"
            ):
                raise EvidenceError("Turn event reference is missing or does not match")
            if t.get("clock") != "host_monotonic":
                raise EvidenceError(
                    "Recorded event references use host_monotonic seconds"
                )
            t = {
                **t,
                "time_s": matches[0]["record"].get("host_monotonic_s"),
                "resolved_event": matches[0],
            }
        turn[key] = t
    start, end = turn.get("start") or {}, turn.get("end") or {}
    turn["verified_interval"] = bool(
        turn.get("provenance")
        and bounded(start)
        and bounded(end)
        and same_clock(start, end)
        and start["time_s"] + start["uncertainty_s"]
        <= end["time_s"] - end["uncertainty_s"]
    )
    turn["verification_basis"] = (
        "evaluation_only_operator_annotation_not_measured_motion"
        if turn["verified_interval"]
        else "missing_or_uncertain_turn_evidence"
    )
    return turn, mappings


def _equal(errors, name, a, b):
    if a is None or b is None or a != b:
        errors.append(name + "_mismatch_or_missing")


def inspect_prediction(archive, trial, row, turn, mappings):
    p, errors, warnings = row["record"], [], []
    out = {
        "record": p,
        "provenance": row["provenance"],
        "errors": errors,
        "warnings": warnings,
        "selected_inputs": [],
        "marked_input": None,
        "eligible": False,
    }
    _equal(errors, "trial_id", p.get("trial_id"), trial.get("trial_id"))
    _equal(
        errors,
        "instruction_revision",
        p.get("instruction_revision"),
        trial.get("instruction_revision"),
    )
    _equal(errors, "instruction", p.get("instruction"), trial.get("instruction"))
    reason = p.get("rejection_reason_at_receipt")
    out["receipt_status"] = p.get("label_at_receipt", "UNKNOWN")
    if (
        isinstance(reason, str)
        and reason.startswith("stale")
        and out["receipt_status"] == "STALE"
    ):
        warnings.append("stale_at_receipt_image_grounding_remains_evaluable")
    elif reason or out["receipt_status"] != "CURRENT":
        errors.append(
            "rejected_or_noncurrent_at_receipt:" + str(reason or out["receipt_status"])
        )
    request = p.get("request_directory")
    if not isinstance(request, str):
        errors.append("missing_request_directory")
        return out
    job = archive.json(f"{request}/job.json")
    snapshot = archive.json(f"{request}/snapshot.json")
    receipt = archive.json(f"{request}/receipt.json")
    native = archive.json(f"{request}/native/result.json")
    if not job or not snapshot or not receipt:
        errors.append("missing_job_snapshot_or_receipt")
        return out
    identity = snapshot.get("identity", {})
    sequence = p.get("request_sequence")
    if (
        type(sequence) is not int
        or p.get("prediction_id") != f"{trial.get('trial_id')}-P{sequence:03d}"
    ):
        errors.append("prediction_id_does_not_match_own_trial_sequence")
    for key in ("session_id", "instruction_revision"):
        _equal(errors, key, identity.get(key), trial.get(key))
    _equal(
        errors, "request_sequence", identity.get("sequence"), p.get("request_sequence")
    )
    _equal(errors, "job_instruction", job.get("instruction"), trial.get("instruction"))
    expected_episode = f"{trial.get('session_id')}.r{trial.get('instruction_revision')}"
    _equal(errors, "job_episode", job.get("episode_id"), expected_episode)
    _equal(errors, "job_epoch", job.get("clock_epoch"), identity.get("clock_epoch"))
    _equal(
        errors, "job_decision", job.get("decision_id"), str(identity.get("sequence"))
    )
    response = receipt.get("response", {})
    _equal(errors, "response_identity", response.get("identity"), identity)
    _equal(
        errors,
        "arrival_time",
        receipt.get("receipt_monotonic_s"),
        p.get("received_host_monotonic_s"),
    )
    if receipt.get("rejection_reason") != reason:
        errors.append("receipt_rejection_disagrees")
    result = response.get("result", {})
    if not native:
        warnings.append("native_result_missing_using_receipt_result")
    elif native != result:
        errors.append("native_result_disagrees_with_receipt")
    for key in ("episode_id", "clock_epoch", "decision_id"):
        _equal(errors, "result_" + key, result.get(key), job.get(key))
    for key in ("native_output", "pointing", "waypoints"):
        _equal(errors, "native_" + key, result.get(key), p.get(key))
    if (
        result.get("execution_authorized") is not False
        or result.get("status") != "decoded"
    ):
        errors.append("not_a_decoded_advisory_result")
    derived = target_fraction(p.get("pointing"))
    declared = p.get("target_fraction")
    if (None if derived is None else list(derived)) != declared:
        errors.append("target_mark_disagrees_with_native_pointing")
    if bool(p.get("opos_not_visible")) != (p.get("pointing", {}).get("opos") == 0):
        errors.append("opos_zero_flag_disagrees")
    convention = "absolute_ego_SE2_x_forward_y_left_yaw_CCW_metres_radians"
    if (
        p.get("waypoint_convention") != convention
        or result.get("waypoint_convention") != convention
    ):
        errors.append("unsupported_waypoint_convention")
    if (
        p.get("waypoint_time_base") is not None
        or result.get("waypoint_time_base") is not None
    ):
        errors.append("unexpected_waypoint_time_base")
    if p.get("native_stop") != result.get("decoder_stop") or p.get(
        "explicit_stop_code"
    ) != result.get("explicit_stop_code"):
        errors.append("native_stop_flags_disagree")
    if (
        snapshot.get("annotations_consumed") is not False
        or snapshot.get("previous_actions_supplied") is not False
    ):
        errors.append("unexpected_annotation_or_action_feedback")
    if (
        snapshot.get("reset_policy")
        != "reconstruct_complete_causal_instruction_history_per_request"
    ):
        errors.append("unsupported_history_reset_policy")
    frames, history = job.get("frames", []), snapshot.get("history", [])
    indices = result.get("native_selected_frame_indices")
    if (
        not isinstance(indices, list)
        or not indices
        or len(frames) != len(history)
        or any(type(i) is not int or not 0 <= i < len(frames) for i in indices)
    ):
        errors.append("missing_or_invalid_selected_input_indices")
        return out
    if indices != sorted(indices):
        errors.append("selected_inputs_not_in_causal_order")
    _equal(errors, "history_frame_count", p.get("history_frames_in_job"), len(frames))
    out["native_segments"] = result.get("native_segments")
    out["snapshot_identity"] = identity
    out["history_policy"] = snapshot.get("history_policy")
    out["reset_policy"] = snapshot.get("reset_policy")
    out["host_submit_monotonic_s"] = snapshot.get("host_submit_monotonic_s")
    submit, arrival = (
        snapshot.get("host_submit_monotonic_s"),
        p.get("received_host_monotonic_s"),
    )
    if not finite(submit) or not finite(arrival) or submit > arrival:
        errors.append("invalid_submit_arrival_order")
    for i, (f, h) in enumerate(zip(frames, history)):
        s = h.get("source", {})
        for key in ("observation_s", "available_s"):
            _equal(errors, f"frame_{i}_{key}", f.get(key), s.get(key))
        _equal(
            errors,
            f"frame_{i}_epoch",
            s.get("clock_epoch"),
            identity.get("clock_epoch"),
        )
        _equal(
            errors, f"frame_{i}_source", s.get("source_id"), identity.get("source_id")
        )
        _equal(
            errors,
            f"frame_{i}_stamp",
            s.get("stamp", {}).get("t"),
            f.get("observation_s"),
        )
        _equal(
            errors,
            f"frame_{i}_clock",
            s.get("stamp", {}).get("clock"),
            snapshot.get("current_source", {}).get("stamp", {}).get("clock"),
        )
        if not all(
            finite(x)
            for x in (
                f.get("observation_s"),
                f.get("available_s"),
                job.get("decision_time_s"),
            )
        ):
            errors.append(f"frame_{i}_missing_causal_times")
        elif not f["observation_s"] <= f["available_s"] <= job["decision_time_s"]:
            errors.append(f"frame_{i}_noncausal")
        received, started = (
            s.get("received_monotonic_s"),
            trial.get("started_host_monotonic_s"),
        )
        # The live receiver stamps first RTP receipt; the app admits the decoded
        # frame later. Fresh instruction history is checked at app admission when
        # that independently recorded timestamp is available. Never relabel capture.
        admission = s.get("provenance", {}).get("session_receive_monotonic_s", received)
        if (
            not finite(admission)
            or (finite(received) and admission < received)
            or (finite(submit) and admission > submit)
        ):
            errors.append(f"frame_{i}_invalid_history_admission_time")
        if finite(admission) and finite(started) and admission < started:
            errors.append(f"frame_{i}_predates_own_trial_history")
        elif finite(received) and finite(started) and received < started:
            warnings.append(
                f"frame_{i}_sensor_receipt_precedes_ask_but_application_admission_follows_reset"
            )
        if finite(received) and finite(submit) and received > submit:
            errors.append(f"frame_{i}_received_after_submission")
    ref = p.get("source_image") or {}
    marked_index = ref.get("job_frame_index")
    if marked_index != indices[-1]:
        errors.append("marked_input_not_last_selected_frame")
    current = snapshot.get("current_source", {})
    if current != history[-1].get("source"):
        errors.append("snapshot_current_source_disagrees")
    if "source_frame" in p:
        expected = {
            k: current.get(k)
            for k in ("frame_id", "observation_s", "available_s", "clock_epoch")
        }
        expected.update(
            {
                k: current.get("stamp", {}).get(k)
                for k in ("clock", "semantics", "uncertainty_s")
            }
        )
        source_frame = p["source_frame"]
        if not isinstance(source_frame, dict):
            errors.append("prediction_current_source_frame_disagrees")
        else:
            if any(
                k not in source_frame or source_frame[k] != v
                for k, v in expected.items()
            ):
                errors.append("prediction_current_source_frame_disagrees")
            optional = ("received_monotonic_s", "exposure_monotonic_s")
            for key in optional:
                if key in source_frame and (
                    key not in current
                    or source_frame[key] != current[key]
                    or (source_frame[key] is not None and not finite(source_frame[key]))
                ):
                    errors.append("prediction_source_frame_" + key + "_disagrees")
            out["source_frame_validation"] = {
                "required_fields_checked": list(expected),
                "optional_timestamps_checked": [
                    k for k in optional if k in source_frame
                ],
                "uninterpreted_additional_fields": {
                    k: v
                    for k, v in source_frame.items()
                    if k not in expected and k not in optional
                },
            }
    out["current_source"] = current  # distinct from the exact image carrying the mark
    for i in indices:
        source = history[i]["source"]
        # Absolute paths from the original job may be relocated. Only the canonical
        # archived layout is opened, never an arbitrary absolute job/image path.
        prepared_path = f"{request}/frame_{i:04d}.png"
        if Path(frames[i].get("path", "")).name != Path(prepared_path).name:
            errors.append(f"frame_{i}_prepared_reference_mismatch")
        prepared = archive.image(prepared_path, frames[i].get("sha256"))
        if not frames[i].get("sha256"):
            errors.append(f"frame_{i}_prepared_hash_missing")
        native_path = (
            prepared_path if p.get("mock") else f"{request}/native/input_{i:04d}.png"
        )
        image = archive.image(
            native_path, ref.get("file_sha256") if i == marked_index else None
        )
        if prepared["status"] != "available" or image["status"] != "available":
            errors.append(f"frame_{i}_missing_or_changed_image")
        capture = source_time(source, trial)
        item = {
            "job_frame_index": i,
            "source": source,
            "capture_time": capture,
            "turn_relation": relation(capture, turn, mappings),
            "image": image,
            "prepared_image": prepared,
            "preprocessing": {k: v for k, v in history[i].items() if k != "source"},
        }
        out["selected_inputs"].append(item)
        if i == marked_index:
            out["marked_input"] = item
            if ref.get("path") != native_path or not ref.get("file_sha256"):
                errors.append("marked_source_image_reference_mismatch")
            out["age_at_receipt"] = age(
                p.get("received_host_monotonic_s"), capture, trial, mappings
            )
    out["eligible"] = not errors
    return out


DIMENSIONS = (
    "marking",
    "visibility",
    "opos0_agreement",
    "path",
    "stopping",
    "navigation",
)


def inspect_review(row, trial, predictions, mappings):
    r = row["record"]
    out = {
        **row,
        "scope": "unknown",
        "prediction_ids": [],
        "errors": [],
        "judgments": {key: "unknown" for key in DIMENSIONS},
    }
    errors, judgments = out["errors"], out["judgments"]
    if r.get("trial_id") != trial["trial_id"]:
        errors.append("review_trial_mismatch")
    if r.get("model_feedback") is not False or r.get("confirmed_by_user") is not True:
        errors.append("not_confirmed_evaluation_only_human_feedback")
    schema = r.get("schema")
    if schema == "go2-vln-guided-answer-review-v1":
        out["scope"] = "one_prediction_target_only"
        ids = [r.get("prediction_id")]
    elif schema == "go2-vln-human-review-v1":
        ids = r.get("applies_to_predictions", [])
        out["scope"] = (
            "legacy_prediction_set"  # preserve even a singleton's original scope
        )
    else:
        errors.append("unsupported_review_schema")
        return out
    out["prediction_ids"] = ids
    if not isinstance(ids, list) or not ids or any(i not in predictions for i in ids):
        errors.append("review_prediction_reference_missing_or_mismatched")
        return out
    if len(ids) != 1:
        out["scope_note"] = (
            "Group/trial judgment retained; never distributed to individual predictions"
        )
        return out
    pred = predictions[ids[0]]
    p = pred["record"]
    reviewed, arrived = (
        r.get("reviewed_host_monotonic_s"),
        p.get("received_host_monotonic_s"),
    )
    if finite(reviewed) and finite(arrived) and reviewed < arrived:
        errors.append("review_precedes_prediction_arrival")
    marked = pred.get("marked_input")
    if marked:
        out["age_at_review"] = age(
            r.get("reviewed_host_monotonic_s"), marked["capture_time"], trial, mappings
        )
    else:
        out["age_at_review"] = {"seconds": None, "basis": "missing_marked_input"}
    if schema == "go2-vln-guided-answer-review-v1":
        if r.get("source_image") != p.get("source_image") or not r.get("source_image"):
            errors.append("review_image_reference_mismatch_or_missing")
        if "displayed_image_sha256" in r and r["displayed_image_sha256"] != (
            p.get("source_image") or {}
        ).get("file_sha256"):
            errors.append("review_displayed_image_hash_mismatch")
        choice = r.get("choice")
        point = p.get("target_fraction") is not None
        opos0 = p.get("opos_not_visible") is True
        permitted = (
            {"correct", "wrong", "unsure", "target_not_in_view"}
            if point
            else {"agree_not_in_view", "disagree_target_in_view", "unsure"}
            if opos0
            else {"no_point", "unsure"}
        )
        if choice not in permitted:
            errors.append("choice_incompatible_with_archived_proposal")
        if choice in ("correct", "wrong"):
            judgments["marking"] = choice
        if choice in ("correct", "disagree_target_in_view"):
            judgments["visibility"] = "present"
        elif choice in ("target_not_in_view", "agree_not_in_view"):
            judgments["visibility"] = "absent"
        if choice in ("agree_not_in_view", "disagree_target_in_view"):
            judgments["opos0_agreement"] = (
                "agree" if choice == "agree_not_in_view" else "disagree"
            )
    else:
        if r.get("marking") in ("correct", "wrong"):
            judgments["marking"] = r["marking"]
        if r.get("target_presence") in ("present", "absent"):
            judgments["visibility"] = r["target_presence"]
        for key in ("path", "stopping"):
            if r.get(key) in ("plausible", "inappropriate", "appropriate"):
                judgments[key] = r[key]
        if judgments["marking"] == "correct" and judgments["visibility"] == "absent":
            errors.append("legacy_review_has_inconsistent_marking_and_visibility")
    out["eligible"] = not errors
    return out


def consensus(reviews, prediction_id):
    exact = [
        r
        for r in reviews
        if r["prediction_ids"] == [prediction_id] and r.get("eligible")
    ]
    result = {}
    for key in DIMENSIONS:
        values = [r["judgments"][key] for r in exact]
        known = set(values) - {"unknown"}
        state = (
            "conflicting"
            if len(known) > 1
            else (
                "ambiguous"
                if known and "unknown" in values
                else next(iter(known))
                if known
                else "unknown"
            )
        )
        result[key] = {
            "status": state,
            "review_count": len(values),
            "provenance": [r["provenance"] for r in exact],
        }
    return result


def inspect_trial(archive, trial_id, *, prediction_id=None, turn=None, mappings=()):
    if (
        not trial_id
        or Path(trial_id).name != trial_id
        or trial_id.startswith(".")
        or "\\" in trial_id
    ):
        raise EvidenceError("Explicit valid trial IDs required")
    root = f"trials/{trial_id}"
    trial = archive.json(f"{root}/trial.json", required=True)
    end = archive.json(f"{root}/trial_end.json", required=True)
    if (
        trial.get("trial_id") != trial_id
        or end.get("trial_id") != trial_id
        or not end.get("end_reason")
    ):
        raise EvidenceError("Trial ID mismatch or incomplete trial end")
    rows = archive.rows(f"{root}/predictions.jsonl")
    predictions = {}
    for row in rows:
        pid = row["record"].get("prediction_id")
        if not isinstance(pid, str) or pid in predictions:
            raise EvidenceError("Missing or duplicated prediction identity")
        predictions[pid] = inspect_prediction(archive, trial, row, turn or {}, mappings)
    if prediction_id is not None and prediction_id not in predictions:
        raise EvidenceError("Explicit prediction ID does not belong to this trial")
    selected = (
        prediction_id
        if prediction_id is not None
        else (next(iter(predictions)) if len(predictions) == 1 else None)
    )
    reviews = [
        inspect_review(r, trial, predictions, mappings)
        for r in archive.rows(f"{root}/human_labels.jsonl")
    ]
    for pid, prediction in predictions.items():
        prediction["human_assessment"] = consensus(reviews, pid)
    return {
        "trial": trial,
        "end": end,
        "predictions": list(predictions.values()),
        "reviews": reviews,
        "selected_prediction_id": selected,
        "selection_basis": "explicit"
        if prediction_id
        else "only_prediction"
        if selected
        else "ambiguous_or_missing",
        "marks": archive.rows(f"{root}/marks.jsonl"),
        "notes": archive.rows(f"{root}/notes.jsonl"),
        "corrections": archive.rows(f"{root}/corrections.jsonl"),
    }


def chosen(view):
    return next(
        (
            p
            for p in view["predictions"]
            if p["record"]["prediction_id"] == view["selected_prediction_id"]
        ),
        None,
    )


def pair_outcome(before, after, target_ok):
    b, a = chosen(before), chosen(after)
    if not target_ok or not b or not a or not b["eligible"] or not a["eligible"]:
        return {
            "code": "insufficient_evidence",
            "description": "Evidence is insufficient for a paired-view assessment.",
            "temporal_verification": "incomplete",
            "reviewed_image_outcome": {
                "code": "insufficient_evidence",
                "description": "Evidence is insufficient for a paired reviewed-image assessment.",
                "scope": "reviewed_images_only",
            },
        }
    bm, am = (p["human_assessment"]["marking"]["status"] for p in (b, a))
    bv = b["human_assessment"]["visibility"]["status"]
    view_change = "unknown_or_no_supported_change"
    if bv == "absent" and am == "correct":
        view_change = "visibility_change_and_correct_mark_in_reviewed_image"
    elif bm == "wrong" and am == "correct":
        view_change = "wrong_to_correct_in_reviewed_images"
    elif bm == "wrong" and am == "wrong":
        view_change = "wrong_mark_in_both_reviewed_images"
    elif all(
        p["record"].get("opos_not_visible") is True
        and p["human_assessment"]["opos0_agreement"]["status"] == "disagree"
        and p["human_assessment"]["visibility"]["status"] == "present"
        for p in (b, a)
    ):
        view_change = "persistent_false_absence_in_reviewed_images"
    image_descriptions = {
        "visibility_change_and_correct_mark_in_reviewed_image": "Human reviewers report absent then present/correctly marked; this is a visibility change, not a recognition-accuracy gain.",
        "wrong_to_correct_in_reviewed_images": "Previously wrong marking becomes correct in the two reviewed images.",
        "wrong_mark_in_both_reviewed_images": "Wrong target marking persists in both reviewed images.",
        "persistent_false_absence_in_reviewed_images": "Persistent false absence: the model emits OPOS-0 in both views and humans report the target present in both.",
    }
    after_relation = a["marked_input"]["turn_relation"]
    result = {
        "view_comparison": view_change,
        "before_marked_image_relation": b["marked_input"]["turn_relation"],
        "after_marked_image_relation": after_relation,
        "temporal_verification": "incomplete",
        "reviewed_image_outcome": {
            "code": view_change,
            "description": image_descriptions.get(
                view_change,
                "Human review is missing, ambiguous, or does not establish a supported image-outcome transition.",
            ),
            "scope": "reviewed_images_only",
        },
    }
    if after_relation != "after":
        return {
            **result,
            "code": "no_verified_post_turn_observation",
            "description": "No verified post-turn observation is available; reviewed images can still be compared.",
        }
    clean = all(i["turn_relation"] == "before" for i in b["selected_inputs"]) and all(
        i["turn_relation"] == "after" for i in a["selected_inputs"]
    )
    if not clean:
        return {
            **result,
            "code": "insufficient_evidence",
            "description": "The marked after-image is post-turn, but the selected histories do not establish a clean before/after pair.",
        }
    result["temporal_verification"] = "relative_to_declared_turn_and_clock_bounds"
    descriptions = {
        "visibility_change_and_correct_mark_in_reviewed_image": "Target becomes visible and is correctly marked; this is a visibility change, not a recognition-accuracy gain.",
        "wrong_to_correct_in_reviewed_images": "Previously wrong target marking becomes correct in the reviewed new view.",
        "wrong_mark_in_both_reviewed_images": "Wrong marking persists in a reviewed post-turn image.",
        "persistent_false_absence_in_reviewed_images": image_descriptions[
            "persistent_false_absence_in_reviewed_images"
        ],
    }
    if view_change in descriptions:
        return {**result, "code": view_change, "description": descriptions[view_change]}
    return {
        **result,
        "code": "human_review_missing_or_ambiguous",
        "description": "Human review is missing, ambiguous, or does not establish one of the declared marking changes.",
    }


def analyze_pair(
    archive,
    before_id,
    after_id,
    *,
    sidecar=None,
    before_prediction=None,
    after_prediction=None,
):
    if before_id == after_id:
        raise EvidenceError("Before and after must be different independent trials")
    if sidecar is not None:
        if not isinstance(sidecar, dict):
            raise EvidenceError("Pair annotation must be a JSON object")
        if (
            sidecar.get("schema") != PAIR_SCHEMA
            or not sidecar.get("episode_id")
            or not sidecar.get("target_wording")
        ):
            raise EvidenceError(
                "Sidecar requires the pair schema, episode ID and target wording"
            )
        for name, tid in (("before_trial_id", before_id), ("after_trial_id", after_id)):
            if sidecar.get(name) != tid:
                raise EvidenceError(
                    "Sidecar trial references must match the explicit pair"
                )
    turn, mappings = read_turn(archive, sidecar)
    before = inspect_trial(
        archive,
        before_id,
        prediction_id=before_prediction,
        turn=turn,
        mappings=mappings,
    )
    after = inspect_trial(
        archive, after_id, prediction_id=after_prediction, turn=turn, mappings=mappings
    )
    bt, at = before["trial"], after["trial"]
    target_ok = bt.get("instruction") == at.get("instruction") and bt.get(
        "operator_text"
    ) == at.get("operator_text")
    if sidecar and bt.get("kind") == "target":
        target_ok = target_ok and sidecar["target_wording"] == bt.get("operator_text")
    independent = (bt.get("session_id"), bt.get("instruction_revision")) != (
        at.get("session_id"),
        at.get("instruction_revision"),
    )
    result = {
        "schema": SUMMARY_SCHEMA,
        "analysis_kind": "two_independent_queries_linked_only_for_evaluation",
        "archive_root": str(archive.root),
        "archive_git_commit": archive.run.get("git_commit"),
        "pair_annotation": sidecar,
        "turn": turn,
        "clock_mappings": mappings,
        "same_target_and_instruction": target_ok,
        "distinct_query_revisions": independent,
        "before": before,
        "after": after,
        "outcome": pair_outcome(before, after, target_ok and independent),
        "limitations": [
            "Descriptive correlated-room comparison; no success rate or navigation claim.",
            "Native paths are unexecuted, not registered to LiDAR or physical robot coordinates.",
            "Guided feedback assesses target grounding/visibility only; path and stopping are unreviewed.",
            "Clock bounds and manual turns are declared evidence, not independent motion validation.",
            "Raw observation NPZ/LiDAR files are not opened or revalidated.",
        ],
    }
    timeline = []
    for role, view in (("before", before), ("after", after)):
        t = view["trial"]
        common = {"role": role, "trial_id": t["trial_id"]}
        timeline.append(
            {
                **common,
                "event": "instruction",
                "instruction": t["instruction"],
                "time": host_stamp(t.get("started_host_monotonic_s"), t),
            }
        )
        for p in view["predictions"]:
            pid = p["record"]["prediction_id"]
            for f in p["selected_inputs"]:
                timeline.append(
                    {
                        **common,
                        "event": "actual_model_input",
                        "prediction_id": pid,
                        "job_frame_index": f["job_frame_index"],
                        "frame_id": f["source"].get("frame_id"),
                        "time": f["capture_time"],
                        "turn_relation": f["turn_relation"],
                        "image": f["image"],
                    }
                )
            timeline.append(
                {
                    **common,
                    "event": "prediction_arrival",
                    "prediction_id": pid,
                    "time": host_stamp(p["record"].get("received_host_monotonic_s"), t),
                    "receipt_status": p["receipt_status"],
                }
            )
        for r in view["reviews"]:
            timeline.append(
                {
                    **common,
                    "event": "human_review",
                    "prediction_ids": r["prediction_ids"],
                    "time": host_stamp(r["record"].get("reviewed_host_monotonic_s"), t),
                    "provenance": r["provenance"],
                }
            )
    for key in ("start", "end"):
        if turn.get(key):
            timeline.append(
                {
                    "event": "manual_turn_" + key,
                    "time": turn[key],
                    "evaluation_only": True,
                }
            )
    result["timeline"] = (
        timeline  # causal links, not a false global sort of unrelated clocks
    )
    archive.assert_unchanged()
    result["consumed_artifact_sha256"] = dict(archive.manifest)
    return result
