"""Import the recorder's append-only pair annotations without running its linker.

Copied prediction/review summaries are references only. Image judgments always
come from viewpoint.analyze_pair reading the original trial/request/review files.
Receipt-time odometry annotations never become camera-exposure error bounds.
"""

import math
from pathlib import Path

from .viewpoint import EvidenceError, PAIR_SCHEMA, chosen, finite

LINK_SCHEMA = "go2-vln-paired-view-evaluation-v1"


def import_linked_pair(
    archive, path, before_id, after_id, *, method_version, record_line=None
):
    """Select an explicit pair/method, retaining every linkage of that same pair."""
    if method_version != 2:
        raise EvidenceError(
            "This adapter supports explicitly declared linkage method version 2"
        )
    path = str(Path(path))
    rows = archive.rows(path)
    matches = [
        row
        for row in rows
        if row["record"].get("schema") == LINK_SCHEMA
        and row["record"].get("before", {}).get("trial_id") == before_id
        and row["record"].get("after", {}).get("trial_id") == after_id
    ]
    candidates = [
        row
        for row in matches
        if row["record"]
        .get("viewpoint_change", {})
        .get("measured", {})
        .get("method_version")
        == method_version
        and (record_line is None or row["provenance"]["line"] == record_line)
    ]
    if len(candidates) != 1:
        raise EvidenceError(
            "Expected one linkage for this pair/method; use --linkage-line for repeated records of the same method"
        )
    selected = candidates[0]
    record = selected["record"]
    if record.get("model_feedback") is not False:
        raise EvidenceError("Linkage must be evaluation-only (model_feedback: false)")
    refs = {}
    sessions = []
    for role, tid in (("before", before_id), ("after", after_id)):
        reference = record[role]
        trial = archive.json(f"trials/{tid}/trial.json", required=True)
        for field in (
            "trial_id",
            "instruction",
            "operator_text",
            "instruction_revision",
        ):
            if reference.get(field) != trial.get(field):
                raise EvidenceError(
                    f"Linkage {role} {field} disagrees with original trial"
                )
        if trial.get("operator_text") != record.get("target_operator_text"):
            raise EvidenceError("Linked target wording disagrees with original trial")
        pid = reference.get("model_output", {}).get("prediction_id")
        if not isinstance(pid, str) or not pid:
            raise EvidenceError("Linkage requires exact prediction references")
        refs[role] = pid
        sessions.append(trial.get("session_id"))
    measured = record.get("viewpoint_change", {}).get("measured", {})
    intervals = [
        measured.get(k) for k in ("deviation_based", "rate_based") if measured.get(k)
    ]
    valid_intervals = all(
        isinstance(i, dict)
        and finite(i.get("turn_start_host_monotonic_s"))
        and finite(i.get("turn_end_host_monotonic_s"))
        and i["turn_start_host_monotonic_s"] <= i["turn_end_host_monotonic_s"]
        for i in intervals
    )
    turn = {
        "direction": record.get("viewpoint_change", {}).get("operator_direction"),
        "provenance": {
            "source": "recorded_odometry_receipt_linkage",
            **selected["provenance"],
        },
        "semantics": "odometry_receipt_turn_estimate_not_camera_exposure_bound",
    }
    if measured.get("available") is True and intervals and valid_intervals:
        for key, value in (
            ("start", min(i["turn_start_host_monotonic_s"] for i in intervals)),
            ("end", max(i["turn_end_host_monotonic_s"] for i in intervals)),
        ):
            turn[key] = {
                "time_s": value,
                "clock": "host_monotonic",
                "clock_epoch": sessions[0] if sessions[0] == sessions[1] else None,
                "uncertainty_s": None,
                "semantics": "odometry_receipt_estimate_unknown_transport_and_detection_error",
            }
    return {
        "schema": PAIR_SCHEMA,
        "episode_id": f"linked:{before_id}:{after_id}",
        "target_wording": record["target_operator_text"],
        "before_trial_id": before_id,
        "after_trial_id": after_id,
        "turn": turn,
        "prediction_references": refs,
        "linkage_import": {
            "schema": LINK_SCHEMA,
            "method_version": method_version,
            "source_file": path,
            "source_file_sha256": archive.manifest[path],
            "selected": selected,
            "all_records_for_this_pair": matches,
            "unique_experiments": 1,
            "selection_policy": "explicit_trial_pair_and_method_version_not_latest_record",
            "copied_summaries_used_as_labels": False,
        },
    }


def _matches(a, b):
    # Serialization comparison tolerance, NOT a sensor timing error bound.
    return finite(a) and finite(b) and math.isclose(a, b, abs_tol=1e-6, rel_tol=0)


def _continuity(archive):
    rows = archive.rows("lidar/index.jsonl")
    dropped = [r["provenance"] for r in rows if r["record"].get("dropped_not_recorded")]
    epochs, discontinuities, missing, previous = set(), [], [], None
    for row in rows:
        r = row["record"]
        if not r.get("file"):
            continue
        epoch = r.get("robot_epoch")
        if epoch:
            epochs.add(epoch)
        if not archive.path("lidar/" + r["file"]).is_file():
            missing.append(r["file"])
        seq = r.get("odom_seq")
        if (
            not isinstance(seq, list)
            or len(seq) != 2
            or any(type(i) is not int for i in seq)
        ):
            discontinuities.append(
                {"reason": "missing_odom_sequence", **row["provenance"]}
            )
            previous = None
            continue
        if r.get("n_odom") != seq[1] - seq[0] + 1 or (
            previous and previous[0] == epoch and seq[0] != previous[1] + 1
        ):
            discontinuities.append(
                {"reason": "odom_sequence_gap_or_overlap", **row["provenance"]}
            )
        previous = (epoch, seq[1])
    camera = archive.summary.get("camera", {})
    bridge = archive.summary.get("camera_bridge") or {}
    return {
        "scope": "closed_run_index_and_summary_metadata_not_raw_cloud_or_odometry_reprocessing",
        "index_available": archive.manifest.get("lidar/index.jsonl") is not None,
        "index_records": len(rows),
        "robot_epochs": sorted(epochs),
        "dropped_lidar_chunks": len(dropped),
        "dropped_chunk_records": dropped,
        "odometry_sequence_issues": discontinuities,
        "missing_chunk_files": missing,
        "camera_delivery_counts": camera,
        "camera_bridge_counts": bridge.get("counts"),
        "camera_final_transport_stats": bridge.get("bridge_stats"),
    }


def receipt_evidence(archive, analysis):
    """Recompute receipt margins from original snapshots and the selected v2 interval.

    Audits clock epochs, saved delivery/transport counters and index continuity.
    It does not rerun the yaw estimator or infer metric registration/physical stop.
    """
    sidecar = analysis["pair_annotation"]
    link = sidecar["linkage_import"]["selected"]["record"]
    measured = link["viewpoint_change"]["measured"]
    claimed = link.get("second_answer_uses_post_turn_view", {})
    turn = analysis["turn"]
    issues, frames, sources, sessions, selected_frames = [], {}, {}, [], {}
    for role in ("before", "after"):
        pred = chosen(analysis[role])
        if pred is None:
            raise EvidenceError(
                "Linked receipt audit requires the exact referenced predictions"
            )
        snapshot = archive.json(
            pred["record"]["request_directory"] + "/snapshot.json", required=True
        )
        original = [h["source"] for h in snapshot["history"]]
        times = [s.get("received_monotonic_s") for s in original]
        selected_times = [
            f["source"].get("received_monotonic_s") for f in pred["selected_inputs"]
        ]
        sessions.append(analysis[role]["trial"].get("session_id"))
        sources[role] = {
            "clock_epochs": sorted({str(s.get("clock_epoch")) for s in original}),
            "source_ids": sorted({str(s.get("source_id")) for s in original}),
        }
        valid = bool(times) and all(finite(t) for t in times)
        frames[role] = {
            "count": len(times),
            "first_s": min(times) if valid else None,
            "last_s": max(times) if valid else None,
            "max_gap_s": max((y - x for x, y in zip(times, times[1:])), default=0)
            if valid
            else None,
            "times_s": times,
            "clock": "host_monotonic_receipt",
            "session_id": sessions[-1],
        }
        selected_valid = bool(selected_times) and all(finite(t) for t in selected_times)
        selected_frames[role] = {
            "first_s": min(selected_times) if selected_valid else None,
            "last_s": max(selected_times) if selected_valid else None,
            "times_s": selected_times,
        }
        copied = link[role].get("input_frames", {})
        if copied.get("count") != len(times) or any(
            not _matches(copied.get(k), frames[role][k]) for k in ("first_s", "last_s")
        ):
            issues.append(role + "_copied_receipt_summary_disagrees_with_snapshot")
        if len(sources[role]["clock_epochs"]) != 1 or sources[role]["clock_epochs"][
            0
        ] != copied.get("clock_epoch"):
            issues.append(role + "_camera_epoch_mismatch")
        if not valid or any(y < x or y - x > 0.375 for x, y in zip(times, times[1:])):
            issues.append(role + "_missing_or_discontinuous_input_receipts")
    if not sessions[0] or sessions[0] != sessions[1]:
        issues.append("different_host_session_clocks")
    if sources["before"] != sources["after"]:
        issues.append("camera_source_or_epoch_changed_between_queries")
    index = _continuity(archive)
    if (
        not index["index_available"]
        or len(index["robot_epochs"]) != 1
        or index["robot_epochs"] != measured.get("robot_epochs")
    ):
        issues.append("robot_epoch_or_index_evidence_missing_or_changed")
    if index["dropped_lidar_chunks"] != measured.get("dropped_lidar_chunks_in_run"):
        issues.append("linkage_drop_count_disagrees_with_index")
    if (
        index["dropped_lidar_chunks"]
        or index["odometry_sequence_issues"]
        or index["missing_chunk_files"]
    ):
        issues.append("incomplete_odometry_index_evidence")
    counts = index["camera_bridge_counts"]
    if counts is None or any(
        counts.get(k) != 0
        for k in ("clock_epoch_changes", "events_lost", "bridge_seq_gaps")
    ):
        issues.append("camera_epoch_or_delivery_continuity_not_confirmed")
    if any(
        index["camera_delivery_counts"].get(k) != 0
        for k in ("inbox_overflow", "receipt_gaps")
    ):
        issues.append("camera_delivery_gaps_or_missing_counters")
    # Preserve periodic counter provenance. These are periodic reports, not exposure clocks.
    events = archive.rows("session/events.jsonl")
    stats = [r for r in events if r["record"].get("event") == "camera_bridge_stats"]
    stat_keys = (
        "rtp_lost",
        "incomplete_units",
        "dropped_bridge_queue",
        "decoder_errors",
    )
    periodic_max = {
        k: max((r["record"].get("stats", {}).get(k, 0) for r in stats), default=None)
        for k in stat_keys
    }
    final = index["camera_final_transport_stats"] or {}
    index["periodic_transport_maxima"] = periodic_max
    index["last_periodic_status"] = (
        {
            "host_monotonic_s": stats[-1]["record"].get("host_monotonic_s"),
            "provenance": stats[-1]["provenance"],
        }
        if stats
        else None
    )
    index["final_counter_increase_after_periodic_status"] = {
        k: final[k] - periodic_max[k]
        for k in stat_keys
        if finite(final.get(k))
        and finite(periodic_max[k])
        and final[k] > periodic_max[k]
    }
    start, end = turn.get("start", {}).get("time_s"), turn.get("end", {}).get("time_s")
    valid = all(
        finite(t)
        for t in (start, end, frames["before"]["last_s"], frames["after"]["first_s"])
    )
    first_margin = start - frames["before"]["last_s"] if valid else None
    margin = frames["after"]["first_s"] - end if valid else None
    check = {
        "margin_s": margin,
        "first_input_margin_s": first_margin,
        "turn_end_s": end,
        "second_input_first_s": frames["after"]["first_s"],
    }
    if not valid or any(not _matches(claimed.get(k), v) for k, v in check.items()):
        issues.append("copied_turn_margins_disagree_with_original_receipts_or_v2_union")
    post = valid and margin > 0
    pre = valid and first_margin > 0
    if claimed.get("verified") != post or claimed.get("first_input_before_turn") != pre:
        issues.append("linker_ordering_flags_disagree_with_recomputed_receipt_order")
    return {
        "schema": "go2-vln-receipt-ordering-audit-v1",
        "method_version": 2,
        "status": "receipt_ordering_supported"
        if pre and post and not issues
        else "incomplete_or_inconsistent_receipt_evidence",
        "issues": issues,
        "original_snapshot_receipts": frames,
        "selected_native_input_receipts": selected_frames,
        "camera_sources": sources,
        "turn_start_s": start,
        **check,
        "first_input_before_turn": bool(pre),
        "second_input_after_turn": bool(post),
        "linker_claim": claimed,
        "odometry_annotation": measured,
        "continuity": index,
        "capture_time_verified": False,
        "qualification": "Receipt-time/odometry evidence only. Unknown camera exposure, odometry transport, threshold and sampling error bounds remain unknown. No physical-stop or navigation-success claim.",
        "raw_odometry_turn_estimator_rerun": False,
    }
