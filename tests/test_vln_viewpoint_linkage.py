"""Current recorder and method-v2 import: CPU-only, archived/synthetic data only."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np
import pytest

from go2_perception.vln.viewpoint import Archive, EvidenceError, analyze_pair
from go2_perception.vln.viewpoint_linkage import import_linked_pair, receipt_evidence
from test_vln_viewpoint import (
    edit,
    edit_native,
    edit_source,
    evaluate,
    pred_path,
    read,
    review_path,
    write,
)

FIXTURE = Path(__file__).parent / "fixtures/vln_viewpoint_current"


@pytest.fixture
def current_case(tmp_path):
    dest = tmp_path / "fixture"
    shutil.copytree(FIXTURE.parent / "vln_viewpoint", dest)
    case = dest / "run", read(dest / "pair.json")
    root, _ = case
    fields = read(FIXTURE / "recorder_fields.json")
    for role in ("before", "after"):
        p = edit(pred_path(root, role), source_frame=fields[role])
        edit(
            review_path(root, role),
            displayed_image_sha256=p["source_image"]["file_sha256"],
            observation_age_at_review_s=3.5,
            trial_already_ended_before_save=False,
        )
    (root / "evaluation").mkdir()
    (root / "evaluation/paired_views.jsonl").write_bytes(
        (FIXTURE / "paired_views.jsonl").read_bytes()
    )
    (root / "lidar").mkdir()
    np.savez(
        root / "lidar/chunk_000001.npz",
        odom_host_recv_wall=np.array([1000.0, 1000.01]),
        odom_wall_minus_monotonic=np.array([890.0, 890.0]),
    )
    write(
        root / "lidar/index.jsonl",
        {
            "file": "chunk_000001.npz",
            "robot_epoch": "synthetic-robot-epoch",
            "dropped_not_recorded": False,
            "odom_seq": [1, 2],
            "n_odom": 2,
        },
    )
    stats = {
        "rtp_lost": 0,
        "incomplete_units": 0,
        "dropped_bridge_queue": 0,
        "decoder_errors": 0,
    }
    edit(
        root / "summary.json",
        camera={"inbox_overflow": 0, "receipt_gaps": 0},
        camera_bridge={
            "counts": {
                "clock_epoch_changes": 0,
                "events_lost": 0,
                "bridge_seq_gaps": 0,
            },
            "bridge_stats": stats,
        },
    )
    write(
        root / "session/events.jsonl",
        {"event": "camera_bridge_stats", "host_monotonic_s": 119, "stats": stats},
    )
    return case


def linked(current_case, *, method=2, line=None):
    archive = Archive(current_case[0])
    sidecar = import_linked_pair(
        archive,
        "evaluation/paired_views.jsonl",
        "synthetic-before",
        "synthetic-after",
        method_version=method,
        record_line=line,
    )
    a = analyze_pair(
        archive,
        "synthetic-before",
        "synthetic-after",
        sidecar=sidecar,
        before_prediction=sidecar["prediction_references"]["before"],
        after_prediction=sidecar["prediction_references"]["after"],
    )
    a["receipt_ordering_evidence"] = receipt_evidence(archive, a)
    archive.assert_unchanged()
    return a


def test_current_recorder_optional_timestamps_match_snapshot(current_case):
    a = evaluate(current_case)
    for role in ("before", "after"):
        p = a[role]["predictions"][0]
        assert p["eligible"]
        assert p["source_frame_validation"]["optional_timestamps_checked"] == [
            "received_monotonic_s",
            "exposure_monotonic_s",
        ]
        assert (
            p["record"]["source_frame"]["received_monotonic_s"]
            == p["current_source"]["received_monotonic_s"]
        )


@pytest.mark.parametrize("field", ["received_monotonic_s", "exposure_monotonic_s"])
def test_inconsistent_added_timestamp_is_rejected(current_case, field):
    path = pred_path(current_case[0])
    p = read(path)
    p["source_frame"][field] = read(FIXTURE / "recorder_fields.json")[
        "inconsistent_optional_timestamp"
    ]["value"]
    write(path, p)
    errors = evaluate(current_case)["after"]["predictions"][0]["errors"]
    assert "prediction_source_frame_" + field + "_disagrees" in errors


def test_required_field_cannot_be_removed_and_unknown_fields_are_retained(current_case):
    path = pred_path(current_case[0])
    p = read(path)
    p["source_frame"]["future_evidence"] = {"retained": True}
    write(path, p)
    a = evaluate(current_case)
    pout = a["after"]["predictions"][0]
    assert pout["eligible"] and pout["record"]["source_frame"]["future_evidence"] == {
        "retained": True
    }
    p["source_frame"].pop("clock_epoch")
    write(path, p)
    assert not evaluate(current_case)["after"]["predictions"][0]["eligible"]


def test_new_displayed_image_hash_is_checked(current_case):
    edit(review_path(current_case[0]), displayed_image_sha256="mismatched")
    assert (
        "review_displayed_image_hash_mismatch"
        in evaluate(current_case)["after"]["reviews"][0]["errors"]
    )


def test_fresh_history_admission_distinct_from_sensor_receipt(current_case):
    case = current_case
    root, _ = case
    # Receiver gets one frame immediately before Ask; app admits it after reset.
    edit(root / "trials/synthetic-before/trial.json", started_host_monotonic_s=100.011)
    edit_source(
        root,
        "before",
        lambda s: s["provenance"].update(
            session_receive_monotonic_s=s["received_monotonic_s"] + 0.02
        ),
    )
    p = evaluate(case)["before"]["predictions"][0]
    assert p["eligible"]
    assert (
        "frame_0_sensor_receipt_precedes_ask_but_application_admission_follows_reset"
        in p["warnings"]
    )
    assert p["selected_inputs"][0]["source"]["received_monotonic_s"] == 100.01
    edit_source(
        root,
        "before",
        lambda s: s["provenance"].update(
            session_receive_monotonic_s=s["received_monotonic_s"]
        ),
    )
    assert (
        "frame_0_predates_own_trial_history"
        in evaluate(case)["before"]["predictions"][0]["errors"]
    )


def test_import_explicit_pair_method_and_preserve_earlier_record(current_case):
    a = linked(current_case)
    info = a["pair_annotation"]["linkage_import"]
    assert (
        info["selected"]["provenance"]["line"] == 2 and info["unique_experiments"] == 1
    )
    assert [r["provenance"]["line"] for r in info["all_records_for_this_pair"]] == [
        1,
        2,
    ]
    data = (current_case[0] / "evaluation/paired_views.jsonl").read_bytes()
    assert info["source_file_sha256"] == hashlib.sha256(data).hexdigest()
    assert (
        info["selected"]["provenance"]["record_sha256"]
        == hashlib.sha256(data.splitlines()[1]).hexdigest()
    )


def test_copied_labels_are_not_source_labels_and_verified_is_not_exposure_bound(
    current_case,
):
    a = linked(current_case)
    assert (
        a["pair_annotation"]["linkage_import"]["selected"]["record"]["after"][
            "operator_verdicts"
        ][0]["choice"]
        == "unsure"
    )
    assert (
        a["after"]["predictions"][0]["human_assessment"]["marking"]["status"]
        == "correct"
    )
    assert (
        a["outcome"]["reviewed_image_outcome"]["code"]
        == "wrong_to_correct_in_reviewed_images"
    )
    assert a["outcome"]["temporal_verification"] == "incomplete"
    assert a["turn"]["end"]["uncertainty_s"] is None
    assert a["receipt_ordering_evidence"]["status"] == "receipt_ordering_supported"
    assert a["receipt_ordering_evidence"]["capture_time_verified"] is False
    assert a["receipt_ordering_evidence"]["margin_s"] == pytest.approx(6.01)


def test_version_required_no_latest_wins(current_case):
    with pytest.raises(EvidenceError, match="version 2"):
        linked(current_case, method=1)
    p = current_case[0] / "evaluation/paired_views.jsonl"
    p.write_text(p.read_text() + p.read_text().splitlines()[1] + "\n")
    with pytest.raises(EvidenceError, match="Expected one"):
        linked(current_case)
    assert (
        linked(current_case, line=2)["pair_annotation"]["linkage_import"][
            "unique_experiments"
        ]
        == 1
    )
    assert (
        len(
            linked(current_case, line=3)["pair_annotation"]["linkage_import"][
                "all_records_for_this_pair"
            ]
        )
        == 3
    )


def change_link(root, change):
    p = root / "evaluation/paired_views.jsonl"
    rows = [json.loads(s) for s in p.read_text().splitlines()]
    change(rows[1])
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_prediction_reference_is_validated_in_original_archive(current_case):
    change_link(
        current_case[0],
        lambda r: r["after"]["model_output"].update(prediction_id="unrelated"),
    )
    with pytest.raises(EvidenceError, match="does not belong"):
        linked(current_case)


def test_wrong_pair_and_target_are_not_selected(current_case):
    archive = Archive(current_case[0])
    with pytest.raises(EvidenceError, match="Expected one"):
        import_linked_pair(
            archive,
            "evaluation/paired_views.jsonl",
            "synthetic-after",
            "synthetic-before",
            method_version=2,
        )
    change_link(current_case[0], lambda r: r.update(target_operator_text="different"))
    with pytest.raises(EvidenceError, match="target wording"):
        linked(current_case)


def test_incorrect_copied_margin_flagged_without_erasing_image_outcome(current_case):
    change_link(
        current_case[0],
        lambda r: r["second_answer_uses_post_turn_view"].update(margin_s=900),
    )
    a = linked(current_case)
    assert (
        a["outcome"]["reviewed_image_outcome"]["code"]
        == "wrong_to_correct_in_reviewed_images"
    )
    assert (
        "copied_turn_margins_disagree_with_original_receipts_or_v2_union"
        in a["receipt_ordering_evidence"]["issues"]
    )


@pytest.mark.parametrize(
    "issue", ["epoch", "dropped", "sequence", "missing_chunk", "camera_epoch"]
)
def test_continuity_qualifications_preserved(current_case, issue):
    root, _ = current_case
    p = root / "lidar/index.jsonl"
    r = read(p)
    if issue == "epoch":
        r["robot_epoch"] = "changed-epoch"
    if issue == "dropped":
        r["dropped_not_recorded"] = True
    if issue == "sequence":
        r["n_odom"] = 1
    if issue == "missing_chunk":
        (root / "lidar/chunk_000001.npz").unlink()
    if issue == "camera_epoch":
        s = read(root / "summary.json")
        s["camera_bridge"]["counts"]["clock_epoch_changes"] = 1
        write(root / "summary.json", s)
    write(p, r)
    a = linked(current_case)
    assert a["receipt_ordering_evidence"]["issues"]
    assert (
        a["receipt_ordering_evidence"]["status"]
        == "incomplete_or_inconsistent_receipt_evidence"
    )
    assert (
        a["outcome"]["reviewed_image_outcome"]["code"]
        == "wrong_to_correct_in_reviewed_images"
    )


def test_closing_transport_counts_are_not_reported_as_zero(current_case):
    p = current_case[0] / "summary.json"
    s = read(p)
    s["camera_bridge"]["bridge_stats"].update(rtp_lost=38, incomplete_units=2)
    write(p, s)
    c = linked(current_case)["receipt_ordering_evidence"]["continuity"]
    assert c["periodic_transport_maxima"]["rtp_lost"] == 0
    assert c["final_counter_increase_after_periodic_status"] == {
        "rtp_lost": 38,
        "incomplete_units": 2,
    }


def test_persistent_false_absence_is_an_image_outcome_even_without_turn_bound(
    current_case,
):
    root, _ = current_case
    for role in ("before", "after"):
        edit_native(
            root,
            role,
            pointing={"opos": 0, "posxy": {}, "sentinels": {}},
            native_output="<opos_0>",
        )
        edit(pred_path(root, role), target_fraction=None, opos_not_visible=True)
        edit(review_path(root, role), choice="disagree_target_in_view")
    a = linked(current_case)
    assert (
        a["outcome"]["reviewed_image_outcome"]["code"]
        == "persistent_false_absence_in_reviewed_images"
    )
    assert a["outcome"]["temporal_verification"] == "incomplete"
    for role in ("before", "after"):
        h = a[role]["predictions"][0]["human_assessment"]
        assert all(
            h[k]["status"] == "unknown" for k in ("path", "stopping", "navigation")
        )
    edit(review_path(root), choice="unsure")
    assert (
        linked(current_case)["outcome"]["reviewed_image_outcome"]["code"]
        != "persistent_false_absence_in_reviewed_images"
    )


def test_cli_linkage_report_and_original_files_unchanged(current_case, tmp_path):
    root, _ = current_case
    before = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    p = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts/analyze_vln_viewpoint_trial.py"),
            "--run",
            str(root),
            "--before-trial",
            "synthetic-before",
            "--after-trial",
            "synthetic-after",
            "--linked-pairs",
            "evaluation/paired_views.jsonl",
            "--method-version",
            "2",
            "--output",
            str(tmp_path / "report"),
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    out = read(tmp_path / "report/summary.json")
    assert out["receipt_ordering_evidence"]["capture_time_verified"] is False
    assert (
        "Receipt-time / odometry evidence"
        in (tmp_path / "report/index.html").read_text()
    )
    after = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    assert before == after


def test_combined_table_does_not_count_relinked_pair_twice(current_case, tmp_path):
    from summarize_vln_viewpoint_reports import summarize

    a = linked(current_case)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    write(first / "summary.json", a)
    write(second / "summary.json", a)
    with pytest.raises(ValueError, match="Repeated trial pair"):
        summarize([first, second], tmp_path / "invalid_comparison")
    result = summarize([first], tmp_path / "comparison")
    assert result["unique_pairs"] == 1
    assert result["rows"][0]["capture_time_verification"] == "incomplete"
    assert result["rows"][0]["before"]["human_choices"] == ["wrong"]
    assert "wrong_to_correct" in (tmp_path / "comparison/summary.json").read_text()
