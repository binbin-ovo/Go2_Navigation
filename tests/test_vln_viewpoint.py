"""CPU-only archive analysis: no weights, GPU, receiver, server, or robot commands."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from go2_perception.vln.viewpoint import (
    Archive,
    EvidenceError,
    analyze_pair,
    inspect_trial,
)
from go2_perception.vln.viewpoint_report import write_report

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/vln_viewpoint"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def case(tmp_path):
    dest = tmp_path / "fixture"
    shutil.copytree(FIXTURE, dest)
    return dest / "run", read(dest / "pair.json")


def evaluate(case, **kwargs):
    root, pair = case
    return analyze_pair(
        Archive(root), "synthetic-before", "synthetic-after", sidecar=pair, **kwargs
    )


def pred_path(root, role="after"):
    return root / f"trials/synthetic-{role}/predictions.jsonl"


def review_path(root, role="after"):
    return root / f"trials/synthetic-{role}/human_labels.jsonl"


def request(root, role="after"):
    return root / f"session/requests/{2 if role == 'after' else 1:06d}"


def edit(path, **updates):
    value = read(path)
    value.update(updates)
    write(path, value)
    return value


def edit_source(root, role, transform):
    folder = request(root, role)
    snap, job = read(folder / "snapshot.json"), read(folder / "job.json")
    for f, h in zip(job["frames"], snap["history"]):
        transform(h["source"])
        for key in ("observation_s", "available_s"):
            f[key] = h["source"][key]
    snap["current_source"] = snap["history"][-1]["source"]
    job["decision_time_s"] = job["frames"][-1]["available_s"]
    write(folder / "snapshot.json", snap)
    write(folder / "job.json", job)


def edit_native(root, role="after", **updates):
    folder = request(root, role)
    native = edit(folder / "native/result.json", **updates)
    receipt = read(folder / "receipt.json")
    receipt["response"]["result"] = native
    write(folder / "receipt.json", receipt)
    p = read(pred_path(root, role))
    for key in ("pointing", "native_output", "waypoints"):
        if key in updates:
            p[key] = updates[key]
    write(pred_path(root, role), p)


def test_confirmed_new_view_is_two_independent_queries(case):
    a = evaluate(case)
    assert a["outcome"]["code"] == "wrong_to_correct_in_reviewed_images"
    assert a["distinct_query_revisions"] and a["same_target_and_instruction"]
    assert (
        a["before"]["trial"]["instruction_revision"]
        != a["after"]["trial"]["instruction_revision"]
    )
    p = a["after"]["predictions"][0]
    assert p["eligible"] and p["marked_input"]["turn_relation"] == "after"
    assert p["marked_input"]["source"]["frame_id"] == "after-frame-2"
    assert len([e for e in a["timeline"] if e["event"] == "actual_model_input"]) == 6


def test_late_preturn_prediction_never_becomes_postturn(case):
    root, pair = case
    # Both independent asks actually happened before the annotated turn; after-label is only a role.
    pair["turn"]["start"]["time_s"] = 125
    pair["turn"]["end"]["time_s"] = 127
    edit(pred_path(root), received_host_monotonic_s=130)
    edit(request(root) / "receipt.json", receipt_monotonic_s=130)
    edit(review_path(root), reviewed_host_monotonic_s=131)
    a = evaluate(case)
    assert a["after"]["predictions"][0]["eligible"]
    assert a["outcome"]["code"] == "no_verified_post_turn_observation"
    assert a["outcome"]["after_marked_image_relation"] == "before"
    assert a["outcome"]["view_comparison"] == "wrong_to_correct_in_reviewed_images"


def test_wrong_mark_persists(case):
    edit(review_path(case[0]), choice="wrong", choice_text="Wrong")
    assert evaluate(case)["outcome"]["code"] == "wrong_mark_in_both_reviewed_images"


def test_exact_marked_selected_image_not_unselected_current_source(case):
    root, pair = case
    edit_native(root, native_selected_frame_indices=[0, 1])
    image_path = request(root) / "native/input_0001.png"
    ref = {
        "path": "session/requests/000002/native/input_0001.png",
        "job_frame_index": 1,
        "file_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "kind": "synthetic",
    }
    edit(pred_path(root), source_image=ref)
    edit(review_path(root), source_image=ref)
    pair["turn"]["end"]["time_s"] = 118.3
    a = evaluate(case)
    p = a["after"]["predictions"][0]
    assert p["eligible"]
    assert p["current_source"]["frame_id"] == "after-frame-2"
    assert p["marked_input"]["source"]["frame_id"] == "after-frame-1"
    assert p["age_at_receipt"]["seconds"] == 1.75
    assert p["marked_input"]["turn_relation"] == "during"
    assert a["outcome"]["code"] == "no_verified_post_turn_observation"


def test_mixed_history_does_not_claim_clean_postturn_query(case):
    case[1]["turn"]["start"]["time_s"] = 118.1
    case[1]["turn"]["end"]["time_s"] = 118.2
    a = evaluate(case)
    assert a["outcome"]["after_marked_image_relation"] == "after"
    assert a["outcome"]["code"] == "insufficient_evidence"


@pytest.mark.parametrize(
    "field,value",
    [
        ("trial_id", "other"),
        ("instruction_revision", 800),
        ("instruction", "Different instruction"),
    ],
)
def test_prediction_validates_against_own_trial(case, field, value):
    edit(pred_path(case[0]), **{field: value})
    a = evaluate(case)
    assert not a["after"]["predictions"][0]["eligible"]
    assert a["outcome"]["code"] == "insufficient_evidence"


@pytest.mark.parametrize(
    "reason,label",
    [
        ("superseded_instruction", "SUPERSEDED"),
        ("cancelled_or_closed", "SUPERSEDED"),
        ("wrong_session", "REJECTED"),
    ],
)
def test_rejected_prediction_stays_in_report_but_not_pair_claim(case, reason, label):
    root, _ = case
    edit(pred_path(root), rejection_reason_at_receipt=reason, label_at_receipt=label)
    edit(request(root) / "receipt.json", rejection_reason=reason)
    a = evaluate(case)
    assert a["after"]["predictions"][0]["receipt_status"] == label
    assert not a["after"]["predictions"][0]["eligible"]
    assert a["outcome"]["code"] == "insufficient_evidence"


def test_stale_receipt_is_distinct_from_grounding_in_its_own_image(case):
    root, _ = case
    edit(
        pred_path(root),
        rejection_reason_at_receipt="stale_observation",
        label_at_receipt="STALE",
    )
    edit(request(root) / "receipt.json", rejection_reason="stale_observation")
    a = evaluate(case)
    p = a["after"]["predictions"][0]
    assert p["eligible"] and p["receipt_status"] == "STALE"
    assert "stale_at_receipt_image_grounding_remains_evaluable" in p["warnings"]
    assert a["outcome"]["code"] == "wrong_to_correct_in_reviewed_images"


def test_review_cannot_precede_its_prediction(case):
    edit(review_path(case[0]), reviewed_host_monotonic_s=119)
    a = evaluate(case)
    assert "review_precedes_prediction_arrival" in a["after"]["reviews"][0]["errors"]
    assert (
        a["after"]["predictions"][0]["human_assessment"]["marking"]["status"]
        == "unknown"
    )


def test_wrong_response_session_rejected(case):
    path = request(case[0]) / "receipt.json"
    r = read(path)
    r["response"]["identity"]["session_id"] = "another-host"
    write(path, r)
    assert (
        "response_identity_mismatch_or_missing"
        in evaluate(case)["after"]["predictions"][0]["errors"]
    )


@pytest.mark.parametrize(
    "change", ["missing", "unknown_bound", "overlap", "clock", "epoch", "no_provenance"]
)
def test_missing_or_uncertain_turn_evidence_does_not_invent_postturn(case, change):
    _, pair = case
    if change == "missing":
        pair.pop("turn")
    elif change == "no_provenance":
        pair["turn"].pop("provenance")
    elif change == "unknown_bound":
        pair["turn"]["end"]["uncertainty_s"] = None
    elif change == "overlap":
        pair["turn"]["end"]["time_s"] = 118.5
        pair["turn"]["end"]["uncertainty_s"] = 0.3
    else:
        for key in ("start", "end"):
            pair["turn"][key]["clock" if change == "clock" else "clock_epoch"] = (
                "different"
            )
    assert evaluate(case)["outcome"]["code"] == "no_verified_post_turn_observation"


def test_no_sidecar_still_compares_reviewed_images(case):
    a = analyze_pair(Archive(case[0]), "synthetic-before", "synthetic-after")
    assert a["outcome"]["view_comparison"] == "wrong_to_correct_in_reviewed_images"
    assert a["outcome"]["temporal_verification"] == "incomplete"


def test_live_estimate_without_bound_is_not_verified_capture_time(case):
    edit_source(
        case[0],
        "after",
        lambda s: s["provenance"].pop("exposure_monotonic_error_bound_s"),
    )
    a = evaluate(case)
    assert a["outcome"]["code"] == "no_verified_post_turn_observation"
    assert a["after"]["predictions"][0]["age_at_receipt"]["seconds"] == 1.5
    assert a["after"]["predictions"][0]["age_at_receipt"]["error_bound_s"] is None


def test_host_receipt_not_capture_and_unknown_age_not_zero(case):
    def change(s):
        s.pop("exposure_monotonic_s")
        s["stamp"]["semantics"] = "host_receipt"

    edit_source(case[0], "after", change)
    p = evaluate(case)["after"]["predictions"][0]
    assert p["age_at_receipt"]["seconds"] is None
    assert p["marked_input"]["turn_relation"] == "unknown"


def test_documented_mapping_required_between_robot_and_host(case):
    def change(s):
        s.pop("exposure_monotonic_s")
        s["stamp"]["clock"] = "robot_clock"
        s["provenance"]["exposure_source_error_bound_s"] = 0.002

    edit_source(case[0], "after", change)
    assert evaluate(case)["outcome"]["code"] == "no_verified_post_turn_observation"
    case[1]["clock_mappings"] = [
        {
            "source_clock": "robot_clock",
            "source_epoch": "synthetic-camera-epoch-0",
            "target_clock": "host_monotonic",
            "target_epoch": "synthetic-pair-session",
            "offset_s": 0,
            "uncertainty_s": 0.01,
            "provenance": "Synthetic explicit correspondence, not real calibration",
        }
    ]
    assert evaluate(case)["outcome"]["code"] == "wrong_to_correct_in_reviewed_images"


def test_source_stamp_sigma_is_preserved_but_not_an_ordering_bound(case):
    def change(s):
        s.pop("exposure_monotonic_s")

    for role in ("before", "after"):
        edit_source(case[0], role, change)
    for name in ("start", "end"):
        case[1]["turn"][name].update(
            clock="synthetic_camera", clock_epoch="synthetic-camera-epoch-0"
        )
    a = evaluate(case)
    t = a["after"]["predictions"][0]["marked_input"]["capture_time"]
    assert t["source_uncertainty_sigma_s"] == 0.002
    assert t["source_stamp"]["uncertainty_s"] == 0.002
    assert t["uncertainty_s"] is None
    assert a["outcome"]["code"] == "no_verified_post_turn_observation"
    for role in ("before", "after"):
        edit_source(
            case[0],
            role,
            lambda s: s["provenance"].update(exposure_source_error_bound_s=0.01),
        )
    assert evaluate(case)["outcome"]["code"] == "wrong_to_correct_in_reviewed_images"


def test_age_at_review_is_not_receipt_freshness(case):
    edit(review_path(case[0]), reviewed_host_monotonic_s=140)
    a = evaluate(case)
    assert a["after"]["predictions"][0]["receipt_status"] == "CURRENT"
    assert a["after"]["predictions"][0]["age_at_receipt"]["seconds"] == 1.5
    assert a["after"]["reviews"][0]["age_at_review"]["seconds"] == 21.5


def test_guided_target_only_never_evaluates_path_stopping_navigation(case):
    assessment = evaluate(case)["after"]["predictions"][0]["human_assessment"]
    for dimension in ("path", "stopping", "navigation"):
        assert assessment[dimension]["status"] == "unknown"
    assert assessment["marking"]["status"] == "correct"


def test_wrong_mark_does_not_imply_absence(case):
    a = evaluate(case)
    assert (
        a["before"]["predictions"][0]["human_assessment"]["visibility"]["status"]
        == "unknown"
    )


def test_visibility_change_is_not_recognition_accuracy_improvement(case):
    edit(review_path(case[0], "before"), choice="target_not_in_view")
    a = evaluate(case)
    assert (
        a["outcome"]["code"] == "visibility_change_and_correct_mark_in_reviewed_image"
    )
    assert "not a recognition-accuracy gain" in a["outcome"]["description"]


@pytest.mark.parametrize(
    "choice,visible,agreement",
    [
        ("agree_not_in_view", "absent", "agree"),
        ("disagree_target_in_view", "present", "disagree"),
        ("unsure", "unknown", "unknown"),
    ],
)
def test_opos_zero_feedback_stays_distinct(case, choice, visible, agreement):
    root, _ = case
    edit_native(
        root,
        pointing={"opos": 0, "posxy": {}, "sentinels": {}},
        native_output="<opos_0>",
    )
    edit(pred_path(root), target_fraction=None, opos_not_visible=True)
    edit(review_path(root), choice=choice)
    h = evaluate(case)["after"]["predictions"][0]["human_assessment"]
    assert h["visibility"]["status"] == visible
    assert h["opos0_agreement"]["status"] == agreement
    assert h["marking"]["status"] == "unknown"


@pytest.mark.parametrize("kind", ["missing", "unsure"])
def test_missing_or_unsure_review_unknown(case, kind):
    if kind == "missing":
        review_path(case[0]).unlink()
    else:
        edit(review_path(case[0]), choice="unsure")
    assert evaluate(case)["outcome"]["code"] == "human_review_missing_or_ambiguous"


@pytest.mark.parametrize(
    "choice,expected",
    [("wrong", "conflicting"), ("unsure", "ambiguous"), ("correct", "correct")],
)
def test_repeated_and_conflicting_reviews_never_latest_wins(case, choice, expected):
    path = review_path(case[0])
    original = path.read_text()
    r = read(path)
    r["choice"] = choice
    r["verbatim_comment"] = "Second independent synthetic review"
    path.write_text(original + json.dumps(r) + "\n")
    a = evaluate(case)
    h = a["after"]["predictions"][0]["human_assessment"]["marking"]
    assert h["status"] == expected and h["review_count"] == 2
    assert [p["line"] for p in h["provenance"]] == [1, 2]
    assert len(a["after"]["reviews"]) == 2


def test_review_mismatched_image_or_prediction_is_not_ground_truth(case):
    r = edit(review_path(case[0]), prediction_id="belongs-to-another-trial")
    a = evaluate(case)
    assert a["after"]["reviews"][0]["errors"]
    assert (
        a["after"]["predictions"][0]["human_assessment"]["marking"]["status"]
        == "unknown"
    )
    r["prediction_id"] = "synthetic-after-P002"
    r["source_image"]["file_sha256"] = "bad"
    write(review_path(case[0]), r)
    assert (
        "review_image_reference_mismatch_or_missing"
        in evaluate(case)["after"]["reviews"][0]["errors"]
    )


def legacy_review(root):
    p = read(pred_path(root))
    return {
        "schema": "go2-vln-human-review-v1",
        "trial_id": "synthetic-after",
        "review_id": "synthetic-old",
        "applies_to_predictions": [p["prediction_id"]],
        "marking": "correct",
        "target_presence": "present",
        "path": "plausible",
        "stopping": "uncertain",
        "model_feedback": False,
        "confirmed_by_user": True,
        "verbatim_comment": "Legacy single-prediction judgment",
        "reviewed_host_monotonic_s": 122,
    }


def test_older_review_preserves_original_singleton_scope(case):
    write(review_path(case[0]), legacy_review(case[0]))
    a = evaluate(case)
    assert a["after"]["reviews"][0]["scope"] == "legacy_prediction_set"
    assert (
        a["after"]["predictions"][0]["human_assessment"]["path"]["status"]
        == "plausible"
    )
    assert (
        a["after"]["predictions"][0]["human_assessment"]["navigation"]["status"]
        == "unknown"
    )


def test_group_review_is_not_duplicated_to_each_prediction(case):
    root, _ = case
    p = read(pred_path(root))
    original = pred_path(root).read_text()
    p["prediction_id"] = "synthetic-extra"
    pred_path(root).write_text(original + json.dumps(p) + "\n")
    r = legacy_review(FIXTURE / "run")
    r["applies_to_predictions"].append("synthetic-extra")
    write(review_path(root), r)
    a = evaluate(case)
    assert a["after"]["selected_prediction_id"] is None
    assert all(
        p["human_assessment"]["marking"]["status"] == "unknown"
        for p in a["after"]["predictions"]
    )
    assert a["after"]["reviews"][0]["scope"] == "legacy_prediction_set"


@pytest.mark.parametrize(
    "artifact",
    ["native/input_0002.png", "frame_0002.png", "snapshot.json", "receipt.json"],
)
def test_missing_or_deleted_artifacts_are_not_rebuilt(case, artifact):
    (request(case[0]) / artifact).unlink()
    a = evaluate(case)
    assert a["outcome"]["code"] == "insufficient_evidence"
    assert not (request(case[0]) / artifact).exists()


def test_deleted_prediction_file_is_explicitly_missing(case):
    pred_path(case[0]).unlink()
    a = evaluate(case)
    assert a["after"]["predictions"] == []
    assert a["outcome"]["code"] == "insufficient_evidence"
    assert a["after"]["reviews"][0]["errors"]


@pytest.mark.parametrize(
    "artifact", ["summary.json", "trials/synthetic-after/trial_end.json"]
)
def test_active_or_unclosed_archive_refused(case, artifact):
    (case[0] / artifact).unlink()
    with pytest.raises(EvidenceError, match="Missing required"):
        evaluate(case)


def test_mutated_archive_detected(case):
    a = Archive(case[0])
    inspect_trial(a, "synthetic-before")
    edit(review_path(case[0], "before"), choice="correct")
    with pytest.raises(EvidenceError, match="changed"):
        a.assert_unchanged()


def test_wrong_sidecar_identity_and_same_trial_refused(case):
    case[1]["after_trial_id"] = "other"
    with pytest.raises(EvidenceError, match="Sidecar trial"):
        evaluate(case)
    with pytest.raises(EvidenceError, match="different"):
        analyze_pair(Archive(case[0]), "synthetic-before", "synthetic-before")


def test_different_target_not_scored_as_paired_effect(case):
    case[1]["target_wording"] = "table"
    assert evaluate(case)["outcome"]["code"] == "insufficient_evidence"


def test_future_frame_and_history_leak_rejected(case):
    path = request(case[0]) / "job.json"
    j = read(path)
    j["decision_time_s"] = 117
    write(path, j)
    assert not evaluate(case)["after"]["predictions"][0]["eligible"]


def test_event_referenced_turn_preserves_line_provenance(case):
    root, pair = case
    events = [
        {"event": "operator_turn_start", "host_monotonic_s": 110},
        {"event": "operator_stopped", "host_monotonic_s": 112},
    ]
    (root / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    for i, key in enumerate(("start", "end")):
        t = pair["turn"][key]
        t.pop("time_s")
        t["event_ref"] = {
            "path": "events.jsonl",
            "line": i + 1,
            "event": events[i]["event"],
        }
    a = evaluate(case)
    assert a["outcome"]["code"] == "wrong_to_correct_in_reviewed_images"
    assert a["turn"]["end"]["resolved_event"]["provenance"]["line"] == 2


def test_path_escape_refused(case):
    edit(pred_path(case[0]), request_directory="../../external")
    with pytest.raises(EvidenceError, match="relative"):
        evaluate(case)


def test_duplicate_prediction_identity_refused(case):
    p = pred_path(case[0])
    p.write_text(p.read_text() * 2)
    with pytest.raises(EvidenceError, match="duplicated"):
        evaluate(case)


def test_corrupt_native_image_hash_rejected(case):
    root, _ = case
    (request(root) / "native/input_0002.png").write_bytes(
        (request(root) / "native/input_0000.png").read_bytes()
    )
    a = evaluate(case)
    assert (
        a["after"]["predictions"][0]["marked_input"]["image"]["status"]
        == "hash_mismatch"
    )
    assert a["outcome"]["code"] == "insufficient_evidence"


def test_explicit_prediction_id_must_belong_to_own_trial(case):
    with pytest.raises(EvidenceError, match="does not belong"):
        evaluate(case, after_prediction="synthetic-before-P001")


def test_new_file_during_analysis_detected(case):
    root, _ = case
    a = Archive(root)
    inspect_trial(a, "synthetic-before")
    write(root / "trials/synthetic-before/notes.jsonl", {"note": "concurrent write"})
    with pytest.raises(EvidenceError, match="changed"):
        a.assert_unchanged()


def test_nonfinite_json_rejected(case):
    p = pred_path(case[0])
    p.write_text(
        p.read_text().replace(
            '"received_host_monotonic_s": 120.0', '"received_host_monotonic_s": NaN'
        )
    )
    with pytest.raises(EvidenceError, match="Invalid JSONL"):
        evaluate(case)


def test_invalid_sidecar_is_a_clear_contract_error(case):
    with pytest.raises(EvidenceError, match="JSON object"):
        analyze_pair(
            Archive(case[0]), "synthetic-before", "synthetic-after", sidecar=[]
        )
    case[1]["turn"]["end"]["uncertainty_s"] = "unknown"
    with pytest.raises(EvidenceError, match="finite seconds"):
        evaluate(case)


def test_realistic_absolute_job_paths_relocate_to_run_only(case):
    path = request(case[0]) / "job.json"
    j = read(path)
    for frame in j["frames"]:
        frame["path"] = "/unavailable/original/archive/" + frame["path"]
    write(path, j)
    assert evaluate(case)["after"]["predictions"][0]["eligible"]


def test_json_html_report_preserves_sources_and_escapes_comments(case, tmp_path):
    root, pair = case
    edit(review_path(root), verbatim_comment='<script>alert("x")</script>')
    before = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    archive = Archive(root)
    a = analyze_pair(archive, "synthetic-before", "synthetic-after", sidecar=pair)
    page = write_report(archive, a, tmp_path / "report")
    html = page.read_text()
    assert "&lt;script&gt;" in html and "<script>alert" not in html
    assert "UNEXECUTED" in html and "frame-2" in html
    assert read(page.parent / "summary.json")["outcome"] == a["outcome"]
    assert read(page.parent / "timeline.json") == a["timeline"]
    after = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    assert before == after
    with pytest.raises(FileExistsError):
        write_report(archive, a, page.parent)
    with pytest.raises(EvidenceError, match="separate"):
        write_report(archive, a, root / "report")


def test_cli_uses_only_lightweight_local_analysis(tmp_path):
    cmd = [
        sys.executable,
        str(ROOT / "scripts/analyze_vln_viewpoint_trial.py"),
        "--run",
        str(FIXTURE / "run"),
        "--before-trial",
        "synthetic-before",
        "--after-trial",
        "synthetic-after",
        "--annotation",
        str(FIXTURE / "pair.json"),
        "--output",
        str(tmp_path / "report"),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert (tmp_path / "report/index.html").exists()
    # Import boundary: no Torch/Transformers, robot SDK, receiver, or live app gets loaded.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import go2_perception.vln.viewpoint_report; "
            "assert not any(k.startswith(('torch', 'transformers', 'unitree', 'cyclonedds', "
            "'go2_perception.vln.interactive_app', 'go2_perception.vln.shadow_live')) for k in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
