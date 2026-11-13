"""Offline Phase 6B contracts. Fixture responses are not model accuracy evidence."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from go2_perception.vln.episodes import load_episode, select_history
from go2_perception.vln.temporal_inputs import (
    build_messages,
    prepare_views,
    pinhole_fov,
)
from go2_perception.vln.temporal_proposals import parse_proposal
from go2_perception.vln.temporal_review import make_video, render_review, summarize

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/vln/episode_v1/episode.json"


def test_temporal_frames_each_carry_calibration_and_times():
    ep = load_episode(FIXTURE)
    selection, views = prepare_views(ep, "d1", allow_synthetic=True)
    assert len(views) == 4
    for f, v in zip(selection["frames"], views):
        assert v.image.dtype == np.uint8 and v.metadata["view"]["rectified"]
        assert not v.intrinsics.D.any()
        assert v.metadata["source"]["stamp_robot"]["t"] == f["observation_s"]
        assert v.metadata["source"]["clock_epoch"] == ep.data["clock"]["epoch"]
        assert 0 < pinhole_fov(v.intrinsics)["horizontal_deg"] < 180
    with pytest.raises(ValueError, match="Measured"):
        prepare_views(ep, "d0")


def test_message_construction_preserves_full_instruction_and_excludes_annotations():
    ep = load_episode(FIXTURE)
    s = select_history(ep, "d1")
    s["acceptable_actions"] = ["SECRET_REFERENCE"]
    s["frames"][0]["target_visibility"] = "SECRET_VISIBILITY"
    instruction = (
        "Leave the room, turn left into the hallway, and stop beside the doorway."
    )
    messages = build_messages(instruction, s, "fixed prompt")
    encoded = json.dumps(messages)
    assert instruction in encoded
    assert "SECRET_REFERENCE" not in encoded and "SECRET_VISIBILITY" not in encoded
    content = messages[0]["content"]
    assert sum(x["type"] == "image" for x in content) == 4
    texts = [x["text"] for x in content if x["type"] == "text"]
    assert texts[1].startswith("history") and any(
        t.startswith("current") for t in texts
    )
    assert "Observation gaps" in encoded


def test_selected_missing_file_and_dimension_change_rejected(tmp_path):
    shutil.copytree(FIXTURE.parent, tmp_path / "ep")
    ep = load_episode(tmp_path / "ep/episode.json")
    (ep.root / "images/005.png").write_bytes(b"changed after validation")
    with pytest.raises(ValueError, match="changed"):
        prepare_views(ep, "d1", allow_synthetic=True)


def test_official_smol_image_processor_uses_half_range_not_imagenet():
    mod = pytest.importorskip("transformers.models.smolvlm.image_processing_smolvlm")
    processor = mod.SmolVLMImageProcessor(
        do_image_splitting=False,
        size={"longest_edge": 384},
        max_image_size={"longest_edge": 384},
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )
    image = np.empty((120, 160, 3), np.uint8)
    image[:] = [255, 0, 128]
    out = processor(images=[[image, image]], return_tensors="np")
    assert out.pixel_values.shape == (1, 2, 3, 384, 384)
    np.testing.assert_allclose(
        out.pixel_values[0, 0, :, 100, 100], [1, -1, 128 / 255 * 2 - 1], atol=1e-6
    )
    assert out.pixel_attention_mask.all()


@pytest.mark.parametrize(
    "action", ["ADVANCE", "TURN_LEFT", "TURN_RIGHT", "STOP", "UNRESOLVED"]
)
def test_parse_categories_never_authorize_execution(action):
    p = parse_proposal(
        json.dumps(
            {"proposal": action, "task_status": "IN_PROGRESS", "reason": "fixture only"}
        )
    )
    assert p["parse_status"] == "accepted" and p["proposal"] == action
    assert p["execution_authorized"] is False and p["metric_goal"] is None
    assert p["native_action_reference_frame"] is None
    assert p["native_distance"] is None and p["native_angle"] is None


def test_stop_does_not_establish_completion():
    p = parse_proposal('{"proposal":"STOP","reason":"wait"}')
    assert (
        p["task_status"] is None
        and not p["model_claims_complete"]
        and p["parse_status"] == "partial"
    )
    p = parse_proposal(
        '{"proposal":"STOP","task_status":"COMPLETE","reason":"I am there"}'
    )
    assert p["model_claims_complete"] and not p["completion_verified"]


def test_explicit_native_fields_preserve_unknown_task_status():
    p = parse_proposal("Proposal: TURN_RIGHT\nReason: The target is on the right.")
    assert p["proposal"] == "TURN_RIGHT" and p["parse_status"] == "partial"
    assert p["task_status"] is None and not p["completion_verified"]
    assert (
        parse_proposal("Proposal: STOP\nReason: wait\nProposal: ADVANCE")[
            "parse_status"
        ]
        == "rejected"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "Move forward.",
        "STOP",
        '[{"proposal":"STOP"}]',
        '{"proposal":"STOP","proposal":"ADVANCE","reason":"duplicate"}',
        '{"proposal":"FLY","reason":"unknown"}',
        '{"proposal":"STOP","reason":"x","execution_authorized":true}',
        '{"proposal":"STOP","task_status":"DONE","reason":"bad enum"}',
        '{"proposal":"STOP","reason":"x","distance":{"value":NaN,"unit":"m"}}',
        '{"proposal":"TURN_LEFT","reason":"x","angle":{"value":30,"unit":"feet"}}',
        'before {"proposal":"STOP","reason":"x"} after',
        '{"proposal":"ADVANCE","reason":"x"',
    ],
)
def test_unstructured_malformed_and_unsupported_outputs_not_forced_into_actions(raw):
    p = parse_proposal(raw)
    assert p["parse_status"] == "rejected" and p["proposal"] == "UNRESOLVED"
    assert p["task_status"] is None and not p["execution_authorized"]


def test_explicit_native_quantities_retain_units_but_no_frame():
    p = parse_proposal(
        '```json\n{"proposal":"ADVANCE","task_status":"IN_PROGRESS","reason":"x","distance":{"value":75,"unit":"cm"}}\n```'
    )
    assert p["native_distance"] == {
        "value": 75,
        "unit": "cm",
        "reference_frame": None,
        "measured": False,
    }
    assert p["metric_goal"] is None


def proposal_fixture(
    episode_id="synthetic_door", decision_id="d1", action="STOP", status="COMPLETE"
):
    p = parse_proposal(
        json.dumps(
            {
                "proposal": action,
                "task_status": status,
                "reason": "synthetic rendering/evaluation fixture",
            }
        )
    )
    p.update(
        episode_id=episode_id,
        decision_id=decision_id,
        instruction="Synthetic test instruction",
        mode="temporal",
        native_model_output="Synthetic test text, not actual model inference",
        history_images_consumed=3,
        selected_observations={"decision_time_s": 105.2},
        timing={"generation_s": 1.0, "replay_observation_age_at_output_s": 1.2},
        memory={"process_lifetime_peak_rss_bytes": 1234},
    )
    return p


def test_only_human_reviewed_references_count_and_stop_not_completion():
    p = proposal_fixture()
    ref = {
        "reviewer": {"name": "fixture", "type": "agent", "status": "reviewed"},
        "decision": {
            "acceptable_actions": ["UNRESOLVED"],
            "completion_status": "incomplete",
            "instruction_progress": "not there",
        },
    }
    refs = {(p["episode_id"], p["decision_id"]): ref}
    a = summarize([p], refs)
    assert (
        a["human_action_agreement"] is None
        and a["false_completion_declarations"] is None
    )
    ref["reviewer"]["type"] = "human"
    ref["reviewer"]["status"] = "pending"
    assert summarize([p], refs)["human_action_comparisons"] == 0
    ref["reviewer"]["status"] = "reviewed"
    a = summarize([p], refs)
    assert a["human_action_agreement"] == 0 and a["false_completion_declarations"] == 1
    assert a["closed_loop_navigation_success"] is None


def test_sequence_flags_do_not_claim_unsafe_or_successful_navigation():
    a = proposal_fixture(action="TURN_LEFT", status="IN_PROGRESS")
    b = deepcopy(a)
    b["decision_id"] = "d2"
    b["proposal"] = "TURN_RIGHT"
    b["selected_observations"]["decision_time_s"] = 106.2
    result = summarize([a, b])
    assert result["opposite_consecutive_turn_pairs"] == 1
    assert result["collision_avoidance_success"] is None


def test_missing_task_status_is_not_scored_as_verified_incompletion():
    p = proposal_fixture(action="STOP", status=None)
    refs = {
        (p["episode_id"], p["decision_id"]): {
            "reviewer": {"name": "fixture", "type": "human", "status": "reviewed"},
            "decision": {
                "acceptable_actions": ["STOP"],
                "completion_status": "incomplete",
                "instruction_progress": None,
            },
        }
    }
    result = summarize([p], refs)
    assert result["human_action_agreement"] == 1
    assert result["human_completion_comparisons"] == 0
    assert result["human_completion_consistency"] is None


def test_review_panel_and_video(tmp_path):
    pytest.importorskip("matplotlib")
    ep = load_episode(FIXTURE)
    selection, views = prepare_views(ep, "d1", allow_synthetic=True)
    prepared = SimpleNamespace(
        selection=selection,
        views=views,
        previews=[cv2.resize(v.image, (384, 384)) for v in views],
    )
    png = tmp_path / "review.png"
    p = proposal_fixture()
    render_review(p, prepared, png)
    assert cv2.imread(str(png)).shape[:2] == (990, 1650)
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg absent; panel verified, video unavailable")
    video = tmp_path / "review.mp4"
    make_video([png, png], video)
    capture = cv2.VideoCapture(str(video))
    ok, image = capture.read()
    capture.release()
    assert ok and image.shape[:2] == (990, 1650)
