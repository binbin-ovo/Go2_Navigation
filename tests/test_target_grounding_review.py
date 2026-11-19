import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from go2_perception.vln.bounded_context import select_context
from go2_perception.vln.episode_app import EpisodeUI, target_spec
from go2_perception.vln.episode_grounding import reference_matches_target, valid_roi
from go2_perception.vln.episode_page import episode_page
from go2_perception.vln.grounding_review import episode_decisions, saved_input
from go2_perception.vln.interactive_trials import RunStore, load_review
from go2_perception.vln.prediction_archive import file_hash
from go2_perception.vln.shadow_display import point_fraction
from go2_perception.vln.supervised import Hold, MODEL_REVISION


@pytest.mark.parametrize(
    "description",
    [
        "box",
        "brown cardboard box",
        "brown cardboard box on the floor",
        "large brown box next to the refrigerator",
    ],
)
def test_typed_and_normalized_description(description):
    entered = " Go to the " + description + " and STOP! "
    result = target_spec(entered)
    assert result["entered_target_text"] == entered
    assert result["instruction"] == "Go to the " + description + " and stop."
    assert result["operator_text"] == description


@pytest.mark.parametrize("value", [None, " ", "Go to the"])
def test_empty_request(value):
    with pytest.raises(ValueError):
        target_spec(value)


def test_reference_binding_keeps_all_words():
    assert reference_matches_target("Brown  cardboard BOX", "brown cardboard box")
    assert not reference_matches_target("box", "large brown box")
    ui = SimpleNamespace(
        grant=None,
        abort=SimpleNamespace(is_set=lambda: False),
        episode={"state": "IDLE"},
        settings={"reference_path": "photo", "reference_target": "box"},
    )
    with pytest.raises(Hold, match="reference_names_another_target.*large brown box"):
        EpisodeUI.authorize(
            ui,
            dict(
                target="large brown box",
                operator="pilot",
                area_inspected=True,
                exclusive_and_intervention_confirmed=True,
            ),
            0,
        )


@pytest.mark.parametrize("roi", [None, [], [0, True, 0.8, 0.9], [0, 0, float("nan"), 1], [0, 0, 2, 2]])
def test_bad_review_rectangle_has_actionable_message(roi):
    assert not valid_roi(roi)
    ui = SimpleNamespace(
        grant={"operator": "pilot"},
        episode={"state": "IDENTIFYING", "last_observation": {"prediction_id": "p", "image_sha256": "h"}},
        app=SimpleNamespace(predictions={"p": {"target_fraction": [0.2, 0.4]}}),
    )
    with pytest.raises(Hold, match="Draw a rectangle"):
        EpisodeUI.review_image(
            ui, dict(action="reject", operator="pilot", prediction_id="p", image_sha256="h", roi=roi)
        )


def put(path, data, line=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + ("\n" if line else ""))


@pytest.fixture
def closed_input(tmp_path):
    root = tmp_path
    tr, pid = "T-fixture", "T-fixture-P001"
    req = root / "session/requests/000001"
    prepared, image = req / "frame_0000.png", req / "native/input_0000.png"
    image.parent.mkdir(parents=True)
    for path in [prepared, image]:
        path.write_bytes(b"test image bytes")
    frame = dict(path=str(prepared), sha256=file_hash(prepared), observation_s=1.0, available_s=1.0)
    context = select_context([frame], 1.0)
    source = dict(frame_id="f", clock_epoch="epoch", observation_s=1.0, available_s=1.0)
    common = dict(decision_id="d", episode_id=tr, clock_epoch="epoch")
    job = dict(**common, instruction="Go to the box and stop.", decision_time_s=1.0, frames=[frame])
    trial = dict(trial_id=tr, instruction=job["instruction"], instruction_revision=1, session_id="s")
    native = dict(
        **common,
        status="decoded",
        native_selected_frame_indices=[0],
        context_policy=context,
        native_output="<opos_0>",
        pointing={"opos": 0},
        explicit_stop_code=False,
        decoder_stop=False,
        waypoints=[[0, 0, 0]] * 10,
        waypoint_convention="absolute_ego_SE2_x_forward_y_left_yaw_CCW_metres_radians",
    )
    pred = {
        k: native[k]
        for k in ("waypoints", "waypoint_convention", "native_output", "pointing", "explicit_stop_code")
    }
    pred.update(
        trial,
        prediction_id=pid,
        request_directory=str(req.relative_to(root)),
        source_image=dict(
            path=str(image.relative_to(root)),
            file_sha256=file_hash(image),
            causal_grid_index=0,
            job_frame_index=0,
        ),
        source_frame=source,
        mock=False,
        native_stop=False,
        target_fraction=None,
    )
    put(req / "job.json", job)
    put(req / "native/result.json", native)
    put(
        req / "snapshot.json",
        dict(
            current_source=source,
            identity={"session_id": "s", "instruction_revision": 1},
            bounded_context=context,
            history=[{"source": source}],
            source_cutoff_s=1.0,
            annotations_consumed=False,
            previous_actions_supplied=False,
        ),
    )
    put(root / "trials" / tr / "trial.json", trial)
    put(root / "trials" / tr / "predictions.jsonl", pred, True)
    put(root / "session/worker/model.json", {"checkpoint": MODEL_REVISION})
    put(root / "summary.json", {"ended_wall_utc": "closed"})
    return root, pid, req


def test_offline_frame_binding_and_tamper(closed_input):
    root, pid, req = closed_input
    source = saved_input(root, pid)
    assert source["frames"][0]["frame_id"] == "f"
    (req / "frame_0000.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="prepared frame"):
        saved_input(root, pid)


def test_offline_refuses_changed_instruction_and_future_history(closed_input):
    root, pid, req = closed_input
    path = req / "snapshot.json"
    snap = json.loads(path.read_text())
    snap["identity"]["instruction_revision"] += 1
    put(path, snap)
    with pytest.raises(ValueError, match="instruction binding"):
        saved_input(root, pid)
    snap["identity"]["instruction_revision"] = 1
    put(path, snap)
    job = json.loads((req / "job.json").read_text())
    job["frames"][0]["available_s"] = 2.0
    put(req / "job.json", job)
    with pytest.raises(ValueError, match="causal availability"):
        saved_input(root, pid)


def test_null_grid_and_review_presentation():
    assert point_fraction(0) is None
    assert point_fraction(698) == pytest.approx((238 / 448, 14.5 / 27))
    page = episode_page()
    assert "Raw model mark" in page and "Matched the identified reference" in page
    assert "s.mode" in page and "s.source" in page
    assert "S.prediction?.prediction_id!==snap.prediction_id" in page


def test_closed_private_trial_separates_rejected_mark_from_dispatch():
    workspace = Path(os.environ.get("GO2_WORKSPACE", Path.home() / "Robotics/Go2"))
    s = workspace / "runs/five_move/trial-THKUcr/session"
    if not (s / "summary.json").is_file():
        pytest.skip("Closed private November 19 box episode unavailable")
    trial = "T20261119-120816-01-3a48"
    summary, rows = episode_decisions(s)
    data = saved_input(s, trial + "-P002")
    assert data["frames"][-1]["sha256"] == "0226c295a4b6f919dc42f417cef2f1dccaead9f31eca98e4eddb771eba999a10"
    assert rows[trial + "-P001"]["rejection"] == "stale_or_future_observation"
    assert rows[trial + "-P002"]["reviews"][0]["decision"] == "reject"
    assert rows[trial + "-P002"]["selection"] is None
    assert rows[trial + "-P003"]["selection"]["mode"] == "SEARCH"
    assert len(rows[trial + "-P003"]["result"]["command_send_statuses"]) == 11
    assert summary["target_found"] is False
    review = load_review(RunStore(s, create=False), trial)
    assert review["predictions"][3]["episode_review"]["reviews"][0]["decision"] == "reject"


@pytest.mark.parametrize("change", ["action", "review_image"])
def test_offline_trace_refuses_mismatched_dispatch_or_review(tmp_path, change):
    from go2_perception.vln.supervised import digest

    proposal = {"identity": {"prediction_id": "p", "source_image_sha256": "image"}}
    evidence = {"prediction_id": "p", "image_sha256": "image", "target": "box"}
    selection = {"prediction_id": "p", "mode": "SEARCH", "action": {"direction": "forward"}}
    binding = {"proposal": digest(proposal), "grounding": digest(evidence), "selection": digest(selection)}
    ticket = {"request_id": digest(binding), "proposal": proposal, "action": selection["action"]}
    events = [
        {"type": "episode_observation", "nonce": "e", "proposal": proposal, "evidence": evidence},
        {
            "type": "episode_dispatch",
            "nonce": "e",
            "ticket": ticket,
            "selection": selection,
            "binding": binding,
        },
    ]
    review = dict(
        event="episode_instance_review",
        prediction_id="p",
        image_sha256="image",
        target="box",
        decision="reject",
    )
    if change == "action":
        ticket["action"] = {"direction": "right"}
    else:
        review["image_sha256"] = "different-frame"
    put(tmp_path / "episode_summary.json", {"episode_id": "e"})
    log = tmp_path / "executor/executor_events.jsonl"
    log.parent.mkdir()
    log.write_text("".join(json.dumps(ev) + "\n" for ev in events))
    put(tmp_path / "events.jsonl", review, True)
    with pytest.raises(ValueError, match="dispatch identity|review bound"):
        episode_decisions(tmp_path)
