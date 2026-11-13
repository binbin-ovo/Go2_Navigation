"""Synthetic saved-output evaluation; no private labels or pretrained inference."""

import json
from pathlib import Path
import runpy
import shutil

import pytest

from go2_perception.vln.episodes import file_sha256
from go2_perception.vln.temporal_proposals import parse_proposal

ROOT = Path(__file__).resolve().parents[1]
evaluate = runpy.run_path(str(ROOT / "scripts/evaluate_temporal_vln.py"))["evaluate"]


def write(path, data):
    path.write_text(json.dumps(data))


@pytest.fixture
def saved_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    declaration = {"episodes": [], "modes": ["temporal"], "decision_filter": "d0"}
    for episode_id in ["navigation", "static"]:
        package = tmp_path / episode_id / "model_input"
        shutil.copytree(ROOT / "tests/fixtures/vln/episode_v1", package)
        manifest = package / "episode.json"
        ep = json.loads(manifest.read_text())
        ep["episode_id"] = episode_id
        write(manifest, ep)
        digest = file_sha256(manifest)
        anns = json.loads((package / "annotations.json").read_text())
        anns.update(episode_id=episode_id, manifest_sha256=digest)
        anns["reviewer"].update(type="human", status="reviewed")
        anns["decisions"][0].update(
            acceptable_actions=["STOP", "TURN_RIGHT"], ambiguity=True
        )
        evaluation = package.parent / "evaluation_only"
        evaluation.mkdir()
        write(evaluation / "annotations.json", anns)
        declaration["episodes"].append(
            {"episode_id": episode_id, "manifest": str(manifest), "sha256": digest}
        )
        out = run / episode_id / "d0" / "temporal"
        out.mkdir(parents=True)
        p = parse_proposal(
            '{"proposal":"TURN_RIGHT","task_status":"IN_PROGRESS","reason":"test only"}'
        )
        p.update(
            episode_id=episode_id,
            decision_id="d0",
            mode="temporal",
            selected_observations={"decision_time_s": ep["decisions"][0]["time_s"]},
            timing={"generation_s": 1},
            memory={"process_lifetime_peak_rss_bytes": 10},
        )
        write(out / "proposal.json", p)
        write(out / "input_provenance.json", {"episode_manifest_sha256": digest})
    write(run / "run_declaration.json", declaration)
    return run


def test_correlated_episodes_and_static_case_are_reported_separately(saved_run):
    result = evaluate(
        saved_run,
        annotations_name="../evaluation_only/annotations.json",
        static_episodes=["static"],
    )
    assert len(result["by_recording"]) == 1
    group = next(iter(result["by_recording"].values()))
    assert group["correlated_examples"]
    assert group["by_mode"]["temporal"]["human_action_agreement"] == 1
    assert (
        group["by_case_kind"]["navigation_proposal"]["temporal"]["inference_calls"] == 1
    )
    assert (
        group["by_case_kind"]["static_absent_target"]["temporal"]["inference_calls"]
        == 1
    )
    assert result["human_reviewed_action_decisions"] == 2
    assert result["pending_action_reference_decisions"] == 0
    assert result["closed_loop_navigation_success"] is None
    # A filtered ambiguity stratum is not an uninterrupted temporal sequence.
    assert (
        "repeated_consecutive_actions"
        not in group["by_reference_ambiguity"]["true"]["temporal"]
    )


@pytest.mark.parametrize("kind", ["missing", "pending", "agent"])
def test_unreviewed_or_absent_reference_does_not_enter_human_score(saved_run, kind):
    path = saved_run.parent / "static/evaluation_only/annotations.json"
    if kind == "missing":
        path.unlink()
    else:
        a = json.loads(path.read_text())
        a["reviewer"]["status" if kind == "pending" else "type"] = kind
        write(path, a)
    result = evaluate(
        saved_run,
        annotations_name="../evaluation_only/annotations.json",
        static_episodes=["static"],
    )
    assert result["human_reviewed_action_decisions"] == 1
    assert result["pending_action_reference_decisions"] == 1
    group = next(iter(result["by_recording"].values()))
    assert group["by_mode"]["temporal"]["human_action_comparisons"] == 1
    row = next(x for x in result["decisions"] if x["episode_id"] == "static")
    assert row["action_agrees"] is None and row["action_reference_status"] == "pending"


def test_input_provenance_mismatch_cannot_be_scored(saved_run):
    p = saved_run / "navigation/d0/temporal/input_provenance.json"
    write(p, {"episode_manifest_sha256": "0" * 64})
    with pytest.raises(ValueError, match="another manifest"):
        evaluate(saved_run, annotations_name="../evaluation_only/annotations.json")


def test_incomplete_inference_remains_pending_without_false_denominator(saved_run):
    (saved_run / "static/d0/temporal/proposal.json").unlink()
    result = evaluate(saved_run, annotations_name="../evaluation_only/annotations.json")
    assert len(result["pending_inference_calls"]) == 1
    group = next(iter(result["by_recording"].values()))
    assert group["by_mode"]["temporal"]["human_action_comparisons"] == 1
    assert result["human_reviewed_action_decisions"] == 2
