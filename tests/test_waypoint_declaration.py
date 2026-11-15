from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from go2_perception.vln.waypoint_declaration import digest, validate_spec

ROOT = Path(__file__).resolve().parents[1]


def draft():
    return yaml.safe_load((ROOT / "configs/waypoint_validation.yaml").read_text())


def test_draft_does_not_authorize_model_runs():
    validate_spec(draft(), ROOT, require_review=False)
    with pytest.raises(ValueError, match="Claude visual review is pending"):
        validate_spec(draft(), ROOT)


def test_review_is_bound_to_scene_report_and_processor_preview(tmp_path):
    spec = draft()
    report, preview = tmp_path / "report.md", tmp_path / "preview.json"
    report.write_text("Synthetic test report; not a real human review.")
    preview.write_text("{}")
    spec["status"] = "reviewed_ready"
    spec["review"] = {
        "report_path": str(report),
        "report_sha256": digest(report),
        "commit": "a" * 40,
        "reviewed_scene_sha256": spec["scene_sha256"],
        "scene_suitable": True,
        "preview_manifest_path": str(preview),
        "preview_manifest_sha256": digest(preview),
    }
    validate_spec(spec, ROOT)
    report.write_text("Changed report")
    with pytest.raises(ValueError, match="Review evidence changed"):
        validate_spec(spec, ROOT)


@pytest.mark.parametrize(
    "change",
    [
        {"physical_execution_authorized": True},
        {"model_revision": "changed"},
        {"code_revision": "changed"},
        {"extra_ignored_setting": 42},
        {"scene_sha256": "changed"},
        {"history": "previous_executed_actions"},
        {"policies": ["oracle"]},
        {"mode": "real_robot"},
    ],
)
def test_reject_unsupported_settings(change):
    spec = deepcopy(draft())
    spec.update(change)
    with pytest.raises(ValueError):
        validate_spec(spec, ROOT, require_review=False)
