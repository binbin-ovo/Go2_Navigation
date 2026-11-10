import json
from copy import deepcopy
from pathlib import Path
import runpy
import pytest
from go2_perception.scene.serialization import read_scenes, validate_scene, write_scenes


def test_fixture_is_deterministic_and_standalone(tmp_path):
    expected = list(read_scenes("tests/fixtures/scene/renderer.jsonl"))
    generated = list(runpy.run_path("scripts/generate_scene_fixture.py")["fixture"]())
    assert expected == generated and len(expected) == 6
    write_scenes(tmp_path / "fixture.jsonl", generated)
    assert list(read_scenes(tmp_path / "fixture.jsonl")) == expected
    assert expected[-1]["objects"][0]["status"] == "stale"
    assert expected[-1]["objects"][0]["observed_box"] is None
    with pytest.raises(FileExistsError):
        write_scenes(tmp_path / "fixture.jsonl", generated)


@pytest.mark.parametrize(
    "change", ["nan", "covariance", "pose", "version", "duplicate", "age"]
)
def test_scene_rejects_bad_contract(change):
    r = deepcopy(
        json.loads(
            Path("tests/fixtures/scene/renderer.jsonl").read_text().splitlines()[0]
        )
    )
    if change == "nan":
        r["timestamp"] = float("nan")
    elif change == "covariance":
        r["objects"][0]["velocity_covariance"][0][0] = -1
    elif change == "pose":
        r["robot_pose"]["T_odom_base"][0][0] = 2
    elif change == "version":
        r["schema"] = "unknown"
    elif change == "duplicate":
        r["objects"].append(r["objects"][0])
    elif change == "age":
        r["objects"][0]["observation_age_s"] = 5
    with pytest.raises(ValueError):
        validate_scene(r)


def test_machine_schema_and_minimal_consumer():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(Path("schemas/go2-scene-v1.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    for row in read_scenes("tests/fixtures/scene/renderer.jsonl"):
        jsonschema.Draft202012Validator(schema).validate(row)
    consumer = runpy.run_path("scripts/read_scene.py")["summarize"]
    rows = list(consumer("tests/fixtures/scene/renderer.jsonl"))
    assert rows[0]["observed_boxes"] == 2 and rows[-1]["stale_boxes"] == 1
    assert all(r["frame"] == "odom" and r["clock"] == "synthetic" for r in rows)
