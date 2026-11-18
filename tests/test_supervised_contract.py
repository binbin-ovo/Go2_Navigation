"""The public example is synthetic evidence, never a hardware capability."""
import copy
import json
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "src/go2_perception/vln/schemas/supervised-navigation-v1.schema.json").read_text())
FIXTURE = json.loads((ROOT / "examples/supervised_navigation/synthetic_cycle.json").read_text())


def test_synthetic_fixture_contract():
    jsonschema.Draft202012Validator.check_schema(SCHEMA)
    jsonschema.validate(FIXTURE, SCHEMA)
    assert FIXTURE["synthetic"] and FIXTURE["capability"]["kind"] == "synthetic"
    assert FIXTURE["capability"]["control"] is None


def test_unknown_mapping_and_time_bound_remain_unavailable():
    fixture = copy.deepcopy(FIXTURE)
    fixture["capability"]["mapping"] = None
    fixture["proposal"]["identity"]["uncertainty_bound_s"] = None
    jsonschema.validate(fixture, SCHEMA)  # represent unknown, do not invent identity/zero


def test_incomplete_identity_rejected():
    fixture = copy.deepcopy(FIXTURE)
    del fixture["proposal"]["identity"]["clock_epoch"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(fixture, SCHEMA)
