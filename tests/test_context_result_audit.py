"""Evaluation-only archive comparisons must distinguish input from output tokens."""

import copy
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def audit(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    import summarize_vln_context_results
    return summarize_vln_context_results


def test_equal_suffix_is_not_equal_input_or_complete_sequence(audit, tmp_path):
    for name, prompt in (("a", [1,2,3]), ("b", [1,4,3])):
        np.savez(tmp_path/f"{name}.npz", input_ids=np.array([prompt]),
                 generated_ids=np.array([prompt+[8,9]]))
    a, b = [audit.archive(tmp_path/f"{name}.npz") for name in ("a","b")]
    assert a["inputs"] != b["inputs"] and a["generated_with_prompt"] != b["generated_with_prompt"]
    assert a["generated_suffix"] == b["generated_suffix"]


def test_wrong_prompt_prefix_rejected(audit,tmp_path):
    p = tmp_path/"bad.npz"
    np.savez(p,input_ids=np.array([[1,2]]),generated_ids=np.array([[1,3,9]]))
    with pytest.raises(ValueError,match="prompt prefix"):
        audit.archive(p)


def test_missing_or_failed_outputs_do_not_count_as_equal(audit):
    assert not audit.native_equal({}, {})
    assert not audit.native_equal({"status":"error"}, {"status":"error"})
    assert not audit.native_equal({"status":"decoded"}, {"status":"decoded"})


def test_last_waypoint_heading_is_part_of_complete_output(audit):
    a={"status":"decoded","native_output":"native token text", "pointing":{"opos":501},
       "rvq_codes":[1,2,3],"waypoints":[[.1*i,0,0] for i in range(10)],
       "decoder_stop":False,"explicit_stop_code":False}
    b=copy.deepcopy(a)
    assert audit.native_equal(a,b)
    b["waypoints"][-1][2]=.5
    assert not audit.native_equal(a,b)


def test_tensor_identity_includes_dtype_shape_and_bits(audit):
    a=np.array([1,2],np.uint16)
    assert audit.signatures({"x":a}) != audit.signatures({"x":a.astype(np.float32)})
    assert audit.signatures({"x":a}) != audit.signatures({"x":a.reshape(1,2)})
    assert audit.signatures({"x":a}) != audit.signatures({"x":a+1})
