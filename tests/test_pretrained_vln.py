"""No weights/downloads/hardware needed for Phase 6C interface regressions."""

from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from go2_perception.vln.pretrained import (
    SimulationRequest,
    causal_history,
    comparison_action,
    native_token_contract,
    validate_runtime,
)


def episode():
    return SimpleNamespace(
        data={
            "clock": {"epoch": "epoch-1"},
            "gaps": [],
            "decisions": [{"decision_id": "d0", "time_s": 1.1}],
            "frames": [
                {
                    "frame_id": str(i),
                    "observation_s": i / 10,
                    "available_s": i / 10 + 0.11,
                    "image": {"path": f"{i}.png"},
                }
                for i in range(20)
            ],
        }
    )


def runtime():
    return {
        "device": "cpu",
        "backend": "hf",
        "dtype": "bfloat16",
        "attention": "sdpa",
        "quantization": None,
        "max_new_tokens": 64,
        "action_feedback": "none_observation_only",
        "execution_authorized": False,
    }


def test_causality_includes_availability_and_uses_actual_history():
    ep = episode()
    result = causal_history(ep, "d0")
    assert result["frames"][-1]["frame_id"] == "9"
    assert len({f["frame_id"] for f in result["frames"]}) == len(result["frames"])
    assert all(f["available_s"] <= 1.1 for f in result["frames"])
    assert all(0 <= e <= 0.101 for e in result["timestamp_quantization_error_s"])
    old = deepcopy(result)
    ep.data["frames"][12]["image"] = {"path": "FUTURE_LABEL_MUST_NOT_LEAK.png"}
    assert causal_history(ep, "d0") == old


def test_new_epoch_has_no_old_episode_history():
    ep = episode()
    first = causal_history(ep, "d0")
    ep.data["clock"]["epoch"] = "epoch-2"
    ep.data["frames"] = ep.data["frames"][8:]
    second = causal_history(ep, "d0")
    assert second["clock_epoch"] != first["clock_epoch"]
    assert len(second["frames"]) == 1


@pytest.mark.parametrize("case", ["gap", "missing", "sparse", "empty"])
def test_regular_stream_refuses_fabrication(case):
    ep = episode()
    if case == "gap":
        ep.data["gaps"] = [{"start_s": 0.4}]
    if case == "missing":
        ep.data["frames"][4]["image"] = None
    if case == "sparse":
        ep.data["frames"] = ep.data["frames"][::5]
    if case == "empty":
        ep.data["frames"] = []
    with pytest.raises(ValueError):
        causal_history(ep, "d0")


@pytest.mark.parametrize(
    "text",
    [
        "Turn right",
        "<act_l0_6>",
        "<act_l0_6><act_l1_122><act_l2_256>",
        "<act_l0_6><act_l0_5><act_l1_122><act_l2_174>",
        "<act_l1_122><act_l0_6><act_l2_174>",
    ],
)
def test_unsupported_native_output_is_not_a_stop(text):
    with pytest.raises(ValueError):
        native_token_contract(text)


def test_native_codes_preserve_pointing_and_stopping_tokens():
    text = "<opos_0><apos_1299><act_l0_6><act_l1_122><act_l2_174><|im_end|>"
    assert native_token_contract(text) == [6, 122, 174]


@pytest.mark.parametrize(
    "row,expected",
    [
        ([0, 0, 0.2], "TURN_LEFT"),
        ([0, 0, -0.2], "TURN_RIGHT"),
        ([0.1, 0, 0], "ADVANCE"),
        ([0.1, 0, 0.2], None),
        ([0, 0.1, 0], None),
        ([-0.1, 0, 0], None),
        ([0, 0, 0], None),
    ],
)
def test_mapping_uses_radians_and_preserves_native_chunk(row, expected):
    wp = np.tile(row, (10, 1)).astype(float)
    saved = wp.copy()
    assert comparison_action(wp, native_stop=False) == expected
    assert np.array_equal(wp, saved)


def test_stop_requires_native_decoder_and_finite_ten_waypoints():
    assert comparison_action(np.zeros((10, 3)), native_stop=True) == "STOP"
    for wp in (np.ones((10, 3)), np.zeros((1, 3)), np.full((10, 3), np.nan)):
        with pytest.raises(ValueError):
            comparison_action(wp, native_stop=True)


@pytest.mark.parametrize(
    "key,value",
    [
        ("dtype", "float16"),
        ("attention", "flash_attention_2"),
        ("quantization", "4bit"),
        ("backend", "vllm_local"),
        ("action_feedback", "executed_proposals"),
        ("execution_authorized", True),
        ("max_new_tokens", 0),
        ("device", "automatic"),
    ],
)
def test_runtime_rejects_unverified_configuration(key, value):
    cfg = runtime()
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_runtime(cfg, {})


def test_turing_and_unavailable_devices_are_rejected():
    cfg = runtime()
    validate_runtime(cfg, {})
    cfg["device"] = "cuda"
    with pytest.raises(ValueError, match="Turing"):
        validate_runtime(cfg, {"cuda_available": True, "compute_capability": [7, 5]})
    with pytest.raises(ValueError, match="14 GiB"):
        validate_runtime(
            cfg,
            {
                "cuda_available": True,
                "compute_capability": [8, 6],
                "free_vram_bytes": 8 * 2**30,
            },
        )
    cfg["device"] = "mps"
    with pytest.raises(ValueError):
        validate_runtime(cfg, {})


@pytest.mark.parametrize(
    "mutation",
    [
        {"episode_id": "old"},
        {"clock_epoch": "old"},
        {"sequence": 0},
        {"execution_authorized": True},
    ],
)
def test_simulation_gate_rejects_wrong_episode_epoch_sequence_and_hardware(mutation):
    request = SimulationRequest("ep", "epoch", 1, 10.0, 12.0)
    response = {
        "episode_id": "ep",
        "clock_epoch": "epoch",
        "sequence": 1,
        "execution_authorized": False,
    }
    assert request.accept(response, 11.0) == response
    with pytest.raises(ValueError):
        request.accept({**response, **mutation}, 11.0)


@pytest.mark.parametrize("now", [9.0, 12.01, math.nan, math.inf])
def test_stale_or_invalid_host_time_is_refused(now):
    request = SimulationRequest("ep", "epoch", 1, 10.0, 12.0)
    with pytest.raises(ValueError):
        request.accept({}, now)
