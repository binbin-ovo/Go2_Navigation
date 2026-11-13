"""Phase 6A contracts. Synthetic unit tests do not measure model semantic accuracy."""

from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from go2_perception.vln import load
from go2_perception.vln.calibration import load_calibration
from go2_perception.vln.grounding import GroundingResult, format_instruction, parse_candidates
from go2_perception.vln.grounding_demo import replay_source, run_observation
from go2_perception.vln.model_preprocessing import calibrated_view, processor_geometry
from go2_perception.vln.replay import Go2RecordingReplay

FIXTURE = Path(__file__).parent / "fixtures/vln/synthetic_recording"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def replay():
    return Go2RecordingReplay(FIXTURE, calibration=FIXTURE / "synthetic_calibration.yaml", allow_synthetic=True)


@pytest.fixture
def obs(replay):
    return replay.observation(8, history=2, instruction="Find the red box.")


def test_synthetic_intrinsics_require_opt_in(obs):
    with pytest.raises(ValueError, match="Measured"):
        calibrated_view(obs)
    assert calibrated_view(obs, allow_synthetic=True).image.dtype == np.uint8


def test_fisheye_rectification_uses_new_K_and_preserves_source(obs):
    # Measured coefficients scaled onto a synthetic test image, not a real-image experiment.
    intr, ext = load_calibration(ROOT / "configs/go2_front_camera_calibration.yaml")
    assert ext is None
    v = obs.visual
    intr = intr.scaled(v.width, v.height)
    v.K, v.D, v.distortion_model, v.intrinsics_status = intr.K, intr.D, intr.model, "synthetic"
    original = deepcopy(v)
    view = calibrated_view(obs, allow_synthetic=True)
    expected = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        intr.K, intr.D, (v.width, v.height), np.eye(3), balance=0, new_size=(v.width, v.height))
    np.testing.assert_allclose(view.intrinsics.K, expected)
    assert not np.allclose(expected, intr.K)
    assert view.intrinsics.model == "pinhole_radtan" and not view.intrinsics.D.any()
    assert view.image.shape == original.image.shape
    np.testing.assert_array_equal(v.image, original.image)
    np.testing.assert_array_equal(v.K, original.K)
    assert v.rectified is False
    assert view.metadata["source"]["clock_epoch"] == obs.clock_epoch
    assert view.metadata["source"]["stamp_robot"]["t"] == v.stamp_robot.t
    assert view.metadata["history_frames_available"] == 2
    assert view.metadata["history_frames_consumed"] == 0


def test_radtan_and_already_rectified_do_not_double_undistort(obs):
    view = calibrated_view(obs, allow_synthetic=True)
    v = obs.visual
    v.image, v.K, v.D = view.image, view.intrinsics.K, np.zeros(5)
    v.distortion_model, v.rectified = "pinhole_radtan", True
    copied = calibrated_view(obs, allow_synthetic=True)
    np.testing.assert_array_equal(copied.image, v.image)
    np.testing.assert_array_equal(copied.intrinsics.K, v.K)
    v.D[0] = .1
    with pytest.raises(ValueError, match="Rectified"):
        calibrated_view(obs, allow_synthetic=True)


@pytest.mark.parametrize("failure", ["normalized", "size", "unknown", "K", "missing_visual"])
def test_invalid_visual_input_fails_explicitly(obs, failure):
    if failure == "normalized":
        obs.visual.image = obs.visual.image.astype(float) / 255
    elif failure == "size":
        obs.visual.intrinsics_size = (320, 240)
    elif failure == "unknown":
        obs.visual.intrinsics_status = "unknown"
    elif failure == "K":
        obs.visual.K = None
    else:
        obs.visual = None
    with pytest.raises(ValueError):
        calibrated_view(obs, allow_synthetic=True)


def test_processor_geometry_carries_pixel_center_resize_and_padding(obs):
    view = calibrated_view(obs, allow_synthetic=True)
    mask = np.zeros((250, 330), dtype=int)
    mask[:240, :320] = 1
    result = processor_geometry(view, mask)
    np.testing.assert_allclose(result["K"], view.intrinsics.scaled(320, 240).K)
    assert result["padding_ltrb"] == [0, 0, 10, 10]
    assert result["resize_scale_xy"] == [2, 2]
    assert result["size_wh"] == [330, 250]
    mask[1, 1] = 0
    with pytest.raises(ValueError, match="layout"):
        processor_geometry(view, mask)
    with pytest.raises(ValueError, match="nonempty"):
        processor_geometry(view, np.zeros((3, 3)))


def test_official_image_processor_normalizes_uint8_once(obs):
    pytest.importorskip("transformers")
    from transformers import GroundingDinoImageProcessor
    v = calibrated_view(obs, allow_synthetic=True)
    v.image[:] = [255, 0, 128]
    ip = GroundingDinoImageProcessor(size={"shortest_edge": 120, "longest_edge": 160})
    batch = ip(images=v.image, return_tensors="np")
    expected = (np.array([1., 0., 128 / 255]) - ip.image_mean) / ip.image_std
    np.testing.assert_allclose(batch.pixel_values[0, :, 50, 50], expected, atol=1e-6)
    g = processor_geometry(v, batch.pixel_mask[0])
    np.testing.assert_allclose(g["K"], v.intrinsics.K)


@pytest.mark.parametrize("instruction,query", [("Find the Door!", "the door."),
        ("Locate the fridge.", "the fridge."), (" Show me the shoes ", "the shoes."),
        ("a tennis racket", "a tennis racket.")])
def test_instruction_format(instruction, query):
    assert format_instruction(instruction)["query"] == query


@pytest.mark.parametrize("instruction", ["Go toward the kitchen entrance.",
    "Which chair is beside the table?", "Turn right 30 degrees", "Find the nearest chair",
    "Find a fridge and then move forward", "Remember the previous room", "x" * 241])
def test_unsupported_instructions_remain_unresolved(instruction):
    result = format_instruction(instruction)
    assert not result["supported"] and result["query"] is None


def test_empty_instruction_is_error():
    with pytest.raises(ValueError):
        format_instruction("  ")


def test_parser_keeps_ambiguity_and_has_no_metric_goal():
    p = parse_candidates([[10, 20, 30, 40], [40, 40, 70, 90]], [.8, .7], ["chair", "chair"], (100, 100))
    assert p["status"] == "unresolved_multiple"
    assert p["proposal_only"] and not p["executable"] and p["metric_goal"] is None
    assert all(c["point_base_m"] is None for c in p["candidates"])
    assert parse_candidates([], [], [], (100, 100))["status"] == "unresolved_no_candidate"


def test_parser_rejects_bad_values_and_reports_clipping():
    p = parse_candidates([[np.nan, 1, 2, 3], [2, 3, 1, 4], [0, 0, 10, 20], [-1, 5, 12, 90]],
                         [.5, .5, 2, .9], ["bad", "bad", "bad", "box"], (100, 50))
    assert p["rejected_candidates"] == 3 and len(p["candidates"]) == 1
    assert p["candidates"][0]["box_xyxy"] == [0, 5, 12, 50]
    assert any("clipped" in w for w in p["warnings"])
    p = parse_candidates([[np.inf, 1, 2, 3]], [.8], ["box"], (100, 100))
    assert p["status"] == "invalid_output"
    with pytest.raises(ValueError, match="Malformed"):
        parse_candidates([[1, 2, 3]], [.8], ["box"], (100, 100))


def test_source_manifest_matches_causal_replay_and_history(replay, obs):
    source = replay_source(replay, 8, history=2)
    frames = source["rgb_frames"]
    assert len(frames) == 3 and sum(f["used_by_model"] for f in frames) == 1
    assert frames[-1]["exposure_estimated_robot_s"] == obs.visual.stamp_robot.t
    assert all(len(f["sha256"]) == 64 for f in frames)
    assert source["reference_robot_s"] == obs.geometric.stamp.t
    with pytest.raises(ValueError):
        replay_source(replay, -1)


def test_replay_without_extrinsics_does_not_invent_identity(tmp_path):
    import yaml
    cal = yaml.safe_load((FIXTURE / "synthetic_calibration.yaml").read_text())
    cal["T_base_camera"] = None
    path = tmp_path / "cal.yaml"
    path.write_text(yaml.safe_dump(cal))
    r = Go2RecordingReplay(FIXTURE, calibration=path, allow_synthetic=True)
    o = r.observation(8)
    assert r.extrinsics is None and o.geometric.projected_depth is None
    assert o.calibration["extrinsics_status"] == "unknown"


def test_demo_serializes_unmodified_observation_and_warnings(obs, tmp_path):
    pytest.importorskip("matplotlib")
    image = obs.visual.image.copy()
    obs.calibration["extrinsics_status"] = "unknown"
    obs.geometric.projected_depth = None
    class FixtureModel:
        info = {"model_id": "UNIT_TEST_FIXTURE_NOT_PRETRAINED"}
        config = {"box_threshold": .35, "text_threshold": .25}
        def predict(self, view, instruction):
            p = parse_candidates([[20, 20, 40, 60]], [.8], ["red box"], (160, 120))
            p.update(instruction=instruction, model_executed=False)
            return GroundingResult(p, {"fixture_logits": np.zeros((1, 4))}, {}, view.image,
                                   {"forward_s": None})
    p = run_observation(obs, FixtureModel(), tmp_path / "run", allow_synthetic=True)
    back = load(tmp_path / "run/observation.vlnobs.npz")
    np.testing.assert_array_equal(back.visual.image, image)
    np.testing.assert_array_equal(back.geometric.grid, obs.geometric.grid)
    assert p["metric_goal"] is None
    assert any("missing/unvalidated" in w for w in p["warnings"])
    assert (tmp_path / "run/demo.png").stat().st_size > 0
    metadata = json.loads((tmp_path / "run/preprocessing.json").read_text())
    assert metadata["source"]["clock_epoch"] == obs.clock_epoch
    with pytest.raises(FileExistsError):
        run_observation(obs, FixtureModel(), tmp_path / "run", allow_synthetic=True)
