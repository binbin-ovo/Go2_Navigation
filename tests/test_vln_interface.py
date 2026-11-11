"""Phase 5D: Go2 VLN observation interface, calibration types, replay and timing provenance."""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from go2_perception.vln import (FREE, OCCUPIED, UNKNOWN, CameraIntrinsics, RigidTransform, Stamp,
                                load, load_calibration, project_points, save, validate)
from go2_perception.vln import geometry
from go2_perception.vln.calibration import save_calibration
from go2_perception.vln.replay import Go2RecordingReplay
from go2_perception.vln.synthetic import (BOX_CENTER, CAMERA_LATENCY_S, OFFSET_S, box_corners,
                                          synthetic_calibration, write_recording)
from go2_perception.vln.tensors import resize, stack_history, to_tensor

FIXTURE = Path(__file__).parent / "fixtures" / "vln" / "synthetic_recording"


@pytest.fixture()
def replay():
    return Go2RecordingReplay(FIXTURE, calibration=FIXTURE / "synthetic_calibration.yaml",
                              allow_synthetic=True)


@pytest.fixture()
def recording(tmp_path):
    return write_recording(tmp_path / "rec")


def test_fixture_is_reproducible(tmp_path):
    fresh = write_recording(tmp_path / "fresh")
    for name in ["lidar/utlidar_cloud_base/index.jsonl", "rgb/index.jsonl", "manifest.json",
                 "synthetic_calibration.yaml", "lidar/utlidar_cloud_base/000003.npy"]:
        assert (fresh / name).read_bytes() == (FIXTURE / name).read_bytes(), name


def test_rgb_decoded_as_rgb_uint8_with_shape(replay):
    obs = replay.observation(8)
    img = obs.visual.image
    assert img.shape == (120, 160, 3) and img.dtype == np.uint8
    u, v = np.round(project_points(
        replay.extrinsics.inverse().apply(BOX_CENTER[None] - [0.2, 0, 0]), replay.intrinsics)[0][0]).astype(int)
    assert tuple(img[v, u]) == (255, 0, 0)  # drawn red in BGR, must read back red in RGB
    assert tuple(img[2, 2]) == (90, 90, 90)


def test_timestamp_provenance_and_camera_clock_conversion(replay):
    obs = replay.observation(8)
    v = obs.visual
    assert v.stamp_host.clock == "host_receipt" and v.stamp_host.semantics == "host_receipt"
    assert v.stamp_robot.clock == "go2_robot" and v.stamp_robot.semantics == "exposure_estimated"
    assert v.stamp_robot.uncertainty_s > 0
    rows = [json.loads(line) for line in open(FIXTURE / "rgb" / "index.jsonl")]
    truth = {r["host_recv_first"]: r["synthetic_exposure_robot_s"] for r in rows}
    assert v.stamp_robot.t == pytest.approx(truth[v.stamp_host.t], abs=1e-6)
    assert obs.host_minus_robot_s == pytest.approx(OFFSET_S)
    assert v.stamp_host.t - obs.host_minus_robot_s - v.stamp_robot.t == pytest.approx(CAMERA_LATENCY_S)


def test_camera_frame_is_causal(replay):
    for k in range(3, len(replay)):
        obs = replay.observation(k)
        assert obs.visual.stamp_robot.t <= obs.geometric.stamp.t + 1e-9


def test_lidar_decoding_window_and_frames(replay):
    obs = replay.observation(11)
    g = obs.geometric
    assert g.frame_id == "base_link" and g.stamp.semantics == "scan_midpoint"
    assert g.points.dtype == np.float32 and g.points.shape[1] == 3
    assert g.frames_used == 8  # 0.5 s / 65 ms
    assert g.display_points.shape[0] > g.points.shape[0] and g.display_window_s == 1.0
    one = np.load(FIXTURE / "lidar" / "utlidar_cloud_base" / "000011.npy")
    assert one.shape[1] == 4  # xyz + intensity on disk; interface keeps xyz


def test_grid_semantics_unknown_free_occupied():
    pts = np.array([[1.05, 0.05, -0.33], [2.05, 0.05, -0.33], [2.05, 0.05, 0.20], [1.05, 1.05, 2.5]])
    grid, elev, origin, band = geometry.occupancy(pts, **geometry.DEFAULTS)
    cell = lambda x, y: (int((x - origin[0]) / 0.1), int((y - origin[1]) / 0.1))  # noqa: E731
    assert grid[cell(1.05, 0.05)] == FREE
    assert grid[cell(2.05, 0.05)] == OCCUPIED  # obstacle return wins over floor return
    assert grid[cell(1.05, 1.05)] == UNKNOWN  # above obstacle_top: no evidence either way
    assert grid[cell(4.05, 0.05)] == UNKNOWN and np.isnan(elev[cell(4.05, 0.05)])
    assert band == pytest.approx((-0.23, 0.87))


def test_sectors_ignore_blind_zone_and_report_unknown():
    pts = np.array([[0.2, 0.0, 0.0], [1.0, 0.01, -0.33], [2.0, 0.02, 0.1], [3.0, 0.03, -0.33]])
    near, clear = geometry.sectors(pts, **geometry.DEFAULTS)
    assert near[0] == pytest.approx(2.0, abs=1e-3)  # 0.2 m return is inside the blind zone
    assert clear[0] == pytest.approx(1.0, abs=1e-3)  # floor beyond the obstacle is not "clear"
    assert np.isnan(near[36]) and np.isnan(clear[36])  # behind the robot: unknown


def test_transforms_inverse_compose_and_direction():
    intr, ext = synthetic_calibration()
    p_cam = np.array([[0.0, 0.0, 2.0]])  # 2 m along optical axis
    assert ext.apply(p_cam)[0] == pytest.approx([2.3, 0.0, 0.1])  # base_link: x forward
    assert ext.inverse().apply(ext.apply(p_cam)) == pytest.approx(p_cam)
    T_lidar_base = RigidTransform(np.eye(4), "base_link", "utlidar_lidar", "firmware")
    chain = T_lidar_base.compose(ext)
    assert chain.source_frame == intr.camera_frame and chain.target_frame == "utlidar_lidar"
    assert chain.status == "synthetic"
    with pytest.raises(ValueError, match="compose"):
        ext.compose(ext)


@pytest.mark.parametrize("matrix", [np.diag([1, 1, -1, 1]), np.diag([2, 1, 1, 1]),
                                    np.full((4, 4), np.nan), np.eye(3)])
def test_invalid_transforms_rejected(matrix):
    with pytest.raises(ValueError):
        RigidTransform(matrix, "a", "b", "measured")


def test_transform_frames_and_status_required():
    with pytest.raises(ValueError):
        RigidTransform(np.eye(4), "a", "a", "measured")
    with pytest.raises(ValueError):
        RigidTransform(np.eye(4), "a", "b", "hand_tuned")


def test_projection_rejects_behind_and_outside():
    intr, _ = synthetic_calibration()
    uv, ok = project_points(np.array([[0, 0, 2.0], [0, 0, -2.0], [10, 0, 1.0], [np.nan, 0, 1]]), intr)
    assert ok.tolist() == [True, False, False, False]
    assert uv[0] == pytest.approx([79.5, 59.5])
    assert np.isnan(uv[1:]).all()


@pytest.mark.parametrize("model,D", [("pinhole_radtan", [-0.3, 0.1, 0.001, -0.001, 0.0]),
                                     ("fisheye_equidistant", [0.05, -0.01, 0.002, 0.0])])
def test_distortion_models_match_opencv_undistort(model, D):
    K = np.array([[600.0, 0, 640], [0, 600, 360], [0, 0, 1]])
    intr = CameraIntrinsics(1280, 720, K, model, D, "synthetic")
    rng = np.random.default_rng(1)
    p = np.c_[rng.uniform(-0.8, 0.8, 50), rng.uniform(-0.4, 0.4, 50), np.ones(50)]
    uv, ok = project_points(p, intr)
    assert ok.all()
    pts = uv[ok].reshape(-1, 1, 2)
    if model == "fisheye_equidistant":
        norm = cv2.fisheye.undistortPoints(pts, K, intr.D).reshape(-1, 2)
    else:
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
        norm = cv2.undistortPointsIter(pts, K, intr.D, None, None, crit).reshape(-1, 2)
    assert norm == pytest.approx(p[ok, :2], abs=1e-4)
    with pytest.raises(ValueError):
        CameraIntrinsics(1280, 720, K, model, D[:-1], "synthetic")


def test_intrinsics_scale_with_resize():
    intr, _ = synthetic_calibration()
    small = intr.scaled(80, 60)
    p = np.array([[0.3, -0.2, 2.0]])
    uv_full = project_points(p, intr)[0][0]
    uv_small = project_points(p, small)[0][0]
    assert uv_small == pytest.approx((uv_full + 0.5) * 0.5 - 0.5)


def test_letterbox_resize_updates_intrinsics_and_tensor():
    intr, _ = synthetic_calibration()
    img = np.zeros((120, 160, 3), np.uint8)
    out, new, info = resize(img, (224, 224), intr)
    assert out.shape == (224, 224, 3) and info["pad_y"] > 0 and info["pad_x"] == 0
    p = np.array([[0.1, 0.1, 1.5]])
    a = project_points(p, intr)[0][0]
    b = project_points(p, new)[0][0]
    assert b == pytest.approx([(a[0] + 0.5) * info["scale_x"] - 0.5 + info["pad_x"],
                               (a[1] + 0.5) * info["scale_y"] - 0.5 + info["pad_y"]])
    t = to_tensor(out)
    assert tuple(t.shape) == (3, 224, 224) and str(t.dtype) == "torch.float32"
    with pytest.raises(ValueError):
        to_tensor(out.astype(np.float32))


def test_missing_calibration_gives_no_depth_and_a_warning():
    obs = Go2RecordingReplay(FIXTURE).observation(10)
    assert obs.visual.K is None and obs.visual.intrinsics_status == "unknown"
    assert obs.geometric.projected_depth is None
    assert any("calibration missing" in w for w in obs.warnings)
    assert obs.calibration["extrinsics_status"] == "unknown"
    assert validate(obs) == []


def test_synthetic_calibration_refused_without_opt_in():
    with pytest.raises(ValueError, match="allow_synthetic"):
        Go2RecordingReplay(FIXTURE, calibration=FIXTURE / "synthetic_calibration.yaml")


def test_projected_depth_with_synthetic_calibration(replay):
    obs = replay.observation(10)
    d = obs.geometric.projected_depth
    assert d.shape == (120, 160) and np.isfinite(d).sum() > 100
    corners = replay.extrinsics.inverse().apply(box_corners())
    uv, ok = project_points(corners, replay.intrinsics)
    u0, v0 = np.floor(np.nanmin(uv, 0)).astype(int)
    u1, v1 = np.ceil(np.nanmax(uv, 0)).astype(int)
    box_depth = d[v0 + 2:v1 - 2, u0 + 2:u1 - 2]
    assert np.nanmedian(box_depth) == pytest.approx(1.4 - 0.3, abs=0.05)  # box front face in camera z


def test_missing_odometry(recording):
    (recording / "state" / "utlidar_robot_odom.jsonl").write_text(
        "".join(line + "\n" for line in (recording / "state" / "utlidar_robot_odom.jsonl")
                .read_text().splitlines()[:5]))
    rp = Go2RecordingReplay(recording)
    obs = rp.observation(10)
    assert obs.geometric is None and obs.robot.T_odom_base is None
    assert any("odometry" in w for w in obs.warnings) and any("pose unknown" in w for w in obs.warnings)


def test_missing_rgb_stream(recording):
    shutil.rmtree(recording / "rgb")
    obs = Go2RecordingReplay(recording).observation(10)
    assert obs.visual is None and validate(obs) == []


def test_clock_domain_mismatch_detected(replay):
    obs = replay.observation(10)
    obs.visual.stamp_robot = None
    assert any(i.startswith("clock_domain_mismatch") for i in validate(obs))
    obs = replay.observation(10)
    obs.visual.stamp_robot = Stamp(obs.visual.stamp_robot.t, "host_receipt", "exposure_estimated")
    assert any("clock_domain_mismatch" in i for i in validate(obs))
    obs = replay.observation(10)
    obs.visual.stamp_robot.t -= 1.0
    assert any("skew" in i for i in validate(obs))


def test_validation_catches_bad_fields(replay):
    obs = replay.observation(10)
    obs.visual.image = obs.visual.image.astype(np.float32)
    obs.geometric.grid[0, 0] = 5
    obs.robot.T_odom_base = np.diag([1, 1, -1, 1.0])
    obs.geometric.points[0, 0] = np.nan
    issues = validate(obs)
    assert len(issues) == 4
    with pytest.raises(ValueError):
        validate(obs, strict=True)


def test_intrinsics_size_mismatch_flagged(replay):
    obs = replay.observation(10)
    assert validate(obs) == []
    obs.visual.image = np.zeros((720, 1280, 3), np.uint8)  # K belongs to 160x120
    assert any("intrinsics are for (160, 120)" in i for i in validate(obs))


def test_serialization_round_trip(replay, tmp_path):
    obs = replay.observation(11, history=2, instruction="Go to the kitchen")
    save(obs, tmp_path / "o.npz")
    back = load(tmp_path / "o.npz")
    assert validate(back) == validate(obs) == []
    assert back.instruction == "Go to the kitchen" and back.clock_epoch == obs.clock_epoch
    assert np.array_equal(back.visual.image, obs.visual.image)
    assert len(back.visual.history) == 2
    assert back.visual.history[0].stamp_host == obs.visual.history[0].stamp_host
    assert np.array_equal(back.geometric.grid, obs.geometric.grid)
    assert np.array_equal(back.geometric.sector_range, obs.geometric.sector_range, equal_nan=True)
    assert back.geometric.origin_xy == obs.geometric.origin_xy
    assert np.allclose(back.robot.T_odom_base, obs.robot.T_odom_base)
    assert stack_history(back.visual).shape == (3, 120, 160, 3)
    with np.load(tmp_path / "o.npz", allow_pickle=False) as z:
        assert all(z[k].dtype != object for k in z.files)


def test_serialization_rejects_unknown_schema(replay, tmp_path):
    obs = replay.observation(10)
    obs.schema_version = "go2-vln-obs-v0"
    save(obs, tmp_path / "o.npz")
    with pytest.raises(ValueError, match="Unsupported schema"):
        load(tmp_path / "o.npz")


def test_calibration_file_metadata_and_units(tmp_path):
    intr, ext = load_calibration(FIXTURE / "synthetic_calibration.yaml")
    data = yaml.safe_load((FIXTURE / "synthetic_calibration.yaml").read_text())
    assert data["schema"] == "go2-camera-calibration-v1" and data["status"] == "synthetic"
    assert data["units"] == {"K": "pixels", "translation": "metres"}
    assert data["T_base_camera"]["direction"] == "p_base = T @ p_camera"
    assert ext.source_frame == intr.camera_frame and ext.target_frame == "base_link"
    save_calibration(tmp_path / "c.yaml", intr, ext)
    i2, e2 = load_calibration(tmp_path / "c.yaml")
    assert np.allclose(i2.K, intr.K) and np.allclose(e2.matrix, ext.matrix)
    data["T_base_camera"]["status"] = "measured"
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="Measured extrinsics"):
        load_calibration(tmp_path / "bad.yaml")
    data["schema"] = "other"
    (tmp_path / "bad2.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="schema"):
        load_calibration(tmp_path / "bad2.yaml")


def test_vln_config_records_user_choice():
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs" / "go2_vln_sensors.yaml").read_text())
    acc = cfg["lidar_accumulation"]
    assert acc["display_window_s"] == 1.0 and acc["obstacle_window_s"] == 0.5
    assert acc["source_topic"] == "rt/utlidar/cloud_base" and acc["unobserved_is_free"] is False
