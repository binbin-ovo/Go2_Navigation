"""Mock tests for the read-only supervised guard provider. No SDK, DDS or robot."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "hardware/supervised_guards.py"
spec = importlib.util.spec_from_file_location("supervised_guards_under_test", PATH)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def floor(x0=0.3, x1=2.0, step=0.02, z=-0.33):
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(-0.6, 0.6, 0.05))
    return np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)], axis=1)


def wall(x, z0=-0.2, z1=0.3):
    ys, zs = np.meshgrid(np.arange(-0.3, 0.3, 0.02), np.arange(z0, z1, 0.05))
    return np.stack([np.full(ys.size, x), ys.ravel(), zs.ravel()], axis=1)


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def test_no_command_or_rpc_surface():
    source = PATH.read_text()
    for forbidden in ("_Call", "AvoidRoute", "claim_api", "release_api", "send_velocity", "send_stop",
                      "SwitchSet", "SportClient", "1001", "1003", "1004"):
        assert forbidden not in source


def test_floor_only_clearance_ends_at_observed_floor():
    clear, detail = g.forward_clearance(floor(x1=2.0))
    assert detail["observed_to_m"] == pytest.approx(2.0, abs=0.11)
    assert clear == pytest.approx(detail["observed_to_m"] - g.FRONT_M - g.MARGIN_M)


def test_obstacle_limits_clearance():
    clear, detail = g.forward_clearance(np.concatenate([floor(), wall(1.2)]))
    assert detail["nearest_obstacle_m"] == pytest.approx(1.2)
    assert clear == pytest.approx(1.2 - 0.5)


def test_unobserved_gap_is_not_free_space():
    pts = np.concatenate([floor(0.3, 0.85), floor(1.2, 2.0)])
    clear, detail = g.forward_clearance(pts)
    assert detail["observed_to_m"] == pytest.approx(0.9, abs=0.011)
    assert clear == pytest.approx(0.9 - g.FRONT_M - g.MARGIN_M)  # stops at the gap, not at 2.0 m


def test_no_floor_means_zero():
    clear, detail = g.forward_clearance(wall(1.0))
    assert clear == 0.0 and detail["reason"] == "floor_not_observed"


def test_obstacle_outside_corridor_ignored_inside_counted():
    side = wall(0.8) + np.array([0, 0.6, 0])
    clear, _ = g.forward_clearance(np.concatenate([floor(), side]))
    assert clear > 1.0


def test_to_current_base_after_forward_motion_brings_points_closer():
    pts = np.array([[1.0, 0.0, 0.0]])
    moved = g.to_current_base(pts, (0.0, 0.0, 0.0), (0.3, 0.0, 0.0))
    assert moved[0, 0] == pytest.approx(0.7)
    turned = g.to_current_base(pts, (0.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2))
    assert turned[0, :2] == pytest.approx([0.0, -1.0])


def test_odom_epoch_changes_on_reboot_and_backwards_stamp():
    clock = Clock()
    track = g.OdomTrack(clock)
    track.update(1000.0, 0, 0, 0)
    e0 = track.epoch()
    clock.t += 0.1
    track.update(1000.1, 0, 0, 0)
    assert track.epoch() == e0
    clock.t += 0.1
    track.update(5.0, 0, 0, 0)  # reboot: stamps restart
    assert track.epoch() != e0 and track.epoch().endswith(":g1")


def test_capture_bound_requires_evidence_mapping_and_boot():
    meta = {"clock_epoch": "host_receipt_wall:bridge=x:g0",
            "monotonic_mapping": "host_monotonic = ... - configured camera latency 0.048 s (sigma 0.016 s ...)"}
    assert g.capture_bound(meta, 50.0) == (None, None)  # no evidence configured
    cfg = dict(g.CAMERA, bound_s=0.1, boot_offset_s=50.0, basis="evidence")
    assert g.capture_bound(meta, 52.0, cfg) == (0.1, "evidence")
    assert g.capture_bound(meta, 70.0, cfg) == (None, None)  # other boot
    other = dict(meta, monotonic_mapping="configured camera latency 0.060 s")
    assert g.capture_bound(other, 50.0, cfg) == (None, None)
    assert g.capture_bound(None, 50.0, cfg) == (None, None)


def make_guards(tmp_path, clock):
    def front_surface_x(xyz, ground_z, half_width=0.35, min_height=0.10):
        ahead = xyz[(xyz[:, 0] > 0.3) & (np.abs(xyz[:, 1]) < half_width) & (xyz[:, 2] > ground_z + min_height)]
        return None if len(ahead) < 20 else float(np.median(ahead[:, 0]))

    def fuse_progress(lidar, odom):
        if odom is None:
            return lidar, lidar is not None
        if lidar is not None and abs(lidar - odom) <= 0.15 + 0.3 * abs(odom):
            return max(lidar, odom), True
        return odom, False

    return g.Guards(monitor=None, capability={}, monotonic=clock, observation_root=tmp_path,
                    front_surface_x=front_surface_x, fuse_progress=fuse_progress)


def test_snapshot_schema_shape_and_forward_progress(tmp_path):
    clock = Clock()
    guards = make_guards(tmp_path, clock)
    with pytest.raises(RuntimeError):
        guards.snapshot()
    guards.odom.update(10.0, 0.0, 0.0, 0.0)
    guards.scans.add(np.concatenate([floor(), wall(1.5)]))
    guards.compute()
    snap = guards.snapshot()
    assert snap["pose"]["frame"] == snap["geometry"]["frame"] == g.POSE_FRAME
    assert snap["geometry"]["directions"] == ["forward"]
    assert snap["geometry"]["forward_clear_m"] == pytest.approx(1.0)
    assert snap["geometry"]["turn_clear_rad"] == 0.0
    assert snap["capture_uncertainty_bound_s"] is None and snap["capture_bound_basis"] is None
    progress = guards.progress({"type": "forward", "direction": "forward"})
    clock.t += 0.5
    guards.odom.update(10.5, 0.2, 0.0, 0.0)
    guards.scans.add(np.concatenate([floor(), wall(1.5)]) - np.array([0.2, 0, 0]))
    assert progress() == pytest.approx(0.2, abs=0.02)


def test_stale_scans_make_geometry_unavailable(tmp_path):
    clock = Clock()
    guards = make_guards(tmp_path, clock)
    guards.odom.update(10.0, 0.0, 0.0, 0.0)
    guards.scans.add(floor())
    clock.t += 1.0
    guards.odom.update(11.0, 0.0, 0.0, 0.0)
    guards.compute()
    geo = guards.snapshot()["geometry"]
    assert geo["available"] is False and geo["directions"] == []


def test_camera_epoch_reads_newest_observation(tmp_path):
    obs = tmp_path / "session" / "observations"
    obs.mkdir(parents=True)
    for i, epoch in ((1, "e1"), (2, "e2")):
        (obs / f"{i:06d}.vlnobs.json").write_text(json.dumps({"clock_epoch": epoch, "monotonic_mapping": "m"}))
    cam = g.CameraEpoch(tmp_path)
    cam.poll()
    assert cam.get()["clock_epoch"] == "e2"


RECORDED = Path.home() / "Desktop/Go2_Navigation-vln-interactive/data/outputs/vln_interactive/live_10/lidar"


@pytest.mark.skipif(not RECORDED.exists(), reason="private recording not present")
def test_recorded_stationary_scans_give_plausible_clearance():
    data = np.load(sorted(RECORDED.glob("chunk_*.npz"))[100])
    points = data["points_xyz_base_link"][: data["scan_offsets"][8]]
    clear, detail = g.forward_clearance(points.astype(float))
    # A level-floor assumption reported the rising floor as an obstacle at 1.59 m here.
    assert detail["nearest_obstacle_m"] is not None and detail["nearest_obstacle_m"] > 2.0
    assert 1.5 < clear < 2.5
