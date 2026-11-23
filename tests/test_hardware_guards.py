"""Mock tests for the read-only supervised guard provider. No SDK, DDS or robot."""
import importlib.util
import json
import math
import os
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


WORKSPACE = Path(os.environ.get("GO2_WORKSPACE", Path.home() / "Robotics/Go2"))
RECORDED = WORKSPACE / "recordings/desktop_worktrees/vln-interactive/data/outputs/vln_interactive/live_10/lidar"


@pytest.mark.skipif(not RECORDED.exists(), reason="private recording not present")
def test_recorded_stationary_scans_give_plausible_clearance():
    data = np.load(sorted(RECORDED.glob("chunk_*.npz"))[100])
    points = data["points_xyz_base_link"][: data["scan_offsets"][8]]
    clear, detail = g.forward_clearance(points.astype(float))
    # A level-floor assumption reported the rising floor as an obstacle at 1.59 m here.
    assert detail["nearest_obstacle_m"] is not None and detail["nearest_obstacle_m"] > 2.0
    assert 1.5 < clear < 2.5


def _published(disc):
    return disc[0]


def _floor_with_side_bias(seed=0):
    """Floor as cloud_base reports it on 2026-11-22: range-dependent rise, +0.05..0.07 m at the sides."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.3, 2.4, 9000)
    a = rng.uniform(-np.pi * 0.75, np.pi * 0.75, 9000)  # the rear is LiDAR-blind
    x, y = r * np.cos(a), r * np.sin(a)
    z = -0.47 + 0.12 * r + rng.normal(0, 0.015, 9000)  # rises with range, not with x; open-floor recordings stay within 0.05 m
    return np.column_stack([x, y, z]).astype(np.float32)


def _box(cx, cy, size=0.2, top=0.30, n=400, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(cx - size / 2, cx + size / 2, n)
    y = rng.uniform(cy - size / 2, cy + size / 2, n)
    z = -0.47 + 0.12 * np.hypot(x, y) + rng.uniform(0.0, top, n)
    return np.column_stack([x, y, z]).astype(np.float32)


def test_turn_disc_measures_side_floor_locally_but_keeps_real_objects():
    floor = _floor_with_side_bias()
    coef = g.ground_plane(floor)
    assert coef is not None
    # The forward plane alone calls the biased side floor an obstacle.
    radius = np.hypot(floor[:, 0], floor[:, 1])
    ring = floor[(radius >= g.NEAR_LIMIT_M) & (radius <= g.DISC_RADIUS_M)]
    h = g.height_above_ground(ring, coef)
    assert np.sum((h > g.OBSTACLE_MIN_DZ) & (h < g.OBSTACLE_MAX_DZ)) > 20
    clear = _published(g.turn_disc(floor, coef))
    assert clear["obstacle_returns"] == 0 and clear["floor_returns"] > 500
    # A 0.3 m object beside the robot, or one filling its whole sector, still vetoes.
    for obj in (_box(0.33, 0.25), _box(0.0, -0.45, size=0.4, top=0.3, n=1500)):
        disc = _published(g.turn_disc(np.vstack([floor, obj]), coef))
        assert disc["obstacle_returns"] >= g.OBSTACLE_RANK
    # Straight beside the robot an in-place turn reaches only ~0.34 m: an object at 0.45 m stays clear.
    assert _published(g.turn_disc(np.vstack([floor, _box(0.0, 0.47, size=0.1)]), coef))["obstacle_returns"] == 0
    assert g.turn_reach(0) == pytest.approx(0.547, abs=0.005) and g.turn_reach(90) < 0.36
    # Two stray returns are rejected like the forward corridor's isolated points.
    stray = np.float32([[0.0, 0.45, -0.20], [0.05, 0.47, -0.21]])
    assert _published(g.turn_disc(np.vstack([floor, stray]), coef))["obstacle_returns"] == 0


def test_turn_disc_without_side_floor_keeps_the_forward_plane():
    floor = _floor_with_side_bias()
    coef = g.ground_plane(floor)
    front = floor[np.abs(np.degrees(np.arctan2(floor[:, 1], floor[:, 0]))) < 30]
    obj = _box(0.35, 0.25, n=60)  # no side floor at all: no refit, the object stays visible
    disc = _published(g.turn_disc(np.vstack([front, obj]), coef))
    assert disc["obstacle_returns"] >= g.OBSTACLE_RANK


def test_turn_disc_publishes_only_the_schema_fields():
    # An extra field made the executor reject every live proposal (2026-11-22, trial QZwmkd).
    schema = json.loads((ROOT / "src/go2_perception/vln/schemas/supervised-navigation-v2.schema.json").read_text())

    def find(node):
        if isinstance(node, dict):
            if "observed_turn_disc" in node.get("properties", {}):
                return node["properties"]["observed_turn_disc"]
            for value in node.values():
                hit = find(value)
                if hit:
                    return hit
        elif isinstance(node, list):
            for value in node:
                hit = find(value)
                if hit:
                    return hit

    floor = _floor_with_side_bias()
    disc, diagnostics = g.turn_disc(floor, g.ground_plane(floor))
    assert set(disc) == set(find(schema)["properties"]) and "floor_model" in diagnostics


def test_turn_disc_publishes_only_the_schema_fields():
    # An extra field made the executor reject every live proposal (2026-11-22, trial QZwmkd).
    schema = json.loads((ROOT / "src/go2_perception/vln/schemas/supervised-navigation-v2.schema.json").read_text())

    def find(node):
        if isinstance(node, dict):
            if "observed_turn_disc" in node.get("properties", {}):
                return node["properties"]["observed_turn_disc"]
            for value in node.values():
                hit = find(value)
                if hit:
                    return hit
        elif isinstance(node, list):
            for value in node:
                hit = find(value)
                if hit:
                    return hit

    floor = _floor_with_side_bias()
    disc, diagnostics = g.turn_disc(floor, g.ground_plane(floor))
    assert set(disc) == set(find(schema)["properties"]) and "floor_model" in diagnostics


def test_turn_disc_is_fast_enough_for_the_guard_loop():
    # Evaluating the swept reach per point took ~70 ms and staled the pose (2026-11-23 N7NzcG).
    import time
    floor = np.vstack([_floor_with_side_bias(seed) for seed in range(2)])
    coef = g.ground_plane(floor)
    start = time.perf_counter()
    for _ in range(10):
        g.turn_disc(floor, coef)
    assert (time.perf_counter() - start) / 10 < 0.02
