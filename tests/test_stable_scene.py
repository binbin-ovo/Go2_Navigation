import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from go2_perception.scene.accumulation import TemporalAccumulator
from go2_perception.scene.background import BackgroundMap
from go2_perception.scene.objects import SceneObjects
from go2_perception.scene.proposals import make_proposals
from go2_perception.scene.pipeline import StableScene
from go2_perception.io.config import load_config


def pose(t):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("z", t * 0.3).as_matrix()
    m[:3, 3] = [0.2 * t, 0, 0.32]
    return m


def box(center=(2.0, 0.0, 0.4), size=(0.5, 0.5, 0.8), n=100, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(-0.5, 0.5, (n, 3)) * size + center
    return p


def proposal(center):
    return dict(
        center=np.array(center, dtype=float),
        dimensions=np.array([0.5, 0.5, 0.8]),
        point_count=40,
    )


def test_frame_motion_rotation_translation_and_provenance():
    a = TemporalAccumulator()
    world = box()
    for i in range(10):
        t = i * 0.065
        m = pose(t)
        body = (world - m[:3, 3]) @ m[:3, :3]
        a.add(body, t, str(i), pose, clock_domain="synthetic")
    cloud = a.snapshot(0.3)
    assert len(set(cloud.scan_ids)) == 5
    for identity in set(cloud.scan_ids):
        np.testing.assert_allclose(
            cloud.xyz[cloud.scan_ids == identity], world, atol=1e-12
        )
    assert cloud.ages.min() >= 0 and cloud.ages.max() <= 0.3
    v = cloud.voxelized(0.1)
    assert set(v.acquired) <= set(cloud.acquired)
    with pytest.raises(ValueError):
        a.add(body, t, str(i), pose, clock_domain="synthetic")
    with pytest.raises(ValueError):
        a.add(body, t + 1, "bad", pose, clock_domain="synthetic", frame_id="odom")


def test_point_deskew_requires_verified_contract():
    times = np.linspace(0, 0.06, 20)
    world = np.tile([2.0, 1.0, 0.5], (20, 1))
    body = np.array(
        [(p - pose(t)[:3, 3]) @ pose(t)[:3, :3] for p, t in zip(world, times)]
    )
    a = TemporalAccumulator()
    with pytest.raises(ValueError):
        a.add(
            body,
            0,
            "a",
            pose,
            clock_domain="synthetic",
            point_offsets=times,
            deskew=True,
        )
    new = a.add(
        body,
        0,
        "a",
        pose,
        clock_domain="synthetic",
        point_offsets=times,
        deskew=True,
        instantaneous_point_frame=True,
    )
    np.testing.assert_allclose(new.xyz, world, atol=1e-12)


@pytest.mark.parametrize("kind", ["missing", "gap", "rewind", "clock", "reset", "jump"])
def test_accumulation_invalidates_history(kind):
    a = TemporalAccumulator()
    a.add(box(), 1, "a", lambda t: np.eye(4), clock_domain="x")
    t = 1.1
    clock = "x"

    def lookup(t):
        return np.eye(4)

    if kind == "missing":

        def lookup(t):
            return None
    elif kind == "gap":
        t = 2.0
    elif kind == "rewind":
        t = 0.5
    elif kind == "clock":
        clock = "y"
    elif kind == "jump":

        def lookup(t):
            return np.array([[1, 0, 0, 3], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    a.add(box(), t, "b", lookup, clock_domain=clock, reset=kind == "reset")
    assert a.epoch == 1 and len(a.frames) <= 1 and a.reset_reason


def test_map_counts_distinct_scans_and_expires_to_unknown():
    m = BackgroundMap(ttl_s=0.5, min_span_s=0.1)
    p = np.array([[1, 0, 0.2]] * 20)
    m.update(p, 0, "a", [0, 0, 0])
    m.update(p, 0.1, "a", [0, 0, 0])
    assert not m.snapshot(0.1)["cells"]
    m.update(p, 0.2, "b", [0, 0, 0])
    m.update(p, 0.3, "c", [0, 0, 0])
    assert len(m.snapshot(0.3)["cells"]) == 1
    assert m.snapshot(1)["cells"] == []
    m.update(np.empty((0, 3)), 1, "d", [0, 0, 0])
    assert not m.cells
    assert m.snapshot(1)["free_space_policy"] == "unobserved_is_unknown"


def test_sparse_empty_nan_missing_pose_pipeline():
    c = load_config("configs/stable_lidar.yaml")
    s = StableScene(c)
    r = s.process(
        np.array([[np.nan, 0, 0], [1, 0, 0]]), 0, "a", pose, clock_domain="synthetic"
    )
    assert r["objects"] == []
    r = s.process(np.empty((0, 3)), 0.1, "b", lambda t: None, clock_domain="synthetic")
    assert r["robot_pose"] is None and not r["objects"] and not r["background"]["cells"]


def test_geometry_smoothing_motion_start_and_occlusion():
    o = SceneObjects()
    rng = np.random.default_rng(7)
    rows = []
    for i in range(70):
        t = i * 0.065
        center = np.array([2.0, max(0, t - 2), 0.4]) + rng.normal(0, 0.01, 3)
        o.update([proposal(center)], t)
        rows.append(o.snapshot(t)[0])
    assert sum(r["motion_state"] == "MOVING" for r in rows[:30]) == 0
    assert any(r["motion_state"] == "STATIONARY" for r in rows[:30])
    assert any(r["motion_state"] == "MOVING" for r in rows[40:])
    assert len({r["object_id"] for r in rows}) == 1
    assert all(np.isfinite(r["velocity"]).all() for r in rows)
    o.update([], 4.55)
    assert o.snapshot(4.55)[0]["status"] == "stale"
    o.update([], 6)
    assert not o.snapshot(6)


def test_two_objects_and_new_conflicting_points_are_not_suppressed():
    c = load_config("configs/stable_lidar.yaml")
    c["proposals"]["split_planes"] = False
    s = StableScene(c)
    for i in range(12):
        t = i * 0.065
        p = box(seed=i)
        m = pose(t)
        body = (p - m[:3, 3]) @ m[:3, :3]
        s.process(body, t, str(i), pose, clock_domain="synthetic")
    # New object touches the displayed old footprint but extends distinctly upward.
    p = np.concatenate((box(), box((2.0, 0.4, 1.4), (0.3, 0.3, 0.6), seed=5)))
    t = 0.8
    m = pose(t)
    r = s.process((p - m[:3, 3]) @ m[:3, :3], t, "new", pose, clock_domain="synthetic")
    assert r["diagnostics"]["new_points"] >= 150
    assert r["diagnostics"]["proposals"] >= 2


def test_planar_background_split_preserves_compact_points():
    rng = np.random.default_rng(3)
    wall = np.column_stack(
        (np.full(500, 3.0), rng.uniform(-2, 2, 500), rng.uniform(0, 2, 500))
    )
    compact = box((2.7, 0, 0.5), (0.4, 0.5, 1.0), n=100)
    p, d = make_proposals(np.vstack((wall, compact)))
    assert d["plane_points"] >= 450 and any(
        np.linalg.norm(x["center"] - [2.7, 0, 0.5]) < 0.2 for x in p
    )


def test_interval_ids_and_missing_point_pose():
    a = TemporalAccumulator()
    a.add(box(), 0, 1, pose, clock_domain="synthetic")
    with pytest.raises(ValueError):
        a.add(box(), 0.1, "1", pose, clock_domain="synthetic")
    b = TemporalAccumulator()

    def missing(t):
        return None if np.isclose(t, 0.02) else np.eye(4)

    assert (
        b.add(
            np.ones((4, 3)),
            0,
            "x",
            missing,
            clock_domain="synthetic",
            point_offsets=np.array([0, 0.02, 0.04, 0.06]),
            deskew=True,
            instantaneous_point_frame=True,
        )
        is None
    )
    assert b.reset_reason == "missing_point_odometry" and not b.frames


def test_runtime_scene_matches_published_schema():
    import json
    from pathlib import Path

    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(
        json.loads(Path("schemas/go2-scene-v1.schema.json").read_text())
    )
    s = StableScene(load_config("configs/stable_lidar.yaml"))
    for i in range(5):
        t = i * 0.1
        m = pose(t)
        p = box(seed=i)
        r = s.process(
            (p - m[:3, 3]) @ m[:3, :3], t, str(i), pose, clock_domain="synthetic"
        )
        validator.validate(r)


def test_geometry_metrics_are_json_serializable_and_do_not_count_stale_coverage():
    import json
    from go2_perception.scene.evaluation import score_frames

    row = dict(
        timestamp=0.0,
        objects=[
            dict(
                center=[1.0, 0.0, 0.0],
                dimensions=[1.0, 1.0, 1.0],
                velocity=[0.0, 0.0, 0.0],
                status="stale",
                motion_state="UNCERTAIN",
            )
        ],
        diagnostics={"processing_ms": 1.0},
    )
    truth = [
        dict(
            id=1,
            center=[1.0, 0.0, 0.0],
            dimensions=[1.0, 1.0, 1.0],
            velocity=[0.0, 0.0, 0.0],
            visible=True,
        )
    ]
    result = score_frames([(row, truth)])
    json.dumps(result, allow_nan=False)
    assert result["coverage"] == 0 and result["stale_fraction"] == 1


def test_accumulated_old_points_cannot_refresh_observation_status():
    c = load_config("configs/stable_lidar.yaml")
    c["object_window_s"] = 0.6
    s = StableScene(c)
    m = pose(0.0)
    p = box()
    first = s.process(
        (p - m[:3, 3]) @ m[:3, :3], 0.0, "a", pose, clock_domain="synthetic"
    )
    assert first["objects"]
    second = s.process(np.empty((0, 3)), 0.1, "b", pose, clock_domain="synthetic")
    assert second["objects"] and all(
        o["status"] == "stale" and o["last_observed"] == 0 for o in second["objects"]
    )
    assert second["diagnostics"]["unsupported_old_proposals"] > 0


def test_sparse_consecutive_observations_can_establish_motion():
    o = SceneObjects()
    moving = []
    for i in range(70):
        t = i * 0.065
        o.update([proposal([2.0, 0.7 * t, 0.4])] if i % 2 == 0 else [], t)
        row = o.snapshot(t)[0]
        if i % 2:
            assert row["status"] == "stale" and row["motion_state"] == "UNCERTAIN"
        else:
            moving.append(row["motion_state"] == "MOVING")
    assert any(moving[10:])
