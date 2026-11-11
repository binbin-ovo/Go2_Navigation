"""Phase 5B viewer: schema parsing, transforms, timing, display state, colours, layers, export,
privacy. Synthetic data only."""

import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest

from go2_perception.diagnostics.attribution import attribute_miss, attribute_not_moving, in_cylinder
from go2_perception.viewer import planning
from go2_perception.viewer import render as R
from go2_perception.viewer.camera import Camera, ChaseCamera
from go2_perception.viewer.layers import LAYER_INFO, Layers, mode_layers
from go2_perception.viewer.privacy import is_private_path, require_private_dir
from go2_perception.viewer.scene import SceneError, loads, parse_row, read_frames
from go2_perception.viewer.session import Session
from go2_perception.viewer.sources import Phase3
from go2_perception.viewer.state import GHOST, STALE, CausalPlayer, DisplayConfig, DisplayState, interpolate_pose

FIXTURE = Path("tests/fixtures/scene/renderer.jsonl")
ROOT = Path(__file__).resolve().parents[1]


def rows():
    return [json.loads(s) for s in FIXTURE.read_text().splitlines()]


def obj(oid=1, status="observed_filtered", motion="UNCERTAIN", center=(2, 0, 0.4), v=(0, 0, 0), t=0.0, last=None, cov=0.01):
    last = t if last is None else last
    return dict(
        object_id=oid, track_id=None, center=list(center), dimensions=[0.5, 0.5, 0.8], orientation_xyzw=None,
        velocity=list(v), velocity_covariance=(np.eye(3) * cov).tolist(), motion_state=motion, status=status,
        last_observed=last, observation_age_s=t - last, age_s=max(t - last, 0.0) + 1.0, observation_score=0.5,
        observed_box=None if status == "stale" else dict(center=list(center), dimensions=[0.5, 0.5, 0.8], point_count=10),
        display_compact=False,
    )  # fmt: skip


def row(t=0.0, objects=(), epoch=0, pose=True, cells=()):
    m = np.eye(4)
    m[:3, 3] = (0.1 * t, 0, 0.32)
    return dict(
        schema="go2-scene-v1", timestamp=t, clock_domain="synthetic", frame_id="odom", units="m", epoch=epoch,
        robot_pose=dict(timestamp=t, parent_frame="odom", child_frame="base_link", T_odom_base=m.tolist()) if pose else None,
        background=dict(voxel_size=0.1, free_space_policy="unobserved_is_unknown", cells=list(cells)),
        objects=list(objects), diagnostics=dict(synthetic=True),
    )  # fmt: skip


# ---- schema parsing and missing fields ----
def test_fixture_parses_and_matches_contract():
    frames = list(read_frames(FIXTURE))
    assert len(frames) == 6
    last = frames[-1]
    assert [o.status for o in last.objects] == ["stale", "observed_filtered"]
    assert last.objects[1].motion_state == "MOVING" and not last.objects[0].observed


def test_fixture_agrees_with_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(Path("schemas/go2-scene-v1.schema.json").read_text())
    for r in rows():
        jsonschema.validate(r, schema)
        parse_row(r)


@pytest.mark.parametrize("field", ["schema", "timestamp", "epoch", "robot_pose", "background", "objects", "diagnostics", "clock_domain"])
def test_missing_row_field_is_named(field):
    r = rows()[0]
    del r[field]
    with pytest.raises(SceneError, match=field):
        parse_row(r)


@pytest.mark.parametrize("field", ["center", "velocity_covariance", "observed_box", "observation_age_s", "motion_state"])
def test_missing_object_field_is_named(field):
    r = rows()[0]
    del r["objects"][0][field]
    with pytest.raises(SceneError, match=field):
        parse_row(r)


def test_rejects_nan_bad_pose_stale_with_box_and_claims_without_pose():
    with pytest.raises(SceneError):
        loads(FIXTURE.read_text().splitlines()[0].replace('"timestamp": 0.0', '"timestamp": NaN', 1))
    r = rows()[0]
    r["robot_pose"]["T_odom_base"][0][0] = 2.0
    with pytest.raises(SceneError, match="rigid"):
        parse_row(r)
    r = rows()[0]
    r["objects"][0]["status"] = "stale"
    with pytest.raises(SceneError, match="observed_box"):
        parse_row(r)
    r = rows()[0]
    r["robot_pose"] = None
    with pytest.raises(SceneError, match="missing pose"):
        parse_row(r)
    r = rows()[0]
    r["background"]["free_space_policy"] = "unobserved_is_free"
    with pytest.raises(SceneError, match="free_space_policy"):
        parse_row(r)


def test_unknown_fields_are_ignored():
    r = rows()[0]
    r["future_optional"] = {"x": 1}
    r["objects"][0]["new_field"] = 3
    assert parse_row(r).objects[0].object_id == 1


# ---- transforms ----
def test_scene_geometry_is_never_transformed_by_pose():
    r = row(1.0, [obj(t=1.0, center=(3.0, 1.0, 0.4))])
    m = np.eye(4)
    m[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    m[:3, 3] = [5, 5, 0.3]
    r["robot_pose"]["T_odom_base"] = m.tolist()
    f = parse_row(r)
    assert np.array_equal(f.objects[0].center, [3.0, 1.0, 0.4])
    cam = Camera((320, 240)).look_at((-3, 0, 4), (3, 0, 0))
    items = DisplayState().update(f)
    far_a, far_b = np.eye(4), np.eye(4)
    far_a[:3, 3] = (-500, 0, 0.32)
    far_b[:3, 3] = (-800, 40, 0.32)
    layers = Layers(labels=False, robot_path=False)
    ov = R.Overlays(floor_z=0.0)
    img_a, _ = R.render_scene((320, 240), cam, f, items, layers, ov, display_pose=far_a)
    img_b, _ = R.render_scene((320, 240), cam, f, items, layers, ov, display_pose=far_b)
    assert np.array_equal(img_a, img_b)  # robot pose off-screen: scene pixels independent of the pose


def test_phase3_boxes_transform_once(tmp_path):
    m = np.eye(4)
    m[:2, :2] = [[0, -1], [1, 0]]
    m[:3, 3] = [1.0, 2.0, 0.3]
    det = dict(position=[1, 0, 0], bbox_min=[0.9, -0.1, -0.2], bbox_max=[1.1, 0.1, 0.2])
    (tmp_path / "frames.jsonl").write_text(json.dumps(dict(timestamp=5.0, robot_pose=dict(T_parent_child=m.tolist()), detections=[det])) + "\n")
    (tmp_path / "tracks.jsonl").write_text(json.dumps(dict(t=5.0, latency_ms=1.0, tracks=[[3, 1.0, 3.0, 0.3, 0.0, 0.5, 0.1, "C", "M", 0]])) + "\n")
    p3 = Phase3(tmp_path, tmp_path / "tracks.jsonl")
    corners = p3.boxes(0)[0]
    assert np.allclose(corners.mean(0), [1.0, 3.0, 0.3])  # base (1,0,0) -> odom (1,3,0.3)
    t = p3.tracks(0)[0]
    assert t["motion"] == "MOVING" and t["observed"] and t["xyz"] == [1.0, 3.0, 0.3]


def test_camera_projection_and_near_clip():
    cam = Camera((200, 100), 60).look_at((0, 0, 0), (1, 0, 0))
    uv, z = cam.project([(5, 0, 0), (-1, 0, 0), (5, 1, 0)])
    assert np.allclose(uv[0], (100, 50)) and np.isnan(uv[1]).all()
    assert uv[2, 0] < 100  # +y (left in odom) projects to the left
    assert cam.clip_segment((-1, 0, 0), (-2, 0, 0)) is None
    assert cam.clip_segment((-1, 0, 0), (5, 0, 0)) is not None


def test_chase_camera_smooths_viewpoint_only():
    cc = ChaseCamera((100, 100), tau_s=1.0)
    p0, p1 = np.eye(4), np.eye(4)
    p1[:3, 3] = (2, 0, 0)
    cc.update(p0, 0.0)
    cc.update(p1, 0.1)
    assert 0 < cc.pos[0] < 0.5  # lags the robot; the robot pose itself is unchanged
    assert np.array_equal(p1[:3, 3], [2, 0, 0])


# ---- timestamp sync and causality ----
def test_causal_player_never_shows_future_frames():
    frames = [parse_row(row(t)) for t in (0.0, 0.1, 0.2, 0.3)]
    player = CausalPlayer(frames)
    assert player.index_at(-0.01) == -1
    assert player.index_at(0.15) == 1
    assert player.index_at(0.1) == 1
    for clock in np.linspace(0, 0.35, 50):
        k = player.index_at(clock)
        assert k < 0 or frames[k].timestamp <= clock


def test_causal_interpolation_uses_only_received_poses_and_respects_epochs():
    frames = [parse_row(row(t, epoch=0 if t < 0.25 else 1)) for t in (0.0, 0.1, 0.2, 0.3)]
    cfg = DisplayConfig(interpolate_pose=True, interpolation_delay_s=0.1)
    player = CausalPlayer(frames, cfg)
    pose, mode = player.pose_at(0.15)  # display time 0.05, both bracketing frames received
    assert mode == "interpolated" and np.isclose(pose[0, 3], 0.005)
    _, mode = player.pose_at(0.28)  # bracket 0.1..0.2 received, fine
    assert mode == "interpolated"
    _, mode = player.pose_at(0.31)  # bracket crosses the epoch change: hold
    assert mode == "held"
    assert player.index_at(0.15) == 0  # objects are delayed with the pose


def test_interpolate_pose_slerp():
    a, b = np.eye(4), np.eye(4)
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    b[:2, :2] = [[c, -s], [s, c]]
    m = interpolate_pose(a, b, 0, 1, 0.5)
    assert np.isclose(np.degrees(np.arctan2(m[1, 0], m[0, 0])), 45)


def test_rgb_alignment_marks_missing_frames():
    from go2_perception.visualization.diagnostic_video import align_rgb

    idx, dt = align_rgb([0.0, 1.0, 2.0], [10.0, 11.05], host_offset=10.0)
    assert idx[:2] == [0, 1] and idx[2] is None and np.isclose(dt[1], 0.05)


# ---- object update, expiry, display persistence ----
def test_stale_fades_and_expires_and_ghost_is_unknown():
    cfg = DisplayConfig(stale_max_s=0.5, ghost_hold_s=0.4)
    st = DisplayState(cfg)
    st.update(parse_row(row(0.0, [obj(t=0.0)])))
    items = st.update(parse_row(row(0.2, [obj(status="stale", t=0.2, last=0.0)])))
    assert items[0].kind == STALE and 0.15 < items[0].alpha < 1.0
    items = st.update(parse_row(row(0.6, [obj(status="stale", t=0.6, last=0.0)])))
    assert [i.kind for i in items] == []  # older than stale_max_s: no box, not yet a ghost
    items = st.update(parse_row(row(0.7, [])))
    assert [i.kind for i in items] == [GHOST] and items[0].colour_key == "UNCERTAIN"
    assert "unknown" in " ".join(R.object_label(items[0], False))
    items = st.update(parse_row(row(1.2, [])))
    assert items == []  # ghost expired after ghost_hold_s


def test_epoch_change_clears_renderer_memory():
    st = DisplayState()
    st.update(parse_row(row(0.0, [obj(t=0.0)])))
    st.update(parse_row(row(0.1, [obj(t=0.1)])))
    assert st.trails[1] and len(st.robot_path) == 2
    items = st.update(parse_row(row(0.2, [], epoch=1)))
    assert items == [] and st.resets == 1 and len(st.robot_path) == 1 and not st.trails


def test_out_of_order_frames_rejected():
    st = DisplayState()
    st.update(parse_row(row(1.0)))
    with pytest.raises(ValueError):
        st.update(parse_row(row(0.5)))


# ---- motion colours and arrows ----
def test_motion_colours_only_for_fresh_evidence():
    st = DisplayState()
    items = st.update(parse_row(row(0.0, [obj(1, motion="MOVING", v=(1, 0, 0)), obj(2, motion="STATIONARY", center=(4, 0, 0.4)), obj(3, status="stale", motion="MOVING", center=(6, 0, 0.4), last=-0.1)])))
    keys = {i.obj.object_id: i.colour_key for i in items}
    assert keys == {1: "MOVING", 2: "STATIONARY", 3: "UNCERTAIN"}
    assert R.MOTION_COLOUR["MOVING"] != R.MOTION_COLOUR["UNCERTAIN"]
    by_id = {i.obj.object_id: i for i in items}
    assert by_id[1].cv_path is not None and by_id[3].cv_path is None
    assert np.all(np.diff(by_id[1].cv_sigma) > 0)  # uncertainty band widens


def test_prior_velocity_draws_no_arrow_or_prediction():
    o = obj(motion="MOVING", v=(0, 0, 0), cov=1.0)
    items = DisplayState().update(parse_row(row(0.0, [o])))
    assert items[0].obj.velocity_is_prior and items[0].cv_path is None
    assert "n/a" in R.object_label(items[0], False)[1]


def test_velocity_arrow_points_along_velocity():
    cam = Camera((300, 300), 60).look_at((0, 0, 8), (0, 0, 0), up=(1, 0, 0))  # top-down, +x up
    f = parse_row(row(0.0, [obj(motion="MOVING", center=(0, 0, 0.4), v=(0, 1.0, 0))]))
    item = DisplayState(DisplayConfig(cv_horizon_s=0)).update(f)[0]
    img = np.zeros((300, 300, 3), np.uint8)
    R.draw_velocity(img, cam, item, 0.0, 1.0)
    ys, xs = np.nonzero(img.any(2))
    c = cam.project([(0, 0, 0)])[0][0]
    assert xs.mean() < c[0] - 5  # +y in odom is to the left in a +x-up top-down view
    img2 = np.zeros_like(img)
    f2 = parse_row(row(0.0, [obj(motion="MOVING", center=(0, 0, 0.4), v=(1.0, 0, 0))]))
    R.draw_velocity(img2, cam, DisplayState(DisplayConfig(cv_horizon_s=0)).update(f2)[0], 0.0, 1.0)
    ys2, _ = np.nonzero(img2.any(2))
    assert ys2.mean() < c[1] - 5  # +x is up


# ---- layer toggles ----
def test_mode_presets_and_toggles():
    clean, hybrid, debug = (mode_layers(m) for m in ("clean", "hybrid", "debug"))
    assert not clean.raw_incremental and not clean.accumulated and clean.fresh_boxes
    assert hybrid.accumulated and hybrid.background_voxels and not hybrid.raw_incremental
    assert all(getattr(debug, n) for n in LAYER_INFO if n != "retrospective_gt") and not debug.retrospective_gt
    assert clean.toggle("raw_incremental") is True
    assert clean.toggle_key("1") == ("raw_incremental", False)
    keys = [k for k, _ in LAYER_INFO.values()]
    assert len(keys) == len(set(keys))
    with pytest.raises(ValueError):
        mode_layers("pretty")


def test_raw_point_layer_changes_pixels_only_when_enabled():
    f = parse_row(row(0.0))
    cam = Camera((200, 200)).look_at((-3, 0, 2), (2, 0, 0))
    ov = R.Overlays(raw_points=np.array([[2.0, y, 0.5] for y in np.linspace(-1, 1, 50)]))
    off, s_off = R.render_scene((200, 200), cam, f, [], mode_layers("clean"), ov)
    on, s_on = R.render_scene((200, 200), cam, f, [], mode_layers("debug"), ov)
    assert s_off["raw"] == 0 and s_on["raw"] == 50 and not np.array_equal(off, on)


def test_layer_hides_kind():
    f = parse_row(row(0.0, [obj(status="stale", last=-0.1)]))
    items = DisplayState().update(f)
    cam = Camera((200, 200)).look_at((-3, 0, 2), (2, 0, 0))
    layers = mode_layers("clean")
    _, s = R.render_scene((200, 200), cam, f, items, layers)
    layers.stale_boxes = False
    _, s2 = R.render_scene((200, 200), cam, f, items, layers)
    assert s["stale"] == 1 and s2["stale"] == 0


# ---- synthetic rendering, replay consistency, video export ----
def test_synthetic_session_renders_all_frames():
    frames = list(read_frames(FIXTURE))
    s = Session(frames, mode="debug")
    for f in frames:
        img, stats = s.render(f.timestamp)
        assert img.shape == (1080, 1920, 3)
    assert stats["fresh"] == 1 and stats["stale"] == 1


def test_replay_is_deterministic_and_seek_consistent():
    frames = list(read_frames(FIXTURE))
    a = Session(frames)
    b = Session(frames)
    seq = [a.render(t)[0] for t in np.arange(0, 1.01, 0.1)]
    b.render(1.0)
    back = b.render(0.5)[0]  # seek backwards replays causally from the start
    assert np.array_equal(back[:, 768:], seq[5][:, 768:])  # 3D panel identical (left HUD shows render ms)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not installed")
def test_video_export_h264(tmp_path):
    from go2_perception.visualization.diagnostic_video import VideoWriter

    frames = list(read_frames(FIXTURE))
    s = Session(frames)
    w = VideoWriter(tmp_path / "fx.mp4", 10)
    for t in np.arange(0, 1.0, 0.1):
        w.write(s.render(t)[0])
    w.close()
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=codec_name,width,height,pix_fmt,nb_read_frames", "-of", "json", str(tmp_path / "fx.mp4")], capture_output=True, text=True, check=True)
    st = json.loads(probe.stdout)["streams"][0]
    assert (st["codec_name"], st["width"], st["height"], st["pix_fmt"], int(st["nb_read_frames"])) == ("h264", 1920, 1080, "yuv420p", 10)
    cap = cv2.VideoCapture(str(tmp_path / "fx.mp4"))
    ok, decoded = cap.read()
    assert ok and np.abs(decoded.astype(int) - s.render(0.0)[0].astype(int)).mean() < 6


# ---- privacy ----
def test_private_output_paths(tmp_path):
    assert is_private_path(tmp_path / "anything.mp4")  # outside any repository
    assert is_private_path(ROOT / "data" / "outputs" / "phase5b" / "x.mp4")
    assert not is_private_path(ROOT / "docs" / "x.mp4")
    assert not is_private_path(ROOT / "scripts" / "out")
    with pytest.raises(SystemExit):
        require_private_dir(ROOT / "docs" / "private_videos")


def test_fsd_viewer_refuses_tracked_output_for_recordings(tmp_path):
    r = subprocess.run(
        [".venv/bin/python" if Path(".venv/bin/python").exists() else "python", "scripts/fsd_viewer.py", "--recording", "nonexistent", "--data", str(tmp_path), "--output", "docs/leak.mp4"],
        capture_output=True, text=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"},
    )  # fmt: skip
    assert r.returncode != 0 and not Path("docs/leak.mp4").exists()


# ---- planner readiness: stale/missing never reads as free ----
def test_planning_query_never_returns_free():
    cells = [dict(center=[5.0, 5.0, 0.2], observation_count=3, last_observed=0.0, age_s=0.0, evidence_score=1.0)]
    f = parse_row(row(0.0, [obj(1, center=(1, 0, 0.4)), obj(2, status="stale", center=(3, 0, 0.4), last=-0.2)], cells=cells))
    assert planning.query(f, (1, 0)) == planning.OBJECT_FRESH
    assert planning.query(f, (3, 0)) == planning.OBJECT_STALE
    assert planning.query(f, (5, 5)) == planning.CELL
    assert planning.query(f, (9, 9)) == planning.UNKNOWN
    rep = planning.readiness([f])
    assert rep["stats"]["stale_objects"] == 1 and any(r["status"].startswith("gap") for r in rep["requirements"])


# ---- failure attribution (synthetic evidence) ----
def test_attribution_stages():
    base = dict(scan_n=10, ng03_n=20, proposals=[])
    assert attribute_miss(dict(base, scan_n=0, win_codes=[]))[0] == "absent_from_current_scan"
    assert attribute_miss(dict(base, win_codes=[0] * 12 + [4] * 2))[0] == "ground_or_background"
    assert attribute_miss(dict(base, win_codes=[4] * 5))[0] == "short_window_support"
    assert attribute_miss(dict(base, ng03_n=5, win_codes=[4] * 5))[0] == "min_support_thresholds"
    assert attribute_miss(dict(base, win_codes=[3] * 12))[0] == "aabb_proposal_rejection"
    assert attribute_miss(dict(base, win_codes=[4] * 12))[0] == "foreground_clustering"
    assert attribute_miss(dict(base, win_codes=[2] * 12, proposals=[dict(dist=0.2, support=1)]))[0] == "min_support_thresholds"
    assert attribute_miss(dict(base, win_codes=[2] * 12, proposals=[dict(dist=0.2, support=5)]))[0] == "lifetime_stale"
    g = dict(enough_history=True, speed_ok=False, displacement_ok=True, shift_ok=True, shape_ok=True)
    internals = dict(gate=g, history=6, span_s=0.5, age_s=1.0, speed=0.4, sigma=0.2, displacement=0.3, shift=0.05, shape_change=0.1, candidate="UNCERTAIN", streak=1)
    assert attribute_not_moving(internals)[0] == "motion_speed_gate"
    assert attribute_not_moving(dict(internals, gate=dict(g, enough_history=False)))[0] == "motion_object_too_young"
    assert in_cylinder(np.array([[0, 0, 1.0], [1, 0, 1.0], [0, 0, 3.0]]), (0, 0), 0.5, 0, 2).tolist() == [True, False, False]


def test_session_retrospective_layer_is_off_in_every_mode():
    frames = list(read_frames(FIXTURE))
    s = Session(frames, mode="debug")
    for m in ("clean", "hybrid", "debug"):
        s.set_mode(m)
        assert not s.layers.retrospective_gt
    s.layers.retrospective_gt = True
    s.set_mode("clean")
    assert s.layers.retrospective_gt  # explicit user choice survives a mode change


def test_fixture_rows_are_unmodified_by_parsing():
    r = rows()
    before = deepcopy(r)
    for x in r:
        parse_row(x)
    assert r == before
