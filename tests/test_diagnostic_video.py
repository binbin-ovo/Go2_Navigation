"""Diagnostic video rendering, event heuristics, annotation edits and privacy on synthetic data."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from go2_perception.tracking import annotation_edits as ae
from go2_perception.tracking import ground_truth as gt
from go2_perception.visualization import diagnostic_video as dv


def track(i, x, y, status="C", motion="S", misses=0, v=(0.0, 0.0)):
    return dv.Track(i, x, y, v[0], v[1], status, motion, misses)


def frame(**kw):
    base = dict(
        t=100.0, t_rel=1.0, frame="odom", robot_pose=np.eye(3), robot_path=np.array([[0, 0], [0.1, 0]]),
        points=np.array([[1.0, 0.0], [2.0, 0.5]]), clusters=[(np.array([1.8, 0.3]), np.array([2.2, 0.7]))],
        tracks=[track(1, 2.0, 0.5)],
    )  # fmt: skip
    base.update(kw)
    return dv.FrameData(**base)


def view():
    return dv.BevView((1.5, 0.0), 0.0, 3.0, (dv.BEV_W, dv.TOP_H))


def bev_pixel(img, xy, v=None):
    v = v or view()
    u, w = np.int32(v.px(xy)[0])
    return img[w, u + dv.RGB_W]


def test_align_rgb_maps_host_receipt_and_flags_missing_frames():
    lidar = [10.0, 10.065, 10.13, 11.0]
    host = [110.002, 110.07, 110.5]  # host = robot + 100
    idx, dt = dv.align_rgb(lidar, host, host_offset=100.0, max_dt=0.15)
    assert idx[:3] == [0, 1, 1]
    assert dt[0] == pytest.approx(0.002) and dt[2] == pytest.approx(-0.06)
    assert idx[3] is None  # nearest camera frame 0.5 s away: missing, not stretched
    assert dv.align_rgb([1.0], [], 0.0) == ([None], [None])


def test_view_round_trip_and_heading_up():
    v = dv.BevView((2.0, 1.0), np.radians(30), 3.0, (800, 800), origin=(100, 0))
    for xy in ((2.0, 1.0), (3.5, -0.2), (0.0, 4.0)):
        assert np.allclose(v.inverse(v.px(xy)[0]), xy)
    ahead = v.px((2.0 + np.cos(np.radians(30)), 1.0 + np.sin(np.radians(30))))[0]
    centre = v.px((2.0, 1.0))[0]
    assert ahead[1] < centre[1] and abs(ahead[0] - centre[0]) < 1e-6  # heading points up
    assert "frame: odom" in dv.frame_label("odom", 5.0)


def test_compose_is_full_hd_and_draws_observed_vs_coasting():
    tracks = [track(1, 2.0, 0.5, "C", "M", 0, (0.5, 0.0)), track(2, 1.3, -0.7, "L", "S", 3), track(3, 1.5, 1.0, "T")]
    img = dv.compose(frame(tracks=tracks, key_ids={1, 2, 3}), view())
    assert img.shape == (dv.HEIGHT, dv.WIDTH, 3) and img.dtype == np.uint8
    assert tuple(bev_pixel(img, (2.0, 0.5))) == dv.ORANGE  # observed: filled, motion colour
    assert tuple(bev_pixel(img, (1.3, -0.7))) == (255, 255, 255)  # coasting: hollow centre


def test_lost_and_coasting_are_not_observed():
    assert track(1, 0, 0, "C", misses=0).observed
    assert not track(1, 0, 0, "C", misses=2).observed
    assert not track(1, 0, 0, "L", misses=0).observed


def test_missing_camera_and_odometry_still_render():
    f = frame(rgb=None, robot_pose=None, robot_path=np.empty((0, 2)), odometry_gap=True)
    img = dv.compose(f, view())
    assert img.shape == (dv.HEIGHT, dv.WIDTH, 3)
    # status bar shows a red warning box on the right
    assert (img[dv.TOP_H + 10 : dv.TOP_H + 40, dv.WIDTH - 420 : dv.WIDTH - 20] == (60, 60, 220)).all(axis=-1).any()
    # RGB panel has no image: only the dark background and text
    assert (img[100:600, 600:1000] == (30, 30, 30)).all(axis=-1).mean() > 0.95


def test_camera_frame_and_detection_are_drawn():
    rgb = np.full((720, 1280, 3), 200, np.uint8)
    f = frame(rgb=rgb, rgb_index=7, rgb_time=100.01, rgb_detections=[("person", 0.9, (100, 100, 400, 600))])
    img = dv.compose(f, view())
    s = min(dv.RGB_W / 1280, (dv.TOP_H - 80) / 720)
    assert tuple(img[40 + int(300 * s), int(100 * s)]) == (0, 220, 255)  # person box edge


def test_track_trails_persist_per_id_and_expire():
    trails = {}
    for k in range(5):
        shown = dv.update_trails(trails, k * 0.5, [track(7, k * 0.1, 0.0), track(8, 1.0, 1.0, misses=k % 2)])
    assert [p[0] for p in shown[7]] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert [p[2] for p in shown[8]] == [True, False, True, False, True]
    shown = dv.update_trails(trails, 2.5, [track(8, 1.0, 1.0)])
    assert 7 not in shown and 7 in trails  # vanished ID keeps its history briefly
    dv.update_trails(trails, 6.0, [])
    assert 7 not in trails


def test_cv_forecast_and_overlay_pixels():
    t = track(1, 1.0, 0.0, v=(0.5, 0.25))
    pred = dv.cv_forecast(t)
    assert pred.shape == (12, 2)
    assert np.allclose(pred[-1], (1.0 + 0.5 * 4.8, 0.25 * 4.8))
    fc = dict(origin_rel=0.5, gt_history=[[0.0, 0.0], [1.0, 0.0]], future=[[1.4, 0.0]], cv_gt=[[1.4, 0.0], [1.8, 0.0]], cv_tracker=None)
    img = dv.compose(frame(tracks=[], forecast=fc), view())
    assert tuple(bev_pixel(img, (1.4, 0.0))) == dv.MAGENTA


def test_heuristic_events_lost_fragmentation_duplicate():
    recs = []
    for k in range(60):
        t = k * 0.1
        rows = []
        if k < 30:
            rows.append([1, 2.0, 0.0, 0, 0, 0, 0.1, "C", "S", 0 if k < 20 else k - 19])
        if k >= 32:
            rows.append([2, 2.1, 0.0, 0, 0, 0, 0.1, "C", "S", 0])
        if k >= 40:
            rows.append([3, 2.2, 0.05, 0, 0, 0, 0.1, "C", "S", 0])
        recs.append(dict(t=t, tracks=rows))
    events = dv.heuristic_events(recs, min_life_s=1.0)
    kinds = [(e.kind, e.ids) for e in events]
    assert ("NEW TRACK", (1,)) in kinds
    assert ("TRACK LOST", (1,)) in kinds
    assert ("TRACK ENDED", (1,)) in kinds
    assert ("POSSIBLE FRAGMENTATION", (1, 2)) in kinds
    assert ("POSSIBLE DUPLICATE", (2, 3)) in kinds
    assert not any(e.confirmed for e in events)  # heuristics are never confirmed


def test_video_export_opens_as_h264(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg not installed")
    path = tmp_path / "synthetic.mp4"
    w = dv.VideoWriter(path, 10)
    for k in range(12):
        f = frame(t_rel=k * 0.1, tracks=[track(1, 2.0 + 0.05 * k, 0.5, "C", "M", 0, (0.5, 0))])
        w.write(dv.compose(f, view()))
    with pytest.raises(ValueError):
        w.write(np.zeros((10, 10, 3), np.uint8))
    w.close()
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_read_frames", "-of", "json", str(path)
    ]))["streams"][0]  # fmt: skip
    assert (probe["codec_name"], probe["pix_fmt"], probe["width"], probe["height"]) == ("h264", "yuv420p", 1920, 1080)
    assert int(probe["nb_read_frames"]) == 12


def test_private_output_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / ".gitignore").write_text("data/outputs/*\n")
    assert dv.private_output_ok(repo / "data/outputs/mac/videos/x.mp4", repo)
    assert not dv.private_output_ok(repo / "docs/media/x.mp4", repo)
    assert dv.private_output_ok(tmp_path / "elsewhere.mp4", repo)


def annotation_doc():
    frames = []
    for k in range(10):
        frames.append(dict(t=100 + 0.1 * k, objects=[
            dict(id="person", status="visible", xy=[1.0 + 0.1 * k, 0.0], region=None, motion="moving", confidence="high", visit=1, velocity=[1.0, 0.0]),
        ]))  # fmt: skip
    return dict(schema=gt.SCHEMA, recording="synthetic", frame="odom", clock_domain="go2_robot", t0=100.0,
                objects=dict(person=dict(type="person")), intervals=[dict(name="all", start=0, end=1)], frames=frames)  # fmt: skip


def test_csv_round_trip_turns_edits_into_review_entries(tmp_path):
    doc = gt.validate(annotation_doc())
    path = tmp_path / "a.csv"
    ae.export_csv(doc, path)
    assert ae.import_csv(doc, path, "tester") == []
    lines = path.read_text().splitlines()
    cells = lines[3].split(",")
    cells[4], cells[5], cells[6] = "occluded", "", ""  # status, x, y of frame 2
    lines[3] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n")
    entries = ae.import_csv(doc, path, "tester")
    assert len(entries) == 1 and entries[0]["set"] == {"status": "occluded", "xy": None}
    frames = {f["t"]: f["objects"] for f in doc["frames"]}
    ae.apply_review(frames, entries, doc["t0"])
    s = doc["frames"][2]["objects"][0]
    assert s["status"] == "occluded" and s["xy"] is None and "velocity" not in s
    assert s["human_reviewed"] == "tester"
    gt.validate(doc)


def test_edit_session_ranges_rename_unidentifiable_and_scoring(tmp_path):
    doc = gt.validate(annotation_doc())
    session = ae.EditSession(doc, "tester")
    session.k = 2
    session.key("b")
    session.k = 4
    session.key("u")  # frames 2-4 unidentifiable
    session.k = 6
    session.key("r", "person_2")  # identity correction at frame 6
    session.k = 8
    session.key("\t")  # back to "person"
    session.click((5.0, 5.0))
    states = [f["objects"][0] for f in session.doc["frames"]]
    assert [bool(s.get("identity_uncertain")) for s in states] == [False, False, True, True, True] + [False] * 5
    assert states[6]["id"] == "person_2" and "person_2" in session.doc["objects"]
    assert states[8]["xy"] == [5.0, 5.0]
    assert not gt.scorable(states[3]) and gt.scorable(states[5])
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(dict(recording="synthetic", objects=dict(person=dict(type="person")))))
    assert session.save(spec) == 3
    saved = json.loads(spec.read_text())
    assert saved["objects"]["person_2"]["manual"] is True
    assert all(e["reviewer"] == "tester" for e in saved["review"])
    # unidentifiable frames are don't-care for scoring: no track there, no penalty
    records = [dict(t=f["t"], tracks=[] if 2 <= k <= 4 else [[1, *f["objects"][0]["xy"], 0, 1.0, 0, 0.1, "C", "M", 0]])
               for k, f in enumerate(session.doc["frames"])]  # fmt: skip
    r = gt.evaluate(session.doc, records)
    assert r["objects"]["person"]["coverage"] == 1.0
    with pytest.raises(ValueError):
        gt.validate(dict(doc, frames=[dict(t=1.0, objects=[dict(states[0], identity_uncertain="yes")])]))
