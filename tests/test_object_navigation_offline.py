from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from evaluate_object_navigation import CASES, BASE, covered_image, mask_history
from simulate_object_search import simulate
from go2_perception.vln.episode_grounding import ground, object_reference, remember_rejection
from go2_perception.vln.motion_profile import Hold


def test_rejections_keep_distinct_recent_reviewed_regions(tmp_path):
    rng = np.random.default_rng(27)
    refs = []
    roi = [0.2, 0.2, 0.8, 0.8]
    for i in range(2):
        path = tmp_path / f"texture{i}.png"
        Image.fromarray(rng.integers(0, 256, (180, 250), dtype=np.uint8)).save(path)
        refs.append(object_reference(path, roi, "box", "pilot"))
    recent = remember_rejection(remember_rejection([], refs[0]), refs[1])
    pred = dict(
        prediction_id="p",
        source_image={"file_sha256": refs[0]["sha256"]},
        target_fraction=[0.5, 0.5],
        opos_not_visible=False,
    )
    view = {
        "view": {
            "rectified": True,
            "distortion_model": "pinhole_radtan",
            "size_wh": [250, 180],
            "K": [[200, 0, 125]],
        }
    }
    e = ground(pred, tmp_path / "texture0.png", view, "box", None, recent, seek_view=True)
    assert e["method"] == "previously_rejected_instance"
    assert [m.get("mark_inside") for m in e["rejected_references"]] == [None, True]
    assert len(remember_rejection(recent, refs[0])) == 2
    for i in range(8):
        recent = remember_rejection(recent, dict(sha256=f"{i:064x}", roi=roi))
    assert [int(r["sha256"], 16) for r in recent] == [7, 6, 5, 4, 3]
    with pytest.raises(Hold, match="current_target_image_unavailable"):
        ground(pred, tmp_path / "missing.png", view, "box", refs[0], seek_view=True)


def test_controlled_cover_preserves_timestamps_and_source_files(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (100, 100), (10, 20, 30)).save(source)
    h = sha256(source.read_bytes()).hexdigest()
    frames = [
        dict(path=str(source), sha256=h, observation_s=t, available_s=t + 0.01) for t in (1.0, 1.25, 1.50)
    ]
    out = tmp_path / "covered"
    out.mkdir()
    altered = mask_history(frames, "partial", out)
    for a, f in zip(altered, frames, strict=True):
        assert a["observation_s"] == f["observation_s"] and a["available_s"] == f["available_s"]
        assert (
            Path(a["path"]).parent == out
            and sha256(Path(a["path"]).read_bytes()).hexdigest() == a["sha256"] != h
        )
        with Image.open(a["path"]) as image:
            assert image.getpixel((50, 50)) == (110, 110, 110) and image.getpixel((10, 10)) == (10, 20, 30)
            assert image.tobytes() == covered_image(source, "partial").tobytes()
    assert h == sha256(source.read_bytes()).hexdigest()
    assert len({c[0] for c in CASES}) == len(CASES) == 10 and len(BASE) == 40


def test_closed_loop_renders_after_actual_selected_motion(tmp_path):
    report = simulate(tmp_path)
    a, b, c = report["decisions"]
    assert [r["visible_in_renderer"] for r in (a, b, c)] == [False, True, True]
    assert [(r["selection"]["mode"], r["selection"]["action"]["direction"]) for r in (a, b)] == [
        ("SEARCH", "left"),
        ("APPROACH", "forward"),
    ]
    assert a["pose"]["yaw_rad"] < b["pose"]["yaw_rad"] and c["pose"]["x_m"] > b["pose"]["x_m"]
    assert a["capture_s"] < b["capture_s"] < c["capture_s"] and c["native_stop"]
    assert report["synthetic"] and report["moves"] == 2 and report["outcome"] == "STOP_ARRIVAL_UNVERIFIED"


def test_similar_cartons_require_agreement_on_the_enrolled_instance(tmp_path):
    # Both packages share a silhouette, shade and label layout. Only the printed texture differs.
    def carton(seed):
        tile = np.full((116, 110, 3), (160, 120, 80), np.uint8)
        tile[12:101, 8:102] = (205, 190, 157)
        ink = np.random.default_rng(seed).integers(0, 256, (67, 78, 1), np.uint8)
        tile[20:87, 16:94] = ink
        return tile

    scene = np.full((256, 448, 3), 47, np.uint8)
    scene[70:186, 70:180] = carton(12)
    original = tmp_path / "enroll.png"
    Image.fromarray(scene).save(original)
    ref = object_reference(original, [70 / 448, 70 / 256, 180 / 448, 186 / 256], "box", "pilot")
    scene[70:186, 270:380] = carton(30)
    current = tmp_path / "two-cartons.png"
    Image.fromarray(scene).save(current)
    view = dict(
        view=dict(
            rectified=True,
            distortion_model="pinhole_radtan",
            size_wh=[448, 256],
            K=[[300, 0, 224], [0, 300, 128], [0, 0, 1]],
        )
    )
    pred = dict(
        prediction_id="p",
        source_image={"file_sha256": sha256(current.read_bytes()).hexdigest()},
        opos_not_visible=False,
        target_fraction=[325 / 448, 0.5],
    )
    wrong = ground(pred, current, view, "box", ref, seek_view=True)
    assert wrong["status"] == "uncertain" and wrong["positive_reference"]["mark_inside"] is False
    correct = ground(dict(pred, target_fraction=[125 / 448, 0.5]), current, view, "box", ref, seek_view=True)
    assert correct["status"] == "supported" and correct["positive_reference"]["mark_inside"] is True
