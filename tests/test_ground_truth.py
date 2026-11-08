"""Ground-truth schema and evaluation on small synthetic sequences (no recordings needed)."""

import copy

import numpy as np
import pytest

from go2_perception.tracking import ground_truth as gt

BOX = [[2.0, -0.2], [2.4, -0.2], [2.4, 0.2], [2.0, 0.2]]


def doc(frames, objects=None):
    return dict(
        schema=gt.SCHEMA,
        recording="synthetic",
        frame="odom",
        clock_domain="go2_robot",
        t0=0.0,
        objects=objects or dict(person=dict(type="person"), box=dict(type="static_obstacle")),
        intervals=[dict(name="all", start=0, end=100)],
        frames=frames,
    )


def person(x, status="visible", motion="moving", velocity=(1.0, 0.0), visit=1):
    s = dict(id="person", status=status, xy=None, region=None, motion=motion, confidence="high")
    if status in gt.DETECTABLE:
        s.update(xy=[x, 1.0], velocity=list(velocity), visit=visit)
    return s


def box(status="visible"):
    return dict(
        id="box",
        status=status,
        xy=None,
        region=None if status in ("occluded", "out_of_view") else BOX,
        motion="stationary",
        confidence="high",
    )


def row(tid, x, y, status="C", motion="M", misses=0, v=(1.0, 0.0)):
    return [tid, x, y, 0.0, v[0], v[1], 0.1, status, motion, misses]


def walk(n=20, switch_at=10):
    frames, records = [], []
    for k in range(n):
        t = 0.1 * k
        frames.append(dict(t=t, objects=[person(0.1 * k)]))
        tid = 1 if k < switch_at else 2
        records.append(dict(t=t, tracks=[row(tid, 0.1 * k + 0.05, 1.0)]))
    return frames, records


def test_validate_rejects_bad_annotations():
    frames, _ = walk(3)
    good = doc(frames)
    gt.validate(good)
    bad = copy.deepcopy(good)
    bad["frames"][1]["objects"][0]["status"] = "hidden"
    with pytest.raises(ValueError):
        gt.validate(bad)
    bad = copy.deepcopy(good)
    bad["frames"][1]["objects"][0].update(status="out_of_view")
    with pytest.raises(ValueError, match="no position"):
        gt.validate(bad)
    bad = copy.deepcopy(good)
    bad["frames"][1]["objects"][0]["region"] = BOX
    with pytest.raises(ValueError, match="not both"):
        gt.validate(bad)
    bad = copy.deepcopy(good)
    bad["frames"][2]["t"] = bad["frames"][1]["t"]
    with pytest.raises(ValueError, match="increasing"):
        gt.validate(bad)


def test_region_distance():
    assert gt.region_distance((2.2, 0.0), BOX) == 0.0
    assert gt.region_distance((2.6, 0.0), BOX) == pytest.approx(0.2)
    assert gt.region_distance((1.7, 0.6), BOX) == pytest.approx(0.5)


def test_switch_idf1_and_velocity():
    frames, records = walk()
    r = gt.evaluate(doc(frames), records)
    p = r["objects"]["person"]
    assert p["coverage"] == 1.0
    assert p["switches"] == 1
    assert p["unique_ids"] == 2
    assert r["idf1"]["idf1"] == pytest.approx(0.5)  # IDTP 10 of 20 GT and 20 hypothesis frames
    assert p["position_error_median"] == pytest.approx(0.05)
    assert p["velocity_error_median"] == pytest.approx(0.0)
    assert p["moving_recall"] == 1.0


def test_unannotated_scene_is_ignored_but_duplicates_count():
    frames, records = walk(switch_at=99)
    for k, rec in enumerate(records):
        rec["tracks"].append(row(50, -3.0, -3.0))  # far from every annotated object
        if k < 5:
            rec["tracks"].append(row(60, 0.1 * k - 0.1, 1.1))  # second track on the person
    r = gt.evaluate(doc(frames), records)
    assert r["objects"]["person"]["duplicate_frames"] == 5
    assert r["idf1"]["idfp"] == 5
    assert r["idf1"]["idf1"] == pytest.approx(40 / 45)


def test_occlusion_excluded_and_recovery():
    frames, records = [], []
    for k in range(30):
        t = 0.1 * k
        occluded = 10 <= k < 18
        frames.append(dict(t=t, objects=[box("occluded" if occluded else "visible")]))
        tid = 7 if k < 10 or k >= 18 else 8
        records.append(dict(t=t, tracks=[] if occluded else [row(tid, 2.2, 0.0, motion="S", v=(0, 0))]))
    r = gt.evaluate(doc(frames, dict(box=dict(type="static_obstacle"))), records)
    b = r["objects"]["box"]
    assert b["detectable"] == 22 and b["coverage"] == 1.0
    assert b["switches"] == 0 and b["fragments"] == 0
    assert [e["kept"] for e in b["recovery"]] == [True]
    assert b["stationary_recall"] == 1.0 and b["static_speed_median"] == 0.0


def test_fragmentation_and_lost_recovery_failure():
    frames, records = [], []
    for k in range(30):
        t = 0.1 * k
        occluded = 10 <= k < 15
        frames.append(dict(t=t, objects=[box("occluded" if occluded else "visible")]))
        tracks = []
        if k < 10:
            tracks = [row(1, 2.2, 0.0)]
        elif 15 <= k < 20 or k >= 24:
            tracks = [row(2, 2.2, 0.0)]
        records.append(dict(t=t, tracks=tracks))
    r = gt.evaluate(doc(frames, dict(box=dict(type="static_obstacle"))), records)
    b = r["objects"]["box"]
    assert b["switches"] == 1
    assert b["fragments"] == 1  # 20-23 unmatched while visible, then matched again
    assert [e["kept"] for e in b["recovery"]] == [False]


def test_observed_selection_drops_coasting_tracks():
    frames, records = walk(switch_at=99)
    for rec in records[5:8]:
        rec["tracks"][0] = row(1, rec["tracks"][0][1], 1.0, status="L", misses=1)
    confirmed = gt.evaluate(doc(frames), records, "confirmed")["objects"]["person"]
    observed = gt.evaluate(doc(frames), records, "observed")["objects"]["person"]
    assert confirmed["matched"] == observed["matched"] == 17
    for rec in records[5:8]:
        rec["tracks"][0][7] = "C"
    assert gt.evaluate(doc(frames), records, "confirmed")["objects"]["person"]["matched"] == 20
    assert gt.evaluate(doc(frames), records, "observed")["objects"]["person"]["matched"] == 17


def test_motion_confusion_and_unlabelled_frames():
    frames, records = walk(switch_at=99)
    for k, f in enumerate(frames):
        f["objects"][0]["motion"] = "moving" if k < 10 else "stationary" if k < 15 else None
        records[k]["tracks"][0][8] = "M" if k % 2 == 0 else "U"
    p = gt.evaluate(doc(frames), records)["objects"]["person"]
    assert p["motion_moving"] == dict(M=5, S=0, U=5)
    assert p["motion_stationary"] == dict(M=3, S=0, U=2)
    assert p["moving_recall"] == 0.5
    assert p["moving_precision"] == pytest.approx(5 / 8)
    assert p["false_moving_rate"] == pytest.approx(0.6)


def test_per_visit_idf1_forgives_switch_across_exit():
    frames, records = [], []
    for k in range(20):
        t = 0.1 * k
        visit = 1 if k < 10 else 2
        out = 10 <= k < 12
        frames.append(dict(t=t, objects=[person(0.1 * k, "out_of_view" if out else "visible", visit=visit)]))
        records.append(dict(t=t, tracks=[] if out else [row(1 if k < 10 else 2, 0.1 * k, 1.0)]))
    r = gt.evaluate(doc(frames), records)
    assert r["idf1_per_visit"]["idf1"] == 1.0
    assert r["idf1"]["idf1"] < 1.0
    assert r["objects"]["person"]["recovery"] == []  # exits are not occlusion recoveries
    assert np.isclose(r["objects"]["person"]["coverage"], 1.0)


def test_previous_match_kept_while_inside_gate():
    frames, records = walk(switch_at=99)
    for k, rec in enumerate(records):
        if 5 <= k < 10:  # a second track appears closer to the person than track 1
            rec["tracks"].append(row(9, 0.1 * k, 1.0))
    p = gt.evaluate(doc(frames), records)["objects"]["person"]
    assert p["switches"] == 0 and p["unique_ids"] == 1 and p["duplicate_frames"] == 5
    matches, _ = gt.match_frame(
        [dict(person(0.5), type="person")], records[5]["tracks"], dict(person=0.5)
    )
    assert matches["person"][0] == 9  # without history the closest track wins


def test_vectorised_distances_match_scalar():
    rng = np.random.default_rng(3)
    points = rng.uniform(-1, 4, size=(200, 2))
    for state in (dict(xy=[1.0, 0.5]), dict(region=BOX), dict(region=BOX[::-1])):
        expected = [gt.state_distance(state, p) for p in points]
        assert np.allclose(gt.distances(state, points), expected)
