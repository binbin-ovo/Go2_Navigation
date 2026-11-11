"""Recovery safety/semantics regressions; synthetic truths never enter runtime."""

from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from go2_perception.io.config import load_config
from go2_perception.scene.background import BackgroundMap
from go2_perception.scene.fast_background import FastBackgroundMap
from go2_perception.scene.coverage import return_coverage
from go2_perception.scene.recovery import RecoveryScene
from go2_perception.scene.recovery_objects import RecoveryObjects
from go2_perception.scene.synthetic import sequence
from go2_perception.viewer.scene import parse_row
from go2_perception.scene.serialization import validate_scene


def proposal(center, dims=(0.4, 0.4, 1.6)):
    return dict(
        center=np.array(center, float), dimensions=np.array(dims), point_count=40
    )


def test_vectorized_map_mask_matches_reference():
    rng = np.random.default_rng(19)
    old, new = BackgroundMap(), FastBackgroundMap()
    p = rng.uniform(-2, 2, (1200, 3))
    for i, t in enumerate([0, 0.1, 0.25, 0.4, 2.8]):
        for m in [old, new]:
            m.update(p, t, str(i), np.zeros(3))
        q = np.vstack([p, p + 0.01, rng.uniform(-2, 2, (500, 3))])
        np.testing.assert_array_equal(old.stable_mask(q, t), new.stable_mask(q, t))
        assert old.snapshot(t) == new.snapshot(t)
    assert new.stable_mask(np.empty((0, 3)), 3).shape == (0,)


def test_fast_scalar_age_validation_matches_numpy_tolerance():
    row = json.loads(
        Path("tests/fixtures/scene/renderer.jsonl").read_text().splitlines()[-1]
    )
    for age in [0, 0.2, 1, 100]:
        for delta in [0, 1e-8 + 1e-5 * age, (1e-8 + 1e-5 * age) * 1.01]:
            r = deepcopy(row)
            c = r["background"]["cells"][0]
            c["last_observed"] = r["timestamp"] - age
            c["age_s"] = age + delta
            if np.isclose(c["age_s"], r["timestamp"] - c["last_observed"]):
                validate_scene(r)
            else:
                with pytest.raises(ValueError):
                    validate_scene(r)


@pytest.mark.parametrize(
    "yaw,pitch,height", [(0, 0, 0.32), (0.7, 0, 0.32), (0.3, 0.2, 0.20)]
)
def test_body_filter_frame_and_nearby_obstacle(yaw, pitch, height):
    c = load_config("configs/detection_recovery.yaml")
    s = RecoveryScene(c)
    body = np.array([[0, 0, 0], [0.26, 0, 0], [0, 0.13, 0], [0, 0, -0.21]])
    raw = body.copy()
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("zy", [yaw, pitch]).as_matrix()
    m[:3, 3] = [1, 2, height]
    r = s.process(body, 0, "a", lambda t: m, clock_domain="synthetic")
    np.testing.assert_array_equal(body, raw)
    assert r["diagnostics"]["self_excluded_points"] == 1
    assert len(s.accumulator.snapshot().xyz) == 3
    np.testing.assert_allclose(
        s.accumulator.snapshot().xyz, body[1:] @ m[:3, :3].T + m[:3, 3]
    )


def test_absent_sector_not_proof_of_occlusion_or_free_space():
    m = np.eye(4)
    absent = return_coverage(np.array([[0, 2, 0]]), [2, 0, 0], [0.4, 0.4, 1], m)
    foreground = return_coverage(np.array([[1, 0, 0]]), [2, 0, 0], [0.4, 0.4, 1], m)
    behind = return_coverage(np.array([[3, 0, 0]]), [2, 0, 0], [0.4, 0.4, 1], m)
    assert absent["state"] == "no_sector_returns"
    assert foreground["state"] == "foreground_returns_occlusion_possible"
    assert behind["state"] == "sector_returns_without_object"
    assert not any(x["proves_visible"] for x in [absent, foreground, behind])


def test_sampling_gap_never_refreshes_v1_observation_and_viewer_parses():
    s = RecoveryScene(load_config("configs/detection_recovery.yaml"))
    f = next(sequence("sampling"))
    first = s.process(f["points"], 0, "a", f["pose_lookup"], clock_domain="synthetic")
    assert first["objects"]
    last = first["objects"][0]["last_observed"]
    r = s.process(
        np.empty((0, 3)), 0.065, "b", f["pose_lookup"], clock_domain="synthetic"
    )
    parse_row(r)
    assert r["objects"]
    for o in r["objects"]:
        assert o["status"] == "stale" and o["observed_box"] is None
        assert o["last_observed"] == last
        assert o["evidence_state"] == "observed_recently"
        assert o["sector_evidence"]["state"] == "no_sector_returns"
    for i in range(1, 12):
        r = s.process(
            np.empty((0, 3)),
            0.065 + i * 0.065,
            str(i + 1),
            f["pose_lookup"],
            clock_domain="synthetic",
        )
    assert not r["objects"]


def test_start_stop_restart_and_sparse_observations():
    o = RecoveryObjects(ttl_s=1.0)
    states = []
    for i in range(180):
        t = i * 0.065
        x = min(t, 3) * 0.6 + max(t - 6, 0) * 0.6
        o.update([proposal([2, x, 0.8])] if i % 3 != 1 else [], t)
        r = o.snapshot(t)
        states.append(r[0]["motion_state"] if r else "UNCERTAIN")
    assert "MOVING" in states[20:40]
    assert "STATIONARY" in states[75:90]
    assert "MOVING" in states[115:140]


def test_no_motion_history_transfer_person_to_low_box():
    o = RecoveryObjects(ttl_s=1.0)
    for i in range(20):
        o.update([proposal([2, i * 0.04, 0.8])], i * 0.065)
    identity = next(iter(o.objects))
    o.update([], 1.35)
    o.update([proposal([2, 0.8, 0.2], (0.4, 0.4, 0.3))], 1.5)
    assert len(o.objects) == 2
    newest = max(o.objects)
    assert (
        newest != identity
        and o.objects[newest]["recovery_motion"]["state"] == "UNCERTAIN"
    )


def test_ambiguous_crossing_discards_motion_evidence():
    o = RecoveryObjects(ttl_s=1.0)
    for i in range(10):
        o.update(
            [proposal([2, -0.6 + i * 0.04, 0.8]), proposal([2, 0.6 - i * 0.04, 0.8])],
            i * 0.065,
        )
    o.update([proposal([2, -0.05, 0.8]), proposal([2, 0.05, 0.8])], 0.65)
    assert all(v["state"] == "UNCERTAIN" for v in o.objects.values())


@pytest.mark.parametrize("kind", ["gap", "missing", "rewind", "clock"])
def test_recovery_reset_clears_direct_history(kind):
    s = RecoveryScene(load_config("configs/detection_recovery.yaml"))
    f = next(sequence("rotation"))
    s.process(f["points"], 1, "a", f["pose_lookup"], clock_domain="synthetic")
    t = {"gap": 2, "rewind": 0.5}.get(kind, 1.1)
    r = s.process(
        f["points"],
        t,
        "b",
        lambda t: None if kind == "missing" else f["pose_lookup"](t),
        clock_domain="changed" if kind == "clock" else "synthetic",
    )
    assert r["epoch"] == 1
    assert all(o["motion_state"] == "UNCERTAIN" for o in r["objects"])


def test_recent_support_plus_one_new_return_updates_existence_not_motion_history():
    s = RecoveryScene(load_config("configs/detection_recovery.yaml"))
    f = next(sequence("sampling"))
    first = s.process(f["points"], 0, "a", f["pose_lookup"], clock_domain="synthetic")
    obj = first["objects"][0]
    center = np.asarray(obj["center"])
    m = f["pose_lookup"](0.065)
    raw = ((center - m[:3, 3]) @ m[:3, :3])[None, :]
    r = s.process(raw, 0.065, "b", f["pose_lookup"], clock_domain="synthetic")
    observed = [o for o in r["objects"] if o["status"] == "observed_filtered"]
    assert observed and observed[0]["observed_box"]["point_count"] == 1
    assert observed[0]["last_observed"] == 0.065
    assert observed[0]["motion_state"] == "UNCERTAIN"


def test_motion_history_survives_unambiguous_short_gap():
    o = RecoveryObjects(ttl_s=1.0)
    for i in range(20):
        o.update([proposal([2, i * 0.0325, 0.8])], i * 0.065)
    identity = next(iter(o.objects))
    prior = len(o.objects[identity]["direct_history"])
    o.update([], 1.1)
    assert o.snapshot(1.1)[0]["motion_state"] == "UNCERTAIN"
    o.update([proposal([2, 0.7, 0.8])], 1.4)
    assert list(o.objects) == [identity]
    assert len(o.objects[identity]["direct_history"]) >= prior - 7
    assert o.objects[identity]["recovery_motion"]["valid"]


@pytest.mark.parametrize(
    "scenario", ["sampling", "rotation", "translation", "sparse_occlusion"]
)
def test_stationary_recovery_with_changing_samples_and_ego_motion(scenario):
    s = RecoveryScene(load_config("configs/detection_recovery.yaml"))
    states = []
    for i, f in enumerate(sequence(scenario, frames=45)):
        r = s.process(
            f["points"],
            f["timestamp"],
            str(i),
            f["pose_lookup"],
            clock_domain="synthetic",
        )
        parse_row(r)
        states.extend(o["motion_state"] for o in r["objects"])
        assert all(np.isfinite(o["velocity"]).all() for o in r["objects"])
    assert states and "MOVING" not in states


def test_ambiguous_single_return_cannot_keep_old_motion_state():
    o = RecoveryObjects(ttl_s=1.0)
    for i in range(20):
        o.update(
            [proposal([2, i * 0.04, 0.8]), proposal([2, 1.8 - i * 0.04, 0.8])],
            i * 0.065,
        )
    # Both estimates compete for almost coincident partial returns.
    parts = [proposal([2, 0.89, 0.8]), proposal([2, 0.91, 0.8])]
    for p in parts:
        p["point_count"] = 1
    o.update(parts, 1.3)
    assert all(x["motion_state"] == "UNCERTAIN" for x in o.snapshot(1.3))


def test_optional_extension_and_original_schema_accept_runtime_and_old_fixture():
    import jsonschema

    validators = [
        jsonschema.Draft202012Validator(json.loads(Path(p).read_text()))
        for p in [
            "schemas/go2-scene-v1.schema.json",
            "schemas/go2-scene-v1-recovery.schema.json",
        ]
    ]
    s = RecoveryScene(load_config("configs/detection_recovery.yaml"))
    f = next(sequence("sampling"))
    row = s.process(f["points"], 0, "a", f["pose_lookup"], clock_domain="synthetic")
    old = json.loads(
        Path("tests/fixtures/scene/renderer.jsonl").read_text().splitlines()[0]
    )
    for v in validators:
        v.validate(row)
        v.validate(old)
    invalid = deepcopy(row)
    invalid["objects"][0]["sector_evidence"]["proves_visible"] = True
    with pytest.raises(jsonschema.ValidationError):
        validators[1].validate(invalid)


def test_scalar_validator_rejects_overflowed_age_difference():
    r = json.loads(
        Path("tests/fixtures/scene/renderer.jsonl").read_text().splitlines()[0]
    )
    r["timestamp"] = 1e308
    r["robot_pose"] = None
    r["objects"] = []
    for c in r["background"]["cells"]:
        c["last_observed"] = -1e308
        c["age_s"] = 1e308
    with pytest.raises(ValueError):
        validate_scene(r)
