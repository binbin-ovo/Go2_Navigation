from copy import deepcopy
import numpy as np
import pytest
from go2_perception.forecasting.data import Window
from go2_perception.forecasting.social_data import context_arrays, social_dataset
from go2_perception.io.config import load_config


def example():
    t = np.arange(12) * 0.4
    p = np.column_stack((t, t * 0))
    w = Window("scene:1", "scene", "file", 1, t, p[:8], p[8:])
    tracks = {
        1: (t, p),
        2: (t, p + [0, 1]),
        3: (t[5:], p[5:] + [0, 2]),
        4: (t, p + [0, 20]),
        5: (t[8:], p[8:] + [0, 0.1]),
    }
    return [w], {"file": tracks}


def smoke_config():
    c = load_config("configs/forecasting_smoke.yaml")
    c["social"] = dict(radius=3.0, max_neighbors=3)
    c["augmentation"] = dict(noise_std=0.0, mask_probability=0.0)
    return c


def test_simultaneous_neighbors_partial_and_radius():
    w, r = example()
    b = context_arrays(w, r, max_neighbors=3)
    assert b.neighbor_ids.tolist() == [[2, 3, -1]]
    assert b.neighbor_mask[0, 1].tolist() == [False] * 5 + [True] * 3
    np.testing.assert_array_equal(b.times[0], w[0].times[:8])
    assert b.counts.tolist() == [2]
    assert (b.relative_positions[~b.neighbor_mask] == 0).all()
    assert context_arrays(w, r, radius=1.5).counts.tolist() == [1]
    assert context_arrays(w, r, max_neighbors=1).neighbor_ids.tolist() == [[2]]


def test_future_neighbor_values_and_availability_do_not_leak():
    w, r = example()
    a = context_arrays(w, r)
    r2 = deepcopy(r)
    for identity, (t, xy) in r2["file"].items():
        xy[t > w[0].times[7]] = 99999
    b = context_arrays(w, r2)
    np.testing.assert_array_equal(a.neighbors, b.neighbors)
    r3 = {
        name: {
            i: (t[t <= 2.8 + 1e-8], xy[t <= 2.8 + 1e-8])
            for i, (t, xy) in tracks.items()
        }
        for name, tracks in r.items()
    }
    c = context_arrays(w, r3)
    np.testing.assert_array_equal(a.neighbors, c.neighbors)
    np.testing.assert_array_equal(a.neighbor_mask, c.neighbor_mask)


def test_no_neighbor_and_translation():
    w, r = example()
    a = context_arrays(w, r, radius=0.01)
    assert not a.neighbor_mask.any()
    assert context_arrays(w, r, max_neighbors=0).neighbors.shape == (1, 0, 8, 2)
    a = context_arrays(w, r)
    offset = np.array([30.0, -20.0])
    w[0].observed = w[0].observed + offset
    for t, xy in r["file"].values():
        xy += offset
    b = context_arrays(w, r)
    np.testing.assert_allclose(a.relative_positions, b.relative_positions, atol=4e-6)


def test_social_retains_exact_phase4_windows():
    from go2_perception.forecasting.data import dataset

    c = smoke_config()
    w, b, m = social_dataset(c, "train")
    old, meta = dataset(c["dataset"], "train")
    assert [v.key for v in w] == [v.key for v in old]
    assert m["window_keys_sha256"] == meta["window_keys_sha256"]
    np.testing.assert_array_equal(
        b.future, np.stack([v.future for v in old]).astype("float32")
    )


@pytest.mark.parametrize("radius,count", [(0, 2), (float("nan"), 2), (2, -1), (2, 1.5)])
def test_invalid_selection(radius, count):
    with pytest.raises(ValueError):
        context_arrays(*example(), radius=radius, max_neighbors=count)


def tiny_model():
    import torch
    from go2_perception.forecasting.social_model import SocialTransformer

    torch.manual_seed(7)
    return SocialTransformer(
        obs_steps=8,
        pred_steps=4,
        d_model=16,
        heads=2,
        layers=1,
        feedforward=32,
        dropout=0,
    ).eval()


def tensors():
    import torch

    b = context_arrays(*example(), max_neighbors=3)
    return tuple(
        torch.from_numpy(x)
        for x in (b.observed, b.neighbors, b.neighbor_mask, b.target_mask)
    )


def test_shapes_permutation_and_padding_invariance():
    import torch

    m = tiny_model()
    x, n, mask, tm = tensors()
    y = m(x, n, mask, tm)
    assert y.shape == (1, 4, 2)
    torch.testing.assert_close(
        y, m(x, n[:, [2, 0, 1]], mask[:, [2, 0, 1]], tm), atol=2e-6, rtol=1e-6
    )
    n[~mask] = float("nan")
    torch.testing.assert_close(y, m(x, n, mask, tm))
    torch.testing.assert_close(m(x), m(x, n, torch.zeros_like(mask), tm))
    torch.testing.assert_close(m(x), m(x, n[:, :0], mask[:, :0], tm))


def test_translation_equivariance_masked_inputs_and_gradient():
    import torch

    m = tiny_model()
    x, n, mask, tm = tensors()
    tm[:, 3] = False
    delta = torch.tensor([12.0, -15.0])
    y = m(x, n, mask, tm)
    torch.testing.assert_close(
        m(x + delta, n + delta, mask, tm), y + delta, atol=3e-6, rtol=1e-6
    )
    y.sum().backward()
    assert m.relative_embedding.weight.grad.abs().sum() > 0
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )


def test_invalid_social_masks():
    m = tiny_model()
    x, n, mask, tm = tensors()
    mask[:, 0, -1] = False
    with pytest.raises(ValueError):
        m(x, n, mask, tm)
    with pytest.raises(ValueError):
        m(x, n, None, tm)
    tm[:, -1] = False
    with pytest.raises(ValueError):
        m(x, target_mask=tm)


def test_social_checkpoint_train_without_test_and_repeatability(tmp_path):
    import shutil
    import torch
    from go2_perception.forecasting.social_training import (
        train_social,
        load_social_checkpoint,
        predict_social,
        inputs,
    )

    c = smoke_config()
    c["training"]["epochs"] = 2
    shutil.copytree(c["dataset"]["root"], tmp_path / "data")
    (tmp_path / "data/test_scene.txt").unlink()
    c["dataset"]["root"] = str(tmp_path / "data")
    path = train_social(c, tmp_path / "first", make_plots=False)
    second = train_social(c, tmp_path / "repeat", make_plots=False)
    m, p = load_social_checkpoint(path)
    _, q = load_social_checkpoint(second)
    assert p["model_version"] == "social-temporal-transformer-v1"
    for key in p["state_dict"]:
        torch.testing.assert_close(
            p["state_dict"][key], q["state_dict"][key], rtol=0, atol=0
        )
    _, b, _ = social_dataset(c, "validation")
    y = predict_social(m, inputs(b), "cpu")
    assert y.shape == b.future.shape and np.isfinite(y).all()
    with pytest.raises(FileExistsError):
        train_social(c, tmp_path / "first", make_plots=False)
    p["model_version"] = "bad"
    torch.save(p, tmp_path / "bad.pt")
    with pytest.raises(ValueError):
        load_social_checkpoint(tmp_path / "bad.pt")


def test_perturbations_are_observation_only_and_paired():
    from go2_perception.forecasting.social_evaluation import perturb

    b = context_arrays(*example())
    for mode in ["clean", "noise", "missing", "partial_neighbors", "no_neighbors"]:
        c, repair = perturb(b, mode)
        np.testing.assert_array_equal(c.future, b.future)
        assert np.isfinite(repair).all()
        assert c.target_mask[:, -1].all()
    c, repair = perturb(b, "missing")
    assert not c.target_mask[:, 4].any()
    np.testing.assert_allclose(repair[:, 4], (b.observed[:, 3] + b.observed[:, 5]) / 2)
    c, _ = perturb(b, "no_neighbors")
    assert not c.neighbor_mask.any()


def tracks():
    from go2_perception.core.tracking_types import TrackOutput

    def one(identifier, first, offset):
        times = np.arange(first, 8) * 0.4
        h = [
            dict(timestamp=float(t), position=[float(t), offset, 0.0], observed=True)
            for t in times
        ]
        return TrackOutput(
            identifier,
            float(times[-1]),
            "odom",
            "sim",
            "odometry",
            np.array(h[-1]["position"]),
            np.array([1.0, 0]),
            np.eye(4),
            None,
            None,
            "person",
            None,
            "Confirmed",
            "MOVING",
            float(times[-1]),
            float(times[-1]),
            len(h),
            0,
            h,
            [],
        )

    return one(1, 0, 0), one(2, 4, 1.0)


def test_multitrack_adapter_partial_neighbors_and_forecast():
    from go2_perception.forecasting.social_adapter import (
        multi_track_input,
        forecast_multitrack,
    )

    a, b = tracks()
    c = multi_track_input(a, [a, b], 8, 0.4, validated_ids={1, 2})
    assert (
        c.neighbor_ids[0] == 2
        and c.neighbor_mask[0].tolist() == [False] * 4 + [True] * 4
    )
    f, c = forecast_multitrack(
        tiny_model(), a, [b], 0.4, model_id="test", validated_ids={1, 2}
    )
    assert f.track_id == a.track_id and f.frame_id == "odom" and len(f.xy) == 4
    assert f.experimental and not hasattr(f, "confidence")
    assert np.isfinite(f.xy).all()


@pytest.mark.parametrize(
    "bad",
    [
        "identity",
        "clock",
        "timestamp",
        "frame",
        "semantics",
        "future",
        "coasting",
        "fragment",
    ],
)
def test_multitrack_rejects_bad_neighbors(bad):
    from go2_perception.forecasting.social_adapter import multi_track_input

    a, b = tracks()
    validated = {1, 2}
    if bad == "identity":
        validated = {1}
    elif bad == "clock":
        b.clock_domain = "different"
    elif bad == "timestamp":
        b.timestamp += 0.1
    elif bad == "frame":
        b.frame_id = "map"
    elif bad == "semantics":
        b.class_name = None
    elif bad == "future":
        b.history[-1]["timestamp"] += 1
    elif bad == "coasting":
        b.history[-1]["observed"] = False
    elif bad == "fragment":
        b.history[0]["track_id"] = 9
    c = multi_track_input(a, [b], 8, 0.4, validated_ids=validated)
    assert not c.neighbor_mask.any() and "2" in c.rejected


def test_multitrack_target_requires_attestation_and_unique_ids():
    from go2_perception.forecasting.social_adapter import multi_track_input

    a, b = tracks()
    with pytest.raises(ValueError):
        multi_track_input(a, [b], 8, 0.4, validated_ids={2})
    with pytest.raises(ValueError):
        multi_track_input(a, [b, b], 8, 0.4, validated_ids={1, 2})
    with pytest.raises(ValueError):
        multi_track_input(a, [b], 8, 0.4, validated_ids={1, 2}, units="cm")


def test_end_to_end_metrics_and_visualization(tmp_path):
    pytest.importorskip("matplotlib")
    from go2_perception.forecasting.social_training import train_social
    from go2_perception.forecasting.social_evaluation import evaluate_social
    from go2_perception.forecasting.social_visualization import plot_social_examples

    c = smoke_config()
    c["training"]["epochs"] = 1
    path = train_social(c, tmp_path / "train", make_plots=False)
    r = evaluate_social(path, tmp_path / "evaluation", split="test")
    assert r["models"]["social"]["transformer"]["count"] == 27
    assert sum(v["count"] for v in r["crowding"].values()) == 27
    data = np.load(tmp_path / "evaluation/predictions.npz")
    from go2_perception.forecasting.metrics import summarize

    assert r["models"]["social"]["transformer"] == summarize(
        data["social"], data["target"]
    )
    selection = plot_social_examples(
        tmp_path / "evaluation/predictions.npz", tmp_path / "plots"
    )
    assert selection and all(
        (tmp_path / "plots" / f"{name}.png").stat().st_size > 1000 for name in selection
    )
    with pytest.raises(ValueError):
        evaluate_social(path, tmp_path / "bad", split="test", corruption="noise")


def test_same_timestamp_other_recording_cannot_be_neighbor():
    w, r = example()
    r["other_file"] = {99: (np.arange(12) * 0.4, np.ones((12, 2)))}
    b = context_arrays(w, r)
    assert 99 not in b.neighbor_ids


def test_future_target_changes_do_not_change_social_inputs():
    w, r = example()
    a = context_arrays(w, r)
    w[0].future = np.zeros_like(w[0].future) + 100
    b = context_arrays(w, r)
    np.testing.assert_array_equal(a.neighbors, b.neighbors)
    np.testing.assert_array_equal(a.observed, b.observed)
    assert not np.array_equal(a.future, b.future)


def test_social_heading_rotation_and_batch_shapes():
    import torch

    m = tiny_model()
    m.heading_align = True
    x, n, mask, tm = tensors()
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    y = m(x, n, mask, tm)
    torch.testing.assert_close(
        m(x @ rotation, n @ rotation, mask, tm), y @ rotation, atol=3e-6, rtol=1e-6
    )
    expanded = m(
        x.repeat(7, 1, 1), n.repeat(7, 1, 1, 1), mask.repeat(7, 1, 1), tm.repeat(7, 1)
    )
    assert expanded.shape == (7, 4, 2)
    torch.testing.assert_close(expanded, y.repeat(7, 1, 1), atol=3e-6, rtol=1e-6)


def test_multitrack_interpolation_is_bounded_and_flagged():
    from go2_perception.forecasting.social_adapter import multi_track_input

    a, b = tracks()
    b.history = [h for h in b.history if not np.isclose(h["timestamp"], 2.0)]
    small = multi_track_input(
        a, [b], 8, 0.4, validated_ids={1, 2}, resample=True, max_gap_s=0.45
    )
    assert not small.neighbor_mask[0, 5]
    large = multi_track_input(
        a, [b], 8, 0.4, validated_ids={1, 2}, resample=True, max_gap_s=0.85
    )
    assert large.neighbor_mask[0, 5] and large.neighbor_interpolated[0, 5]
    assert not large.neighbor_mask[
        0, :4
    ].any()  # No support before first observed sample.
