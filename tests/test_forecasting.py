import json
import numpy as np
import pytest
import torch
from go2_perception.io.config import load_config
from go2_perception.forecasting.data import (
    dataset,
    load_recording,
    validate_trajectory,
    resample_observed,
)
from go2_perception.forecasting.geometry import local_frame, to_global
from go2_perception.forecasting.baseline import (
    constant_velocity,
    phase3_constant_velocity,
)
from go2_perception.forecasting.metrics import errors, summarize, motion_categories
from go2_perception.forecasting.model import TemporalTransformer
from go2_perception.forecasting.training import (
    train,
    load_checkpoint,
    predict,
    select_device,
)
from go2_perception.forecasting.adapter import track_history, forecast_track
from go2_perception.core.tracking_types import TrackOutput


def configuration():
    c = load_config("configs/forecasting_transformer.yaml")
    c["dataset"].update(
        root="tests/fixtures/forecasting",
        manifest="tests/fixtures/forecasting/manifest.yaml",
        train=["train_scene"],
        validation=["validation_scene"],
        test=["test_scene"],
        obs_steps=4,
        pred_steps=6,
        cohort_obs=6,
        cohort_pred=6,
        stride=3,
    )
    c["model"].update(d_model=16, heads=2, layers=1, feedforward=32, dropout=0)
    c["training"].update(epochs=3, patience=3, batch_size=8, threads=1)
    return c


def track():
    times = np.arange(8) * 0.2
    history = [
        dict(timestamp=float(t), position=[t, 2 * t, 0], observed=True) for t in times
    ]
    return TrackOutput(
        7,
        float(times[-1]),
        "odom",
        "sim",
        "odometry",
        np.array(history[-1]["position"]),
        np.array([1, 2]),
        np.eye(4),
        None,
        None,
        "person",
        None,
        "Confirmed",
        "MOVING",
        float(times[-1]),
        float(times[-1]),
        8,
        0,
        history,
        [],
    )


def test_window_extraction_and_scene_separation():
    c = configuration()["dataset"]
    a, ma = dataset(c, "train")
    b, mb = dataset(c, "test")
    assert len(a) == 27 and len(b) == 27
    assert not {w.recording for w in a} & {w.recording for w in b}
    assert a[0].observed.shape == (4, 2) and a[0].future.shape == (6, 2)
    assert a[0].times[-1] - a[0].times[3] == pytest.approx(2.4)
    assert a[0].times[3] - a[0].times[0] == pytest.approx(1.2)
    assert ma["window_keys_sha256"] != mb["window_keys_sha256"]


def test_history_ablation_has_identical_forecast_anchors():
    c = configuration()["dataset"]
    a, ma = dataset(c, "train")
    c["obs_steps"] = 6
    b, mb = dataset(c, "train")
    assert ma["window_keys_sha256"] == mb["window_keys_sha256"]
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.future, y.future)


@pytest.mark.parametrize("bad", ["overlap", "time", "units", "duplicate_file"])
def test_bad_dataset_protocol(tmp_path, bad):
    import yaml

    c = configuration()["dataset"]
    if bad == "overlap":
        c["test"] = c["train"]
    elif bad == "time":
        c["dt"] = 0.2
    else:
        spec = load_config(c["manifest"])
        if bad == "units":
            spec["units"] = "pixels"
        else:
            spec["scenes"]["test_scene"] = spec["scenes"]["train_scene"]
        p = tmp_path / "manifest.yaml"
        p.write_text(yaml.safe_dump(spec))
        c["manifest"] = str(p)
    with pytest.raises(ValueError):
        dataset(c, "test")


@pytest.mark.parametrize(
    "rows",
    [
        [[0, 1, 0, 0], [0, 1, 1, 1]],
        [[10, 1, 0, 0], [0, 1, 1, 1]],
        [[0, 1, float("nan"), 0]],
        [[0, 1, 0]],
    ],
)
def test_malformed_rows(tmp_path, rows):
    p = tmp_path / "bad.txt"
    np.savetxt(p, rows)
    with pytest.raises(ValueError):
        load_recording(p, frame_stride=10, dt=0.4)


def test_missing_data_does_not_create_windows(tmp_path):
    from go2_perception.forecasting.data import windows_for_recording

    p = tmp_path / "gap.txt"
    a = np.array([[i * 10, 1, i, 0] for i in range(24) if i != 10])
    np.savetxt(p, a)
    w = windows_for_recording(
        p,
        "a",
        "a",
        obs_steps=8,
        pred_steps=12,
        cohort_obs=8,
        cohort_pred=12,
        stride=1,
        dt=0.4,
        frame_stride=10,
    )
    assert w == []


def test_normalization_inverse_heading_and_translation():
    x = torch.tensor([[[3.0, 5.0], [3.0, 6.0], [3.0, 7.0]]], dtype=torch.float64)
    local, origin, r = local_frame(x, True)
    torch.testing.assert_close(local[0, -1], torch.zeros(2, dtype=x.dtype))
    assert abs(float(local[0, 0, 1])) < 1e-10
    torch.testing.assert_close(to_global(local, origin, r), x)
    shifted = local_frame(x + 100, True)[0]
    torch.testing.assert_close(local, shifted)


def test_baseline_analytic_and_phase3_equivalence():
    x = np.array([[[0, 0], [0.4, 0.8], [0.8, 1.6]]])
    for method in ["mean", "last"]:
        out = constant_velocity(x, 3, 0.4, method=method)
        np.testing.assert_allclose(out[0], [[1.2, 2.4], [1.6, 3.2], [2.0, 4.0]])
        np.testing.assert_allclose(
            out, phase3_constant_velocity(x, 3, 0.4, method=method)
        )
    np.testing.assert_allclose(constant_velocity(np.ones((2, 4, 2)), 2, 0.4), 1)


def test_metrics_euclidean_not_mse():
    target = np.zeros((2, 3, 2))
    pred = np.tile([3, 4], (2, 3, 1))
    assert summarize(pred, target) == dict(count=2, ade_m=5.0, fde_m=5.0)
    pred[:, 0] = 0
    assert errors(pred, target)[0][0] == pytest.approx(10 / 3)
    with pytest.raises(ValueError):
        errors(pred[:, :0], target[:, :0])


@pytest.mark.parametrize("batch", [1, 2, 7])
@pytest.mark.parametrize("representation", ["displacement", "local_position"])
def test_model_shapes_determinism_and_translation(batch, representation):
    torch.manual_seed(7)
    model = TemporalTransformer(
        4,
        6,
        d_model=16,
        heads=2,
        layers=1,
        feedforward=32,
        representation=representation,
    ).eval()
    x = torch.randn(batch, 4, 2)
    y = model(x)
    assert y.shape == (batch, 6, 2)
    torch.testing.assert_close(y, model(x), rtol=0, atol=0)
    torch.testing.assert_close(model(x + 10), y + 10, rtol=1e-5, atol=1e-5)


def test_observation_only_features_and_gradient():
    model = TemporalTransformer(
        4,
        6,
        d_model=16,
        heads=2,
        layers=1,
        feedforward=32,
        dropout=0,
        heading_align=True,
    )
    x = torch.randn(2, 4, 2)
    y = model(x)
    y.square().mean().backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    with pytest.raises(ValueError):
        model(torch.zeros(2, 3, 2))


def test_bounded_explicit_resampling():
    t = np.array([0, 0.2, 0.6, 0.8])
    p = np.stack((t, 2 * t), axis=1)
    with pytest.raises(ValueError):
        resample_observed(t, p, [0.4, 0.8], max_gap=0.3)
    out, mask = resample_observed(t, p, [0.4, 0.8], max_gap=0.41)
    np.testing.assert_allclose(out, [[0.4, 0.8], [0.8, 1.6]])
    assert mask.tolist() == [True, False]
    with pytest.raises(ValueError):
        resample_observed(t, p, [0.8, 1.0], max_gap=1)
    with pytest.raises(ValueError):
        validate_trajectory([0, 0], [[0, 0], [1, 1]])


def test_track_conversion_and_forecast_contract():
    t = track()
    x, times, mask = track_history(t, 4, 0.4)
    np.testing.assert_allclose(x[:, 0], times)
    assert not mask.any()
    model = TemporalTransformer(4, 6, d_model=16, heads=2, layers=1, feedforward=32)
    out = forecast_track(model, t, 0.4, model_id="untrained-unit-test").to_dict()
    assert (
        out["track_id"] == 7
        and out["frame_id"] == "odom"
        and out["clock_domain"] == "sim"
    )
    assert out["future_timestamps"][0] == pytest.approx(1.8)
    assert out["future_timestamps"][-1] == pytest.approx(3.8)
    assert "confidence" not in out and "covariance" not in out


@pytest.mark.parametrize(
    "case",
    [
        "class",
        "frame",
        "clock",
        "miss",
        "short",
        "order",
        "identity",
        "units",
        "unflagged",
    ],
)
def test_reject_invalid_track_history(case):
    t = track()
    opts = {}
    if case == "class":
        t.class_name = None
    if case == "frame":
        t.frame_id = "base_link"
    if case == "clock":
        t.history[0]["clock_domain"] = "other"
    if case == "miss":
        t.consecutive_misses = 1
    if case == "short":
        t.history = t.history[-3:]
    if case == "order":
        t.history[3]["timestamp"] = t.history[2]["timestamp"]
    if case == "identity":
        t.history[3]["track_id"] = 99
    if case == "units":
        opts["units"] = "mm"
    if case == "unflagged":
        del t.history[2]["observed"]
    with pytest.raises(ValueError):
        track_history(t, 4, 0.4, **opts)


def test_predicted_history_is_not_evidence():
    t = track()
    t.history[3].update(observed=False, position=[999, 999, 0])
    with pytest.raises(ValueError):
        track_history(t, 4, 0.4)
    with pytest.raises(ValueError):
        track_history(t, 4, 0.4, resample=True, max_gap_s=0.2)
    x, times, mask = track_history(t, 4, 0.4, resample=True, max_gap_s=0.41)
    assert mask[1]
    np.testing.assert_allclose(x[:, 0], times)


def test_default_tracker_history_too_short_for_forecaster():
    t = track()
    t.history = [
        dict(timestamp=i * 0.065, position=[i * 0.065, 0, 0], observed=True)
        for i in range(40)
    ]
    t.timestamp = t.last_observation_timestamp = t.history[-1]["timestamp"]
    with pytest.raises(ValueError):
        track_history(t, 8, 0.4, resample=True)


def test_synthetic_training_checkpoint_and_test_isolation(tmp_path):
    c = configuration()
    # Deliberately nonexistent test file: training must not open it.
    import yaml

    spec = load_config(c["dataset"]["manifest"])
    spec["scenes"]["test_scene"] = ["absent.txt"]
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.safe_dump(spec))
    c["dataset"]["manifest"] = str(p)
    checkpoint = train(c, tmp_path / "run", make_plots=False)
    model, payload = load_checkpoint(checkpoint)
    x = np.zeros((3, 4, 2), dtype=np.float32)
    np.testing.assert_array_equal(
        predict(model, x, select_device("cpu")),
        predict(load_checkpoint(checkpoint)[0], x, select_device("cpu")),
    )
    repeat = train(c, tmp_path / "repeat", make_plots=False)
    repeated_model, _ = load_checkpoint(repeat)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(
            value, repeated_model.state_dict()[name], rtol=0, atol=0
        )
    assert payload["metadata"]["train"]["windows"] == 27
    assert (
        json.loads((checkpoint.parent / "training_summary.json").read_text())[
            "test_evaluated"
        ]
        is False
    )
    assert len(json.loads((checkpoint.parent / "history.json").read_text())) == 3


def test_motion_categories_are_evaluation_only():
    t = np.arange(8) * 0.4
    x = np.stack((t, np.zeros(8)), axis=1)
    straight = np.stack((t[-1] + np.arange(1, 13) * 0.4, np.zeros(12)), axis=1)
    turn = np.stack((np.full(12, t[-1]), np.arange(1, 13) * 0.4), axis=1)
    stop = np.repeat(x[-1:, :], 12, axis=0)
    assert motion_categories(
        np.array([x, x, x]), np.array([straight, turn, stop]), 0.4
    ).tolist() == ["straight", "turning", "stopping"]


def test_adapter_rejects_wrong_model_interval():
    model = TemporalTransformer(4, 6, d_model=16, heads=2, layers=1, feedforward=32)
    with pytest.raises(ValueError, match="sampling interval"):
        forecast_track(model, track(), 0.2, model_id="unit-test")


def test_checksum_prevents_silent_dataset_change(tmp_path):
    import yaml

    c = configuration()["dataset"]
    spec = load_config(c["manifest"])
    spec["file_sha256"] = {"test_scene.txt": "wrong-hash"}
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.safe_dump(spec))
    c["manifest"] = str(p)
    with pytest.raises(ValueError, match="checksum"):
        dataset(c, "test")


def test_empty_model_and_baseline_inputs_rejected():
    with pytest.raises(ValueError):
        local_frame(torch.empty(0, 4, 2))
    with pytest.raises(ValueError):
        constant_velocity(np.empty((0, 4, 2)), 6, 0.4)
