"""Pinned decoder and processor, without neural inference or robot clients."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ASSETS = os.getenv("LIGHTNAV_ASSETS")
pytestmark = [
    pytest.mark.model_integration,
    pytest.mark.skipif(
        not ASSETS, reason="Requires isolated pinned LightNav source/RVQ"
    ),
]


def source_path():
    sys.path.insert(0, str(Path(ASSETS) / "source-local-v2/src"))


def test_official_se2_composition_returns_future_absolute_headings():
    source_path()
    from lightnav.traj_vocab import compose_to_abs

    result = compose_to_abs(np.array([[[0.1, 0, np.pi / 2], [0.2, 0, 0]]]))[0]
    assert result[0] == pytest.approx([0.1, 0, np.pi / 2])
    assert result[1] == pytest.approx([0.1, 0.2, np.pi / 2])
    assert result.shape == (2, 3)  # The initial origin is not a stored row.


def test_official_native_stop_and_moving_chunk_are_distinct():
    source_path()
    from lightnav.traj_vocab import load_rvq_bundle
    from lightnav.tracking import TrackingAgent
    from lightnav.serving.protocol import decode_prediction_signals

    bundle = load_rvq_bundle(Path(ASSETS) / "checkpoint/action_tokenizer", 10)
    policy = TrackingAgent(SimpleNamespace(), rvq_bundle=bundle)
    stop, _ = policy.decode_waypoints(
        "<apos_1299><opos_0><act_l0_6><act_l1_122><act_l2_174>"
    )
    moving, _ = policy.decode_waypoints(
        "<apos_1299><opos_0><act_l0_231><act_l1_186><act_l2_18>"
    )
    assert bundle.stop_l0 == 6 and np.all(stop == 0) and moving.shape == (10, 3)
    assert not np.all(moving == 0)
    # APOS completion token alone never substitutes for a native action STOP.
    assert not decode_prediction_signals(
        "<apos_1299><opos_0>", is_rvq=True, waypoints=moving
    ).stop


def test_official_pinhole_preprocessing_does_not_apply_go2_fisheye_calibration():
    source_path()
    from lightnav.inference.frame_preprocessing import rgb_frame_to_model_tensor

    rgb = np.random.default_rng(1).integers(0, 256, (256, 448, 3), dtype=np.uint8)
    tensor = rgb_frame_to_model_tensor(rgb, (256, 448))
    recovered = (
        np.rint((tensor.permute(1, 2, 0).numpy() + 1) * 127.5)
        .clip(0, 255)
        .astype(np.uint8)
    )
    assert np.array_equal(rgb, recovered)


def test_streaming_native_buffers_match_reconstructed_causal_history():
    source_path()
    from lightnav.inference.policies import NavigationPolicy
    from go2_perception.vln.native_history import NativeObservationSession

    resets = []
    engine = SimpleNamespace(
        bundle=SimpleNamespace(
            slowfast_tiers=[{"pool_spatial": 1}], video_size=(256, 448)
        ),
        new_vit_cache=lambda: {},
        reset_episode_state=lambda: resets.append(1),
    )
    stream, native = NativeObservationSession(), NavigationPolicy(engine)
    frames = [np.full((256, 448, 3), i * 40, dtype=np.uint8) for i in range(4)]
    for n in range(1, 5):
        job = {
            "episode_id": "test",
            "clock_epoch": "one",
            "decision_id": str(n),
            "instruction": "Doorway",
            "decision_time_s": (n - 1) * 0.25,
            "frames": [
                {
                    "path": str(i),
                    "sha256": str(i),
                    "observation_s": i * 0.25,
                    "available_s": i * 0.25,
                }
                for i in range(n)
            ],
        }
        stream.observe(native, job, lambda f: frames[int(f["path"])])
        rebuilt = NavigationPolicy(
            SimpleNamespace(bundle=engine.bundle, reset_episode_state=lambda: None)
        )
        rebuilt.reset("Doorway")
        for rgb in frames[:n]:
            rebuilt.observe(rgb)
        assert np.array_equal(
            native._get_video_tensor().numpy(), rebuilt._get_video_tensor().numpy()
        )
        assert native._history_frame_ids == rebuilt._history_frame_ids == list(range(n))
    assert len(resets) == 1
