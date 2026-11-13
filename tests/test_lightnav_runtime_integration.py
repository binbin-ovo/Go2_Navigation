"""Opt-in tests of the actual pinned upstream processor's CPU/MPS pooling path."""

import os
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.model_integration
ASSETS = os.getenv("LIGHTNAV_ASSETS")


@pytest.mark.skipif(not ASSETS, reason="Needs audited local LightNav source")
def test_native_reset_clears_full_history_and_vision_cache():
    import numpy as np
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(ASSETS) / "source-local-v2/src"))
    from lightnav.inference.policies import NavigationPolicy

    resets = []
    cache = {}
    engine = SimpleNamespace(
        bundle=SimpleNamespace(
            slowfast_tiers=[{"pool_spatial": 1}], video_size=(256, 448)
        ),
        new_vit_cache=lambda: cache,
        reset_episode_state=lambda: resets.append(True),
    )
    policy = NavigationPolicy(engine)
    policy.reset("First episode")
    policy.observe(np.zeros((90, 160, 3), dtype=np.uint8))
    policy.observe(np.full((90, 160, 3), 80, dtype=np.uint8))
    cache["previous_epoch"] = True
    assert policy._get_video_tensor().shape[0] == 2
    policy.reset("Second epoch")
    assert policy._history_frame_ids == [] and not cache
    assert policy._buffer_len == 0 and policy._sf_buffer is None
    policy.observe(np.full((90, 160, 3), 255, dtype=np.uint8))
    assert policy._history_frame_ids == [0]
    assert policy._get_video_tensor().shape[0] == 1
    assert policy._get_video_tensor().min() == 1  # only the new white observation
    assert policy.instruction == "Second epoch" and len(resets) == 2


@pytest.mark.skipif(
    not ASSETS, reason="Needs isolated LightNav 5.8 runtime and audited local source"
)
def test_native_pooling_cpu_mps_preserves_geometry_dtype_and_deepstack():
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available in this execution context")
    sys.path.insert(0, str(Path(ASSETS) / "source-local-v2/src"))
    from lightnav.processing import post_vit_spatial_pool

    torch.manual_seed(5)
    # Merged H=8,W=14 with factor=4 -> output 2x4, non-divisible W.
    grid = torch.tensor([[2, 16, 28], [2, 16, 28]])
    embeddings = torch.randn(448, 32, dtype=torch.bfloat16)
    deepstack = [embeddings.clone(), embeddings * 0.5]
    expected = post_vit_spatial_pool(embeddings, grid, [4, 1], 2, deepstack)
    actual = post_vit_spatial_pool(
        embeddings.to("mps"),
        grid.to("mps"),
        [4, 1],
        2,
        [x.to("mps") for x in deepstack],
    )
    assert torch.equal(expected[0], actual[0].cpu())
    assert torch.equal(expected[1], actual[1].cpu())
    assert actual[0].dtype == torch.bfloat16 and actual[0].device.type == "mps"
    assert all(torch.equal(a, b.cpu()) for a, b in zip(expected[2], actual[2]))
