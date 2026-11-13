"""Opt-in actual pretrained processor/model tests. Never download during pytest."""

import os
from pathlib import Path

import pytest
import yaml

from go2_perception.vln.episodes import load_episode
from go2_perception.vln.temporal_inputs import build_model_input

ROOT = Path(__file__).resolve().parents[1]
CACHE = os.environ.get("SMOLVLM_MODEL_CACHE")
pytestmark = [
    pytest.mark.model_integration,
    pytest.mark.skipif(
        not CACHE, reason="set SMOLVLM_MODEL_CACHE to a populated pinned local cache"
    ),
]


def processor():
    from transformers import SmolVLMProcessor

    snapshot = (
        Path(CACHE)
        / "models--HuggingFaceTB--SmolVLM2-2.2B-Instruct/snapshots/be92141c8d0ceb98596a923c308ac2bbb4a7c466"
    )
    p = SmolVLMProcessor.from_pretrained(snapshot, local_files_only=True)
    p.do_image_splitting = p.image_processor.do_image_splitting = False
    p.image_size = p.image_processor.size = {"longest_edge": 384}
    return p


def test_pinned_processor_passes_every_selected_image_and_preserves_K():
    ep = load_episode(ROOT / "tests/fixtures/vln/episode_v1/episode.json")
    p = processor()
    result = build_model_input(
        ep, "d1", p, ROOT / "configs/temporal_vln_prompt.txt", allow_synthetic=True
    )
    assert result.inputs.pixel_values.shape == (1, 4, 3, 384, 384)
    assert result.metadata["image_tokens"] == 4 * 81
    assert result.metadata["model_input_frame_count"] == 4
    assert result.metadata["per_frame"][0]["official_processor_stages"][-1][
        "size_wh"
    ] == [384, 384]
    assert result.inputs.pixel_values[0, 0].ne(result.inputs.pixel_values[0, -1]).any()


def test_pretrained_model_vision_encoder_consumes_history():
    from go2_perception.vln.temporal_model import SmolTemporalModel

    cfg = yaml.safe_load((ROOT / "configs/temporal_vln.yaml").read_text())
    cfg["max_new_tokens"] = 16
    model = SmolTemporalModel(cfg, CACHE)
    p, _, raw = model.infer(
        load_episode(ROOT / "tests/fixtures/vln/episode_v1/episode.json"),
        "d1",
        ROOT / "configs/temporal_vln_prompt.txt",
        allow_synthetic=True,
    )
    assert p["vision_encoder_calls"] == [[4, 3, 384, 384]]
    assert p["history_images_consumed"] == 3 and p["model_executed"]
    assert not p["execution_authorized"] and len(raw["generated_token_ids"]) > 0
