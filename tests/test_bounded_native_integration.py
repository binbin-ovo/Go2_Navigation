"""Optional pinned processor checks with synthetic pixels; no neural inference."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from go2_perception.vln.bounded_context import sample_factory, select_context

ASSETS = os.getenv("LIGHTNAV_ASSETS")
pytestmark = [pytest.mark.model_integration, pytest.mark.usefixtures("isolated_native_lightnav"),
              pytest.mark.skipif(not ASSETS, reason="Needs isolated pinned LightNav processor/source; no weights loaded")]


def setup_native(monkeypatch):
    from lightnav.inference.policies import NavigationPolicy
    from lightnav.slowfast import validate_slowfast_tiers
    original = json.loads((Path(ASSETS)/"checkpoint/eval_config.json").read_text())["tasks"]["vlnce"]["slowfast_tiers"]
    bundle = SimpleNamespace(slowfast_tiers=validate_slowfast_tiers(original), video_size=(256,448), action_method="rvq",
                             video_fps=4, predict_horizon=10)
    resets = []
    agent = NavigationPolicy(SimpleNamespace(bundle=bundle, reset_episode_state=lambda: resets.append(True)))
    return agent, resets


@pytest.mark.parametrize("n", [1,3,5,7,8,19,39,241])
def test_native_selector_and_bounded_storage(monkeypatch,n):
    agent, resets = setup_native(monkeypatch)
    original = agent.engine.bundle.slowfast_tiers
    frames = [{"observation_s": i*.25, "available_s":i*.25+.01} for i in range(n)]
    plan = select_context(frames,n*.25)
    job = {"instruction":"Original instruction", "frames":[frames[i] for i in plan["source_indices"]]}
    load = lambda _: np.full((90,160,3),128,np.uint8)
    sample, tensors, metadata = sample_factory(plan)(agent,job,load)
    assert set(tensors) == set(plan["source_indices"])
    assert sum(len(s["frame_indices"]) for s in sample["video_segments"]) <= 8
    assert agent._sf_buffer is None and agent._history == [] and len(resets)==1
    assert agent.engine.bundle.slowfast_tiers is original
    assert agent.instruction == "Original instruction"
    assert max(tensors) == n-1 and metadata["processor_timestamp_hz"] == 1000
    assert all(t.shape==(3,256,448) for t in tensors.values())
    # Reconstruct/reset does not move the time origin or consume past proposals.
    sample_factory(plan)(agent,{**job,"instruction":"new epoch instruction"},load)
    assert len(resets)==2 and not agent._history and agent._sf_buffer is None


def test_real_processor_accepts_millisecond_metadata_and_keeps_gap(monkeypatch):
    agent, _ = setup_native(monkeypatch)
    from lightnav.data_processor import Qwen3VLDataProcessor
    from lightnav.processing import VLNQwen3VLProcessor
    from lightnav.inference.model import _position_id_func_from_config
    monkeypatch.setenv("VLN_TIMESTAMP_RELATIVE","1")
    checkpoint = str(Path(ASSETS)/"checkpoint")
    processor = VLNQwen3VLProcessor.from_pretrained(checkpoint,local_files_only=True)
    dp = Qwen3VLDataProcessor(processor, video_fps=1000, video_pool_stage="post_vit",
                             model_max_length=8192, position_id_func=_position_id_func_from_config(checkpoint))
    frames = [{"observation_s":100+i*.25-(.031 if i%2 else 0), "available_s":100+i*.25+.01} for i in range(39)]
    plan = select_context(frames,110)
    job = {"instruction":"Turn right", "frames":[frames[i] for i in plan["source_indices"]]}
    sample, _, _ = sample_factory(plan)(agent,job,lambda _:np.zeros((90,160,3),np.uint8))
    inputs = dp.process_sample(sample,add_generation_prompt=True,validate_video_shapes=False)
    prompt = processor.tokenizer.decode(inputs["input_ids"],skip_special_tokens=False)
    assert "<9.4 seconds>" in prompt  # anchor mean relative to true current time
    assert "<1.1 seconds>" in prompt and "<0.1 seconds>" in prompt
    assert "<0.6 seconds>" in prompt
    assert len(inputs["pixel_values_videos"]) > 0


def test_worker_restores_baseline_after_candidate_and_generate_error(monkeypatch,tmp_path):
    """Actual sample builders/conversion, mocked generation (no weights or GPU)."""
    import torch
    from PIL import Image
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from lightnav_worker import run_job
    agent, _ = setup_native(monkeypatch)
    fps_seen=[]
    dp=SimpleNamespace(video_fps=4, process_sample=lambda sample: {}, _sync_processor_settings=lambda:None)
    model=SimpleNamespace(generate=lambda **kw:torch.tensor([[1,2,3]]))
    agent.engine.bundle.model=model
    agent.engine.bundle.max_seq_len=8192
    agent.engine.bundle.tokenizer=SimpleNamespace(decode=lambda *a,**kw:"mock prompt")
    agent.engine.bundle.data_processor=dp
    def generate(sample):
        fps_seen.append(dp.video_fps)
        dp.process_sample(sample)
        model.generate(input_ids=torch.tensor([[1,2]]))
        return "<apos_1298><opos_0><act_l0_6><act_l1_0><act_l2_0>",1.0
    agent.engine.generate=generate
    agent.decode_waypoints=lambda _: (np.zeros((10,3)),None)
    agent.rvq=SimpleNamespace(is_stop=lambda _:True)
    frames=[]
    for i in range(19):
        path=tmp_path/f"{i}.png"
        Image.fromarray(np.full((40,60,3),i,np.uint8)).save(path)
        frames.append({"path":str(path),"observation_s":i*.25,"available_s":i*.25+.01})
    plan=select_context(frames,5)
    full={"instruction":"keep instruction", "episode_id":"one", "clock_epoch":"a", "decision_id":"d", "frames":frames,
          "decision_time_s":5}
    baseline=run_job(agent,full,tmp_path/"full_before","cpu")
    assert baseline["status"]=="decoded", baseline.get("error")
    bounded={**full,"frames":[frames[i] for i in plan["source_indices"]]}
    candidate=run_job(agent,bounded,tmp_path/"bounded","cpu",sample_factory=sample_factory(plan))
    assert candidate["status"]=="decoded" and dp.video_fps==4 and agent._sf_buffer is None
    def fail(sample):
        assert dp.video_fps==1000
        raise RuntimeError("Mock generation failure")
    agent.engine.generate=fail
    failure=run_job(agent,bounded,tmp_path/"failure","cpu",sample_factory=sample_factory(plan))
    assert failure["status"]=="error" and dp.video_fps==4
    agent.engine.generate=generate
    after=run_job(agent,full,tmp_path/"full_after","cpu")
    assert fps_seen==[4,1000,4]
    assert baseline["native_segments"]==after["native_segments"]
    assert baseline["native_output"]==after["native_output"]
    assert (tmp_path/"full_before/native_sample.json").read_bytes()==(tmp_path/"full_after/native_sample.json").read_bytes()
    assert model.generate.__name__=="<lambda>"
