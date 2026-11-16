"""Opt-in CPU native-format tests; pinned processor/source only, NO weights/GPU."""

import copy
import os
from pathlib import Path

import numpy as np
import pytest

from go2_perception.vln.context_ablation import canonical_segments, sample_factory, verify_native_layout
from go2_perception.vln.bounded_context import sample_factory as bounded_factory
from test_vln_context_ablation import contract, segments
from test_bounded_native_integration import setup_native

ASSETS=os.getenv("LIGHTNAV_ASSETS")
pytestmark=[pytest.mark.model_integration,pytest.mark.skipif(not ASSETS,reason="Pinned native CPU processor/source not supplied; no neural inference")]


@pytest.mark.parametrize("n",[8,25,29,36,39])
def test_real_native_groups_timestamps_pools_and_episode_reset(monkeypatch,n):
    import torch
    torch.set_num_threads(2)
    agent,resets=setup_native(monkeypatch)
    original_tiers=copy.deepcopy(agent.engine.bundle.slowfast_tiers)
    load=lambda f:np.full((40,60,3),int(Path(f["path"]).stem),np.uint8)
    job,spec=contract(n)
    verify_native_layout(spec,agent.engine.bundle)
    assert canonical_segments({"video_segments":segments(n)})==spec["baseline_segments"]
    old,_,_=bounded_factory(spec["bounded_context"])(agent,job,load)
    samples={}
    for c in ("B","T","P","TP"):
        job,spec=contract(n,c)
        job.update(instruction=f"Fresh instruction {c}",episode_id=c,clock_epoch=c)
        sample,tensors,meta=sample_factory(spec)(agent,job,load)
        samples[c]=sample
        assert agent.instruction==job["instruction"] and agent._history==[] and agent._sf_buffer is None
        assert set(tensors)==set(spec["bounded_context"]["source_indices"])
        assert agent.engine.bundle.slowfast_tiers==original_tiers
        assert meta["action_feedback"] is False
    assert len(resets)==5
    assert canonical_segments(samples["B"])==canonical_segments(old)
    for b,t,p in zip(samples["B"]["video_segments"],samples["T"]["video_segments"],samples["P"]["video_segments"],strict=True):
        assert t["frame_indices"]==b["source_frame_indices"] and t["total_frames"]==n
        assert t["pool_spatial"]==b["pool_spatial"]
        assert p["frame_indices"]==b["frame_indices"] and p["total_frames"]==b["total_frames"]
        assert torch.equal(b["video"],t["video"]) and torch.equal(b["video"],p["video"])
    assert samples["T"]["video_fps"]==samples["TP"]["video_fps"]==4
    assert samples["B"]["video_fps"]==samples["P"]["video_fps"]==1000
    if n==8:assert canonical_segments(samples["TP"])==segments(8)
    if n in (36,39):
        assert canonical_segments(samples["P"])==canonical_segments(samples["B"])
        assert canonical_segments(samples["TP"])==canonical_segments(samples["T"])


def test_rejects_saved_metadata_that_disagrees_with_native_tiers(monkeypatch):
    agent,_=setup_native(monkeypatch)
    _,spec=contract(29)
    spec["baseline_segments"][0]["pool_spatial"]=4
    with pytest.raises(ValueError,match="disagrees"):
        verify_native_layout(spec,agent.engine.bundle)


def test_real_processor_eight_image_tp_f_equivalence_and_original_b(monkeypatch,tmp_path):
    """Real processor/positional metadata/HF input preparation; never a model."""
    from PIL import Image
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    monkeypatch.setenv("VLN_TIMESTAMP_RELATIVE","1")
    monkeypatch.setenv("HF_HUB_OFFLINE","1")
    from check_vln_context_inputs import cpu_agent, construct, signatures
    from go2_perception.vln.episodes import file_sha256
    agent=cpu_agent(Path(ASSETS))
    built={}
    for c in ("F","TP","B","T","P"):
        job,spec=contract(8,c)
        for i,f in enumerate(job["frames"]):
            p=tmp_path/f"{i}.png"
            # Asymmetric spatial detail catches transpose/mirroring/normalization drift.
            im=np.indices((64,96)).sum(axis=0).astype(np.uint8)
            Image.fromarray(np.stack([im,im+i,np.full_like(im,i*20)],axis=-1)).save(p)
            f.update(path=str(p),sha256=file_sha256(p))
        built[c]=construct(agent,job,spec)
        assert agent.engine.bundle.data_processor.video_fps==4
        assert not hasattr(agent.engine.bundle,"model")
    f,tp=built["F"],built["TP"]
    for key in ("processed","generation_inputs","pool_arrays"):
        assert signatures(f[key])==signatures(tp[key]),key
    assert f["pool_state"]==tp["pool_state"]
    assert canonical_segments(f["sample"])==canonical_segments(tp["sample"])
    assert f["sample"]["conversations"]==tp["sample"]["conversations"]
    assert signatures(built["B"]["generation_inputs"])!=signatures(f["generation_inputs"])
    assert len(f["sample"]["video_segments"])==2 and len(built["P"]["sample"]["video_segments"])==3
