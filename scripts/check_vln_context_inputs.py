"""Stage A: CPU native input construction only. Never loads neural weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from go2_perception.vln.context_ablation import canonical_segments, layout_audit, sample_factory, validate_contract, verify_native_layout
from go2_perception.vln.episodes import file_sha256
from go2_perception.vln.pretrained import CODE_REVISION, MODEL_REVISION


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def cpu_agent(assets):
    """Processor + config + NavigationPolicy, with no model or inference engine."""
    import torch
    torch.set_num_threads(2)
    source=json.loads((assets/"source_local_v2_identity.json").read_text())
    checkpoint=json.loads((assets/"checkpoint_identity.json").read_text())
    if source["upstream_revision"]!=CODE_REVISION or checkpoint["revision"]!=MODEL_REVISION:
        raise ValueError("Unpinned source/checkpoint")
    for f in source["files"]:
        if file_sha256(assets/"source-local-v2"/f["path"])!=f["sha256"]:
            raise ValueError("Changed native source")
    # Stage A does not read large model weights. Verify all non-weight processor
    # assets used here; Stage B's unchanged bootstrap verifies the entire bundle.
    for f in checkpoint["files"]:
        if Path(f["path"]).suffix in (".json",".txt",".model",".jinja"):
            if file_sha256(assets/"checkpoint"/f["path"])!=f["sha256"]:
                raise ValueError("Changed processor/config asset")
    sys.path.insert(0,str(assets/"source-local-v2/src"))
    from lightnav.data_processor import Qwen3VLDataProcessor
    from lightnav.processing import VLNQwen3VLProcessor
    from lightnav.inference.model import _position_id_func_from_config
    from lightnav.inference.policies import NavigationPolicy
    from lightnav.slowfast import validate_slowfast_tiers
    cfg=json.loads((assets/"checkpoint/eval_config.json").read_text())
    common,task=cfg["common"],cfg["tasks"]["vlnce"]
    processor=VLNQwen3VLProcessor.from_pretrained(str(assets/"checkpoint"),local_files_only=True)
    dp=Qwen3VLDataProcessor(processor,model_max_length=common["max_seq_len"],video_fps=task["video_fps"],
         video_pool_stage=common["pool_stage"],position_id_func=_position_id_func_from_config(str(assets/"checkpoint")))
    bundle=SimpleNamespace(**common,slowfast_tiers=validate_slowfast_tiers(task["slowfast_tiers"]),
         video_fps=task["video_fps"],predict_horizon=task["predict_horizon"],action_method="rvq",
         processor=processor,data_processor=dp,tokenizer=processor.tokenizer,device=torch.device("cpu"))
    agent=NavigationPolicy(SimpleNamespace(bundle=bundle,reset_episode_state=lambda:None))
    return agent


def construct(agent,job,spec):
    """Reuse the actual builders and processor. Return exact model-input tensors."""
    import torch
    from lightnav.inference.samples import build_vln_traj_sample
    from lightnav.inference.engine import VLNInferenceEngine
    from lightnav_worker import tensor_archive, validate_jobs
    validate_jobs({"schema":"lightnav-local-jobs-v1","jobs":[job]})
    validate_contract(spec,job)
    verify_native_layout(spec,agent.engine.bundle)
    def load_rgb(frame):
        with Image.open(frame["path"]) as im:
            return np.asarray(im.convert("RGB"))
    factory=sample_factory(spec)
    if factory is None:
        agent.reset(instruction=job["instruction"])
        for f in job["frames"]: agent.observe(load_rgb(f))
        sample=build_vln_traj_sample(agent._get_video_tensor(),job["instruction"],None,agent.engine.bundle)
        tensors={i:agent._get_video_tensor()[i] for s in sample["video_segments"] for i in s["frame_indices"]}
        metadata={"ablation_condition":"F"}
    else:
        sample,tensors,metadata=factory(agent,job,load_rgb)
    dp=agent.engine.bundle.data_processor
    old=dp.video_fps
    try:
        dp.video_fps=sample["video_fps"]
        processed=dp.process_sample(sample,add_generation_prompt=True,validate_video_shapes=False)
    finally:
        dp.video_fps=old
        dp._sync_processor_settings()
    processed_arrays,processed_types=tensor_archive(processed)
    # Call the pinned engine's preparation method on a CPU bundle, without an
    # engine/model instance. Keep post-ViT configuration as separate evidence.
    gen=VLNInferenceEngine._prepare_inputs(agent.engine,processed)
    gen.pop("position_ids",None)  # same stock-HF path as pinned _generate_hf
    pool_state={k:gen.pop(k,None) for k in ("_original_video_grid_thw","_post_vit_pool_factors","_post_vit_pool_spatial")}
    arrays,types=tensor_archive(gen)
    pool_arrays,pool_types=tensor_archive(pool_state)
    plain_pool={k:v for k,v in pool_state.items() if not isinstance(v,torch.Tensor)}
    return {"sample":sample,"normalized":tensors,"metadata":metadata,"processed":processed_arrays,
            "processed_types":processed_types,"generation_inputs":arrays,"generation_types":types,
            "pool_arrays":pool_arrays,"pool_types":pool_types,"pool_state":plain_pool}


def signatures(arrays):
    return {k:{"dtype":str(v.dtype),"shape":list(v.shape),"sha256":hashlib.sha256(v.tobytes()).hexdigest()}
            for k,v in arrays.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    a=p.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",
                      VLN_TIMESTAMP_RELATIVE="1",GO2_HARDWARE="0",MUJOCO_RENDER_TEST="0")
    data=json.loads((a.run/"stage_a/prepared_jobs.json").read_text())
    plan=json.loads((a.run/"plan.json").read_text())
    seal=json.loads((a.run/"stage_a/prepared_manifest.json").read_text())
    plan_sha=file_sha256(a.run/"plan.json")
    root=Path(__file__).resolve().parents[1]
    if seal["plan_sha256"]!=plan_sha or [{"case":e["case"],"condition":e["condition"]} for e in data]!=plan["matrix"]:
        raise ValueError("Prepared inputs do not match the frozen matrix")
    for p,h in {**seal["files"],**{str(root/p):h for p,h in plan["code_sha256"].items()}}.items():
        if file_sha256(p)!=h: raise ValueError(f"Prepared source/code changed: {p}")
    agent=cpu_agent(a.assets)
    records=[]
    for entry in data:
        folder=Path(entry["folder"])
        out=folder/"native"
        out.mkdir(exist_ok=False)
        c=construct(agent,entry["job"],entry["ablation"])
        indices=(list(range(entry["ablation"]["bounded_context"]["source_count"])) if entry["condition"]=="F"
                 else entry["ablation"]["bounded_context"]["source_indices"])
        pixels={}
        for i,f in zip(indices,entry["job"]["frames"],strict=True):
            with Image.open(f["path"]) as im: pixels[str(i)]=np.asarray(im.convert("RGB"))
        norm={str(i):v.numpy() for i,v in c["normalized"].items()}
        for i,v in c["normalized"].items():
            rgb=np.clip(np.rint((v.permute(1,2,0).numpy()+1)*127.5),0,255).astype(np.uint8)
            Image.fromarray(rgb).save(out/f"input_{i:04d}.png")
        np.savez_compressed(out/"generation_inputs.npz",**c["generation_inputs"])
        sample={k:v for k,v in c["sample"].items() if k!="video_segments"}
        sample["video_segments"]=[{k:v for k,v in s.items() if k!="video"} for s in c["sample"]["video_segments"]]
        dump(out/"native_sample.json",sample)
        prompt=agent.engine.bundle.tokenizer.decode(c["generation_inputs"]["input_ids"][0],skip_special_tokens=False)
        (out/"prompt.txt").write_text(prompt)
        previous=entry["previous_folder"]
        prior_job=json.loads((Path(previous)/"job.json").read_text())
        old_pixels={}
        for i,f in zip(indices,prior_job["frames"],strict=True):
            with Image.open(f["path"]) as im: old_pixels[str(i)]=np.asarray(im.convert("RGB"))
        identity_fields=("episode_id","clock_epoch","decision_id","decision_time_s","instruction")
        identity_equal=all(entry["job"][k]==prior_job[k] for k in identity_fields)
        stamps_equal=all((x["observation_s"],x["available_s"])==(y["observation_s"],y["available_s"])
                         for x,y in zip(entry["job"]["frames"],prior_job["frames"],strict=True))
        prior_equal=None
        if entry["condition"] in ("F","B"):
            with np.load(Path(previous)/"native/model_inputs_and_generated_ids.npz") as old:
                old_keys=set(old.files)-{"generated_ids"}
                prior_equal=old_keys==set(c["generation_inputs"]) and all(np.array_equal(old[k],c["generation_inputs"][k]) for k in old_keys)
        record={"case":entry["case"],"condition":entry["condition"],"folder":str(folder),
                "status":"input_construction_only_no_inference", "normalized":signatures(norm),
                "calibrated_pixels":signatures(pixels),
                "original_calibrated_pixels_equal":signatures(pixels)==signatures(old_pixels),
                "original_instruction_identity_timestamps_equal":identity_equal and stamps_equal,
                "processed":signatures(c["processed"]),"generation_inputs":signatures(c["generation_inputs"]),
                "processed_types":c["processed_types"],"generation_types":c["generation_types"],
                "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
                "pool_arrays":signatures(c["pool_arrays"]),"pool_state":c["pool_state"],
                "canonical_segments":canonical_segments(c["sample"]),
                "layout_audit":layout_audit(c["sample"],entry["ablation"]["baseline_segments"]),
                "original_F_or_B_generation_inputs_equal":prior_equal,
                "generated_outputs":"pending Stage B; no neural weights loaded"}
        dump(folder/"construction.json",record)
        records.append(record)
        print(json.dumps({"case":entry["case"],"condition":entry["condition"],"segments":len(record["canonical_segments"]),"old_input_equal":prior_equal}),flush=True)
    bycase={}
    for r in records:bycase.setdefault(r["case"],{})[r["condition"]]=r
    checks={}
    for name,g in bycase.items():
        def same(a,b,keys):return all(g[a][k]==g[b][k] for k in keys)
        checks[name]={"TP_equals_F_model_inputs":same("TP","F",["generation_inputs","pool_arrays","pool_state"]),
                      "TP_equals_F_calibrated_pixels":same("TP","F",["calibrated_pixels"]),
                      "TP_equals_F_normalized":same("TP","F",["normalized"]),
                      "TP_equals_F_all_processor_tensors":same("TP","F",["processed"]),
                      "TP_equals_F_prompt_tokens":same("TP","F",["prompt_sha256"]),
                      "TP_equals_F_native_metadata":same("TP","F",["canonical_segments"]),
                      "all_shared_normalized_equal":all(g[c]["normalized"][i]==g["F"]["normalized"][i]
                                                         for c in ("B","T","P","TP") for i in g[c]["normalized"]),
                      "B_equals_P_model_inputs":same("B","P",["generation_inputs","pool_arrays","pool_state"]),
                      "T_equals_TP_model_inputs":same("T","TP",["generation_inputs","pool_arrays","pool_state"])}
    control=checks["p6b_test_c_e3_forward_bottles_d0"]
    passed=(all(control[k] for k in control if k.startswith("TP_equals_F"))
            and all(r["original_F_or_B_generation_inputs_equal"] is True for r in records if r["condition"] in ("F","B"))
            and all(r["original_calibrated_pixels_equal"] and r["original_instruction_identity_timestamps_equal"] for r in records)
            and all(v["all_shared_normalized_equal"] for v in checks.values()))
    dump(a.run/"stage_a/input_checks.json",{"passed":passed,"checks":checks,"requests_constructed":len(records),
          "records":records,"model_weights_loaded":False,"device":"cpu","threads":2,"execution_authorized":False,
          "plan_sha256":plan_sha,"generated_outputs_equal":"pending Stage B"})
    if not passed:raise ValueError("Input-control gate failed; Stage B is blocked")


if __name__=="__main__":main()
