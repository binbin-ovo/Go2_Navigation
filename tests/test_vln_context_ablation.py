"""Stage A CPU contract/mocked orchestration tests; no model or hardware."""

import ast
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from go2_perception.vln.bounded_context import select_context
from go2_perception.vln.context_ablation import (
    CASES, CONDITIONS, canonical_segments, filtered_full_layout, layout_audit,
    pooling_assignments, sample_factory, validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def segments(n):
    # Actual layouts from the saved full baseline; verified against pinned tiers
    # by the optional native tests, not an alternative model selector.
    ids = {8: [(list(range(6)), 2), ([6,7], 1)],
           25: [(list(range(23))+[22], 2), ([23,24], 1)],
           29: [(list(range(27))+[26], 2), ([27,28], 1)],
           36: [([0,1], 4), (list(range(2,34)), 2), ([34,35], 1)],
           39: [([0,1], 4), ([3,4], 2), (list(range(5,37)), 2), ([37,38], 1)]}[n]
    return [{"frame_indices":i,"pool_spatial":p,"pool_mode":"avg","total_frames":n} for i,p in ids]


def contract(n=29, condition="B"):
    frames = [{"path":f"{i}.png", "sha256":"mock", "observation_s":100+i*.25-(.031 if i%2 else 0),
               "available_s":100+i*.25+.01} for i in range(n)]
    cutoff=frames[-1]["available_s"]
    context=select_context(frames,cutoff)
    job={"episode_id":"one", "clock_epoch":"epoch", "decision_id":"d2", "instruction":"Turn toward the doorway",
         "decision_time_s":cutoff, "frames":frames if condition=="F" else [frames[i] for i in context["source_indices"]]}
    return job,{"condition":condition,"bounded_context":context,"baseline_segments":segments(n)}


@pytest.mark.parametrize("n",[8,25,29,36,39])
def test_contract_and_filtered_layout(n):
    for c in CONDITIONS:
        job,spec=contract(n,c)
        validate_contract(spec,job)
        assert sample_factory(spec) is None if c=="F" else callable(sample_factory(spec))
    before=copy.deepcopy(spec)
    layout=filtered_full_layout(spec["baseline_segments"],spec["bounded_context"]["source_indices"])
    assert sum(len(s["frame_indices"]) for s in layout)==8
    assert {i for s in layout for i in s["frame_indices"]}==set(spec["bounded_context"]["source_indices"])
    assert spec==before
    pools=pooling_assignments(layout)
    assert all(v==pooling_assignments(spec["baseline_segments"])[i] for i,v in pools.items())


def test_control_and_partial_pair_confounds_are_explicit():
    assert filtered_full_layout(segments(8),range(8))==segments(8)
    job,spec=contract(29)
    layout=filtered_full_layout(segments(29),spec["bounded_context"]["source_indices"])
    audit=layout_audit({"video_segments":layout,"video_fps":4,"conversations":[]},segments(29))
    assert audit["segments"][0]["pairs_not_in_full_baseline"]==[[23,24],[25,26]]
    assert audit["segments"][0]["timestamp_prefixes"][0]=="<6.9 seconds>"
    assert audit["segments"][1]["source_pairs"]==[[27,28]]


@pytest.mark.parametrize("change",["labels","action","unknown","pool","ids","total","odd","context","anchor"])
def test_closed_contract_rejects_bad_inputs(change):
    job,spec=contract()
    if change=="labels":spec["human_reference"]="STOP"
    if change=="action":spec["executed_action"]="turn"
    if change=="unknown":spec["condition"]="new_context_budget"
    if change=="pool":spec["baseline_segments"][0]["pool_spatial"]=0
    if change=="ids":spec["baseline_segments"][0]["frame_indices"][0]=-1
    if change=="total":spec["baseline_segments"][0]["total_frames"]=30
    if change=="odd":spec["baseline_segments"][0]["frame_indices"].pop()
    if change=="context":spec["bounded_context"]["timestamp_ticks"][2]+=1
    if change=="anchor":spec["bounded_context"]["source_indices"][0]=2
    with pytest.raises(ValueError):validate_contract(spec,job)


def test_no_guessed_pool_or_silent_extra_padding():
    with pytest.raises(ValueError,match="absent"):
        filtered_full_layout(segments(39),[0,1,2,33,34,35,37,38])
    layout=[{"frame_indices":[i,i],"total_frames":8,"pool_spatial":2,"pool_mode":"avg"} for i in range(8)]
    with pytest.raises(ValueError,match="eight-slot"):
        filtered_full_layout(layout,range(8))
    with pytest.raises(ValueError,match="Conflicting"):
        pooling_assignments([segments(8)[0],{**segments(8)[0],"pool_spatial":4}])


def import_runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    import run_vln_context_ablation
    return run_vln_context_ablation


def test_fixed_25_matrix_and_no_stage_b_without_window(monkeypatch,tmp_path):
    module=import_runner(monkeypatch)
    rows=module.matrix()
    assert len(rows)==25 and len({(r["case"],r["condition"]) for r in rows})==25
    assert [rows[i*5]["condition"] for i in range(5)]==list(CONDITIONS)
    assert [rows[i*5]["case"] for i in range(5)]==list(CASES)
    monkeypatch.setattr(module,"AblationWorker",lambda **kw:pytest.fail("Must not start worker"))
    with pytest.raises(ValueError,match="resource window"):
        module.run(SimpleNamespace(resource_window_confirmed=False,output=tmp_path))


def test_worker_envelope_excludes_annotations_and_passes_identity(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    import vln_context_ablation_worker as worker
    import sys
    calls,sent=[],[]
    monkeypatch.setattr(worker,"verify_native_layout",lambda *a:None)
    def mock_run(*args,**kw):
        calls.append(kw);return {"status":"mocked_no_inference"}
    for c in CONDITIONS:
        job,spec=contract(8,c)
        message={"identity":{"episode_id":"one","sequence":len(calls)},"job":job,
                 "destination":str(tmp_path),"ablation":spec}
        monkeypatch.setattr(sys,"stdin",io.StringIO(json.dumps(message)+"\n"))
        worker.serve(SimpleNamespace(engine=SimpleNamespace(bundle=None)),
                     SimpleNamespace(device="cpu",profile_stages=False),sent.append,None,mock_run,lambda _:None)
        assert sent[-1]["identity"]==message["identity"]
    assert calls[0]["sample_factory"] is None and all(callable(c["sample_factory"]) for c in calls[1:])
    message["annotations"]={"action":"STOP"}
    monkeypatch.setattr(sys,"stdin",io.StringIO(json.dumps(message)+"\n"))
    with pytest.raises(ValueError,match="envelope"):
        worker.serve(None,None,None,None,None,None)


@pytest.mark.parametrize("wrong_identity",[False,True])
def test_sequential_runner_mocked_no_retry_and_rejects_wrong_epoch(monkeypatch,tmp_path,wrong_identity):
    module=import_runner(monkeypatch)
    plan={"matrix":module.matrix(),"cases":[{"case":name,"age_offset_s":.2} for name in CASES],
          "model_load_budget_s":150,"request_budget_s":45,"total_host_budget_s":600,"advisory_age_s":3.,"source_files":{}}
    monkeypatch.setenv("GO2_HARDWARE","0")
    monkeypatch.setenv("MUJOCO_RENDER_TEST","0")
    monkeypatch.setattr(module,"verify_plan",lambda _:plan)
    monkeypatch.setattr(module,"input_gate",lambda _:None)
    monkeypatch.setattr(module,"specification",lambda *a:{})
    monkeypatch.setattr(module,"validate_contract",lambda *a:None)
    monkeypatch.setattr(module,"render_result",lambda *a:None)
    monkeypatch.setattr(module,"PreparedFrameCache",lambda _:SimpleNamespace(status=lambda:{}))
    def prepare(case,condition,folder,cache):
        folder.mkdir()
        return {"episode_id":"one","clock_epoch":"epoch","frames":[]},{"source_indices":[]}
    monkeypatch.setattr(module,"prepare",prepare)
    events=[]
    class MockWorker:
        ready=True
        tensor_evidence_summary={"mock":True}
        def __init__(self,**kw):events.append("loaded_mock")
        def submit(self,request,now):
            events.append("submit");self.identity=request.context["identity"]
        def poll(self,now):
            events.append("receive")
            identity={**self.identity,"clock_epoch":"wrong"} if wrong_identity else self.identity
            return {"identity":identity,"result":{"status":"mocked"},"inference_duration_s":0,"host_inference_end_s":now}
        def close(self):events.append("closed")
    monkeypatch.setattr(module,"AblationWorker",MockWorker)
    args=SimpleNamespace(output=tmp_path,model_python="unused",assets="unused",resource_window_confirmed=True)
    if wrong_identity:
        with pytest.raises(ValueError,match="Wrong session"):
            module.run(args)
    else:
        module.run(args)
    summary=json.loads((tmp_path/"results/summary.json").read_text())
    assert summary["completed"]==(0 if wrong_identity else 25)
    assert len(summary["remaining_or_failed"])==(25 if wrong_identity else 0)
    assert events==["loaded_mock"]+[v for _ in range(1 if wrong_identity else 25) for v in ("submit","receive")]+["closed"]


def test_viewer_escapes_native_output_and_never_fabricates_predictions(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    import review_vln_context_ablation as viewer
    assert "NO NEW INFERENCE" in viewer.output_html(None)
    rendered=viewer.output_html({"status":"error","native_output":"<script>danger</script>"})
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered


def test_imports_do_not_reach_live_demo_or_hardware():
    files=[ROOT/"src/go2_perception/vln/context_ablation.py"]
    files+=list((ROOT/"scripts").glob("*context_ablation*.py"))
    files+=[ROOT/"scripts/check_vln_context_inputs.py"]
    for file in files:
        tree=ast.parse(file.read_text())
        imports=[n.module or "" for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        assert not any(any(bad in m for bad in ("unitree","cyclonedds","shadow_live","go2_control","sensor_bridge","interactive")) for m in imports)
