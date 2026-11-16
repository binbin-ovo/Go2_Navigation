"""Evaluation-only aggregation; no model import and no modification of source labels."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

import cv2
import numpy as np

from go2_perception.vln.episodes import file_sha256, load_annotations, load_episode
from go2_perception.vln.pretrained import comparison_action
from go2_perception.vln.shadow import write_json
from go2_perception.vln.shadow_display import escaped, relative


def read(path):
    return json.loads(Path(path).read_text())


def score(result, reference):
    action = (comparison_action(result["waypoints"], native_stop=result["decoder_stop"])
              if result.get("status")=="decoded" else None)
    yaw = result.get("waypoints", [[0,0,0]])[0][2]
    direction = "left" if yaw>=math.radians(5) else "right" if yaw<=-math.radians(5) else None
    accept = [] if reference is None else reference["acceptable_actions"]
    expected = "left" if accept==["TURN_LEFT"] else "right" if accept==["TURN_RIGHT"] else None
    return {"mapped_action":action, "human_reference":reference,
            "agreement":None if reference is None else action in accept,
            "mapping_supported":action is not None,
            "first_waypoint_turn_sign":direction,
            "wrong_direction":expected is not None and direction is not None and direction!=expected,
            "missed_native_stop":accept==["STOP"] and result.get("decoder_stop") is not True,
            "repeated_turn_after_reference_completion":accept==["STOP"] and direction is not None,
            "native_stop":result.get("decoder_stop"),
            "completion_claim":result.get("pointing",{}).get("apos")==1299,
            "absence_recognition": "not established; search-turn agreement does not establish recognition" if reference and reference["target_visibility"]=="absent" else "not an absence probe"}


def equivalent_images(left, right):
    """Intersection equality checks spatial preprocessing only, never context equivalence."""
    count=0
    for frame in right["frames"]:
        other = next(f for f in left["frames"] if f["observation_s"]==frame["observation_s"])
        if other["available_s"] != frame["available_s"]:
            return False, count
        if file_sha256(frame["path"]) != frame["sha256"] or file_sha256(other["path"]) != other["sha256"]:
            raise ValueError("Prepared evidence changed")
        if not np.array_equal(cv2.imread(frame["path"]),cv2.imread(other["path"])):
            return False, count
        count+=1
    return True,count


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    a=p.parse_args()
    plan=read(a.run/"plan.json")
    root=a.run/"results"
    rows=read(root/"rows.json")
    output=a.run/"evaluation"
    output.mkdir(exist_ok=False)
    cases={c["case"]:c for c in plan["cases"]}
    refs={}
    for case in plan["cases"]:
        if case["kind"]=="episode":
            ep=load_episode(case["episode"])
            path=ep.root.parent/"evaluation_only/annotations.json"
            if file_sha256(path)!=plan["source_files"][str(path)]:
                raise ValueError("Annotation changed after declaration")
            labels=load_annotations(path,ep)
            refs[case["case"]]=next((r for r in labels["decisions"] if r["decision_id"]==case["decision_id"]),None) if labels["reviewer"]["status"]=="reviewed" else None
    bycase=defaultdict(dict)
    enriched=[]
    for row in rows:
        folder=Path(row["folder"])
        result=read(folder/"native/result.json")
        result_score=score(result,refs.get(row["case"]))
        enriched.append({**row, **result_score, "native_output":result.get("native_output"),
                         "pointing":result.get("pointing"),"waypoints":result.get("waypoints"),
                         "stage_profile":result.get("stage_profile"),"timing":result.get("timing"),"memory":result.get("memory")})
        bycase[(row["case"],row["repeat"])][row["condition"]]=(row,result,read(folder/"job.json"))
    comparisons=[]
    for (name,repeat),conditions in bycase.items():
        if set(conditions)!={"full","bounded"}:
            comparisons.append({"case":name,"repeat":repeat,"incomplete_pair":True});continue
        f,fr,fj=conditions["full"]; b,br,bj=conditions["bounded"]
        equal,count=equivalent_images(fj,bj)
        ids=("instruction","episode_id","clock_epoch","decision_id","decision_time_s")
        if any(fj[k]!=bj[k] for k in ids):
            raise ValueError("Comparison changed instruction/identity/cutoff")
        same={k:fr.get(k)==br.get(k) for k in ("native_output","pointing","waypoints","decoder_stop","explicit_stop_code")}
        baseline_original=None
        baseline_tensor_equal=None
        if cases[name]["kind"]=="live_recording":
            number=int(name[-1])
            original=read(Path(plan["source_run"])/f"requests/{number:06d}/native/result.json")
            baseline_original={k:original.get(k)==fr.get(k) for k in same}
            if not equivalent_images(cases[name]["job"],fj)[0]:
                raise ValueError("Baseline pixels differ from Claude's original job")
            old_archive=Path(plan["source_run"])/f"requests/{number:06d}/native/model_inputs_and_generated_ids.npz"
            new_archive=Path(f["folder"])/"native/model_inputs_and_generated_ids.npz"
            with np.load(old_archive) as old, np.load(new_archive) as new:
                baseline_tensor_equal=set(old.files)==set(new.files) and all(np.array_equal(old[k],new[k]) for k in old.files)
        comparisons.append({"case":name,"repeat":repeat,"shared_calibrated_pixels_equal":equal,
                            "shared_images_checked":count,"native_equal":same,"baseline_original_native_equal":baseline_original,
                            "baseline_original_all_tensor_bits_equal":baseline_tensor_equal,
                            "full_to_bounded_delivery_ratio":b["request_to_review_file_s"]/f["request_to_review_file_s"],
                            "full_score":score(fr,refs.get(name)),"bounded_score":score(br,refs.get(name))})
    performance={}
    for name in ("live_r1","live_r2","live_r3"):
        performance[name]={}
        for condition in ("full","bounded"):
            group=[r for r in enriched if r["case"]==name and r["condition"]==condition]
            performance[name][condition]={"n":len(group),"cold_first_request_included":any(r["cold_after_load"] for r in group)}
            for key in ("prepare_s","worker_s","response_delivery_s","request_to_review_file_s","reconstructed_age_s"):
                values=[r[key] for r in group]
                performance[name][condition][key]={"values":values,"median":statistics.median(values)} if values else None
    totals={}
    for condition in ("full","bounded"):
        group=[r for r in enriched if r["condition"]==condition and r["human_reference"] is not None and "e4_" not in r["case"]]
        totals[condition]={"reviewed_E1_E3":len(group),"action_agreements":sum(r["agreement"] for r in group),
                           "supported_mappings":sum(r["mapping_supported"] for r in group),
                           "wrong_direction":sum(r["wrong_direction"] for r in group),
                           "missed_native_stop":sum(r["missed_native_stop"] for r in group),
                           "repeated_turn_after_completion":sum(r["repeated_turn_after_reference_completion"] for r in group)}
    report={"policy":plan["policy"],"matrix":plan["matrix"],"performance":performance,
            "E1_E3":totals,"comparisons":comparisons,"requests":enriched,
            "new_human_review":"pending; existing labels only", "live_latency_measured":False,
            "age_limit_s":3.0,"execution_authorized":False}
    write_json(output/"comparison.json",report)
    # Every row is linked with native history/chunk; human labels appear ONLY here.
    body=[]
    for row in enriched:
        link=relative(Path(row["folder"])/"index.html",a.run)
        body.append(f'<section><h2>{escaped(row["case"])} / {row["condition"]} / repeat {row["repeat"]}</h2>'
                    f'<a href="../{link}">Native inputs, pointing and full waypoint chunk</a>'
                    f'<p>Delivered in {row["request_to_review_file_s"]:.3f}s; reconstructed age {row["reconstructed_age_s"]:.3f}s; stale={row["stale_under_3s"]}</p>'
                    f'<pre>{escaped(row["native_output"])}</pre><p>Mapped: {escaped(row["mapped_action"])}; native STOP: {row["native_stop"]}; completion claim: {row["completion_claim"]}</p>'
                    f'<details><summary>EVALUATION ONLY — existing human reference</summary><pre>{escaped(json.dumps(row["human_reference"],indent=2))}</pre></details></section>')
    (output/"index.html").write_text('<meta charset="utf-8"><style>body{font:16px sans-serif;max-width:1100px;margin:auto}section{border-top:1px solid #aaa;padding:15px}pre{white-space:pre-wrap}</style>'
        '<h1>All bounded-context diagnostic results</h1><p>Private offline development examples. No execution. New outputs await human visual review. E4 search agreement is not absence recognition.</p>'+''.join(body))
    print(json.dumps({"evaluation":str(output),"E1_E3":totals,"requests":len(rows)}))


if __name__=="__main__":
    main()
