"""Declare or run ONE fixed offline full-vs-bounded comparison. No sensor access.

Use --declare-only first. --run consumes the immutable plan, refuses code/source
drift, and writes into a new results directory. Evaluation labels never enter jobs.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import time
from types import SimpleNamespace

import cv2
from go2_perception.vln.bounded_context import POLICY, TIERS, select_context
from go2_perception.vln.episodes import file_sha256, load_episode
from go2_perception.vln.pretrained import causal_history
from go2_perception.vln.shadow import write_json
from go2_perception.vln.shadow_cache import PreparedFrameCache
from go2_perception.vln.shadow_display import escaped, path_svg, relative, target_fraction
from go2_perception.vln.shadow_worker import ProcessWorker
from go2_perception.vln.temporal_inputs import prepare_views

ROOT = Path(__file__).resolve().parents[1]
EPISODES = ["p6b_test_c_e1_turn_right", "p6b_test_c_e2_turn_left",
            "p6b_test_c_e3_forward_bottles", "p6b_test_c_e4_absent_sofa"]


def read(path):
    return json.loads(Path(path).read_text())


def manifest(paths):
    return {str(p): file_sha256(p) for root in paths for p in sorted(root.rglob("*")) if p.is_file()}


def code_identity():
    files = [ROOT / "scripts" / name for name in ("run_vln_bounded_context.py", "summarize_vln_bounded_context.py", "vln_bounded_worker.py",
             "lightnav_worker.py", "vln_shadow_worker.py")]
    files += sorted((ROOT / "src/go2_perception/vln").glob("*.py"))
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in files}


def declare(a):
    a.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for number in (1, 2, 3):
        request = a.source_run / "requests" / f"{number:06d}"
        job, snapshot = read(request / "job.json"), read(request / "snapshot.json")
        if len(job["frames"]) != [5, 19, 39][number - 1]:
            raise ValueError("Expected Claude's verified cached live 5/19/39 jobs")
        context = select_context(job["frames"], job["decision_time_s"])
        cases.append({"case": f"live_r{number}", "kind": "live_recording", "job": job,
                      "snapshot": snapshot, "bounded_context": context,
                      "age_offset_s": snapshot["host_submit_monotonic_s"] - snapshot["current_source"]["exposure_monotonic_s"]})
    for name in EPISODES:
        ep = load_episode(a.episodes / name / "model_input/episode.json")
        for decision in ep.data["decisions"]:
            selection = causal_history(ep, decision["decision_id"])
            context = select_context(selection["frames"], selection["decision_time_s"])
            cases.append({"case": f"{name}_{decision['decision_id']}", "kind": "episode",
                          "episode": str(ep.path), "decision_id": decision["decision_id"],
                          "selection": selection, "bounded_context": context,
                          "age_offset_s": selection["current_observation_age_s"]})
    matrix = []
    # Two repetitions of each live cutoff, with reversed condition order. No retries.
    for repeat in range(2):
        for case in cases[:3]:
            for condition in (["full", "bounded"] if repeat == 0 else ["bounded", "full"]):
                matrix.append({"case": case["case"], "condition": condition, "repeat": repeat})
    # All 11 original pilot decisions, once per condition; counterbalance first condition.
    for i, case in enumerate(cases[3:]):
        for condition in (["full", "bounded"] if i % 2 == 0 else ["bounded", "full"]):
            matrix.append({"case": case["case"], "condition": condition, "repeat": 0})
    plan = {"schema": "go2-bounded-context-experiment-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "base_commit": "1f93d7eff57906ce26c4b3ab4109effb4ca71edd", "policy": POLICY, "tiers": TIERS,
            "max_images": 8, "max_native_slots": 8, "advisory_age_s": 3.0,
            "model_load_budget_s": 150, "request_budget_s": 45, "total_host_budget_s": 900,
            "max_source_frames": 1024, "max_history_s": 60, "matrix": matrix, "cases": cases,
            "source_files": manifest([a.source_run, *[a.episodes / n for n in EPISODES]]),
            "code_sha256": code_identity(), "source_run": str(a.source_run), "episodes_root": str(a.episodes),
            "model_revision": "826dc5fbfa37afa8293d2e336d329b6ffc0bfb64",
            "comparison": {"mapping": "unchanged first_waypoint_diagnostic_v1",
                           "native_chunks_retained": True, "apos_1299": "unverified completion claim",
                           "absent_target": "E4 search turns may agree with existing labels; not evidence of absence recognition",
                           "acceptable_observed_behavior": "no lost E1-E3 action agreements, no new wrong direction or missed native STOP; inspect all changed pointing/chunks; human acceptability remains review pending",
                           "material_latency_help": "at least 20 percent lower median delivery duration for 39-frame case; no freshness claim unless reconstructed age <= 3 s",
                           "correlation": "E1-E4 all test_c; live cutoffs one other session; development only"},
            "timing": "fixed-cutoff offline, cached preparation in both conditions; not live; one loaded model; first request cold-after-load reported separately",
            "physical_execution_authorized": False, "annotations_consumed_by_model": False}
    write_json(a.output / "plan.json", plan)
    print(json.dumps({"declaration": str(a.output / "plan.json"), "requests": len(matrix),
                      "cases": [(c["case"], c["bounded_context"]["source_count"]) for c in cases]}))


class ComparisonWorker(ProcessWorker):
    def submit(self, request, now):
        if not self.ready or self.busy or self.closed:
            raise ValueError("Only one request may be in flight")
        message = {"identity": request.context["identity"], "job": request.job,
                   "destination": str(request.folder / "native"),
                   "bounded_context": request.context["bounded_context"]}
        self.process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
        self.process.stdin.flush()
        self.busy = True


def prepare(case, condition, folder, cache):
    context = case["bounded_context"] if condition == "bounded" else None
    indices = context["source_indices"] if context else list(range(case["bounded_context"]["source_count"]))
    folder.mkdir(parents=True, exist_ok=False)
    frames, provenance, stages = [], [], []
    if case["kind"] == "live_recording":
        job = case["job"]
        for i in indices:
            old = job["frames"][i]
            source = case["snapshot"]["history"][i]["source"]
            path = folder / f"frame_{i:04d}.png"
            sha, metadata, identity, timing = cache.prepare(source, path)
            frames.append({**old, "path": str(path), "sha256": sha})
            provenance.append({"source_index": i, "source": source, "rectification": metadata, "cache": identity})
            stages.append(timing)
    else:
        ep = load_episode(case["episode"], verify_files=False)
        chosen = [case["selection"]["frames"][i] for i in indices]
        selected = replace(ep, data={**ep.data, "frames": chosen})
        # Existing calibrated_view via prepare_views, once, with unchanged package settings.
        horizon = case["selection"]["decision_time_s"] - ep.data["bounds"]["start_s"] + 1
        _, views = prepare_views(selected, case["decision_id"], max_frames=len(chosen), horizon_s=horizon)
        for i, frame, view in zip(indices, chosen, views, strict=True):
            path = folder / f"frame_{i:04d}.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(view.image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                raise OSError("Cannot save calibrated image")
            frames.append({"path": str(path), "sha256": file_sha256(path),
                           "observation_s": frame["observation_s"], "available_s": frame["available_s"]})
            provenance.append({"source_index": i, "source": frame, "rectification": view.metadata})
        job = {"instruction": ep.data["instruction"], "episode_id": ep.data["episode_id"],
               "clock_epoch": ep.data["clock"]["epoch"], "decision_id": case["decision_id"],
               "decision_time_s": case["selection"]["decision_time_s"]}
    return {**job, "frames": frames}, {"frames": provenance, "stages": stages,
            "bounded_context": context, "source_indices": indices, "annotations_consumed": False}


def render_result(folder, job, result, row, indices):
    """Reuse existing pointing/path renderers; annotations are absent at inference."""
    images = []
    for i, frame in zip(indices, job["frames"], strict=True):
        image = folder / "native" / f"input_{i:04d}.png"
        if image.is_file():
            images.append(f'<figure><img width="224" src="{relative(image, folder)}"><figcaption>{i}: {frame["observation_s"]:.6f}s</figcaption></figure>')
    point = target_fraction(result.get("pointing", {}))
    overlay = "" if point is None else f'<span style="position:absolute;left:{point[0]*100}%;top:{point[1]*100}%;color:red;font-size:32px;transform:translate(-50%,-50%)">+</span>'
    current = folder / "native" / f"input_{indices[-1]:04d}.png"
    view = "" if not current.exists() else f'<div style="position:relative;width:448px"><img width="448" src="{relative(current,folder)}">{overlay}</div>'
    (folder / "index.html").write_text('<meta charset="utf-8"><title>Offline bounded-context diagnostic</title>'
        '<style>body{font:16px sans-serif;max-width:1200px;margin:auto}figure{display:inline-block;margin:4px}svg{width:440px}pre{white-space:pre-wrap}</style>'
        '<h1>OFFLINE ONLY — no execution or navigation success</h1>'
        f'<h2>{escaped(job["instruction"])}</h2>{view}{path_svg(result.get("waypoints"))}'
        f'<pre>{escaped(json.dumps({"native":result.get("native_output"), "pointing":result.get("pointing"), "STOP":result.get("decoder_stop"), "completion_claim":result.get("pointing",{}).get("apos")==1299, "waypoints":result.get("waypoints"), "timing":row},indent=2))}</pre>'
        '<h2>Actual native history — timestamps preserved</h2>' + ''.join(images))


def run(a):
    if os.getenv("GO2_HARDWARE") != "0" or os.getenv("MUJOCO_RENDER_TEST") != "0":
        raise ValueError("Explicitly disable physical and rendering opt-ins")
    plan = read(a.output / "plan.json")
    if plan["code_sha256"] != code_identity():
        raise ValueError("Code changed after declaration; preserve it and declare a fresh experiment")
    for path, sha in plan["source_files"].items():
        if file_sha256(path) != sha:
            raise ValueError(f"Source changed: {path}")
    result_root = a.output / "results"
    result_root.mkdir(exist_ok=False)
    started = time.monotonic()
    worker = ComparisonWorker(python=a.model_python, assets=a.assets, config=ROOT / "configs/pretrained_vln.yaml",
                              device="mps", output=result_root / "worker", profile_stages=True,
                              preparation_mode="cached", worker_script=ROOT / "scripts/vln_bounded_worker.py")
    rows, caches = [], {}
    failure = None
    try:
        while not worker.ready:
            worker.poll(time.monotonic())
            if time.monotonic() - started > plan["model_load_budget_s"]:
                raise TimeoutError("Declared load budget exceeded")
            time.sleep(.01)
        cases = {c["case"]: c for c in plan["cases"]}
        for number, item in enumerate(plan["matrix"]):
            if time.monotonic() - started > plan["total_host_budget_s"]:
                raise TimeoutError("Declared total host budget exceeded; remaining rows not attempted")
            case = cases[item["case"]]
            key = (item["condition"], item["repeat"])
            if key not in caches:
                caches[key] = PreparedFrameCache(result_root / f"cache_{key[0]}_{key[1]}")
            folder = result_root / f"{number:03d}_{item['case']}_{item['condition']}_r{item['repeat']}"
            begin = time.monotonic()
            job, provenance = prepare(case, item["condition"], folder, caches[key])
            write_json(folder / "job.json", job)
            write_json(folder / "preparation.json", provenance)
            request = SimpleNamespace(job=job, folder=folder, context={
                "identity": {"sequence": number, "episode_id": job["episode_id"], "clock_epoch": job["clock_epoch"]},
                "bounded_context": provenance["bounded_context"]})
            prepared = time.monotonic()
            worker.submit(request, prepared)
            response = None
            while response is None:
                response = worker.poll(time.monotonic())
                if time.monotonic() - begin > plan["request_budget_s"]:
                    raise TimeoutError("Declared request budget exceeded; no retry")
                if response is None:
                    time.sleep(.005)
            received = time.monotonic()
            if response["identity"] != request.context["identity"]:
                raise ValueError("Wrong session/epoch/sequence response")
            result = response["result"]
            row = {**item, "number": number, "folder": str(folder), "status": result["status"],
                   "supplied_images": len(job["frames"]), "source_candidate_images": case["bounded_context"]["source_count"],
                   "native_unique_images": len(result.get("native_selected_frame_indices", [])),
                   "native_slots": sum(len(s["frame_indices"]) for s in result.get("native_segments", [])),
                   "elapsed_history_s": case["bounded_context"]["selected_observation_s"][-1] - case["bounded_context"]["origin_observation_s"],
                   "cold_after_load": number == 0, "prepare_s": prepared-begin,
                   "worker_s": response["inference_duration_s"],
                   "response_delivery_s": received-response["host_inference_end_s"],
                   "request_to_receipt_s": received-begin, "queue_pending": 0,
                   "reconstructed_age_at_receipt_s": case["age_offset_s"]+received-begin,
                   "stale_at_receipt_under_3s": case["age_offset_s"]+received-begin > 3,
                   "age_offset_s": case["age_offset_s"], "cache": caches[key].status()}
            render_result(folder, job, result, row, provenance["source_indices"])
            delivered = time.monotonic()
            row.update(request_to_review_file_s=delivered-begin, review_file_write_s=delivered-received,
                       reconstructed_age_s=case["age_offset_s"]+delivered-begin,
                       stale_under_3s=case["age_offset_s"]+delivered-begin > 3)
            write_json(folder / "response.json", response)
            write_json(folder / "timing.json", row)
            rows.append(row)
            write_json(result_root / "rows.json", rows)
            print(json.dumps({k:row[k] for k in ("number","case","condition","status","supplied_images","request_to_review_file_s","reconstructed_age_s")}), flush=True)
            # Failed decode/output remains a row; no selective retry or instruction reset.
        links = ''.join(f'<li><a href="{relative(Path(r["folder"])/"index.html",result_root)}">{escaped(r["case"])} {r["condition"]} r{r["repeat"]}</a></li>' for r in rows)
        (result_root / "index.html").write_text('<h1>All declared offline requests</h1><ul>'+links+'</ul>')
    except Exception as error:
        failure = {"error": repr(error), "next_or_failed_request": plan["matrix"][len(rows)] if len(rows)<len(plan["matrix"]) else None,
                   "note": "Inspect request directory/worker log. No retries; remaining declared requests were not completed."}
        write_json(result_root / "failure.json", failure)
        raise
    finally:
        worker.close()
        changed = [p for p,h in plan["source_files"].items() if not Path(p).is_file() or file_sha256(p)!=h]
        write_json(result_root / "summary.json", {"completed":len(rows), "declared":len(plan["matrix"]),
                   "not_attempted":plan["matrix"][len(rows):], "host_s":time.monotonic()-started,
                   "source_files_checked":len(plan["source_files"]), "changed_sources":changed,
                   "failure":failure,
                   "parent_lifetime_peak_rss_bytes_macos":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "tensor_evidence":worker.tensor_evidence_summary, "physical_execution_authorized":False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path)
    p.add_argument("--episodes", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets", type=Path)
    p.add_argument("--model-python", type=Path)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--declare-only", action="store_true")
    group.add_argument("--run", action="store_true")
    a = p.parse_args()
    a.output = a.output.absolute()
    if a.declare_only:
        if a.source_run is None or a.episodes is None:
            p.error("Declaration requires source run and episode root")
        declare(a)
    else:
        if a.assets is None or a.model_python is None:
            p.error("Inference requires existing isolated model Python and cached assets")
        run(a)


if __name__ == "__main__":
    main()
