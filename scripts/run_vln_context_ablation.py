"""Declare/prepare the fixed context ablation; model execution is Stage B only.

No receiver, hardware client or annotation reader. Stage A uses only saved causal
observations and the existing calibrated preparation path. Never overwrites a run.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import time
from types import SimpleNamespace

from go2_perception.vln.context_ablation import CASES, CONDITIONS, canonical_segments, validate_contract
from go2_perception.vln.episodes import file_sha256
from go2_perception.vln.shadow import write_json
from go2_perception.vln.shadow_cache import PreparedFrameCache
from go2_perception.vln.shadow_worker import ProcessWorker
from run_vln_bounded_context import code_identity as shared_code_identity, manifest, prepare, read, render_result

ROOT = Path(__file__).resolve().parents[1]
BASE = "d3c227a112436d1b93c2d9b67bb69d5b89288295"
REVIEW = "bc8815319da554c6a3dee2523da9f2a72cf352aa"
PREVIOUS_PLAN_SHA = "55c73c298afe3df47ba58ac9109d39509d485d96d0a1091e0320518be50b408d"


def code_identity():
    files = [ROOT / "scripts" / name for name in (
        "run_vln_context_ablation.py", "check_vln_context_inputs.py",
        "review_vln_context_ablation.py", "vln_context_ablation_worker.py")]
    files.append(ROOT / "configs/pretrained_vln.yaml")
    return {**shared_code_identity(), **{str(p.relative_to(ROOT)): file_sha256(p) for p in files}}


def matrix():
    # Rotate the first condition across cases, fixed before model outcomes.
    return [{"case": case, "condition": c} for i, case in enumerate(CASES)
            for c in (*CONDITIONS[i:], *CONDITIONS[:i])]


def changed_files(files):
    return [p for p, h in files.items() if not Path(p).is_file() or file_sha256(p) != h]


def verify_plan(output):
    plan = read(output / "plan.json")
    if (plan["matrix"] != matrix() or plan["code_sha256"] != code_identity()
            or plan["advisory_age_s"] != 3.0 or plan["base_commit"] != BASE):
        raise ValueError("Declaration/code drift; preserve this run and declare a fresh one")
    if changed_files(plan["source_files"]):
        raise ValueError("Original package/result changed; do not run")
    return plan


def declare(a):
    prior = a.previous_run.resolve()
    if file_sha256(prior / "plan.json") != PREVIOUS_PLAN_SHA:
        raise ValueError("Expected the completed, immutable bounded-context declaration")
    old = read(prior / "plan.json")
    if changed_files(old["source_files"]):
        raise ValueError("Original source package changed")
    rows = read(prior / "results/rows.json")
    cases = []
    for name in CASES:
        case = next(c for c in old["cases"] if c["case"] == name)
        folders = {c: next(r["folder"] for r in rows if r["case"] == name
                   and r["condition"] == old_c and r["repeat"] == 0)
                   for c, old_c in (("F", "full"), ("B", "bounded"))}
        native = Path(folders["F"]) / "native/native_sample.json"
        cases.append({**case, "previous_folders": folders,
                      "baseline_segments": canonical_segments(read(native)),
                      "baseline_metadata_path": str(native), "baseline_metadata_sha256": file_sha256(native)})
    # Hash all existing results, not only selected outputs; annotations stay opaque
    # hashes in this process and are read only by the separate evaluation viewer.
    files = {**old["source_files"], **manifest([prior])}
    plan = {"schema": "go2-context-ablation-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "base_commit": BASE, "review_commit": REVIEW, "previous_run": str(prior),
            "previous_plan_sha256": PREVIOUS_PLAN_SHA, "cases": cases, "matrix": matrix(),
            "source_files": files, "code_sha256": code_identity(),
            "model_revision": old["model_revision"], "model_config": "configs/pretrained_vln.yaml",
            "max_images_bounded": 8, "max_native_slots_bounded": 8, "advisory_age_s": 3.0,
            "model_load_budget_s": 150, "request_budget_s": 45, "total_host_budget_s": 600,
            "conditions": {
                "F": "Original full history path; no custom factory",
                "B": "Unchanged anchor2_recent6_actual_time_v1 input builder",
                "T": "B with original logical indices, total_frames and native 4 Hz metadata only",
                "P": "B with each image's actual full-baseline pooling only; B grouping unchanged",
                "TP": "T plus original pools AND filtered full-baseline segment partition/prompt placeholders"},
            "confounds": ["TP changes grouping/prompt as well as time/pooling; not a clean factorial interaction",
                          "Removing intermediate observations may regroup incomplete temporal pairs",
                          "Native 4 Hz logical-grid timing is an approximation; actual capture/availability retained",
                          "All E1-E3 examples share test_c; doorway is one saved live cutoff; development only"],
            "runtime": {"device": "mps", "dtype": "bfloat16", "attention": "sdpa", "quantization": None,
                        "reset": "every decision; reconstruct causal history", "action_feedback": None,
                        "seed": 0, "max_new_tokens": 64, "one_loaded_model": True},
            "comparison": {"control": "E3 d0 TP/F pixels, normalized tensors, tokens, positions, pools; then generated IDs/full outputs",
                           "behavior": "complete native pointing/chunks/yaw/STOP and APOS completion separately; human review pending",
                           "diagnostic_mapping": "unchanged first_waypoint_diagnostic_v1, secondary only",
                           "recommendation": "No promotion without behavior retention; full baseline also misses stopping",
                           "timing": "one sample per case/condition; cold load separate, only request 0 cold-after-load; offline ages only"},
            "stage_b_status": "pending operator resource window; Stage A only currently authorized",
            "physical_execution_authorized": False, "annotations_consumed_by_model": False}
    a.output.mkdir(parents=True, exist_ok=False)
    write_json(a.output / "plan.json", plan)
    print(json.dumps({"plan": str(a.output / "plan.json"), "sha256": file_sha256(a.output / "plan.json"),
                      "requests": len(plan["matrix"]), "preserved_files": len(files)}))


def specification(case, condition):
    return {"condition": condition, "bounded_context": case["bounded_context"],
            "baseline_segments": case["baseline_segments"]}


def prepare_inputs(a):
    plan = verify_plan(a.output)
    root = a.output / "stage_a"
    root.mkdir(exist_ok=False)
    cases = {c["case"]: c for c in plan["cases"]}
    caches = {c: PreparedFrameCache(root / f"cache_{c}") for c in CONDITIONS}
    entries = []
    for number, item in enumerate(plan["matrix"]):
        c, case = item["condition"], cases[item["case"]]
        folder = root / f"{number:03d}_{item['case']}_{c}"
        job, provenance = prepare(case, "full" if c == "F" else "bounded", folder, caches[c])
        spec = specification(case, c)
        validate_contract(spec, job)
        write_json(folder / "job.json", job)
        write_json(folder / "preparation.json", provenance)
        write_json(folder / "ablation.json", spec)
        entries.append({**item, "number": number, "folder": str(folder), "job": job, "ablation": spec,
                        "previous_folder": case["previous_folders"]["F" if c == "F" else "B"]})
    write_json(root / "prepared_jobs.json", entries)
    write_json(root / "prepared_manifest.json", {"plan_sha256": file_sha256(a.output / "plan.json"),
                                                "files": manifest([root])})
    print(json.dumps({"prepared": len(entries), "root": str(root), "model_loaded": False}))


def input_gate(output):
    seal = read(output / "stage_a/prepared_manifest.json")
    checks = read(output / "stage_a/input_checks.json")
    if (seal["plan_sha256"] != file_sha256(output / "plan.json") or changed_files(seal["files"])
            or checks.get("plan_sha256") != seal["plan_sha256"] or checks.get("passed") is not True
            or checks.get("requests_constructed") != 25
            or checks.get("model_weights_loaded") is not False):
        raise ValueError("Stage A native-input gate failed or evidence changed")


class AblationWorker(ProcessWorker):
    def submit(self, request, now):
        if not self.ready or self.busy or self.closed:
            raise ValueError("Only one request may be in flight")
        message = {"identity": request.context["identity"], "job": request.job,
                   "destination": str(request.folder / "native"), "ablation": request.context["ablation"]}
        self.process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
        self.process.stdin.flush()
        self.busy = True


def run(a):
    # A CLI flag records coordination; it is not an automatic grant of permission.
    if not a.resource_window_confirmed:
        raise ValueError("Stage B requires a separately agreed resource window")
    if os.getenv("GO2_HARDWARE") != "0" or os.getenv("MUJOCO_RENDER_TEST") != "0":
        raise ValueError("Disable hardware/rendering opt-ins explicitly")
    plan = verify_plan(a.output)
    input_gate(a.output)
    root = a.output / "results"
    root.mkdir(exist_ok=False)
    started = time.monotonic()
    rows, failure, worker = [], None, None
    try:
        worker = AblationWorker(python=a.model_python, assets=a.assets, config=ROOT / "configs/pretrained_vln.yaml",
                                device="mps", output=root / "worker", profile_stages=True,
                                preparation_mode="cached", worker_script=ROOT / "scripts/vln_context_ablation_worker.py")
        while not worker.ready:
            worker.poll(time.monotonic())
            if time.monotonic() - started > plan["model_load_budget_s"]:
                raise TimeoutError("Declared model-load budget exceeded")
            time.sleep(.01)
        cases = {c["case"]: c for c in plan["cases"]}
        caches = {c: PreparedFrameCache(root / f"cache_{c}") for c in CONDITIONS}
        for number, item in enumerate(plan["matrix"]):
            if time.monotonic() - started > plan["total_host_budget_s"]:
                raise TimeoutError("Declared total host budget exceeded")
            case, c = cases[item["case"]], item["condition"]
            folder = root / f"{number:03d}_{item['case']}_{c}"
            begin = time.monotonic()
            job, provenance = prepare(case, "full" if c == "F" else "bounded", folder, caches[c])
            spec = specification(case, c)
            validate_contract(spec, job)
            write_json(folder / "job.json", job)
            write_json(folder / "preparation.json", provenance)
            write_json(folder / "ablation.json", spec)
            request = SimpleNamespace(job=job, folder=folder, context={"ablation": spec, "identity": {
                "sequence": number, "episode_id": job["episode_id"], "clock_epoch": job["clock_epoch"]}})
            prepared = time.monotonic()
            worker.submit(request, prepared)
            response = None
            while response is None:
                response = worker.poll(time.monotonic())
                if time.monotonic() - begin > plan["request_budget_s"] or time.monotonic() - started > plan["total_host_budget_s"]:
                    raise TimeoutError("Declared request/total budget exceeded; no retry")
                if response is None:
                    time.sleep(.005)
            received = time.monotonic()
            if response["identity"] != request.context["identity"]:
                raise ValueError("Wrong session/epoch/sequence response")
            result = response["result"]
            row = {**item, "number": number, "folder": str(folder), "status": result["status"],
                   "cold_after_load": number == 0, "supplied_images": len(job["frames"]),
                   "prepare_s": prepared - begin, "worker_s": response["inference_duration_s"],
                   "response_delivery_s": received - response["host_inference_end_s"],
                   "request_to_receipt_s": received - begin, "age_offset_s": case["age_offset_s"],
                   "reconstructed_age_at_receipt_s": case["age_offset_s"] + received - begin,
                   "cache": caches[c].status(), "live_latency_measured": False}
            render_result(folder, job, result, row, provenance["source_indices"])
            delivered = time.monotonic()
            row.update(request_to_review_file_s=delivered - begin, review_file_write_s=delivered - received,
                       reconstructed_age_s=case["age_offset_s"] + delivered - begin,
                       stale_under_3s=case["age_offset_s"] + delivered - begin > plan["advisory_age_s"])
            write_json(folder / "response.json", response)
            write_json(folder / "timing.json", row)
            rows.append(row)
            write_json(root / "rows.json", rows)
            print(json.dumps({k: row[k] for k in ("number", "case", "condition", "status", "request_to_review_file_s")}), flush=True)
    except Exception as error:
        failure = {"error": repr(error), "failed_request_index": len(rows), "no_retry": True}
        write_json(root / "failure.json", failure)
        raise
    finally:
        if worker is not None:
            worker.close()
        write_json(root / "summary.json", {"completed": len(rows), "declared": 25,
                   "remaining_or_failed": plan["matrix"][len(rows):], "failure": failure,
                   "host_s": time.monotonic() - started, "changed_sources": changed_files(plan["source_files"]),
                   "parent_lifetime_peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "tensor_evidence": None if worker is None else worker.tensor_evidence_summary,
                   "physical_execution_authorized": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--previous-run", type=Path)
    p.add_argument("--model-python", type=Path)
    p.add_argument("--assets", type=Path)
    p.add_argument("--resource-window-confirmed", action="store_true")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--declare-only", action="store_true")
    group.add_argument("--prepare-inputs", action="store_true")
    group.add_argument("--run", action="store_true")
    a = p.parse_args()
    a.output = a.output.resolve()
    if a.declare_only:
        if a.previous_run is None: p.error("--previous-run is required")
        declare(a)
    elif a.prepare_inputs:
        prepare_inputs(a)
    else:
        if a.model_python is None or a.assets is None: p.error("Existing --model-python and --assets are required")
        run(a)


if __name__ == "__main__":
    main()
