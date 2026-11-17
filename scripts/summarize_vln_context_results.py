"""Post-run, evaluation-only audit of the frozen 25-request context experiment.

No model imports or new inference. Reads complete saved archives and the separate
review report; never writes a model job, annotation, plan or original result.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from go2_perception.vln.context_ablation import CASES, CONDITIONS, canonical_segments
from go2_perception.vln.episodes import file_sha256
from summarize_vln_bounded_context import score

NATIVE_FIELDS = ("native_output", "pointing", "rvq_codes", "waypoints", "decoder_stop", "explicit_stop_code")


def read(path):
    return json.loads(Path(path).read_text())


def signatures(arrays):
    return {k: {"dtype": str(v.dtype), "shape": list(v.shape),
                "sha256": hashlib.sha256(v.tobytes()).hexdigest()} for k, v in arrays.items()}


def archive(path):
    with np.load(path) as saved:
        inputs = {k: saved[k] for k in saved.files if k != "generated_ids"}
        ids, generated = inputs["input_ids"], saved["generated_ids"]
        if generated.ndim != 2 or ids.ndim != 2 or not np.array_equal(generated[:, :ids.shape[-1]], ids):
            raise ValueError("Generated archive lacks the exact recorded prompt prefix")
        return {"inputs": signatures(inputs),
                "generated_with_prompt": signatures({"ids": generated}),
                "generated_suffix": signatures({"ids": generated[:, ids.shape[-1]:]})}


def native_equal(a, b):
    # Missing outputs must never be reported as equal successful predictions.
    return (a.get("status") == b.get("status") == "decoded"
            and all(k in a and k in b for k in NATIVE_FIELDS)
            and isinstance(a["waypoints"], list) and len(a["waypoints"]) == 10
            and isinstance(a["native_output"], str) and bool(a["native_output"])
            and all(a.get(k) == b.get(k) for k in NATIVE_FIELDS))


def compare(a, b):
    return {"inputs_equal": a["archive"]["inputs"] == b["archive"]["inputs"],
            "generated_suffix_equal": a["archive"]["generated_suffix"] == b["archive"]["generated_suffix"],
            "generated_with_prompt_equal": a["archive"]["generated_with_prompt"] == b["archive"]["generated_with_prompt"],
            "complete_native_equal": native_equal(a["native"], b["native"])}


def summarize(root):
    plan = read(root / "plan.json")
    summary = read(root / "results/summary.json")
    rows = read(root / "results/rows.json")
    if summary["completed"] != 25 or summary["failure"] or len(rows) != 25:
        raise ValueError("This aggregate requires all 25 rows; retain partial results/failure report instead")
    if [{"case": r["case"], "condition": r["condition"]} for r in rows] != plan["matrix"]:
        raise ValueError("Results do not match the frozen matrix/order")
    stage_a = read(root / "stage_a/input_checks.json")
    frozen = {(r["case"], r["condition"]): r for r in stage_a["records"]}
    refs = {(r["case"], r["condition"]): r["human_reference_evaluation_only"]
            for r in read(root / "review_stage_b.json")["requests"]}
    cases = {c["case"]: c for c in plan["cases"]}
    by_case, details, prior_comparisons = {}, [], []
    for row in rows:
        name, c = row["case"], row["condition"]
        folder = Path(row["folder"])
        native = read(folder / "native/result.json")
        evidence = archive(folder / "native/model_inputs_and_generated_ids.npz")
        old = frozen[name, c]
        job, old_job = read(folder / "job.json"), read(Path(old["folder"]) / "job.json")
        identity = ("instruction", "episode_id", "clock_epoch", "decision_id", "decision_time_s")
        stamp_pixels = ("observation_s", "available_s", "sha256")
        native_images = [folder / "native" / f"input_{i:04d}.png" for i in native["native_selected_frame_indices"]]
        checks = {
            "model_generate_tensors_equal_stage_a": evidence["inputs"] == old["generation_inputs"],
            "tensor_types_equal_stage_a": read(folder / "native/tensor_types.json") == old["generation_types"],
            "native_metadata_equal_stage_a": canonical_segments(read(folder / "native/native_sample.json")) == old["canonical_segments"],
            "identity_instruction_cutoff_equal_stage_a": all(job[k] == old_job[k] for k in identity),
            "calibrated_files_timestamps_equal_stage_a": len(job["frames"]) == len(old_job["frames"])
                and all(all(a[k] == b[k] for k in stamp_pixels) and file_sha256(a["path"]) == a["sha256"]
                        for a, b in zip(job["frames"], old_job["frames"])),
            "normalized_image_evidence_equal_stage_a": all(file_sha256(p) == file_sha256(Path(old["folder"]) / "native" / p.name) for p in native_images),
        }
        w = native.get("waypoints")
        path = None if w is None else {
            "first": w[0], "last": w[-1],
            "first_yaw_deg": math.degrees(w[0][2]), "last_yaw_deg": math.degrees(w[-1][2]),
            "max_abs_lateral_m": max(abs(p[1]) for p in w),
            "yaw_range_deg": [math.degrees(min(p[2] for p in w)), math.degrees(max(p[2] for p in w))],
            "all_ten_waypoints": w, "units": "absolute ego SE2 metres/radians; no time base or execution"}
        detail = {**row, "input_checks": checks, "pointing": native.get("pointing"),
                  "native_output": native.get("native_output"), "path": path,
                  "native_stop": native.get("decoder_stop"), "explicit_stop_code": native.get("explicit_stop_code"),
                  "completion_claim": native.get("pointing", {}).get("apos") == 1299,
                  "memory": native["memory"], "timing": native.get("timing"), "stage_profile": native["stage_profile"],
                  "existing_reference_diagnostic": score(native, refs[name, c]),
                  "new_human_review": "pending; old review is not transferred"}
        details.append(detail)
        by_case.setdefault(name, {})[c] = {"archive": evidence, "native": native}
        if c in ("F", "B"):
            prior = Path(cases[name]["previous_folders"][c])
            prev = {"archive": archive(prior / "native/model_inputs_and_generated_ids.npz"),
                    "native": read(prior / "native/result.json")}
            prior_comparisons.append({"case": name, "condition": c, **compare(by_case[name][c], prev),
                                      "previous_native_output": prev["native"].get("native_output"),
                                      "current_native_output": native.get("native_output")})
    pairwise = {name: {f"{a}_{b}": compare(group[a], group[b]) for a, b in
                       (("T", "B"), ("P", "B"), ("TP", "B"), ("TP", "F"), ("P", "F"), ("TP", "T"))}
                for name, group in by_case.items()}
    totals = {}
    for c in CONDITIONS:
        group = [r for r in details if r["condition"] == c]
        labeled = [r for r in group if r["existing_reference_diagnostic"]["human_reference"] is not None]
        totals[c] = {"requests": len(group), "decoded": sum(r["status"] == "decoded" for r in group),
                     "native_stop": sum(r["native_stop"] is True for r in group),
                     "completion_claim": sum(r["completion_claim"] for r in group),
                     "offline_age_under_3s": sum(not r["stale_under_3s"] for r in group),
                     "reference_cases": len(labeled),
                     "secondary_mapped_agreements": sum(r["existing_reference_diagnostic"]["agreement"] is True for r in labeled),
                     "missed_reference_stop": sum(r["existing_reference_diagnostic"]["missed_native_stop"] for r in labeled)}
    model = read(root / "results/worker/model.json")
    memory = [model["load_memory"], *[r["memory"] for r in details]]
    return {"schema": "go2-context-ablation-results-v1", "plan_sha256": file_sha256(root / "plan.json"),
            "analysis_script_sha256": file_sha256(__file__), "analysis_is_post_run_only": True,
            "all_stage_a_input_checks_pass": all(all(r["input_checks"].values()) for r in details),
            "requests": details, "pairwise": pairwise, "previous_F_B": prior_comparisons,
            "totals": totals, "model": model, "run_summary": summary,
            "memory_maxima_bytes": {k: max(r[k] for r in memory) for k in (
                "sampled_rss_bytes", "mps_allocated_bytes", "mps_driver_bytes", "process_lifetime_peak_rss_bytes")},
            "memory_caveat": "Unified-memory figures are not additive; sampled GPU peaks are lower bounds; caches persist across conditions",
            "new_human_review": "pending for all 25 outputs", "physical_execution_authorized": False,
            "latency_scope": "one offline request per case/condition; no percentiles, live-latency or navigation-success claim"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    report = summarize(a.run)
    with (a.run / "analysis_stage_b.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"input_checks_pass": report["all_stage_a_input_checks_pass"],
                      "requests": len(report["requests"]), "previous_F_B_inputs_equal": sum(r["inputs_equal"] for r in report["previous_F_B"]),
                      "previous_F_B_native_equal": sum(r["complete_native_equal"] for r in report["previous_F_B"]),
                      "same_image_control": report["pairwise"][CASES[3]]["TP_F"], "totals": report["totals"]}))


if __name__ == "__main__":
    main()
