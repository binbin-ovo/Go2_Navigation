"""Offline fixed-cutoff profiling of saved advisory observations; no sensor access.

One pass through the declared 5/17/37-frame jobs, in that order. Ingestion is fast
replay between fixed cutoffs, NOT a live or continuously paced replay experiment.
Original timestamps are retained; new host delivery times describe replay only.
"""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from go2_perception.vln.episodes import file_sha256
from go2_perception.vln.observation import load
from go2_perception.vln.shadow import ShadowSession, write_json
from go2_perception.vln.shadow_display import render
from go2_perception.vln.shadow_sources import Delivery
from go2_perception.vln.shadow_worker import MockWorker, ProcessWorker

ROOT = Path(__file__).resolve().parents[1]


def source_manifest(root):
    return {str(p.relative_to(root)): file_sha256(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-python", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--mode", choices=["baseline", "cached"], default="baseline")
    a = p.parse_args()
    jobs = [json.loads((a.source_run / f"requests/{i:06d}/job.json").read_text()) for i in (1, 2, 3)]
    snapshots = [json.loads((a.source_run / f"requests/{i:06d}/snapshot.json").read_text()) for i in (1, 2, 3)]
    if [len(j["frames"]) for j in jobs] != [5, 17, 37]:
        p.error("This declaration requires the recorded 5/17/37-frame jobs")
    if len({j["instruction"] for j in jobs}) != 1:
        p.error("Declared jobs must share the same instruction")
    before = source_manifest(a.source_run)
    options = {} if a.mode == "baseline" else {"preparation_mode": "cached"}
    session = ShadowSession(a.output.resolve(), MockWorker(), mode="replay_fast", **options)
    started = time.monotonic()
    declaration = {
        "source_run": str(a.source_run.resolve()), "mode": a.mode,
        "history_lengths": [5, 17, 37], "requests_per_length": 1,
        "order": ["5_cold_after_model_load", "17_warm", "37_warm"],
        "age_limit_s": session.limits.advisory_age_s,
        "source_files": before,
        "implementation_sha256": {str(f.relative_to(ROOT)): file_sha256(f) for f in
            [Path(__file__).resolve(), ROOT / "scripts/lightnav_worker.py", ROOT / "scripts/vln_shadow_worker.py",
             *sorted((ROOT / "src/go2_perception/vln").glob("shadow*.py"))]},
        "replay": "fixed cutoffs; no live source, no concurrent ingestion during inference",
        "physical_execution_authorized": False,
    }
    write_json(session.output / "declaration.json", declaration)
    rows, ingest = [], []
    try:
        worker_options = {} if a.mode == "baseline" else {"preparation_mode": "cached"}
        session.worker = ProcessWorker(python=a.model_python, assets=a.assets,
            config=ROOT / "configs/pretrained_vln.yaml", device="mps", output=session.output / "worker",
            profile_stages=True, **worker_options)
        while not session.worker.ready:
            session.tick(time.monotonic(), allow_submit=False)
            if time.monotonic() - started > 120:
                raise TimeoutError("Model readiness exceeded declared 120 s budget")
            time.sleep(0.01)
        session.instruction_changed(jobs[0]["instruction"], time.monotonic())
        metadata = [json.loads(f.read_text()) for f in sorted((a.source_run / "observations").glob("*.json"))]
        delivered = 0
        for original, snapshot in zip(jobs, snapshots):
            cutoff = original["decision_time_s"]
            for meta in metadata[delivered:]:
                if meta["available_s"] > cutoff:
                    break
                read_started = time.monotonic()
                observation = load(meta["observation_path"])
                read_s = time.monotonic() - read_started
                now = time.monotonic()
                session.receive(Delivery(observation, meta["source_id"], meta["frame_id"],
                    meta["available_s"], {"recorded_delivery": meta}, received_monotonic_s=now), now)
                ingest.append({"source_frame_id": meta["frame_id"], "source_load_s": read_s,
                               **session.last_receive_profile})
                delivered += 1
            if session.history_error:
                raise ValueError(session.history_error)
            submit = time.monotonic()
            session.tick(submit)
            request = session.inflight
            if request is None:
                raise RuntimeError("Declared request was not submitted")
            selected = request.context["history"]
            assert [x["source"]["frame_id"] for x in selected] == [x["source"]["frame_id"] for x in snapshot["history"]]
            for new, old in zip(request.job["frames"], original["frames"], strict=True):
                assert (new["observation_s"], new["available_s"]) == (old["observation_s"], old["available_s"])
                assert np.array_equal(cv2.imread(new["path"]), cv2.imread(old["path"]))
            equivalence_done = time.monotonic()
            # This diagnostic comparison happens after dispatch; time is reported,
            # not excluded from end-to-end receipt. No labels reach the child.
            while session.inflight is not None:
                session.tick(time.monotonic(), allow_submit=False)
                if time.monotonic() - submit > 40:
                    raise TimeoutError("Request exceeded declared 40 s profile budget")
                time.sleep(0.005)
            display_start = time.monotonic()
            render(session, display_start)
            displayed = time.monotonic()
            record = session.displayed[1]
            response = record["response"]
            result = response["result"]
            historical_age_at_submit = snapshot["host_submit_monotonic_s"] - snapshot["current_source"]["exposure_monotonic_s"]
            row = {
                "history_frames": len(original["frames"]),
                "elapsed_source_history_s": original["frames"][-1]["observation_s"] - original["frames"][0]["observation_s"],
                "input_pixels_and_selection_equal_original": True,
                "post_dispatch_equivalence_check_s": equivalence_done - submit,
                "preparation": json.loads((request.folder / "preparation_profile.json").read_text()),
                "prepare_and_dispatch_s": response["host_inference_start_s"] - submit,
                "worker_s": response["inference_duration_s"],
                "result_delivery_s": record["receipt_monotonic_s"] - response["host_inference_end_s"],
                "display_s": displayed - display_start,
                "request_to_display_s": displayed - submit,
                "recorded_estimated_exposure_age_at_submit_s": historical_age_at_submit,
                "counterfactual_estimated_exposure_age_s": historical_age_at_submit + displayed - submit,
                "counterfactual_age_basis": "recorded estimated exposure-to-submit offset plus measured fixed-cutoff replay duration; NOT live latency",
                "replay_rejection": record["rejection_reason"],
                "native": result, "counts": session.counts.copy(),
            }
            rows.append(row)
            write_json(session.output / "profile.json", {"requests": rows, "ingestion": ingest})
            print(json.dumps({"mode": a.mode, "frames": row["history_frames"], "status": result["status"],
                "worker_s": row["worker_s"], "prepare_dispatch_s": row["prepare_and_dispatch_s"],
                "native_s": result.get("timing", {}).get("native_generate_s")}), flush=True)
            if result["status"] != "decoded":
                raise RuntimeError("Retained failed native request; no retries")
    finally:
        session.close(time.monotonic())
        render(session, time.monotonic(), snapshot=True)
        write_json(session.output / "preservation.json", {"unchanged": before == source_manifest(a.source_run),
            "files_checked": len(before)})
        write_json(session.output / "summary.json", {"counts": session.counts,
            "history_error": session.history_error, "requests_completed": len(rows),
            "parent_host_s": time.monotonic() - started, "physical_execution_authorized": False})


if __name__ == "__main__":
    main()
