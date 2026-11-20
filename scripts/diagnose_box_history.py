"""Compare three original eight-frame decisions to current-frame-only inference."""

from contextlib import closing
import os
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic
from types import SimpleNamespace

from evaluate_object_navigation import IDENTITY_KEYS, load_case, model_worker, wait, write
from go2_perception.vln.bounded_context import select_context
from go2_perception.vln.prediction_archive import file_hash, load, require

CASES = (("hall", 2), ("edge", 2), ("center", 3))


def main():
    workspace = Path(os.environ["GO2_WORKSPACE"]).resolve()
    base = workspace / "runs/target_grounding"
    base.mkdir(exist_ok=True, parents=True)
    out = Path(mkdtemp(prefix="box-history-", dir=base))
    print(out, flush=True)
    write(
        out / "design.json",
        dict(
            cases=CASES,
            instruction="Go to the box and stop.",
            baseline="archived eight-frame decisions",
            intervention="current frame only",
            matched="same cutoff, current image, instruction, model, processor and runtime",
        ),
    )
    rows = []
    with closing(model_worker(workspace, out)) as worker:
        wait(worker, startup=True)
        identity = load(out / "worker/model.json")
        for key, seq in CASES:
            session, saved = load_case(workspace, key, seq)
            original = load(session / "session/worker/model.json")
            require(all(identity[k] == original[k] for k in IDENTITY_KEYS), "different runtime")
            folder = out / key
            folder.mkdir()
            job = dict(saved["job"], frames=saved["job"]["frames"][-1:])
            require(
                job["instruction"] == "Go to the box and stop." and len(saved["job"]["frames"]) == 8,
                "unexpected control",
            )
            plan = select_context(job["frames"], job["decision_time_s"])
            context = dict(identity=saved["snapshot"]["identity"], bounded_context=plan)
            write(folder / "job.json", job)
            write(folder / "context.json", context)
            worker.submit(SimpleNamespace(context=context, job=job, folder=folder), monotonic())
            response = wait(worker)
            native = response["result"]
            previous = saved["native"]
            require(
                response["identity"] == context["identity"]
                and native["status"] == "decoded"
                and native["native_selected_frame_indices"] == [0]
                and file_hash(folder / "native/input_0000.png") == saved["frames"][-1]["sha256"],
                "changed current image or wrong native result",
            )
            rows.append(
                dict(
                    case=key,
                    prediction_id=saved["prediction"]["prediction_id"],
                    source_frame=saved["frames"][-1],
                    original_indices=previous["native_selected_frame_indices"],
                    archived_result=str(saved["request"] / "native/result.json"),
                    original_opos=previous["pointing"]["opos"],
                    single_frame_opos=native["pointing"]["opos"],
                    original_inference_s=saved["prediction"]["inference_duration_s"],
                    single_frame_inference_s=response["inference_duration_s"],
                    result=str(folder / "native/result.json"),
                )
            )
            write(out / "results.json", rows)
            print(key, rows[-1]["original_opos"], rows[-1]["single_frame_opos"], flush=True)


if __name__ == "__main__":
    main()
