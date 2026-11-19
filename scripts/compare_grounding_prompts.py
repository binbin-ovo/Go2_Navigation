"""Fixed, sequential replay of the November 19 box prompt comparison. Model only."""

import argparse
import json
import os
from contextlib import closing
from html import escape
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic, sleep
from types import SimpleNamespace

from go2_perception.vln.grounding_review import saved_input
from go2_perception.vln.interactive_trials import expand_target
from go2_perception.vln.prediction_archive import file_hash, load, require
from go2_perception.vln.shadow_display import target_fraction
from go2_perception.vln.shadow_worker import ProcessWorker
from review_target_grounding import marked_image, page, link

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    "box",
    "brown cardboard box",
    "brown cardboard box on the floor",
    "brown cardboard box on the floor near the stainless steel refrigerator",
)
CASES = (
    ("Right-edge box", "trial-THKUcr", "T20261119-120816-01-3a48-P002"),
    ("Clearly visible box", "trial-bYheuc", "T20261119-115525-01-6b80-P002"),
    ("Hallway distractor", "trial-cdyo3G", "T20261119-114417-01-4c94-P002"),
)


def wait(worker, *, startup=False):
    end = monotonic() + (180 if startup else 45)
    while monotonic() < end:
        response = worker.poll(monotonic())
        if (startup and worker.ready) or response is not None:
            return response
        sleep(0.02)
    raise TimeoutError("LightNav worker time limit exceeded")


def main():
    workspace = Path(os.environ["GO2_WORKSPACE"]).resolve()
    base = workspace / "runs/target_grounding"
    base.mkdir(exist_ok=True)
    out = Path(mkdtemp(prefix="prompts-", dir=base))
    print(out, flush=True)
    inputs = []
    for name, key, pid in CASES:
        s = workspace / "runs/five_move" / key / "session"
        data = saved_input(s, pid)
        run = load(s / "run.json")
        require(
            file_hash(ROOT / "configs/pretrained_vln.yaml") == run["runtime_config_sha256"]
            and all(
                file_hash(ROOT / p) == run["implementation_sha256"][p]
                for p in ("scripts/vln_shadow_worker.py", "scripts/lightnav_worker.py")
            ),
            "worker and config revision",
        )
        inputs.append((name, data, load(s / "session/worker/model.json")))
    declaration = dict(
        cases=CASES,
        targets=TARGETS,
        control="repeat box on right-edge view last",
        agent_edge_roi=[0.925, 0.53, 1, 0.70],
        clear_reference="archived identify ROI on the same image",
        distractor_reference="archived reject ROI, operator session assessment and agent image review",
        mode="offline prompts only, no reference or evaluation supplied to worker",
    )
    (out / "design.json").write_text(json.dumps(declaration, indent=2))
    jobs = [(i, t) for i in range(len(inputs)) for t in TARGETS] + [(0, TARGETS[0])]
    rows = []
    with closing(
        ProcessWorker(
            python=workspace / "envs/model/bin/python",
            assets=workspace / "weights/lightnav",
            config=ROOT / "configs/pretrained_vln.yaml",
            device="mps",
            output=out / "worker",
            preparation_mode="cached",
            tensor_evidence="skip",
            episode_context=True,
        )
    ) as worker:
        wait(worker, startup=True)
        identity = load(out / "worker/model.json")
        for seq, (i, target) in enumerate(jobs, 1):
            case, data, original_model = inputs[i]
            require(
                all(
                    identity[k] == original_model[k]
                    for k in (
                        "runtime",
                        "checkpoint",
                        "upstream_code",
                        "patch_sha256",
                        "processor",
                        "versions",
                        "reset_policy",
                        "preparation_mode",
                        "tensor_evidence",
                    )
                ),
                "runtime or processor drift",
            )
            folder = out / f"{seq:02d}"
            folder.mkdir()
            job = dict(data["job"], instruction=expand_target(target)["instruction"])
            context = data["snapshot"]
            (folder / "job.json").write_text(json.dumps(job, indent=2))
            worker.submit(SimpleNamespace(context=context, job=job, folder=folder), monotonic())
            response = wait(worker)
            res = response["result"]
            require(
                response["identity"] == context["identity"] and res["status"] == "decoded", "worker response"
            )
            require(
                res["native_selected_frame_indices"] == data["native"]["native_selected_frame_indices"]
                and res["native_segments"] == data["native"]["native_segments"]
                and all(
                    file_hash(folder / "native" / Path(f["path"]).name) == f["sha256"] for f in data["frames"]
                ),
                "model-space image or time selection drift",
            )
            prompt = folder / "native/prompt.txt"
            archived_prompt = (data["request"] / "native/prompt.txt").read_text()
            require(
                prompt.read_text() == archived_prompt.replace(data["job"]["instruction"], job["instruction"])
                and job["instruction"] in prompt.read_text(),
                "change outside requested prompt",
            )
            point = target_fraction(res["pointing"])
            row = dict(
                case=case,
                prediction_id=data["prediction"]["prediction_id"],
                target=target,
                instruction=job["instruction"],
                prompt=str(prompt),
                native=res["native_output"],
                point=point,
                opos=res["pointing"]["opos"],
                stop=res["decoder_stop"],
                native_result=str(folder / "native/result.json"),
                image=data["frames"][-1]["path"],
                input_hashes=[f["sha256"] for f in data["frames"]],
                archive_baseline_exact=res["native_output"] == data["native"]["native_output"],
                latency_s=response["inference_duration_s"],
                repeat=(seq == len(jobs)),
            )
            rows.append(row)
            with (out / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(seq, case, target, res["pointing"]["opos"], flush=True)
    render_comparison(rows, out / "results.jsonl", out)


def render_comparison(rows, results_file, out):
    panels = []
    for case, _, _ in CASES:
        panels.append(f"<h2>{escape(case)}</h2><div class='grid'>")
        for row in [r for r in rows if r["case"] == case and not r["repeat"]]:
            claim = (
                "Model reports absence"
                if row["opos"] == 0
                else "Raw model mark"
                if row["point"] is not None
                else "No decodable mark"
            )
            panels.append(
                "<article><h3>"
                + escape(row["target"])
                + "</h3>"
                + marked_image(row["image"], row["point"])
                + f"<p><strong>{claim}</strong> · OPOS {row['opos']}</p>"
                + "<p>"
                + escape(row["instruction"])
                + "</p><code>"
                + escape(row["native"])
                + "</code><p>"
                + link(row["prompt"], "Full prompt")
                + " · "
                + link(row["native_result"], "Decoded waypoints")
                + "</p></article>"
            )
        panels.append("</div>")
    page(
        out / "comparison.html",
        "Same images, four box descriptions",
        "<p>Magenta marks are raw model proposals. Each column uses identical camera pixels and timestamps within its row.</p>"
        + "".join(panels)
        + "<p>"
        + link(results_file, "All records, including the repeated control")
        + "</p>",
    )
    print(out / "comparison.html", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render-run", type=Path, help="Render saved results in a fresh private folder without inference."
    )
    args = parser.parse_args()
    if args.render_run:
        results = args.render_run.resolve() / "results.jsonl"
        rows = [json.loads(line) for line in results.read_text().splitlines()]
        matrix = [(name, target, False) for name, _, _ in CASES for target in TARGETS]
        require(
            [(r["case"], r["target"], r["repeat"]) for r in rows]
            == matrix + [(CASES[0][0], TARGETS[0], True)],
            "unfinished or unexpected comparison",
        )
        for row in rows:
            require(
                file_hash(Path(row["image"])) == row["input_hashes"][-1]
                and load(Path(row["native_result"]))["native_output"] == row["native"],
                "saved comparison provenance",
            )
        out = Path(mkdtemp(prefix="panels-", dir=results.parent))
        render_comparison(rows, results, out)
    else:
        main()
