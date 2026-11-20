"""Two fixed causal cutoffs on six recorded frames. A controlled occluder is removed after frame three."""

from contextlib import closing
from datetime import datetime
import os
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic
from types import SimpleNamespace

from evaluate_object_navigation import (
    IDENTITY_KEYS,
    ROOT,
    decision_inputs,
    enrollment,
    load_case,
    mask_history,
    replay_selection,
    wait,
    write,
)
from go2_perception import navigation
from go2_perception.vln import episode_grounding
from go2_perception.vln.bounded_context import select_context
from go2_perception.vln.prediction_archive import file_hash, iter_records, load, require
from go2_perception.vln.shadow_display import target_fraction
from go2_perception.vln.shadow_worker import ProcessWorker


def main():
    workspace = Path(os.environ["GO2_WORKSPACE"]).resolve()
    session, data = load_case(workspace, "center", 3)
    base = workspace / "runs/target_grounding"
    base.mkdir(parents=True, exist_ok=True)
    out = Path(mkdtemp(dir=base, prefix="object-reveal-"))
    print(out, flush=True)
    write(
        out / "design.json",
        dict(
            cutoffs=[3, 6],
            source=data["prediction"]["prediction_id"],
            frames="latest six chronologically sampled observations",
            full_cover=[0.40, 0.39, 0.69, 0.81],
            covered_indices=[0, 1, 2],
            motion="recorded history; no action applied",
        ),
    )
    original = data["job"]["frames"][-6:]
    require(len(original) == 6, "six source frames")
    stream = mask_history(original[:3], "full", out) + original[3:]
    grant, cap, prop, views, inspected, aligned, _ = decision_inputs(
        session, data["prediction"]["prediction_id"]
    )
    earlier = next(
        r
        for r in iter_records(session / "events.jsonl")
        if r.get("event") == "episode_instance_review"
        and r["decision"] == "identify"
        and r["prediction_id"].endswith("-P002")
    )
    review_time = datetime.fromisoformat(earlier["host_wall_utc"]).timestamp()
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
        ident = load(out / "worker/model.json")
        require(
            all(ident[k] == load(session / "session/worker/model.json")[k] for k in IDENTITY_KEYS), "model"
        )
        for n in (3, 6):
            folder = out / f"cutoff-{n}"
            folder.mkdir()
            stamp = stream[n - 1]["available_s"]
            job = dict(
                data["job"],
                episode_id="controlled-center-reveal",
                decision_id=str(n),
                decision_time_s=stamp,
                frames=stream[:n],
            )
            plan = select_context(job["frames"], stamp)
            ctx = dict(
                identity=dict(session_id=job["episode_id"], sequence=n, clock_epoch=job["clock_epoch"]),
                bounded_context=plan,
            )
            write(folder / "job.json", job)
            write(folder / "context.json", ctx)
            worker.submit(SimpleNamespace(context=ctx, job=job, folder=folder), monotonic())
            response = wait(worker)
            native = response["result"]
            require(
                response["identity"] == ctx["identity"]
                and native["status"] == "decoded"
                and native["native_selected_frame_indices"] == list(range(n))
                and all(native[k] == job[k] for k in ("decision_id", "episode_id", "clock_epoch")),
                "causal response",
            )
            png = folder / "native" / f"input_{n - 1:04d}.png"
            point = target_fraction(native["pointing"])
            row = dict(
                cutoff=n,
                source_observation_s=stream[n - 1]["observation_s"],
                available_s=stamp,
                point=point,
                opos=native["pointing"]["opos"],
                native=native["native_output"],
                latency_s=response["inference_duration_s"],
                image=str(png),
                sha256=file_hash(png),
                source_ids=[f["frame_id"] for f in data["frames"][-6:][:n]],
            )
            if n == 6:
                require(
                    file_hash(png) == data["frames"][-1]["sha256"]
                    and review_time < stream[-1]["observation_s"],
                    "revealed source or enrollment time",
                )
                for i in range(3):
                    name = f"input_{i:04d}.png"
                    require(
                        file_hash(folder / "native" / name) == file_hash(out / "cutoff-3/native" / name),
                        "history rewritten",
                    )
                ref = enrollment(
                    workspace, ("center", 2), "identify", data, grant["operator"], prop["identity"]
                )
                pred = dict(
                    data["prediction"],
                    target_fraction=point,
                    opos_not_visible=native["pointing"]["opos"] == 0,
                )
                require(pred["source_image"]["file_sha256"] == row["sha256"], "revealed image")
                ev = episode_grounding.ground(
                    pred,
                    png,
                    data["snapshot"]["history"][-1]["rectification"],
                    grant["target"],
                    ref,
                    seek_view=True,
                )
                prop.update(
                    point_fraction=point,
                    native_stop=native["decoder_stop"],
                    waypoints=native["waypoints"],
                    native_output=native["native_output"],
                )
                row.update(
                    grounding=ev,
                    selection=replay_selection(
                        navigation, grant, cap, prop, ev, views, inspected, aligned, current=True
                    ),
                )
            else:
                row["selection"] = None  # There is no archived guard snapshot at this cutoff.
            rows.append(row)
    write(out / "reveal.json", rows)
    print([(r["opos"], (r["selection"] or {}).get("decision")) for r in rows])


if __name__ == "__main__":
    main()
