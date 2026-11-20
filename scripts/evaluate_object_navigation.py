"""Repeat the fixed five-move box diagnostic on closed local recordings with cached LightNav."""

import argparse
from collections import Counter
from contextlib import closing
from copy import deepcopy
from datetime import datetime
from html import escape
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
from tempfile import mkdtemp
from time import monotonic
from types import SimpleNamespace

from PIL import Image, ImageChops, ImageDraw

from compare_grounding_prompts import wait
from review_target_grounding import link, marked_image, page
from go2_perception import navigation
from go2_perception.vln import episode_grounding
from go2_perception.vln.chunk_replay import chunk_events
from go2_perception.vln.grounding_review import episode_decisions, saved_input
from go2_perception.vln.motion_profile import Hold
from go2_perception.vln.prediction_archive import file_hash, iter_records, load, require
from go2_perception.vln.shadow_display import target_fraction
from go2_perception.vln.shadow_worker import ProcessWorker
from go2_perception.vln.supervised import digest

ROOT = Path(__file__).resolve().parents[1]
BASE = "2fa98f20b415003bdb6dae3518ea422a757f45ec"
SESSIONS = {
    "hall": ("trial-cdyo3G", "T20261119-114417-01-4c94"),
    "center": ("trial-231Apz", "T20261119-114856-01-5e54"),
    "clear": ("trial-bYheuc", "T20261119-115525-01-6b80"),
    "edge": ("trial-THKUcr", "T20261119-120816-01-3a48"),
}
# Positives and rejections here are earlier enrollments. Current-frame reviews are evaluation only.
CASES = (
    ("hallway-mark", "hall", 2, None, None, "box", None),
    ("absent-after-rejection", "hall", 3, None, ("hall", 2), "box", None),
    ("absent-described", "hall", 2, None, None, "brown cardboard box", None),
    ("center-fresh", "center", 3, ("center", 2), None, "box", None),
    ("center-closer", "center", 4, ("center", 2), None, "box", None),
    ("clear-cross-view", "clear", 2, ("center", 2), None, "box", None),
    ("clear-local-reference", "clear", 3, ("clear", 2), None, "box", None),
    ("edge-hallway-mark", "edge", 2, ("clear", 2), None, "box", None),
    ("center-partial-cover", "center", 3, ("center", 2), None, "box", "partial"),
    ("center-full-cover", "center", 3, ("center", 2), None, "box", "full"),
)
# Interventions on every selected rectified frame, including episode anchors.
MASKS = {"partial": (0.46, 0.44, 0.55, 0.74), "full": (0.40, 0.39, 0.69, 0.81)}
IDENTITY_KEYS = (
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


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_case(workspace, key, seq):
    folder, trial = SESSIONS[key]
    session = workspace / "runs/five_move" / folder / "session"
    data = saved_input(session, f"{trial}-P{seq:03}")
    run = load(session / "run.json")
    require(
        file_hash(ROOT / "configs/pretrained_vln.yaml") == run["runtime_config_sha256"]
        and all(
            file_hash(ROOT / p) == run["implementation_sha256"][p]
            for p in ("scripts/vln_shadow_worker.py", "scripts/lightnav_worker.py")
        ),
        "worker source",
    )
    return session, data


def covered_image(path, mask):
    with Image.open(path) as im:
        image = im.convert("RGB")
    w, h = image.size
    x0, y0, x1, y1 = (round(v * n) for v, n in zip(MASKS[mask], (w, h, w, h), strict=True))
    ImageDraw.Draw(image).rectangle((x0, y0, x1 - 1, y1 - 1), fill=(110, 110, 110))
    return image


def mask_history(frames, mask, folder):
    result = []
    for i, frame in enumerate(frames):
        path = folder / f"covered_{i:02}.png"
        covered_image(frame["path"], mask).save(path)
        result.append(dict(frame, path=str(path), sha256=file_hash(path)))
    return result


def model_worker(workspace, out):
    return ProcessWorker(
        python=workspace / "envs/model/bin/python",
        assets=workspace / "weights/lightnav",
        config=ROOT / "configs/pretrained_vln.yaml",
        device="mps",
        output=out / "worker",
        preparation_mode="cached",
        tensor_evidence="skip",
        episode_context=True,
    )


def infer(workspace, out):
    with closing(model_worker(workspace, out)) as worker:
        wait(worker, startup=True)
        model = load(out / "worker/model.json")
        for name, key, seq, _, _, target, mask in CASES:
            folder = out / name
            folder.mkdir()
            session, data = load_case(workspace, key, seq)
            original = load(session / "session/worker/model.json")
            require(all(model[k] == original[k] for k in IDENTITY_KEYS), "model or processor drift")
            job = dict(data["job"], instruction=navigation.expand_target(target)["instruction"])
            if mask:
                job["frames"] = mask_history(job["frames"], mask, folder)
            write(folder / "job.json", job)
            context = data["snapshot"]
            worker.submit(SimpleNamespace(job=job, context=context, folder=folder), monotonic())
            response = wait(worker)
            require(
                response["identity"] == context["identity"] and response["result"]["status"] == "decoded",
                "response identity",
            )
            write(
                folder / "response.json",
                {k: response[k] for k in ("identity", "inference_duration_s", "backend")},
            )
            print(name, response["result"]["pointing"]["opos"], flush=True)


def inference_hashes(source):
    """Bind every published decision, official prompt and model-input PNG on later replay."""
    paths = [source / "worker/model.json", source / "design.json"]
    for name, *_ in CASES:
        folder = source / name
        paths.extend(
            folder / p for p in ("job.json", "response.json", "native/prompt.txt", "native/result.json")
        )
        paths.extend(sorted((folder / "native").glob("input_*.png")))
    return {str(p.relative_to(source)): file_hash(p) for p in paths}


def pinned_module(out, source, package):
    """Load only the two pure comparison modules from the fixed integrated Git object."""
    path = out / ("baseline_" + Path(source).name)
    path.write_bytes(subprocess.check_output(["git", "show", f"{BASE}:{source}"], cwd=ROOT))
    spec = importlib.util.spec_from_file_location(package + "._baseline_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def enrollment(workspace, source, decision, data, operator, observed):
    if source is None:
        return None
    session, enrolled = load_case(workspace, *source)
    pid = enrolled["prediction"]["prediction_id"]
    _, decisions = episode_decisions(session)
    records = [
        r
        for r in iter_records(session / "events.jsonl")
        if r.get("event") == "episode_instance_review"
        and r["prediction_id"] == pid
        and r["decision"] == decision
    ]
    require(len(records) == 1, "ambiguous enrollment")
    r = records[0]
    require(
        r["operator"] == operator
        and r["target"] == "box"
        and any(
            review["roi"] == r["roi"] and review["image_sha256"] == r["image_sha256"]
            for review in decisions[pid]["reviews"]
        )
        and r["image_sha256"] == enrolled["frames"][-1]["sha256"]
        and observed["uncertainty_bound_s"] >= 0
        and r["host_monotonic_s"] < observed["observed_monotonic_s"] - observed["uncertainty_bound_s"]
        and datetime.fromisoformat(r["host_wall_utc"]).timestamp()
        < data["job"]["frames"][-1]["observation_s"] - observed["uncertainty_bound_s"],
        "unbound or future enrollment",
    )
    path = enrolled["frames"][-1]["path"]
    detail = episode_grounding.request_detail(enrolled["prediction"], enrolled["request"], path, enrolled["snapshot"])
    ref = episode_grounding.object_reference(path, r["roi"], r["target"], operator, detail=detail)
    require(ref["sha256"] == r["image_sha256"], "enrollment image")
    return ref


def decision_inputs(session, pid):
    summary, decisions = episode_decisions(session)
    events = list(iter_records(session / "executor/executor_events.jsonl"))
    grant = next(e["grant"] for e in events if e["type"] == "episode_authorization")
    cap = load(session.parent / "capability.json")
    require(digest(cap) == grant["capability_sha256"], "recorded capability")
    previous = []
    for event in events:
        if event["type"] != "episode_observation" or event["nonce"] != grant["episode_id"]:
            continue
        proposal = event["proposal"]
        previous.append(proposal["identity"]["pose"])
        if proposal["identity"]["prediction_id"] == pid:
            break
    else:
        raise ValueError("No accepted historical observation for this decision")
    views = [
        dict(a["decision_pose"], attempt=a)
        for a in summary["attempts"]
        if a["prediction_id"] < pid and "decision_pose" in a
    ]
    last = views[-1]["attempt"]["prediction_id"] if views else None
    aligned = bool(last and decisions[last]["selection"]["aligned"])
    return grant, cap, proposal, views, previous, aligned, decisions[pid]


def replay_selection(module, grant, cap, proposal, ev, views, inspected, aligned, *, current):
    try:
        selected = module.select_action(
            grant,
            cap,
            proposal,
            ev,
            views,
            len(views),
            aligned,
            **({"inspected": inspected} if current else {}),
        )
    except Hold as exc:
        return dict(decision="HOLD", reason=str(exc), refusal=exc.details)
    action = selected.get("action")
    return dict(
        decision=selected["mode"] + " " + action["direction"]
        if action
        else ("REVIEW" if selected.get("review") else "STOP"),
        reason=selected["reason"],
        action=action,
    )


def native_inputs(folder, data, target, mask):
    job, native = load(folder / "job.json"), load(folder / "native/result.json")
    require(
        job["instruction"] == navigation.expand_target(target)["instruction"]
        and all(
            job[k] == data["job"][k] for k in ("decision_id", "episode_id", "clock_epoch", "decision_time_s")
        )
        and all(native[k] == job[k] for k in ("decision_id", "episode_id", "clock_epoch"))
        and native["status"] == "decoded"
        and native["execution_authorized"] is False,
        "decision",
    )
    require(
        native["native_selected_frame_indices"] == data["native"]["native_selected_frame_indices"]
        and native["native_segments"] == data["native"]["native_segments"],
        "temporal selection",
    )
    expected = (
        (data["request"] / "native/prompt.txt")
        .read_text()
        .replace(data["job"]["instruction"], job["instruction"])
    )
    require((folder / "native/prompt.txt").read_text() == expected, "model prompt")
    files = []
    for i, (frame, orig, saved) in enumerate(
        zip(job["frames"], data["job"]["frames"], data["frames"], strict=True)
    ):
        require(
            all(frame[k] == orig[k] for k in ("observation_s", "available_s"))
            and file_hash(Path(frame["path"])) == frame["sha256"],
            "prepared input or timestamps",
        )
        png = folder / "native" / Path(saved["path"]).name
        digest_png = file_hash(png)
        if mask:
            with Image.open(frame["path"]) as im:
                require(
                    ImageChops.difference(covered_image(orig["path"], mask), im.convert("RGB")).getbbox()
                    is None,
                    "controlled cover pixels",
                )
            require(
                Path(frame["path"]).resolve() == folder / f"covered_{i:02}.png"
                and frame["sha256"] != orig["sha256"]
                and digest_png != saved["sha256"],
                "unapplied cover",
            )
        else:
            require(frame == orig and digest_png == saved["sha256"], "native input drift")
        files.append(
            dict(
                index=saved["index"],
                frame_id=saved["frame_id"],
                observation_s=frame["observation_s"],
                available_s=frame["available_s"],
                image=str(png),
                sha256=digest_png,
                original_native_sha256=saved["sha256"],
            )
        )
    return native, files


def analyze(workspace, source, out):
    old_ground = pinned_module(out, "src/go2_perception/vln/episode_grounding.py", "go2_perception.vln")
    old_policy = pinned_module(out, "src/go2_perception/navigation.py", "go2_perception")
    model = load(source / "worker/model.json")
    rows = []
    for name, key, seq, pos, neg, target, mask in CASES:
        folder = source / name
        session, data = load_case(workspace, key, seq)
        original = load(session / "session/worker/model.json")
        response = load(folder / "response.json")
        require(
            all(model[k] == original[k] for k in IDENTITY_KEYS)
            and response["identity"] == data["snapshot"]["identity"]
            and response["backend"] == "lightnav",
            "worker or recorded response identity",
        )
        pred = deepcopy(data["prediction"])
        pid = pred["prediction_id"]
        grant, cap, proposal, views, inspected, aligned, archived = decision_inputs(session, pid)
        observed = proposal["identity"]
        require(
            observed["observation_id"] == pred["source_frame"]["frame_id"]
            and observed["source_image_sha256"] == data["frames"][-1]["sha256"]
            and observed["observed_monotonic_s"] == pred["source_frame"]["exposure_monotonic_s"]
            and observed["instruction"] == pred["instruction"]
            and proposal["waypoints"] == pred["waypoints"]
            and proposal["native_output"] == pred["native_output"]
            and proposal["point_fraction"] == pred["target_fraction"],
            "model proposal bound to accepted executor geometry",
        )
        native, frames = native_inputs(folder, data, target, mask)
        point = target_fraction(native["pointing"])
        image = frames[-1]
        grant.update(target=target, instruction=navigation.expand_target(target)["instruction"])
        proposal.update(
            waypoints=native["waypoints"],
            point_fraction=point,
            native_stop=native["decoder_stop"],
            native_output=native["native_output"],
        )
        proposal["identity"].update(source_image_sha256=image["sha256"], instruction=grant["instruction"])
        pred.update(instruction=grant["instruction"], target_fraction=point, opos_not_visible=native["pointing"]["opos"] == 0)
        pred["source_image"]["file_sha256"] = image["sha256"]
        refs = [
            enrollment(workspace, src, decision, data, grant["operator"], proposal["identity"])
            for src, decision in ((pos, "identify"), (neg, "reject"))
        ]
        kwargs = (pred, image["image"], data["snapshot"]["history"][-1]["rectification"], target, *refs)
        row = dict(
            case=name,
            prediction_id=pid,
            session=str(session),
            target=target,
            mask=mask,
            frames=frames,
            native_result=str(folder / "native/result.json"),
            opos=native["pointing"]["opos"],
            point=point,
            archive_output_exact=native["native_output"] == data["native"]["native_output"],
            inference_s=load(folder / "response.json")["inference_duration_s"],
            original_reviews=archived["reviews"],
            archived_selection=archived["selection"],
            original_job=str(data["request"] / "job.json"),
            positive_enrollment=pos,
            rejected_enrollment=neg,
        )
        for label, module, policy, current in (
            ("before", old_ground, old_policy, False),
            ("after", episode_grounding, navigation, True),
        ):
            detail = (episode_grounding.prepared_detail(pred, data["snapshot"], load(folder / "job.json"),
                                                        image["image"]) if current else None)
            start = monotonic()
            g = module.ground(*kwargs, **({"seek_view": True, "detail": detail} if current else {}))
            row[label] = dict(
                grounding=g,
                verify_ms=(monotonic() - start) * 1000,
                **replay_selection(
                    policy, grant, cap, proposal, g, views, inspected, aligned, current=current
                ),
            )
        unchanged = mask is None and target == "box"
        decisions = [r["decision"] for r in archived["reviews"]] if unchanged else []
        row["current_image_review"] = decisions or ["unreviewed"]
        assessment = session.parent / "operator_assessment.json"
        row["session_assessment"] = dict(
            source=str(assessment), sha256=file_hash(assessment), scope="original session only"
        )
        if unchanged:
            require(
                row["archive_output_exact"]
                and native["waypoints"] == data["native"]["waypoints"]
                and native["pointing"] == data["native"]["pointing"],
                "unmodified native control drift",
            )
        rows.append(row)
    write(out / "results.json", rows)
    return rows


def audit_hardware(workspace, rows):
    """Count every original native output separately from eligibility, reviews and dispatch."""
    observations, dispatches, columns = [], [], Counter()
    lidar_counts = {}
    for name in ("cXouta", "cdyo3G", "231Apz", "bYheuc", "ivBOES", "THKUcr"):
        session = workspace / "runs/five_move" / ("trial-" + name) / "session"
        summary, decisions = episode_decisions(session)
        require(
            summary["state"] == "TERMINAL" and load(session / "summary.json")["termination"],
            "open or incomplete recording",
        )
        lidar_counts[name] = dict(Counter(kind for _, kind, _, _ in chunk_events(session / "lidar")))
        [predictions] = session.glob("trials/*/predictions.jsonl")
        for pred in iter_records(predictions):
            pid = pred["prediction_id"]
            data = saved_input(session, pid)
            native, ev = data["native"], decisions.get(pid, {})
            point = target_fraction(native["pointing"])
            require(
                (list(point) if point else None) == pred["target_fraction"]
                and native["native_output"] == pred["native_output"]
                and native["waypoints"] == pred["waypoints"]
                and pred["source_image"]["file_sha256"] == data["frames"][-1]["sha256"],
                "native archive differs",
            )
            token = native["pointing"]["opos"]
            prompt = (data["request"] / "native/prompt.txt").read_text()
            with Image.open(data["frames"][-1]["path"]) as image:
                require(image.size == (448, 256), "unexpected native input resolution")
            require(data["job"]["instruction"] in prompt and
                    re.findall(r"<opos_(\d+)>", native["native_output"]) == [str(token)],
                    "native prompt or raw OPOS token differs")
            columns[str((token - 1) % 48) if token else "absent"] += 1
            observations.append(
                dict(
                    session=name,
                    prediction_id=pid,
                    opos=token,
                    point=point,
                    input_sha256=data["frames"][-1]["sha256"],
                    history_indices=native["native_selected_frame_indices"],
                    native_heading_hint_rad=navigation.initial_heading_sweep(native["waypoints"]),
                    archived_age_s=pred["display_timing_at_receipt"]["observation_age_at_display_s"],
                    inference_s=pred["inference_duration_s"],
                    stop=native["decoder_stop"],
                    receipt_refusal=pred["rejection_reason_at_receipt"],
                    executor_refusal=ev.get("rejection"),
                    reviews=ev.get("reviews", []),
                )
            )
            if not ev.get("selection"):
                continue
            grant, cap, proposal, views, inspected, aligned, _ = decision_inputs(session, pid)
            geo, action = proposal["identity"]["geometry"], ev["selection"]["action"]
            current = replay_selection(
                navigation, grant, cap, proposal, ev["grounding"], views, inspected, aligned, current=True
            )
            dispatches.append(
                dict(
                    session=name,
                    prediction_id=pid,
                    recorded_mode=ev["selection"]["mode"],
                    recorded_direction=action["direction"],
                    disc=geo["observed_turn_disc"],
                    directions=geo["directions"],
                    turn_clear_rad=geo["turn_clear_rad"],
                    with_current_policy=current,
                )
            )
    # Change one recorded predicate only. Keep native path, identity, inspection grant,
    # area, pose and all other guard fields, including the unobserved rear.
    local = next(row for row in rows if row["case"] == "clear-local-reference")
    grant, cap, prop, views, seen, aligned, _ = decision_inputs(
        Path(local["session"]), local["prediction_id"]
    )
    require(prop["identity"]["geometry"]["observed_turn_disc"]["obstacle_returns"] > 0, "recorded disc")
    prop["identity"]["geometry"]["observed_turn_disc"]["obstacle_returns"] = 0
    control = replay_selection(
        navigation, grant, cap, prop, local["after"]["grounding"], views, seen, aligned, current=True
    )
    return dict(
        columns_zero_based=dict(columns),
        observations=observations,
        recorded_dispatches=dispatches,
        lidar_messages=lidar_counts,
        single_field_controller_diagnostic=dict(
            case=local["case"], changed="obstacle_returns=0", selection=control, hardware_execution=False
        ),
    )


def render(out, rows):
    panels = []
    for r in rows:
        before, after = r["before"], r["after"]
        g = after["grounding"]
        source = "Controlled cover on recorded pixels" if r["mask"] else "Recorded camera pixels"
        review = ", ".join(r["current_image_review"])
        semantic = r.get("semantic")
        detector = (
            ""
            if semantic is None
            else f"<p>DINO: {len(semantic['boxes'])} region(s); mark overlap {semantic['mark_in_candidate']}</p>"
        )
        regions = []
        if semantic:
            regions.extend(
                (b["xyxy_norm"], "#00b9e8", f"Category candidate {b['score']:.2f}") for b in semantic["boxes"]
            )
        if g.get("positive_reference", {}).get("localized"):
            regions.append((g["positive_reference"]["roi"], "#40c66d", "Located instance reference"))
        overlays = []
        for box, color, label in regions:
            x0, y0, x1, y1 = (min(1, max(0, v)) for v in box)
            overlays.append(
                f'<span title="{escape(label)}" style="position:absolute;box-sizing:border-box;'
                f"left:{100 * x0:.3f}%;top:{100 * y0:.3f}%;width:{100 * (x1 - x0):.3f}%;"
                f'height:{100 * (y1 - y0):.3f}%;border:2px solid {color};pointer-events:none"></span>'
            )
        panel = (
            marked_image(r["frames"][-1]["image"], r["point"]).removesuffix("</div>")
            + "".join(overlays)
            + "</div>"
        )
        panels.append(
            "<article><h2>"
            + escape(r["case"])
            + "</h2>"
            + panel
            + f"<p>{source} · Image review: {escape(review)} · OPOS {r['opos']} · {r['inference_s']:.2f} s</p>"
            + f"<p>{escape(before['decision'])} → <b>{escape(after['decision'])}</b></p>"
            + f"<p>{escape(g['method'])} · {escape(after['reason'])}</p>"
            + detector
            + link(r["native_result"], "Decoded native output and waypoints")
            + "</article>"
        )
    page(
        out / "index.html",
        "Box recognition and five-move decisions",
        "<p>Magenta marks the native point. Cyan boxes are category candidates, green outlines a located reference. The two decisions use identical inputs, enrollment and archived geometry. "
        "Selections are offline. Original reviews and recorded dispatches are in the JSON.</p><div class='grid'>"
        + "".join(panels)
        + "</div>"
        + link(out / "results.json", "Inputs, evidence and selection details")
        + (link(out / "six-trial-audit.json", "All six original trials and the clearance control")
           if (out / "six-trial-audit.json").is_file() else ""),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dino-snapshot", type=Path, help="Optional local pinned detector for a sequential CPU comparison"
    )
    parser.add_argument(
        "--reuse-inference",
        type=Path,
        help="Check a saved run and produce a fresh comparison without rerunning LightNav",
    )
    parser.add_argument("--hardware-report", action="store_true", help="Also audit all six closed Nov 19 trials")
    args = parser.parse_args()
    workspace = Path(os.environ["GO2_WORKSPACE"]).resolve()
    parent = workspace / "runs/target_grounding"
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(mkdtemp(prefix="object-navigation-", dir=parent))
    print(out, flush=True)
    design = dict(
        baseline=BASE,
        cases=CASES,
        masks=MASKS,
        semantic_query="box.",
        dino_box_threshold=0.35,
        dino_text_threshold=0.25,
        comparison="point inside any category box; no instance or movement authority",
    )
    write(out / "design.json", design)
    source = args.reuse_inference.resolve() if args.reuse_inference else out
    if args.reuse_inference:
        require(load(source / "design.json") == load(out / "design.json"), "different experiment design")
        require(load(source / "inference_hashes.json") == inference_hashes(source), "saved inference changed")
        write(out / "inference_source.json", {"directory": str(source)})
    else:
        infer(workspace, out)
        write(out / "inference_hashes.json", inference_hashes(out))
    rows = analyze(workspace, source, out)
    if args.hardware_report:
        write(out / "six-trial-audit.json", audit_hardware(workspace, rows))
    if args.dino_snapshot:
        from compare_box_regions import compare

        compare(args.dino_snapshot, rows, out)
        write(out / "results.json", rows)
    render(out, rows)
    print(out / "index.html", flush=True)


if __name__ == "__main__":
    main()
