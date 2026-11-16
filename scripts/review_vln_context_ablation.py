"""Local evaluation-only comparison, reusing the existing pointing/path renderers.

Stage A renders actual processed inputs with explicit pending output panels. Stage
B renders every declared request, including failures and unattempted requests.
Neither mode writes annotations or constructs model jobs.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from go2_perception.vln.context_ablation import CASES, CONDITIONS
from go2_perception.vln.episodes import file_sha256, load_annotations, load_episode
from go2_perception.vln.shadow import write_json
from go2_perception.vln.shadow_display import escaped, path_svg, point_fraction, relative, target_fraction


OLD_REVIEW = {
    CASES[0]: "OLD outputs: bounded path/extra turn inappropriate; should stop. Full also missed STOP, not separately judged.",
    CASES[1]: "OLD outputs: rack identifiable; full target correct, bounded wrong. Moving closer was noted, but instruction/reference still require facing/STOP.",
    CASES[2]: "OLD complete paths: full acceptable, bounded not. This finding did not rely on the lossy action mapping.",
    CASES[3]: "OLD same-eight-image control was not human-reviewed.",
    CASES[4]: "OLD outputs: bounded more plausible than full. Comparative only; target/path not independently judged, no success label.",
}


def read(path):
    return json.loads(Path(path).read_text())


def reference(case, plan):
    if case["kind"] != "episode":
        return None
    ep = load_episode(case["episode"], verify_files=False)
    path = ep.root.parent / "evaluation_only/annotations.json"
    if file_sha256(path) != plan["source_files"][str(path)]:
        raise ValueError("Existing human reference changed")
    annotations = load_annotations(path, ep)
    if annotations["reviewer"]["status"] != "reviewed":
        return None
    return next((d for d in annotations["decisions"] if d["decision_id"] == case["decision_id"]), None)


def image_html(path, root, result=None):
    if not path.is_file():
        return "<p>No processed image saved.</p>"
    marks = ""
    if result:
        pointing = result.get("pointing", {})
        for p, color, label in ((target_fraction(pointing), "#e00", "OPOS target"),
                                (point_fraction(pointing.get("apos")), "#06c", "APOS affordance")):
            if p:
                marks += f'<span class="mark" style="left:{p[0]*100}%;top:{p[1]*100}%;color:{color}" title="{label}">+</span>'
    return f'<div class="image"><img src="{relative(path,root)}">{marks}</div>'


def output_html(result):
    if result is None:
        return '<p class="pending">NO NEW INFERENCE — output, timing and memory pending Stage B.</p>'
    values = {k: result.get(k) for k in ("status", "error", "decode_error", "native_output", "pointing", "decoder_stop", "explicit_stop_code")}
    values["APOS1299_completion_claim_not_verified"] = result.get("pointing", {}).get("apos") == 1299
    rows = ""
    for i, wp in enumerate(result.get("waypoints", [])):
        rows += "<tr>" + "".join(f"<td>{escaped(v)}</td>" for v in (i, *wp)) + "</tr>"
    return (f'<pre>{escaped(json.dumps(values,indent=2))}</pre>{path_svg(result.get("waypoints"))}'
            '<table><tr><th>#</th><th>x forward (m)</th><th>y left (m)</th><th>yaw CCW (rad)</th></tr>'
            f'{rows}</table><p>Absolute decision-frame waypoints; no execution or time base.</p>')


def equality(left, right):
    if not left or not right:
        return {"status": "pending/incomplete"}
    a, b = read(Path(left["folder"]) / "native/result.json"), read(Path(right["folder"]) / "native/result.json")
    comparison = {k: a.get(k) == b.get(k) for k in (
        "native_output", "pointing", "waypoints", "decoder_stop", "explicit_stop_code")}
    paths = [Path(r["folder"]) / "native/model_inputs_and_generated_ids.npz" for r in (left, right)]
    if not all(p.is_file() for p in paths):
        comparison["archive_status"] = "missing; do not infer equality"
        return comparison
    with np.load(paths[0]) as x, np.load(paths[1]) as y:
        comparison["generated_ids_equal"] = ("generated_ids" in x and "generated_ids" in y
                                              and np.array_equal(x["generated_ids"], y["generated_ids"]))
        keys = set(x.files) - {"generated_ids"}
        comparison["all_generation_inputs_equal"] = (keys == set(y.files) - {"generated_ids"}
                                                       and all(np.array_equal(x[k], y[k]) for k in keys))
    return comparison


def render(a):
    plan = read(a.run / "plan.json")
    stage_a = read(a.run / "stage_a/prepared_jobs.json")
    checks = read(a.run / "stage_a/input_checks.json")
    prepared = {(e["case"], e["condition"]): e for e in stage_a}
    rows = [] if a.inputs_only else read(a.run / "results/rows.json") if (a.run / "results/rows.json").exists() else []
    actual = {(r["case"], r["condition"]): r for r in rows}
    report = {"stage": "A inputs only" if a.inputs_only else "B", "requests": [], "control": checks["checks"],
              "new_human_review": "pending for ALL new outputs, including repeated F/B", "live_latency_measured": False,
              "physical_execution_authorized": False, "age_limit_s": 3.0}
    sections = []
    for case in plan["cases"]:
        name = case["case"]
        ref = reference(case, plan)  # evaluation-only; never passed to worker
        cards = []
        for c in CONDITIONS:
            entry, row = prepared[name,c], actual.get((name,c))
            folder = Path(entry["folder"] if row is None else row["folder"])
            result_path = folder / "native/result.json"
            result = read(result_path) if row and result_path.is_file() else None
            if row is not None:
                job = read(folder / "job.json")
                provenance = read(folder / "preparation.json")
            else:
                job = entry["job"]
                provenance = read(folder / "preparation.json")
            construction = read(Path(entry["folder"]) / "construction.json")
            ids = sorted({i for g in construction["layout_audit"]["segments"] for i in g["source_indices"]})
            current = folder / "native" / f"input_{ids[-1]:04d}.png"
            hist = []
            frames = dict(zip(provenance["source_indices"],job["frames"],strict=True))
            for i in ids:
                frame = frames[i]
                hist.append(f'<figure>{image_html(folder/"native"/f"input_{i:04d}.png",a.run)}'
                            f'<figcaption>Source {i}: capture {frame["observation_s"]:.6f}s; '
                            f'available {frame["available_s"]:.6f}s</figcaption></figure>')
            values = {"case": name, "condition": c, "status": "pending Stage B" if row is None else row["status"],
                      "human_reference_evaluation_only": ref, "new_human_review": "pending",
                      "native": result, "timing": row}
            report["requests"].append(values)
            cards.append(f'<article><h3>{c}</h3><p>{escaped(plan["conditions"][c])}</p>{image_html(current,a.run,result)}'
                         f'{output_html(result)}<details><summary>Timing/memory — offline only</summary>'
                         f'<pre>{escaped(json.dumps({"timing":row,"memory":None if result is None else result.get("memory"),"stages":None if result is None else result.get("stage_profile")},indent=2))}</pre></details>'
                         f'<details><summary>Actual native history ({len(ids)} unique)</summary>{"".join(hist)}</details>'
                         f'<details><summary>Timing, pooling, pairs, padding and prompt structure</summary>'
                         f'<pre>{escaped(json.dumps(construction["layout_audit"],indent=2))}</pre>'
                         f'<a href="{relative(folder/"native/prompt.txt",a.run)}">Exact tokenized prompt</a></details></article>')
        old = []
        for c in ("F", "B"):
            old_result = read(Path(case["previous_folders"][c]) / "native/result.json")
            old.append(f'<h4>Historical {c} — previously saved output</h4>{output_html(old_result)}')
        instruction = prepared[name,"F"]["job"]["instruction"]
        sections.append(f'<section><h2>{escaped(name)}</h2><p class="instruction">{escaped(instruction)}</p>'
                        f'<p>Cutoff: {prepared[name,"F"]["job"]["decision_time_s"]:.6f}s. '
                        'Current image is processed model space. Red + OPOS, blue + APOS.</p>'
                        f'<div class="cards">{"".join(cards)}</div>'
                        f'<details><summary>EVALUATION ONLY — unchanged human reference and historical review</summary>'
                        f'<pre>{escaped(json.dumps(ref,indent=2))}</pre><p>{escaped(OLD_REVIEW[name])}</p>'
                        '<p>These judgements concern OLD outputs only. New outputs require human review. '
                        'Matching F does not resolve its stopping failures.</p>' + "".join(old) + '</details></section>')
    if not a.inputs_only:
        report["same_image_control_outputs"] = equality(actual.get((CASES[3],"TP")), actual.get((CASES[3],"F")))
        report["runtime_noop_controls"] = {name: {"B_P": equality(actual.get((name,"B")),actual.get((name,"P"))),
                                                        "T_TP": equality(actual.get((name,"T")),actual.get((name,"TP")))}
                                                 for name in (CASES[1],CASES[4])}
    suffix = "stage_a" if a.inputs_only else "stage_b"
    html = a.run / f"index_{suffix}.html"
    if html.exists(): raise FileExistsError("Preserve the previous review artifact")
    write_json(a.run / f"review_{suffix}.json", report)
    html.write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>Context ablation — private offline diagnostic</title>'
        '<style>body{font:16px system-ui;margin:24px;background:#fafafa;color:#17232d}section{padding:20px 0;border-top:2px solid #999}'
        '.cards{display:flex;gap:16px;overflow-x:auto}article{flex:0 0 448px;background:white;padding:12px;border:1px solid #aaa}'
        '.image{position:relative;width:100%}.image img{width:100%;display:block}.mark{position:absolute;transform:translate(-50%,-50%);font-size:28px}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}figure{margin:4px 0}figcaption{font-size:12px}svg{width:100%}'
        'td,th{padding:3px;border-bottom:1px solid #ddd;font-size:12px}.pending{color:#703b00;font-weight:bold}'
        '.instruction{font-size:22px}details{padding:10px 0}summary{cursor:pointer;font-weight:bold}</style>'
        f'<h1>Context ablation — {escaped(report["stage"])}</h1><p>25 declared requests. Private, correlated development diagnostics. '
        'No robot execution. New output panels remain pending until Stage B actually runs.</p>'
        '<p>TP also restores segment/prompt grouping; this is not a clean two-factor interaction. '
        'Native 4 Hz timing is a grid approximation. Every source timestamp remains visible.</p>' + ''.join(sections))
    print(json.dumps({"viewer":str(html),"panels":len(report["requests"]),"new_model_outputs":len(rows)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--inputs-only", action="store_true")
    render(p.parse_args())
