"""Write an immutable offline review of a closed five-move trial into a fresh private folder."""

import argparse
from html import escape
import json
import os
from pathlib import Path
from tempfile import mkdtemp

from go2_perception.vln.grounding_review import episode_decisions, saved_input
from go2_perception.vln.prediction_archive import records, require
from go2_perception.vln.shadow_display import path_svg


def link(path, label):
    return f'<a href="{escape(Path(path).resolve().as_uri())}">{escape(label)}</a>'


def marked_image(path, point):
    mark = ""
    if point is not None:
        u, v = point
        mark = f'<span aria-label="Unverified native mark" class="mark" style="left:{u * 100:.5f}%;top:{v * 100:.5f}%">+</span>'
    return f'<div class="image"><img alt="Exact model input" src="{escape(Path(path).resolve().as_uri())}">{mark}</div>'


def page(path, title, body):
    css = """body{font:16px/1.45 system-ui;color:#1b3040;margin:2rem auto;padding:0 1rem;max-width:96rem}
    a{color:#07557c} h2{margin-top:2.2rem} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:1rem}
    article,details{border:1px solid #c8d1d7;padding:.8rem;border-radius:6px;margin:.5rem 0} .image{position:relative}
    img,svg{width:100%;display:block} .mark{position:absolute;color:#fc39b5;text-shadow:0 0 2px black,0 0 4px black;
    transform:translate(-50%,-50%);font:bold 28px/1 monospace} code{overflow-wrap:anywhere} .history img{width:100%}
    .history{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.5rem} figure{margin:0}"""
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title><style>{css}</style><main><h1>{escape(title)}</h1>{body}</main></html>"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument(
        "--prediction",
        help="Open this prediction first. Defaults to the last prediction in a single-trial session.",
    )
    args = p.parse_args(argv)
    root = args.session.resolve()
    summary, decisions = episode_decisions(root)
    require(summary["state"] == "TERMINAL" and (root / "summary.json").is_file(), "closed session required")
    if args.prediction is None:
        candidates = list((root / "trials").glob("*/predictions.jsonl"))
        if len(candidates) != 1 or not (predictions := records(candidates[0])):
            p.error("Specify --prediction for a session with multiple trials or no default prediction")
        args.prediction = predictions[-1]["prediction_id"]
    require(Path(args.prediction).name == args.prediction, "prediction ID")
    trial_dir = root / "trials" / args.prediction.rsplit("-P", 1)[0]
    selected = saved_input(root, args.prediction)
    base = Path(os.environ["GO2_WORKSPACE"]).resolve() / "runs/target_grounding"
    require(base != root and root not in base.parents, "report destination inside recording")
    base.mkdir(exist_ok=True)
    out = Path(mkdtemp(prefix="trace-", dir=base))
    result, panels = [], []
    for pred in records(trial_dir / "predictions.jsonl"):
        pid = pred["prediction_id"]
        data = selected if pid == args.prediction else saved_input(root, pid)
        ev = decisions.get(pid, {})
        grounding = ev.get("grounding") or {}
        if grounding:
            require(
                grounding["image_sha256"] == pred["source_image"]["file_sha256"],
                "executor/model image binding",
            )
        sel, res = ev.get("selection"), ev.get("result")
        reviews = ev.get("reviews", [])
        action = sel["action"] if sel else None
        text = (
            f"{sel['mode']} · {sel['source']} · {action['direction']} {action['magnitude']:.3f} {action['unit']}"
            if action
            else "No dispatch recorded for this prediction"
        )
        review = ", ".join(r["decision"] for r in reviews) or "No image-bound human decision"
        progress = (
            f"{len(res['command_send_statuses'])} movement send returns, settled estimate "
            f"{res['measured_progress']} {action['unit']}, released {res['ownership_released']}"
            if action and res and res["measured_progress"] is not None
            else "See executor record"
        )
        source = data["frames"][-1]["path"]
        row = dict(
            prediction_id=pid,
            requested_target=data["trial"]["operator_text"],
            entered_target_text=data["trial"].get("entered_target_text"),
            instruction=pred["instruction"],
            source_image=source,
            history=data["frames"],
            age_s=pred["display_timing_at_receipt"]["observation_age_at_display_s"],
            native_output=pred["native_output"],
            point_fraction=pred["target_fraction"],
            waypoints=pred["waypoints"],
            receipt_status=pred["label_at_receipt"],
            refusal=pred["rejection_reason_at_receipt"],
            evidence=ev,
        )
        result.append(row)
        history = "".join(
            '<figure><img loading="lazy" alt="History frame" src="'
            + Path(f["path"]).as_uri()
            + f'"><figcaption>Index {f["index"]} · {f["observation_s"]:.3f} s</figcaption></figure>'
            for f in data["frames"]
        )
        panels.append(
            f"<details{' open' if pid == args.prediction else ''}><summary>{escape(pid)} · {escape(text)}</summary>"
            + f'<p>Recorded request <b>{escape(row["requested_target"])}</b> · {escape(row["instruction"])}</p><div class="grid"><div>'
            + marked_image(source, row["point_fraction"])
            + link(source, "Open unmarked model input")
            + "</div><div>"
            + path_svg(row["waypoints"])
            + "</div></div><p>Model mark (unverified) · "
            + escape(str(row["point_fraction"]) if row["point_fraction"] else "no decoded point")
            + " · "
            + escape(str(grounding.get("status", "no executor grounding")))
            + " · "
            + escape(str(grounding.get("method", ev.get("rejection", ""))))
            + f"</p><p>Image review · {escape(review)}</p><p>Dispatch · {escape(text)}. {escape(progress) if res else ''}</p>"
            + f"<p>Native output <code>{escape(row['native_output'])}</code></p>"
            + link(data["request"] / "native/prompt.txt", "Exact model prompt")
            + " · "
            + link(data["request"] / "native/result.json", "Native point, ten waypoints and STOP")
            + f'<h3>Actual input history</h3><div class="history">{history}</div></details>'
        )
    trial = selected["trial"]
    (out / "trace.json").write_text(
        json.dumps(dict(episode=summary, trial=trial, predictions=result), indent=2)
    )
    page(
        out / "trace.html",
        "Recorded grounding and movement",
        f"<p>Episode outcome <b>{escape(summary['outcome'])}</b> · {summary['moves']} recorded movements. "
        "Magenta is the model’s claim. The separate path diagram shows its full proposal in the decision frame.</p>"
        + "".join(panels)
        + link(out / "trace.json", "Complete trace and source hashes"),
    )
    print(out / "trace.html")


if __name__ == "__main__":
    main()
