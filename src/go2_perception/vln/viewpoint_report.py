"""Static local report for archived paired observations; no web server or inference."""

import html
import json
from pathlib import Path

from ..viewer.privacy import require_private_dir
from .shadow_display import path_svg
from .viewpoint import EvidenceError


def esc(value):
    return html.escape(str(value), quote=True)


def details(title, data):
    return f"<details><summary>{esc(title)}</summary><pre>{esc(json.dumps(data, indent=2, ensure_ascii=False))}</pre></details>"


def time_text(t):
    value = "UNKNOWN" if t.get("time_s") is None else f"{t['time_s']:.6f} s"
    u = t.get("uncertainty_s")
    bound = "error bound UNKNOWN" if u is None else f"error bound ±{u:g} s"
    return f"{value} · {t.get('clock')} · epoch {t.get('clock_epoch')} · {bound}"


def render_report(analysis, assets):
    def picture(item, point=None, caption=""):
        if not item or item["image"]["status"] != "available":
            return '<p class="warning">Input image unavailable or changed. No target mark displayed.</p>'
        im = item["image"]
        mark = ""
        if point is not None:
            mark = (
                f'<span class="cross" style="left:{100 * point[0]:.5f}%;top:{100 * point[1]:.5f}%" '
                'aria-label="Archived native target mark">+</span>'
            )
        return (
            f'<figure><div class="image"><img src="{esc(assets[im["path"]])}" '
            f'alt="Actual model input, frame {item["job_frame_index"]}">{mark}</div>'
            f"<figcaption>{esc(caption)}<br>{esc(time_text(item['capture_time']))}<br>"
            f"Turn relation: {esc(item['turn_relation'])}</figcaption></figure>"
        )

    panels = []
    for role in ("before", "after"):
        view = analysis[role]
        t = view["trial"]
        content = [
            f"<section><h2>{role.title()} · independent query</h2><p>{esc(t['trial_id'])}</p>",
            f'<p class="instruction">{esc(t["instruction"])}</p>',
            f"<p>Selected prediction: {esc(view['selected_prediction_id'])} "
            f"({esc(view['selection_basis'])}) · revision {esc(t.get('instruction_revision'))}</p>",
        ]
        if not view["predictions"]:
            content.append(
                '<p class="warning">No saved predictions; possibly intentionally deleted.</p>'
            )
        for p in view["predictions"]:
            r = p["record"]
            content += [
                f"<article><h3>{esc(r['prediction_id'])}</h3>",
                f"<p>Eligible evidence: {esc(p['eligible'])} · receipt label: {esc(p['receipt_status'])}</p>",
                picture(
                    p["marked_input"],
                    None
                    if "target_mark_disagrees_with_native_pointing" in p["errors"]
                    else r.get("target_fraction"),
                    "This prediction’s own marked model input",
                ),
                f"<p>Native OPOS-0 / not-visible claim: {esc(r.get('opos_not_visible'))}</p>",
                details(
                    "Validation issues",
                    {"errors": p["errors"], "warnings": p["warnings"]},
                ),
                details(
                    "Human assessment by scope (all reviews retained below)",
                    p["human_assessment"],
                ),
                details(
                    "Receipt freshness vs exposure age",
                    {
                        "archived_display_timing_at_receipt": r.get(
                            "display_timing_at_receipt"
                        ),
                        "marked_input_age_at_receipt": p.get("age_at_receipt"),
                        "note": "Receipt CURRENT does not imply freshness when reviewed. Review ages appear with each review.",
                    },
                ),
                '<h4>Actual selected model inputs</h4><div class="frames">',
            ]
            content.extend(
                picture(
                    f,
                    caption=f"job index {f['job_frame_index']} · {f['source'].get('frame_id')}",
                )
                for f in p["selected_inputs"]
            )
            content += [
                "</div><h4>UNEXECUTED native output</h4>",
                f"<pre>{esc(r.get('native_output'))}</pre>",
                "<p>Unsupported path convention: see original output below.</p>"
                if "unsupported_waypoint_convention" in p["errors"]
                else path_svg(r.get("waypoints")),
                "<p>Native local path only: metres/radians, x forward, y left, yaw CCW. "
                "No time base, metric clearance, LiDAR registration, or physical execution.</p>",
                details(
                    "Native path / stop claims (not human verified)",
                    {
                        k: r.get(k)
                        for k in (
                            "waypoints",
                            "waypoint_convention",
                            "waypoint_time_base",
                            "native_stop",
                            "explicit_stop_code",
                            "model_completion_claim",
                            "completion_verified",
                            "execution_status",
                        )
                    },
                ),
                "</article>",
            ]
        content.append("<h3>Human feedback · evaluation only</h3>")
        for review in view["reviews"]:
            r = review["record"]
            content += [
                f"<p><b>{esc(r.get('choice_text', r.get('marking', 'Review')))}</b><br>"
                f"Scope: {esc(review['scope'])} · {esc(review['provenance']['path'])}:"
                f"{review['provenance']['line']}</p>",
                details(
                    "Original review, provenance, validation and age at review", review
                ),
            ]
        if not view["reviews"]:
            content.append("<p>Human review missing.</p>")
        content += [
            details(
                "Other original evaluation records (not inferred verdicts)",
                {k: view[k] for k in ("marks", "notes", "corrections")},
            ),
            "</section>",
        ]
        panels.append("".join(content))
    groups = {}
    for event in analysis["timeline"]:
        t = event["time"]
        groups.setdefault((str(t.get("clock")), str(t.get("clock_epoch"))), []).append(
            event
        )
    timeline = []
    for key, events in groups.items():
        timeline.append(
            f"<h3>{esc(key[0])} · epoch {esc(key[1])}</h3><table>"
            "<tr><th>Time and uncertainty</th><th>Event / linked evidence</th></tr>"
        )
        for e in sorted(
            events,
            key=lambda e: (
                e["time"].get("time_s") is None,
                e["time"].get("time_s") or 0,
            ),
        ):
            timeline.append(
                f"<tr><td>{esc(time_text(e['time']))}</td><td><b>{esc(e['event'])}</b>"
                f"<br>{esc(e.get('trial_id', 'evaluation-only turn annotation'))}"
                f"<br>{esc(e.get('prediction_id', e.get('prediction_ids', '')))}"
                f"<br>{esc(e.get('frame_id', e.get('instruction', '')))}</td></tr>"
            )
        timeline.append("</table>")
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self'; style-src 'unsafe-inline'">
<title>Offline paired-view evidence</title><style>
body{{font:16px/1.5 system-ui,sans-serif;background:#edf2f5;color:#172c3b;margin:0;padding:24px}}
main{{max-width:1500px;margin:auto}}h1,h2,h3{{line-height:1.2}}section,header,article,.timeline{{background:white;padding:20px;border-radius:10px;margin:14px 0}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.pair>section{{min-width:0}}article{{border:1px solid #c7d5df}}
.instruction{{font-size:1.2em;font-weight:600}}.image{{position:relative;line-height:0}}img{{width:100%;height:auto}}
.cross{{position:absolute;transform:translate(-50%,-50%);font:36px/1 monospace;color:#ff375f;text-shadow:1px 1px white,-1px -1px white}}
figure{{margin:10px 0}}figcaption{{font-size:12px;overflow-wrap:anywhere}}.frames{{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}details{{margin:10px 0}}summary{{cursor:pointer}}
.warning{{color:#9a3514}}svg{{width:100%;max-height:320px}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #c7d5df;text-align:left;padding:8px;overflow-wrap:anywhere}}@media(max-width:800px){{.pair{{grid-template-columns:1fr}}}}
</style><main><header><h1>Offline paired-view evidence</h1>
<p>Two independent queries · no cross-view model memory · no inference rerun · no robot commands</p>
<h2>{esc(analysis["outcome"]["description"])}</h2>
{details("Supported comparison and timing qualification", analysis["outcome"])}
<p>Guided judgments concern target grounding/visibility only. Path, stopping and navigation remain unreviewed.</p>
<p><a href="summary.json">Machine-readable summary</a> · <a href="timeline.json">Causal/evaluation timeline</a></p>
{details("Manual turn annotation and clock mappings (evaluation only)", {"turn": analysis["turn"], "mappings": analysis["clock_mappings"]})}
</header><div class="pair">{"".join(panels)}</div><section class="timeline"><h2>Evidence timeline</h2>
<p>Sorted only within each clock/epoch. Input timestamps describe capture estimates, not prediction arrival.
Unknown bounds stay unknown. No ordering is inferred across clock groups.</p>{"".join(timeline)}</section>
{details("Limits of this comparison", analysis["limitations"])}</main></html>"""


def write_report(archive, analysis, output):
    output = require_private_dir(Path(output).resolve())
    if output.is_relative_to(archive.root) or archive.root.is_relative_to(output):
        raise EvidenceError("Report output must be separate from the source archive")
    archive.assert_unchanged()
    output.mkdir(parents=True, exist_ok=False)
    (output / "assets").mkdir()
    assets = {}
    for role in ("before", "after"):
        for pred in analysis[role]["predictions"]:
            for frame in pred["selected_inputs"]:
                image = frame["image"]
                if image["status"] == "available":
                    dest = "assets/" + image["sha256"] + ".png"
                    assets[image["path"]] = dest
                    # Archive.read rejects concurrent changes before copying any bytes.
                    (output / dest).write_bytes(
                        archive.read(image["path"], required=True)
                    )
    archive.assert_unchanged()
    (output / "summary.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    (output / "timeline.json").write_text(
        json.dumps(analysis["timeline"], indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    (output / "index.html").write_text(render_report(analysis, assets))
    return output / "index.html"
