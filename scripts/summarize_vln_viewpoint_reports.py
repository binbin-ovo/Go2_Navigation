#!/usr/bin/env python3
"""Combine existing paired-view analyses without counting relinkages as new trials."""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from go2_perception.viewer.privacy import require_private_dir  # noqa: E402
from go2_perception.vln.viewpoint import SUMMARY_SCHEMA, chosen  # noqa: E402


def summarize(paths, output):
    output = require_private_dir(Path(output).resolve())
    rows, seen = [], set()
    for path in paths:
        path = Path(path).resolve()
        source = path / "summary.json" if path.is_dir() else path
        raw = source.read_bytes()
        data = json.loads(raw)
        if data.get("schema") != SUMMARY_SCHEMA:
            raise ValueError("Expected a viewpoint analysis summary")
        key = (
            str(Path(data["archive_root"]).resolve()),
            data["before"]["trial"]["trial_id"],
            data["after"]["trial"]["trial_id"],
        )
        if key in seen:
            raise ValueError(
                "Repeated trial pair: different linkage versions are not separate experiments"
            )
        if output.is_relative_to(Path(key[0])) or output == source.parent:
            raise ValueError(
                "Combined output must be separate from the archive and source report"
            )
        seen.add(key)
        row = {
            "archive_root": key[0],
            "before_trial_id": key[1],
            "after_trial_id": key[2],
            "target": data["before"]["trial"]["operator_text"],
            "recorder_commit": data.get("archive_git_commit"),
            "reviewed_image_outcome": data["outcome"]["reviewed_image_outcome"],
            "capture_time_verification": data["outcome"]["temporal_verification"],
            "source_summary": str(source),
            "source_summary_sha256": hashlib.sha256(raw).hexdigest(),
            "report": quote(os.path.relpath(source.parent / "index.html", output)),
            "receipt_ordering_evidence": data.get("receipt_ordering_evidence"),
            "linkage": (data.get("pair_annotation") or {}).get("linkage_import"),
        }
        for role in ("before", "after"):
            pred = chosen(data[role])
            pid = None if pred is None else pred["record"]["prediction_id"]
            record = {} if pred is None else pred["record"]
            reviews = [
                r
                for r in data[role]["reviews"]
                if r.get("eligible") and r["prediction_ids"] == [pid]
            ]
            row[role] = {
                "prediction_id": pid,
                "model": "OPOS-0"
                if record.get("opos_not_visible")
                else "target point"
                if record.get("target_fraction")
                else "no usable point",
                "human_choices": [
                    r["record"].get("choice", "legacy scoped review") for r in reviews
                ],
                "receipt_label": record.get("label_at_receipt"),
                "path_stopping_navigation": None
                if pred is None
                else {
                    k: pred["human_assessment"][k]["status"]
                    for k in ("path", "stopping", "navigation")
                },
            }
        rows.append(row)
    result = {
        "schema": "go2-vln-viewpoint-comparison-v1",
        "unique_pairs": len(rows),
        "rows": rows,
        "scope": "Descriptive correlated-room observations. No success rate, navigation success, executed path, or guaranteed exposure ordering.",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    cells = []
    markdown = [
        "| Target | Before: model / human | After: model / human | Reviewed-image outcome | Post-turn receipt margin | Capture timing |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        evidence = row["receipt_ordering_evidence"] or {}
        margin = evidence.get("margin_s")
        values = [
            row["target"],
            *(
                row[r]["model"] + " / " + ", ".join(row[r]["human_choices"])
                for r in ("before", "after")
            ),
            row["reviewed_image_outcome"]["description"],
            "unknown" if margin is None else f"{margin:.3f} s ({evidence['status']})",
            row["capture_time_verification"],
        ]
        cells.append(
            "<tr>"
            + "".join("<td>" + html.escape(x) + "</td>" for x in values)
            + '<td><a href="'
            + html.escape(row["report"], quote=True)
            + '">Pair report</a></td></tr>'
        )
        markdown.append(
            "| "
            + " | ".join(v.replace("|", "\\|").replace("\n", " ") for v in values)
            + " |"
        )
    (output / "comparison.md").write_text(
        "\n".join(markdown) + "\n\n" + result["scope"] + "\n"
    )
    (output / "index.html").write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Paired-view comparison</title>
<style>body{font:16px/1.5 system-ui;margin:24px;color:#182c39}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bdcbd5;padding:12px;text-align:left}th{background:#eef3f7}</style>
<h1>Human-reviewed paired views</h1><p>Four concerns stay separate: reviewed-image judgment, receipt ordering, capture-time uncertainty, and unreviewed navigation behavior.</p>
<table><tr><th>Target</th><th>Before: model / human</th><th>After: model / human</th><th>Reviewed-image outcome</th><th>Post-turn receipt margin</th><th>Capture timing</th><th>Evidence</th></tr>"""
        + "".join(cells)
        + "</table><p>"
        + html.escape(result["scope"])
        + '</p><a href="summary.json">Combined JSON and provenance</a></html>'
    )
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        result = summarize(a.reports, a.output)
    except (ValueError, OSError) as error:
        p.exit(2, f"Cannot combine reports: {error}\n")
    print(f"{result['unique_pairs']} unique pairs; report: {a.output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
