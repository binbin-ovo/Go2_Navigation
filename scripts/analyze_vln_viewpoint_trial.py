#!/usr/bin/env python3
"""Analyze two completed independent queries, using archived evidence only."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from go2_perception.vln.viewpoint import Archive, EvidenceError, analyze_pair  # noqa: E402
from go2_perception.vln.viewpoint_report import write_report  # noqa: E402
from go2_perception.vln.viewpoint_linkage import import_linked_pair, receipt_evidence  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Completed interactive run (final summary.json required)",
    )
    parser.add_argument("--before-trial", required=True)
    parser.add_argument("--after-trial", required=True)
    parser.add_argument(
        "--before-prediction",
        help="Required to select among multiple predictions; no latest-wins",
    )
    parser.add_argument("--after-prediction")
    annotation = parser.add_mutually_exclusive_group()
    annotation.add_argument(
        "--annotation",
        type=Path,
        help="Evaluation-only go2-vln-viewpoint-pair-v1 sidecar",
    )
    annotation.add_argument(
        "--linked-pairs",
        help="Run-relative recorder JSONL, e.g. evaluation/paired_views.jsonl",
    )
    parser.add_argument(
        "--method-version",
        type=int,
        help="Explicit linker method version; current adapter supports 2",
    )
    parser.add_argument(
        "--linkage-line",
        type=int,
        help="Optional exact JSONL line to disambiguate the same pair/method",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh Git-ignored local report directory",
    )
    args = parser.parse_args(argv)
    if bool(args.linked_pairs) != (args.method_version is not None) or (
        args.linkage_line is not None and not args.linked_pairs
    ):
        parser.error(
            "--linked-pairs requires --method-version; --linkage-line applies only to that importer"
        )
    try:
        raw = args.annotation.read_bytes() if args.annotation else None
        sidecar = json.loads(raw) if raw else None
        archive = Archive(args.run)
        if args.linked_pairs:
            sidecar = import_linked_pair(
                archive,
                args.linked_pairs,
                args.before_trial,
                args.after_trial,
                method_version=args.method_version,
                record_line=args.linkage_line,
            )
            for role in ("before", "after"):
                attr = role + "_prediction"
                referenced = sidecar["prediction_references"][role]
                if (
                    getattr(args, attr) is not None
                    and getattr(args, attr) != referenced
                ):
                    raise EvidenceError(
                        "Explicit prediction selection disagrees with the selected linkage record"
                    )
                setattr(args, attr, referenced)
        analysis = analyze_pair(
            archive,
            args.before_trial,
            args.after_trial,
            sidecar=sidecar,
            before_prediction=args.before_prediction,
            after_prediction=args.after_prediction,
        )
        if raw is not None:
            if args.annotation.read_bytes() != raw:
                raise EvidenceError("Pair annotation changed during analysis")
            analysis["pair_annotation_sha256"] = hashlib.sha256(raw).hexdigest()
        if args.linked_pairs:
            analysis["receipt_ordering_evidence"] = receipt_evidence(archive, analysis)
            archive.assert_unchanged()
            analysis["consumed_artifact_sha256"] = dict(archive.manifest)
            analysis["limitations"].append(
                "Receipt ordering uses the archived linker v2 turn annotation and audited index/counter metadata; the raw yaw estimator was not rerun."
            )
        root = Path(__file__).resolve().parents[1]
        analysis["evaluator_files_sha256"] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "scripts/analyze_vln_viewpoint_trial.py",
                "src/go2_perception/vln/viewpoint.py",
                "src/go2_perception/vln/viewpoint_report.py",
                "src/go2_perception/vln/viewpoint_linkage.py",
            )
        }
        analysis["python_version"] = sys.version.split()[0]
        report = write_report(archive, analysis, args.output)
    except (EvidenceError, OSError, ValueError) as error:
        parser.exit(2, f"Cannot analyze pair: {error}\n")
    print(analysis["outcome"]["reviewed_image_outcome"]["description"])
    print("Capture-time verification: " + analysis["outcome"]["temporal_verification"])
    print(f"Report: {report}\nSummary: {report.parent / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
