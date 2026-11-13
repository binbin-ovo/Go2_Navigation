"""Score saved proposals against separate references; keep correlated recordings grouped."""

import argparse
import json
from pathlib import Path
import re

from go2_perception.vln.episodes import file_sha256, load_annotations, load_episode
from go2_perception.vln.temporal_review import summarize


def human_action_reference(ref):
    return bool(
        ref
        and ref["reviewer"]["type"] == "human"
        and ref["reviewer"]["status"] == "reviewed"
        and ref["decision"]["acceptable_actions"]
    )


def evaluate(run, *, annotations_name=None, static_episodes=()):
    """Read declared manifests/results only. No model loading, frame rebuilding or relabeling."""
    run = Path(run)
    declaration = json.loads((run / "run_declaration.json").read_text())
    static_episodes = set(static_episodes)
    metadata, refs, proposals, rows, pending, annotation_sources = (
        {},
        {},
        [],
        [],
        [],
        [],
    )
    expected_decisions = set()
    for declared in declaration["episodes"]:
        ep = load_episode(declared["manifest"])
        episode_id = ep.data["episode_id"]
        if ep.sha256 != declared["sha256"] or episode_id != declared["episode_id"]:
            raise ValueError("Declared episode changed after inference")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", episode_id):
            raise ValueError("Unsafe episode identifier")
        metadata[episode_id] = {
            "recording_id": ep.data["recording"]["recording_id"],
            "source_manifest_sha256": ep.data["recording"]["source_manifest_sha256"],
            "clock_epoch": ep.data["clock"]["epoch"],
            "case_kind": "static_absent_target"
            if episode_id in static_episodes
            else "navigation_proposal",
        }
        annotation = ep.root / annotations_name if annotations_name else None
        labels = (
            load_annotations(annotation, ep)
            if annotation and annotation.is_file()
            else None
        )
        annotation_sources.append(
            {
                "episode_id": episode_id,
                "path": str(annotation) if annotation else None,
                "sha256": file_sha256(annotation) if labels else None,
                "status": "loaded" if labels else "missing; evaluation pending",
            }
        )
        if labels:
            refs.update(
                {
                    (episode_id, d["decision_id"]): {
                        "reviewer": labels["reviewer"],
                        "decision": d,
                    }
                    for d in labels["decisions"]
                }
            )
        for decision in ep.data["decisions"]:
            decision_id = decision["decision_id"]
            if (
                declaration["decision_filter"]
                and declaration["decision_filter"] != decision_id
            ):
                continue
            if not re.fullmatch(r"[A-Za-z0-9_-]+", decision_id):
                raise ValueError("Unsafe decision identifier")
            key = (episode_id, decision_id)
            expected_decisions.add(key)
            ref = refs.get(key)
            reviewed = human_action_reference(ref)
            for mode in declaration["modes"]:
                if mode not in {"temporal", "current_only"}:
                    raise ValueError("Unknown declared inference mode")
                dest = run / episode_id / decision_id / mode
                if not (dest / "proposal.json").is_file():
                    pending.append(
                        {
                            "episode_id": episode_id,
                            "decision_id": decision_id,
                            "mode": mode,
                        }
                    )
                    continue
                p = json.loads((dest / "proposal.json").read_text())
                provenance = json.loads((dest / "input_provenance.json").read_text())
                if (p["episode_id"], p["decision_id"], p["mode"]) != (
                    episode_id,
                    decision_id,
                    mode,
                ):
                    raise ValueError(
                        "Proposal identity differs from its declared location"
                    )
                if provenance["episode_manifest_sha256"] != ep.sha256:
                    raise ValueError("Proposal inputs bound to another manifest")
                if p["selected_observations"]["decision_time_s"] != decision["time_s"]:
                    raise ValueError("Proposal decision cutoff differs from manifest")
                proposals.append(p)
                rows.append(
                    {
                        "episode_id": episode_id,
                        "decision_id": decision_id,
                        "mode": mode,
                        **metadata[episode_id],
                        "proposal": p["proposal"],
                        "task_status": p["task_status"],
                        "parse_status": p["parse_status"],
                        "action_reference_status": "human_reviewed"
                        if reviewed
                        else "pending",
                        "acceptable_actions": ref["decision"]["acceptable_actions"]
                        if reviewed
                        else None,
                        "action_agrees": (
                            p["parse_status"] != "rejected"
                            and p["proposal"] in ref["decision"]["acceptable_actions"]
                        )
                        if reviewed
                        else None,
                        "reference_ambiguity": ref["decision"]["ambiguity"]
                        if reviewed
                        else None,
                        "reference_completion": ref["decision"]["completion_status"]
                        if reviewed
                        else None,
                    }
                )
    if static_episodes - metadata.keys():
        raise ValueError("Static episode identifier not present in run declaration")

    def modes(selected, *, sequence_metrics=True):
        result = {
            m: summarize([p for p in selected if p["mode"] == m], refs)
            for m in declaration["modes"]
        }
        if not sequence_metrics:
            # Filtering on a reference label can make nonadjacent cutoffs neighbors.
            for summary in result.values():
                for key in (
                    "repeated_consecutive_actions",
                    "opposite_consecutive_turn_pairs",
                    "sequence_flags_note",
                ):
                    summary.pop(key)
        return result

    recordings = {}
    for recording_id in sorted({x["recording_id"] for x in metadata.values()}):
        ids = [
            e for e, meta in metadata.items() if meta["recording_id"] == recording_id
        ]
        subset = [p for p in proposals if p["episode_id"] in ids]
        recordings[recording_id] = {
            "correlated_examples": True,
            "episode_ids": ids,
            "source_manifest_sha256": sorted(
                {metadata[e]["source_manifest_sha256"] for e in ids}
            ),
            "clock_epochs": sorted({metadata[e]["clock_epoch"] for e in ids}),
            "by_mode": modes(subset),
            "by_case_kind": {
                kind: modes(
                    [
                        p
                        for p in subset
                        if metadata[p["episode_id"]]["case_kind"] == kind
                    ]
                )
                for kind in sorted({metadata[e]["case_kind"] for e in ids})
            },
            "by_reference_ambiguity": {
                str(ambiguous).lower(): modes(
                    [
                        p
                        for p in subset
                        if human_action_reference(
                            refs.get((p["episode_id"], p["decision_id"]))
                        )
                        and refs[(p["episode_id"], p["decision_id"])]["decision"][
                            "ambiguity"
                        ]
                        == ambiguous
                    ],
                    sequence_metrics=False,
                )
                for ambiguous in [False, True]
            },
        }
    reviewed_keys = {
        key for key in expected_decisions if human_action_reference(refs.get(key))
    }
    return {
        "schema_version": "go2-temporal-evaluation-v1",
        "scope": "Offline action-reference agreement only; correlated examples, no navigation-success estimate",
        "run_declaration_sha256": file_sha256(run / "run_declaration.json"),
        "unique_declared_decisions": len(expected_decisions),
        "human_reviewed_action_decisions": len(reviewed_keys),
        "pending_action_reference_decisions": len(expected_decisions - reviewed_keys),
        "pending_inference_calls": pending,
        "annotation_sources": annotation_sources,
        "by_recording": recordings,
        "by_episode": {
            e: modes([p for p in proposals if p["episode_id"] == e]) for e in metadata
        },
        "decisions": rows,
        "closed_loop_navigation_success": None,
        "collision_avoidance_success": None,
        "execution_authorized": False,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument(
        "--annotations-name", help="evaluation path relative to each manifest folder"
    )
    ap.add_argument(
        "--static-episode",
        action="append",
        default=[],
        help="episode ID for a static absent-target case",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = evaluate(
        args.run,
        annotations_name=args.annotations_name,
        static_episodes=args.static_episode,
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        f"{report['human_reviewed_action_decisions']} human-reviewed / {report['unique_declared_decisions']} decisions; {report['pending_action_reference_decisions']} pending references"
    )


if __name__ == "__main__":
    main()
