"""Inventory a five-move evidence bundle for human review. No control or sensor imports."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA = (
    Path(__file__).resolve().parents[1] / "src/go2_perception/vln/schemas/continuous-evidence-v1.schema.json"
)


def verify(report, root):
    Draft202012Validator(json.loads(SCHEMA.read_bytes())).validate(report)
    root = Path(root).resolve(strict=True)
    ids = {}
    for entry in report["artifacts"]:
        rel = Path(entry["path"])
        path = (root / rel).resolve(strict=True)
        if rel.is_absolute() or not path.is_relative_to(root) or not path.is_file() or entry["id"] in ids:
            raise ValueError("duplicate identifier or artifact outside the local bundle")
        digest = sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError("artifact hash mismatch: " + entry["id"])
        ids[entry["id"]] = path
    assessed = []
    for topic, f in report["topics"].items():
        if f["status"] == "unassessed":
            if f["measurements"] or f["evidence_ids"]:
                raise ValueError("unassessed topic contains claims: " + topic)
            continue
        if not (
            f["notes"].strip()
            and f["evidence_ids"]
            and set(f["evidence_ids"]) <= ids.keys()
            and f["measurements"]
        ):
            raise ValueError(
                "assessed topic needs a measured value, note and matching local artifact: " + topic
            )
        assessed.append(topic)
    kind = report["source_kind"]
    if kind == "pending" and assessed:
        raise ValueError("pending report contains assessed claims")
    if kind == "physical_five_move" and (
        not assessed
        or not all(
            report[k] for k in ("run_id", "reviewer", "navigation_commit", "controller_commit", "sdk_commit")
        )
    ):
        raise ValueError("physical report requires exact source revisions, reviewer, run and evidence")
    if kind == "physical_five_move":
        roles = {
            "run_manifest": "run.json",
            "episode_summary": "episode_summary.json",
            "session_summary": "summary.json",
        }
        if (
            any(k not in ids or ids[k].name != name for k, name in roles.items())
            or len({ids[k].parent for k in roles}) != 1
        ):
            raise ValueError(
                "physical evidence requires three hashed manifests from the same baseline session"
            )
        run, episode, session = (json.loads(ids[k].read_bytes()) for k in roles)
        if not (
            run.get("mode") == "LIVE"
            and run.get("navigation_mode") == "live"
            and run.get("backend") == "lightnav"
            and run.get("integration") == "go2-five-move-episode-v1"
            and run.get("git_commit") == report["navigation_commit"]
            and run.get("canonical_control_revision") == report["controller_commit"]
            and episode.get("episode_id") == report["run_id"]
            and episode.get("schema") == "go2-five-move-episode-v1"
            and episode.get("state") == "TERMINAL"
            and session.get("ended_wall_utc")
        ):
            raise ValueError("manifests do not describe a matching closed physical five-move run")
    return dict(
        source_kind=kind,
        assessed_topics=assessed,
        artifacts_verified=len(ids),
        disposition={
            "pending": "awaiting_claude",
            "physical_five_move": "operator_attested_for_human_review",
        }.get(kind, "offline_reference"),
        physical_activation=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--bundle-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(json.loads(args.report.read_bytes()), args.bundle_root or args.report.parent), indent=2
        )
    )
