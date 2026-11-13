"""Validate Claude's exact packages and hash every package/baseline file, read-only."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from go2_perception.vln.episodes import load_annotations, load_episode
from go2_perception.vln.pretrained import causal_history

PACKAGES = [
    "p6b_test_c_e1_turn_right",
    "p6b_test_c_e2_turn_left",
    "p6b_test_c_e3_forward_bottles",
    "p6b_test_c_e4_absent_sofa",
]


def file_sha256(path):
    """Hash a complete stable file; an early EOF must never certify an empty file."""
    path = Path(path)
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as f:
        before = os.fstat(f.fileno())
        for block in iter(lambda: f.read(1024 * 1024), b""):
            count += len(block)
            digest.update(block)
        after = os.fstat(f.fileno())
    current = path.stat()

    def identity(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns

    if (
        count != before.st_size
        or identity(before) != identity(after)
        or identity(after) != identity(current)
    ):
        raise OSError(
            f"Short read or concurrently changed file: {path} ({count}/{before.st_size} bytes)"
        )
    return digest.hexdigest()


def inventory(root):
    result = {"packages": [], "package_files": {}, "baseline_files": {}}
    for name in PACKAGES:
        path = root / "episodes" / name
        ep = load_episode(path / "model_input/episode.json")
        annpath = path / "evaluation_only/annotations.json"
        ann = load_annotations(annpath, ep)  # inventory/evaluation only; no model here
        result["packages"].append(
            {
                "episode_id": name,
                "manifest": str(ep.path),
                "manifest_sha256": ep.sha256,
                "annotation_sha256": file_sha256(annpath),
                "frames": len(ep.data["frames"]),
                "reviewer": ann["reviewer"],
                "decisions": [
                    {
                        "id": d["decision_id"],
                        "cutoff_s": d["time_s"],
                        "causal_4hz_frames": len(
                            causal_history(ep, d["decision_id"])["frames"]
                        ),
                    }
                    for d in ep.data["decisions"]
                ],
            }
        )
        for f in sorted(path.rglob("*")):
            if f.is_file():
                result["package_files"][str(f)] = file_sha256(f)
        print(f"Verified {name}", flush=True)
    for f in sorted((root / "claude_pilot_v1").rglob("*")):
        if f.is_file():
            result["baseline_files"][str(f)] = file_sha256(f)
    if not result["baseline_files"]:
        raise FileNotFoundError("Saved Claude pilot SmolVLM baseline missing")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase6b-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--compare", type=Path)
    a = ap.parse_args()
    result = inventory(a.phase6b_root)
    mismatch = False
    if a.compare:
        before = json.loads(a.compare.read_text())
        comparison = {}
        for key in ("package_files", "baseline_files"):
            changed = {
                p: {"before": before[key].get(p), "after": result[key].get(p)}
                for p in set(before[key]) | set(result[key])
                if before[key].get(p) != result[key].get(p)
            }
            comparison[key] = {
                "matched": sum(result[key].get(p) == h for p, h in before[key].items()),
                "mismatches": changed,
            }
            mismatch |= bool(changed)
        result["compared_against" if mismatch else "unchanged_against"] = str(a.compare)
        result["preservation_comparison"] = comparison
        result["preservation_passed"] = not mismatch
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({key: len(result[key]) for key in ("package_files", "baseline_files")})
    if mismatch:
        raise ValueError(
            f"Source preservation mismatch retained in {a.output}; inspect before claiming unchanged"
        )


if __name__ == "__main__":
    main()
