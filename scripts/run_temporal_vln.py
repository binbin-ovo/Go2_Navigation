"""Offline causal episode -> pretrained temporal text proposal -> saved review evidence."""

import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np
import yaml

from go2_perception.vln.episodes import file_sha256, load_annotations, load_episode
from go2_perception.vln.temporal_model import SmolTemporalModel
from go2_perception.vln.temporal_review import make_video, render_review, summarize

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode", type=Path)
    group.add_argument("--episodes-root", type=Path)
    ap.add_argument("--decision", help="optional single decision ID")
    ap.add_argument(
        "--mode", choices=["temporal", "current_only", "both"], default="both"
    )
    ap.add_argument("--config", type=Path, default=ROOT / "configs/temporal_vln.yaml")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--video", action="store_true")
    ap.add_argument(
        "--annotations-name",
        default=None,
        help="evaluation only; filename relative to episode, read AFTER inference",
    )
    a = ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        ap.error("Output must be new or empty")
    paths = [a.episode] if a.episode else sorted(a.episodes_root.glob("*/episode.json"))
    if not paths:
        ap.error("No episode packages")
    episodes = [load_episode(p) for p in paths]
    if len({ep.data["episode_id"] for ep in episodes}) != len(episodes):
        ap.error("Duplicate episode identifiers")
    for ep in episodes:
        ids = [ep.data["episode_id"]] + [d["decision_id"] for d in ep.data["decisions"]]
        if not all(re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in ids):
            ap.error(
                "Output directory identifiers must contain only letters, digits, underscore or hyphen"
            )
    cfg = yaml.safe_load(a.config.read_text())
    prompt_path = a.config.parent / cfg["prompt_file"]
    a.output.mkdir(parents=True, exist_ok=True)
    modes = ["temporal", "current_only"] if a.mode == "both" else [a.mode]
    dump(
        a.output / "run_declaration.json",
        {
            "episodes": [
                {
                    "episode_id": e.data["episode_id"],
                    "manifest": str(e.path),
                    "sha256": e.sha256,
                }
                for e in episodes
            ],
            "model_config": cfg,
            "config_sha256": file_sha256(a.config),
            "prompt_sha256": file_sha256(prompt_path),
            "decision_filter": a.decision,
            "modes": modes,
        },
    )
    model = SmolTemporalModel(cfg, a.cache_dir, allow_download=a.allow_download)
    dump(a.output / "model.json", model.info)
    proposals = []
    panels = []
    refs = {}
    for ep in episodes:
        for d in ep.data["decisions"]:
            if a.decision and d["decision_id"] != a.decision:
                continue
            for mode in modes:
                dest = a.output / ep.data["episode_id"] / d["decision_id"] / mode
                dest.mkdir(parents=True, exist_ok=False)
                p, prepared, raw = model.infer(
                    ep,
                    d["decision_id"],
                    prompt_path,
                    mode=mode,
                    allow_synthetic=a.allow_synthetic,
                )
                dump(dest / "proposal.json", p)
                dump(dest / "input_provenance.json", prepared.metadata)
                dump(dest / "messages.json", prepared.messages)
                (dest / "prompt.txt").write_text(prepared.prompt_text)
                (dest / "native_output.txt").write_text(p["native_model_output"])
                np.savez_compressed(dest / "model_inputs_and_generated_ids.npz", **raw)
                for i, (view, preview) in enumerate(
                    zip(prepared.views, prepared.previews)
                ):
                    cv2.imwrite(
                        str(dest / f"view_{i:02d}_rectified.png"),
                        cv2.cvtColor(view.image, cv2.COLOR_RGB2BGR),
                    )
                    cv2.imwrite(
                        str(dest / f"view_{i:02d}_model_input.png"),
                        cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
                    )
                ref = None
                if a.annotations_name:
                    # Deliberately after model.infer; never forwarded to model or prompt construction.
                    anns = load_annotations(ep.root / a.annotations_name, ep)
                    label = next(
                        x
                        for x in anns["decisions"]
                        if x["decision_id"] == d["decision_id"]
                    )
                    ref = {"reviewer": anns["reviewer"], "decision": label}
                    refs[(ep.data["episode_id"], d["decision_id"])] = ref
                    dump(dest / "evaluation_reference.json", ref)
                render_review(p, prepared, dest / "review.png", reference=ref)
                panels.append(dest / "review.png")
                proposals.append(p)
                dump(a.output / "summary.json", summarize(proposals, refs))
                print(
                    json.dumps(
                        {
                            "episode": ep.data["episode_id"],
                            "decision": d["decision_id"],
                            "mode": mode,
                            "proposal": p["proposal"],
                            "task_status": p["task_status"],
                            "parse_status": p["parse_status"],
                            "generation_s": p["timing"]["generation_s"],
                            "peak_rss_gib": p["memory"][
                                "process_lifetime_peak_rss_bytes"
                            ]
                            / 2**30,
                            "history_images_encoded": p["history_images_consumed"],
                        }
                    ),
                    flush=True,
                )
    if not proposals:
        raise ValueError("No matching decisions were run")
    dump(
        a.output / "summary_by_mode.json",
        {m: summarize([p for p in proposals if p["mode"] == m], refs) for m in modes},
    )
    if a.video:
        make_video(panels, a.output / "review.mp4", fps=1)


if __name__ == "__main__":
    main()
