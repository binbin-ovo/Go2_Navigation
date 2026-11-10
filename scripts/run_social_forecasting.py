"""Fixed Phase 4.1 experiments; preserve original runs and never select on test scores."""

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from go2_perception.io.config import load_config
from go2_perception.forecasting.training import save_json
from go2_perception.forecasting.social_training import train_social
from go2_perception.forecasting.social_evaluation import evaluate_social

SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]
SEEDS = [42, 43, 44]


def single_checkpoint(root, phase4, seed, scene):
    if seed == 42:
        return (
            phase4 / "eth_base/best.pt"
            if scene == "eth"
            else phase4 / "loso" / scene / "training/best.pt"
        )
    return root / f"single_seed{seed}" / scene / "training/best.pt"


def fold(base, seed, scene):
    c = deepcopy(base)
    val = SCENES[(SCENES.index(scene) + 1) % 5]
    c["dataset"].update(
        train=[s for s in SCENES if s not in (scene, val)],
        validation=[val],
        test=[scene],
    )
    c["training"]["seed"] = seed
    return c


def run(args):
    base = load_config(args.config)
    root = Path(args.output)
    phase4 = Path(args.phase4_results)
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "single":
        for seed in [43, 44]:
            c = load_config("configs/forecasting_transformer.yaml")
            c["training"]["seed"] = seed
            path = root / f"single_seed{seed}.json"
            save_json(path, c)
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_forecasting_crossvalidation.py",
                    "--config",
                    str(path),
                    "--output",
                    str(root / f"single_seed{seed}"),
                ],
                check=True,
            )
    elif args.stage == "social":
        for seed in SEEDS:
            for scene in SCENES:
                c = fold(base, seed, scene)
                out = root / f"social_seed{seed}" / scene
                print(f"SOCIAL seed={seed} test={scene}", flush=True)
                checkpoint = train_social(c, out / "training")
                evaluate_social(
                    checkpoint,
                    out / "test",
                    single_checkpoint=single_checkpoint(root, phase4, seed, scene),
                )
    elif args.stage == "robustness":
        c = fold(base, 42, "eth")
        c["augmentation"] = dict(noise_std=0.05, mask_probability=0.2)
        checkpoint = train_social(c, root / "robustness/augmented/training")
        for name, path in [
            ("clean_trained", root / "social_seed42/eth/training/best.pt"),
            ("augmented", checkpoint),
        ]:
            for mode in [
                "clean",
                "noise",
                "missing",
                "partial_neighbors",
                "no_neighbors",
            ]:
                evaluate_social(
                    path,
                    root / "robustness" / name / mode,
                    split="validation",
                    corruption=mode,
                    single_checkpoint=single_checkpoint(root, phase4, 42, "eth"),
                )
    elif args.stage == "robustness_controls":
        # Validation-only factorial controls; never promoted into held-out comparisons.
        for name, augmentation in [
            ("noise_only", dict(noise_std=0.05, mask_probability=0.0)),
            ("mask_only", dict(noise_std=0.0, mask_probability=0.2)),
        ]:
            c = fold(base, 42, "eth")
            c["augmentation"] = augmentation
            checkpoint = train_social(c, root / "robustness" / name / "training")
            for mode in [
                "clean",
                "noise",
                "missing",
                "partial_neighbors",
                "no_neighbors",
            ]:
                evaluate_social(
                    checkpoint,
                    root / "robustness" / name / mode,
                    split="validation",
                    corruption=mode,
                    single_checkpoint=single_checkpoint(root, phase4, 42, "eth"),
                )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        required=True,
        choices=["single", "social", "robustness", "robustness_controls"],
    )
    p.add_argument("--config", default="configs/forecasting_social.yaml")
    p.add_argument("--output", default="data/outputs/social_forecasting")
    p.add_argument("--phase4-results", default="data/outputs/forecasting")
    run(p.parse_args())
