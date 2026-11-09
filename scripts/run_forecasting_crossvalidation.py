"""Fixed five scene-identifier-held-out runs; next scene is validation, remaining three train."""

import argparse
from copy import deepcopy
from pathlib import Path
import yaml
from go2_perception.io.config import load_config
from go2_perception.forecasting.training import train, save_json, load_checkpoint
from go2_perception.forecasting.evaluation import evaluate

SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/forecasting_transformer.yaml")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--eth-checkpoint",
        help="Reuse a trained ETH-fold checkpoint with identical configuration",
    )
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use fresh cross-validation output directory")
    out.mkdir(parents=True, exist_ok=True)
    base = load_config(a.config)
    results = {}
    for i, scene in enumerate(SCENES):
        c = deepcopy(base)
        validation = SCENES[(i + 1) % len(SCENES)]
        c["dataset"].update(
            train=[s for s in SCENES if s not in (scene, validation)],
            validation=[validation],
            test=[scene],
        )
        (out / f"{scene}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
        if scene == "eth" and a.eth_checkpoint:
            _, payload = load_checkpoint(a.eth_checkpoint)
            if payload["config"] != c:
                raise ValueError(
                    "Reused checkpoint config differs from predeclared fold"
                )
            checkpoint = Path(a.eth_checkpoint)
        else:
            checkpoint = train(c, out / scene / "training")
        results[scene] = evaluate(
            checkpoint,
            out / scene / "test",
            split="test",
            device=c["training"]["device"],
        )
        save_json(out / "scenes.json", results)
    aggregate = {}
    for method in ["constant_velocity", "transformer"]:
        n = sum(r[method]["count"] for r in results.values())
        aggregate[method] = dict(
            count=n,
            micro_ade_m=sum(
                r[method]["count"] * r[method]["ade_m"] for r in results.values()
            )
            / n,
            micro_fde_m=sum(
                r[method]["count"] * r[method]["fde_m"] for r in results.values()
            )
            / n,
            macro_ade_m=sum(r[method]["ade_m"] for r in results.values())
            / len(results),
            macro_fde_m=sum(r[method]["fde_m"] for r in results.values())
            / len(results),
        )
    save_json(out / "aggregate.json", aggregate)
