"""Predeclared small matrix, validation-only by design. Test evaluation is a separate command."""

import argparse
from copy import deepcopy
from pathlib import Path
import yaml
from go2_perception.io.config import load_config
from go2_perception.forecasting.training import train, save_json
from go2_perception.forecasting.evaluation import evaluate


def variants(config):
    rows = {}
    for name, changes in [
        ("history4", dict(obs_steps=4)),
        ("history12", dict(obs_steps=12)),
        ("capacity128", dict(d_model=128, feedforward=256)),
        ("local_positions", dict(representation="local_position")),
    ]:
        c = deepcopy(config)
        for k, v in changes.items():
            c["dataset" if k == "obs_steps" else "model"][k] = v
        rows[name] = c
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/forecasting_transformer.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument(
        "--variants",
        nargs="+",
        choices=["history4", "history12", "capacity128", "local_positions"],
        default=["history4", "history12", "capacity128", "local_positions"],
    )
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use fresh ablation output directory")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config)
    results = {}
    results["base"] = evaluate(
        a.base_checkpoint,
        out / "base",
        split="validation",
        device=cfg["training"]["device"],
    )
    for name in a.variants:
        c = variants(cfg)[name]
        path = out / name
        (out / f"{name}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
        checkpoint = train(c, path / "training")
        results[name] = evaluate(
            checkpoint,
            path / "validation",
            split="validation",
            device=c["training"]["device"],
        )
        if (
            results[name]["dataset"]["window_keys_sha256"]
            != results["base"]["dataset"]["window_keys_sha256"]
        ):
            raise RuntimeError("Ablations must share validation forecast anchors")
    for mode in ["noise", "missing"]:
        results[mode] = evaluate(
            a.base_checkpoint,
            out / mode,
            split="validation",
            device=cfg["training"]["device"],
            corruption=mode,
        )
    save_json(out / "matrix.json", results)
