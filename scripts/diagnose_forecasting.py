"""Inspect frozen original checkpoints without changing the held-out protocol."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from go2_perception.forecasting.data import dataset
from go2_perception.forecasting.training import (
    arrays,
    predict,
    load_checkpoint,
    save_json,
    seed_everything,
)
from go2_perception.forecasting.metrics import summarize

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase4-results", default="data/outputs/forecasting")
    p.add_argument("--results", default="data/outputs/social_forecasting")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    rows = {}
    for seed in [42, 43, 44]:
        root = (
            Path(a.phase4_results) / "loso"
            if seed == 42
            else Path(a.results) / f"single_seed{seed}"
        )
        rows[str(seed)] = {}
        for scene in ["eth", "hotel", "univ", "zara1", "zara2"]:
            train = (
                Path(a.phase4_results) / "eth_base"
                if seed == 42 and scene == "eth"
                else root / scene / "training"
            )
            model, payload = load_checkpoint(train / "best.pt")
            c = payload["config"]
            seed_everything(seed, torch.device("cpu"), c["training"]["threads"])
            row = {
                "history": json.loads((train / "history.json").read_text()),
                "best_epoch": payload["best_epoch"],
                "test": json.loads((root / scene / "test/metrics.json").read_text()),
            }
            for split in ["train", "validation"]:
                windows, meta = dataset(c["dataset"], split)
                x, y = arrays(windows)
                forecast = predict(model, x, "cpu", c["training"]["batch_size"])
                row[split] = dict(
                    **summarize(forecast, y),
                    smooth_l1=float(
                        torch.nn.functional.smooth_l1_loss(
                            torch.from_numpy(forecast), torch.from_numpy(y)
                        )
                    ),
                    dataset=meta,
                )
            rows[str(seed)][scene] = row
    save_json(out, rows)
    for scene in rows["42"]:
        scores = np.array(
            [
                [
                    rows[str(seed)][scene]["test"]["transformer"][metric]
                    for metric in ["ade_m", "fde_m"]
                ]
                for seed in [42, 43, 44]
            ]
        )
        print(scene, "mean", scores.mean(0), "sample_std", scores.std(0, ddof=1))
