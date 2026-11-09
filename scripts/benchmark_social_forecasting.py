"""Dedicated quiet-process warm timing; run after training jobs have finished."""

import argparse
from pathlib import Path
import time
import numpy as np
import torch
from go2_perception.forecasting.social_training import load_social_checkpoint, inputs
from go2_perception.forecasting.social_data import social_dataset
from go2_perception.forecasting.social_evaluation import social_latency
from go2_perception.forecasting.training import (
    load_checkpoint,
    seed_everything,
    save_json,
)
from go2_perception.forecasting.evaluation import latency
from go2_perception.forecasting.baseline import constant_velocity

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--social-checkpoint", required=True)
    p.add_argument("--single-checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    social, payload = load_social_checkpoint(a.social_checkpoint)
    single, _ = load_checkpoint(a.single_checkpoint)
    c = payload["config"]
    dev = torch.device("cpu")
    seed_everything(42, dev, c["training"]["threads"])
    _, b, _ = social_dataset(c, "test")
    v = inputs(b)
    off = list(v)
    off[2] = np.zeros_like(off[2])
    r = dict(
        social=social_latency(social, v, dev, repeats=100),
        social_disabled=social_latency(social, off, dev, repeats=100),
        single=latency(single, b.observed, dev, 128, repeats=100),
        constant_velocity={},
    )
    for n in sorted({1, min(128, len(b.observed))}):
        x = b.observed[:n]
        times = []
        for _ in range(5):
            constant_velocity(
                x, c["dataset"]["pred_steps"], c["dataset"]["dt"], **c["baseline"]
            )
        for _ in range(100):
            t = time.perf_counter()
            constant_velocity(
                x, c["dataset"]["pred_steps"], c["dataset"]["dt"], **c["baseline"]
            )
            times.append((time.perf_counter() - t) * 1000)
        r["constant_velocity"][str(n)] = dict(
            mean_ms=float(np.mean(times)),
            p95_ms=float(np.percentile(times, 95)),
            trajectories_per_s=n * 1000 / float(np.mean(times)),
        )
    r["scope"] = (
        "CPU four threads, five warmups/100 repeats, same ETH window prefixes; excludes transfer/dataset/tracking. Social path includes masks/relative context; CV NumPy."
    )
    save_json(out, r)
    print(r)
