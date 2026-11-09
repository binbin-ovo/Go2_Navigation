"""Paired held-out evaluation. No optimization, checkpoint selection or test-time sampling."""

import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from .data import dataset
from .baseline import constant_velocity, phase3_constant_velocity
from .training import (
    load_checkpoint,
    select_device,
    seed_everything,
    predict,
    arrays,
    save_json,
    synchronize,
)
from .metrics import paired_report, summarize


def latency(model, x, device, batch_size, repeats=30):
    # Device-resident forward includes normalization/model/reconstruction; excludes host transfer.
    result = {}
    for n in sorted({1, min(batch_size, len(x))}):
        batch = torch.as_tensor(x[:n], dtype=torch.float32, device=device)
        times = []
        with torch.inference_mode():
            for _ in range(5):
                model(batch)
            synchronize(device)
            for _ in range(repeats):
                start = time.perf_counter()
                model(batch)
                synchronize(device)
                times.append((time.perf_counter() - start) * 1000)
        result[str(n)] = dict(
            batch_size=n,
            mean_ms=float(np.mean(times)),
            p95_ms=float(np.percentile(times, 95)),
            trajectories_per_s=n * 1000 / float(np.mean(times)),
        )
    return result


def evaluate(
    checkpoint,
    output,
    *,
    split="test",
    device="cpu",
    data_root=None,
    corruption="clean",
    seed=4242,
):
    if split not in ("test", "validation"):
        raise ValueError("Evaluation split must be test or validation")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use fresh evaluation output directory")
    dev = select_device(device)
    model, payload = load_checkpoint(checkpoint, dev)
    cfg = payload["config"]
    seed_everything(cfg["training"]["seed"], dev, cfg["training"]["threads"])
    if data_root:
        cfg["dataset"]["root"] = data_root
    windows, meta = dataset(cfg["dataset"], split)
    x, y = arrays(windows)
    clean_x = x.copy()
    expected = payload["metadata"].get(split)
    if expected and (
        meta["recording_sha256"] != expected["recording_sha256"]
        or meta["window_keys_sha256"] != expected["window_keys_sha256"]
    ):
        raise ValueError("Evaluation data does not match checkpoint provenance")
    corruption_metadata = dict(mode=corruption, seed=seed)
    if corruption == "noise":
        x = x + np.random.default_rng(seed).normal(0, 0.05, x.shape).astype("float32")
        corruption_metadata["sigma_m"] = 0.05
    elif corruption == "missing":
        if x.shape[1] < 3:
            raise ValueError(
                "Missing-point experiment needs at least three observations"
            )
        # Explicit artificial removal of an interior observation; causal-with-respect-to-forecast
        # interpolation uses only adjacent past observations. No future target enters input.
        k = x.shape[1] // 2
        x[:, k] = (x[:, k - 1] + x[:, k + 1]) / 2
        corruption_metadata["interpolated_index"] = k
    elif corruption != "clean":
        raise ValueError("Unknown corruption")
    dt = cfg["dataset"]["dt"]
    h = cfg["dataset"]["pred_steps"]
    bcfg = cfg["baseline"]
    start = time.perf_counter()
    baseline = constant_velocity(x, h, dt, **bcfg)
    baseline_s = time.perf_counter() - start
    learned = predict(model, x, dev, cfg["training"]["batch_size"])
    result = paired_report(clean_x, y, learned, baseline, dt)
    old = phase3_constant_velocity(x, h, dt, **bcfg)
    result["phase3_adapter"] = dict(
        **summarize(old, y),
        max_abs_difference_from_numpy_m=float(np.max(np.abs(old - baseline))),
    )
    result.update(
        dataset=meta,
        config=cfg,
        model_version=payload["model_version"],
        best_epoch=payload["best_epoch"],
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        corruption=corruption_metadata,
        parameter_count=sum(p.numel() for p in model.parameters()),
        device=str(dev),
        latency=latency(model, x, dev, cfg["training"]["batch_size"]),
        cv_total_batch_ms=baseline_s * 1000,
        cv_trajectories_per_s=len(x) / baseline_s,
        timing_scope="warm device-resident forward; excludes transfer, perception and tracking",
    )
    result["per_scene"] = {
        name: paired_report(clean_x[mask], y[mask], learned[mask], baseline[mask], dt)
        for name in sorted({w.scene for w in windows})
        for mask in [np.array([w.scene == name for w in windows])]
    }
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "metrics.json", result)
    np.savez_compressed(
        out / "predictions.npz",
        observed=x,
        target=y,
        constant_velocity=baseline,
        transformer=learned,
        scenes=np.asarray([w.scene for w in windows]),
        keys=np.asarray([w.key for w in windows]),
        dt=dt,
    )
    return result
