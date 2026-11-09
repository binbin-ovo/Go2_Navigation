"""Paired social diagnostics; corruption selection is validation-only in the experiment runner."""

from copy import deepcopy
from pathlib import Path
import hashlib
import time
import numpy as np
import torch
from .social_data import social_dataset
from .social_training import inputs, predict_social, load_social_checkpoint
from .training import (
    select_device,
    seed_everything,
    save_json,
    synchronize,
    load_checkpoint,
    predict,
)
from .baseline import constant_velocity
from .metrics import paired_report


def perturb(batch, mode, seed=4242):
    b = deepcopy(batch)
    rng = np.random.default_rng(seed)
    if mode == "noise":
        b.observed += rng.normal(0, 0.05, b.observed.shape).astype("float32")
        b.neighbors += np.where(
            b.neighbor_mask[..., None], rng.normal(0, 0.05, b.neighbors.shape), 0
        ).astype("float32")
    elif mode == "missing":
        b.target_mask[:, b.observed.shape[1] // 2] = False
    elif mode == "partial_neighbors":
        b.neighbor_mask[:, :, 1:-1] &= (
            rng.random(b.neighbor_mask[:, :, 1:-1].shape) >= 0.5
        )
    elif mode == "no_neighbors":
        b.neighbor_mask[:] = False
    elif mode != "clean":
        raise ValueError("Unknown corruption mode")
    # Explicit repair for models without mask support; only observed-time neighbors used.
    repaired = b.observed.copy()
    for i in range(len(repaired)):
        for coordinate in range(2):
            indices = np.arange(repaired.shape[1])
            mask = b.target_mask[i]
            repaired[i, :, coordinate] = np.interp(
                indices, indices[mask], b.observed[i, mask, coordinate]
            )
    return b, repaired


def social_latency(model, values, device, batch_size=128, repeats=30):
    result = {}
    model.eval()
    for n in sorted({1, min(batch_size, len(values[0]))}):
        batch = [torch.as_tensor(v[:n], device=device) for v in values]
        times = []
        with torch.inference_mode():
            for _ in range(5):
                model(*batch)
            synchronize(device)
            for _ in range(repeats):
                start = time.perf_counter()
                model(*batch)
                synchronize(device)
                times.append((time.perf_counter() - start) * 1000)
        result[str(n)] = dict(
            batch_size=n,
            mean_ms=float(np.mean(times)),
            p95_ms=float(np.percentile(times, 95)),
            trajectories_per_s=n * 1000 / float(np.mean(times)),
        )
    return result


def evaluate_social(
    checkpoint, output, *, split="test", corruption="clean", single_checkpoint=None
):
    if split not in ("test", "validation"):
        raise ValueError("Invalid split")
    if split == "test" and corruption != "clean":
        raise ValueError("Perturbation studies are validation-only")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use fresh evaluation output")
    model, payload = load_social_checkpoint(checkpoint)
    c = payload["config"]
    dev = select_device(c["training"]["device"])
    model.to(dev)
    seed_everything(c["training"]["seed"], dev, c["training"]["threads"])
    start = time.perf_counter()
    w, clean, meta = social_dataset(c, split)
    preparation = time.perf_counter() - start
    expected = payload["metadata"].get(split)
    if expected and (
        expected["recording_sha256"] != meta["recording_sha256"]
        or expected["window_keys_sha256"] != meta["window_keys_sha256"]
    ):
        raise ValueError("Checkpoint data provenance mismatch")
    b, repaired = perturb(clean, corruption)
    values = inputs(b)
    dt = c["dataset"]["dt"]
    baseline = constant_velocity(
        repaired, c["dataset"]["pred_steps"], dt, **c["baseline"]
    )
    forecasts = {
        "social": predict_social(model, values, dev, c["training"]["batch_size"])
    }
    no_social = list(values)
    no_social[2] = np.zeros_like(no_social[2])
    forecasts["social_disabled"] = predict_social(
        model, no_social, dev, c["training"]["batch_size"]
    )
    single_sha = None
    if single_checkpoint:
        single, p = load_checkpoint(single_checkpoint, dev)
        # Compare the complete experimental settings, not only horizon shapes.
        for key in ["dataset", "model", "training", "baseline"]:
            if p["config"][key] != c[key]:
                raise ValueError("Single/social experiment configurations differ")
        forecasts["single"] = predict(
            single, repaired, dev, c["training"]["batch_size"]
        )
        single_sha = hashlib.sha256(Path(single_checkpoint).read_bytes()).hexdigest()
    reports = {
        name: paired_report(clean.observed, clean.future, pred, baseline, dt)
        for name, pred in forecasts.items()
    }
    crowd = {}
    for name, mask in [
        ("zero", clean.counts == 0),
        ("one", clean.counts == 1),
        ("two_plus", clean.counts >= 2),
    ]:
        crowd[name] = dict(
            count=int(mask.sum()),
            models={
                key: paired_report(
                    clean.observed[mask],
                    clean.future[mask],
                    pred[mask],
                    baseline[mask],
                    dt,
                )
                for key, pred in forecasts.items()
            }
            if mask.any()
            else {},
        )
    result = dict(
        models=reports,
        crowding=crowd,
        dataset=meta,
        config=c,
        corruption=corruption,
        perturbation_seed=4242,
        parameter_count=sum(p.numel() for p in model.parameters()),
        model_version=model.version,
        best_epoch=payload["best_epoch"],
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        single_checkpoint_sha256=single_sha,
        latency=social_latency(model, values, dev),
        data_preparation_s=preparation,
        timing_scope="warm device-resident model forward including masks and relative coordinates; no transfer/dataset/tracking",
    )
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "metrics.json", result)
    np.savez_compressed(
        out / "predictions.npz",
        observed=clean.observed,
        target=clean.future,
        constant_velocity=baseline,
        **forecasts,
        neighbors=b.neighbors,
        neighbor_mask=b.neighbor_mask,
        target_mask=b.target_mask,
        keys=np.asarray([v.key for v in w]),
        scenes=np.asarray([v.scene for v in w]),
        dt=dt,
    )
    return result
