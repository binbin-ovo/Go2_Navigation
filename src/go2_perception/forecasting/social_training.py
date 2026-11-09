"""Social training is separate so Phase 4 checkpoint behavior remains unchanged."""

from pathlib import Path
import hashlib
import platform
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from .social_data import social_dataset
from .social_model import SocialTransformer, SOCIAL_VERSION
from .training import select_device, seed_everything, save_json, synchronize
from .metrics import summarize


def make_social_model(config):
    d = config["dataset"]
    return SocialTransformer(
        d["obs_steps"], d["pred_steps"], dt=d["dt"], **config["model"]
    )


def inputs(batch):
    return batch.observed, batch.neighbors, batch.neighbor_mask, batch.target_mask


def predict_social(model, values, device, batch_size=128):
    if batch_size < 1 or not len(values[0]):
        raise ValueError("Nonempty batches and positive batch size required")
    model.eval()
    result = []
    with torch.inference_mode():
        for i in range(0, len(values[0]), batch_size):
            tensors = [
                torch.as_tensor(v[i : i + batch_size], device=device) for v in values
            ]
            y = model(*tensors)
            if not torch.isfinite(y).all():
                raise FloatingPointError("Nonfinite forecast")
            result.append(y.cpu().numpy())
    return np.concatenate(result)


def load_social_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("model_version") != SOCIAL_VERSION:
        raise ValueError("Unsupported social checkpoint")
    model = make_social_model(payload["config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(select_device(str(device))).eval()
    return model, payload


def augment(values, noise_std, mask_probability):
    x, n, nm, tm = [v.clone() for v in values]
    if noise_std:
        x = x + torch.randn_like(x) * noise_std
        n = torch.where(nm[..., None], n + torch.randn_like(n) * noise_std, 0.0)
    if mask_probability:
        tm[:, 1:-1] &= (
            torch.rand_like(tm[:, 1:-1], dtype=torch.float32) >= mask_probability
        )
        nm[:, :, 1:-1] &= (
            torch.rand_like(nm[:, :, 1:-1], dtype=torch.float32) >= mask_probability
        )
    return x, n, nm, tm


def train_social(config, output, *, make_plots=True):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use fresh training directory")
    t = config["training"]
    aug = config.get("augmentation", dict(noise_std=0.0, mask_probability=0.0))
    if (
        min(t["batch_size"], t["epochs"], t["patience"], t["accumulate_steps"]) < 1
        or not all(
            np.isfinite(t[k]) and t[k] > 0 for k in ("learning_rate", "clip_grad_norm")
        )
        or not np.isfinite(t["weight_decay"])
        or t["weight_decay"] < 0
    ):
        raise ValueError("Invalid training configuration")
    if (
        not np.isfinite(aug["noise_std"])
        or aug["noise_std"] < 0
        or not 0 <= aug["mask_probability"] < 1
    ):
        raise ValueError("Invalid augmentation")
    dev = select_device(t["device"])
    seed_everything(t["seed"], dev, t["threads"])
    _, a, ma = social_dataset(config, "train")
    _, b, mb = social_dataset(config, "validation")
    loader = DataLoader(
        TensorDataset(*(torch.from_numpy(v) for v in (*inputs(a), a.future))),
        batch_size=t["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(t["seed"]),
    )
    model = make_social_model(config).to(dev)
    opt = torch.optim.AdamW(
        model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"]
    )
    meta = dict(
        train=ma,
        validation=mb,
        torch=str(torch.__version__),
        numpy=np.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        device=str(dev),
        parameter_count=sum(p.numel() for p in model.parameters()),
    )
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.json", config)
    save_json(out / "metadata.json", meta)
    history = []
    best = float("inf")
    stale = 0
    start = time.perf_counter()
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        total = 0.0
        count = 0
        accumulated = 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            vals = [v.to(dev) for v in batch]
            x = augment(vals[:-1], **aug)
            loss = torch.nn.functional.smooth_l1_loss(model(*x), vals[-1])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            size = len(vals[0])
            (loss * size).backward()
            total += float(loss.detach()) * size
            count += size
            accumulated += size
            if (i + 1) % t["accumulate_steps"] == 0 or i + 1 == len(loader):
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(accumulated)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), t["clip_grad_norm"], error_if_nonfinite=True
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
                accumulated = 0
        prediction = predict_social(model, inputs(b), dev, t["batch_size"])
        val = summarize(prediction, b.future)
        vloss = float(
            torch.nn.functional.smooth_l1_loss(
                torch.from_numpy(prediction), torch.from_numpy(b.future)
            )
        )
        synchronize(dev)
        history.append(
            dict(
                epoch=epoch,
                train_smooth_l1=total / count,
                validation=val,
                validation_smooth_l1=vloss,
                elapsed_s=time.perf_counter() - start,
            )
        )
        save_json(out / "history.json", history)
        print(
            f"epoch {epoch:02d} loss={total / count:.5f} validation ADE={val['ade_m']:.4f} FDE={val['fde_m']:.4f}",
            flush=True,
        )
        if val["ade_m"] < best:
            best = val["ade_m"]
            stale = 0
            torch.save(
                dict(
                    model_version=SOCIAL_VERSION,
                    state_dict={
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    config=config,
                    metadata=meta,
                    best_epoch=epoch,
                    validation=val,
                ),
                out / "best.pt",
            )
        else:
            stale += 1
        if stale >= t["patience"]:
            break
    checkpoint = out / "best.pt"
    save_json(
        out / "training_summary.json",
        dict(
            epochs=len(history),
            best_validation_ade_m=best,
            wall_s=time.perf_counter() - start,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            test_evaluated=False,
            metadata=meta,
        ),
    )
    if make_plots:
        from .visualization import plot_training

        plot_training(history, out / "training_curve.png")
    return checkpoint
