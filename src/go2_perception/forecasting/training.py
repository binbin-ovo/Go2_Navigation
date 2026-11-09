"""CPU-portable checkpoints; validation-only selection; test data is never loaded here."""

import hashlib
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from .data import dataset
from .model import TemporalTransformer
from . import MODEL_VERSION
from .metrics import summarize


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def select_device(name):
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if name not in ("cpu", "cuda", "mps"):
        raise ValueError("Device must be cpu, cuda, mps or auto")
    if (
        name == "cuda"
        and not torch.cuda.is_available()
        or name == "mps"
        and not torch.backends.mps.is_available()
    ):
        raise RuntimeError(f"Requested device unavailable: {name}")
    return torch.device(name)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if device.type == "mps":
        torch.mps.synchronize()


def seed_everything(seed, device, threads=4):
    if threads < 1:
        raise ValueError("Positive CPU thread count required")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Strict determinism is tested on CPU. MPS kernels may lack deterministic implementations.
    torch.use_deterministic_algorithms(device.type != "mps")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def arrays(windows):
    return np.stack([w.observed for w in windows]).astype("float32"), np.stack(
        [w.future for w in windows]
    ).astype("float32")


def make_model(config):
    return TemporalTransformer(
        config["dataset"]["obs_steps"],
        config["dataset"]["pred_steps"],
        dt=config["dataset"]["dt"],
        **config["model"],
    )


def predict(model, x, device, batch_size=128):
    if batch_size < 1:
        raise ValueError("Invalid batch size")
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            y = model(
                torch.as_tensor(
                    x[start : start + batch_size], dtype=torch.float32, device=device
                )
            )
            if not torch.isfinite(y).all():
                raise FloatingPointError("Nonfinite prediction")
            output.append(y.cpu().numpy())
    if not output:
        raise ValueError("Empty inference input")
    return np.concatenate(output)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("model_version") != MODEL_VERSION:
        raise ValueError("Unsupported checkpoint model version")
    model = make_model(payload["config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(select_device(str(device)))
    model.eval()
    return model, payload


def train(config, output, *, make_plots=True):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh training output directory")
    t = config["training"]
    d = config["dataset"]
    if (
        min(t["batch_size"], t["epochs"], t["patience"], t["accumulate_steps"]) < 1
        or not all(
            np.isfinite(t[k]) and t[k] > 0 for k in ("learning_rate", "clip_grad_norm")
        )
        or not np.isfinite(t["weight_decay"])
        or t["weight_decay"] < 0
    ):
        raise ValueError("Invalid training configuration")
    device = select_device(t["device"])
    seed_everything(t["seed"], device, t["threads"])
    train_windows, train_meta = dataset(d, "train")
    val_windows, val_meta = dataset(d, "validation")
    x, y = arrays(train_windows)
    vx, vy = arrays(val_windows)
    generator = torch.Generator().manual_seed(t["seed"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=t["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"]
    )
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.json", config)
    metadata = dict(
        train=train_meta,
        validation=val_meta,
        torch=str(torch.__version__),
        numpy=np.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        device=str(device),
        deterministic_algorithms=device.type != "mps",
        parameter_count=sum(p.numel() for p in model.parameters()),
        loss="SmoothL1 on reconstructed XY; equivalent to offsets about observed origin; beta=1 metre",
        scheduler="none; fixed AdamW learning rate with early stopping",
    )
    save_json(out / "metadata.json", metadata)
    history = []
    best = float("inf")
    stale = 0
    start = time.perf_counter()
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        total = 0.0
        count = 0
        optimizer.zero_grad(set_to_none=True)
        # Accumulate loss sums, divide by actual example count at each optimizer step.
        accumulated = 0
        for i, (bx, by) in enumerate(loader):
            bx, by = bx.to(device), by.to(device)
            loss = torch.nn.functional.smooth_l1_loss(model(bx), by, reduction="mean")
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            (loss * len(bx)).backward()
            accumulated += len(bx)
            total += float(loss.detach()) * len(bx)
            count += len(bx)
            if (i + 1) % t["accumulate_steps"] == 0 or i + 1 == len(loader):
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(accumulated)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), t["clip_grad_norm"], error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
        val = summarize(predict(model, vx, device, t["batch_size"]), vy)
        synchronize(device)
        row = dict(
            epoch=epoch,
            train_smooth_l1=total / count,
            validation=val,
            elapsed_s=time.perf_counter() - start,
        )
        history.append(row)
        save_json(out / "history.json", history)
        print(
            f"epoch {epoch:02d} loss={total / count:.5f} validation ADE={val['ade_m']:.4f} FDE={val['fde_m']:.4f}",
            flush=True,
        )
        if val["ade_m"] < best:
            best = val["ade_m"]
            stale = 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                dict(
                    model_version=MODEL_VERSION,
                    state_dict=state,
                    config=config,
                    metadata=metadata,
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
            metadata=metadata,
        ),
    )
    if make_plots:
        from .visualization import plot_training

        plot_training(history, out / "training_curve.png")
    return checkpoint
