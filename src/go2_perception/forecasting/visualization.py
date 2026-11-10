"""Headless metric XY research plots; same +x right/+y up convention as Phase 3 BEV."""

from pathlib import Path
import numpy as np
from .metrics import errors, motion_categories


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_training(history, path):
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    epoch = [h["epoch"] for h in history]
    axes[0].plot(epoch, [h["train_smooth_l1"] for h in history])
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Smooth L1 (beta = 1 m)")
    axes[1].plot(epoch, [h["validation"]["ade_m"] for h in history], label="ADE")
    axes[1].plot(epoch, [h["validation"]["fde_m"] for h in history], label="FDE")
    axes[1].set(title="Validation only", xlabel="Epoch", ylabel="Error (m)")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_examples(observed, target, baseline, learned, scenes, keys, dt, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    plt = pyplot()
    a, f = errors(learned, target)
    b, g = errors(baseline, target)
    labels = motion_categories(observed, target, dt)
    choices = {}
    for name in ["straight", "turning", "stopping", "starting"]:
        indices = np.flatnonzero(labels == name)
        if len(indices):
            choices[name] = int(
                indices[np.argsort(b[indices], kind="stable")[len(indices) // 2]]
            )
    wins = np.flatnonzero(a < b)
    losses = np.flatnonzero(a > b)
    if len(wins):
        choices["largest_transformer_advantage"] = int(wins[np.argmin((a - b)[wins])])
    if len(losses):
        choices["largest_cv_advantage"] = int(losses[np.argmax((a - b)[losses])])
    manifest = {}
    for name, i in choices.items():
        fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
        ax.plot(*observed[i].T, "o-", color="black", markersize=3, label="Observed")
        for values, label, color, style in [
            (target[i], "Ground truth", "forestgreen", "-"),
            (baseline[i], f"CV ADE {b[i]:.3f}, FDE {g[i]:.3f} m", "royalblue", "--"),
            (
                learned[i],
                f"Transformer ADE {a[i]:.3f}, FDE {f[i]:.3f} m",
                "darkorange",
                "--",
            ),
        ]:
            curve = np.vstack((observed[i, -1], values))
            ax.plot(*curve.T, style, color=color, label=label)
        points = np.vstack((observed[i], target[i], baseline[i], learned[i]))
        low, high = points.min(0), points.max(0)
        center = (low + high) / 2
        span = max(float(np.max(high - low)), 1.0) * 1.3
        ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
        ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set(
            xlabel="World x (m)",
            ylabel="World y (m)",
            title=f"{scenes[i]} | {labels[i]} | {name}\nobs {len(observed[i])} ({(len(observed[i]) - 1) * dt:g} s span), future {len(target[i]) * dt:g} s",
        )
        ax.legend(fontsize=8)
        fig.savefig(out / f"{name}.png", dpi=140)
        plt.close(fig)
        manifest[name] = dict(
            index=i,
            key=str(keys[i]),
            scene=str(scenes[i]),
            category=str(labels[i]),
            transformer_ade_m=float(a[i]),
            cv_ade_m=float(b[i]),
            selection="largest error difference"
            if "advantage" in name
            else "median CV-ADE example within GT category",
        )
    return manifest
