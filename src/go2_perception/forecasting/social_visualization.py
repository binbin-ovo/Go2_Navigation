"""Public-annotation examples with observed neighbors and both successful/failed forecasts."""

from pathlib import Path
import numpy as np
from .visualization import pyplot
from .metrics import errors, motion_categories
from .training import save_json


def plot_social_examples(predictions, output):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use fresh plot directory")
    with np.load(predictions, allow_pickle=False) as archive:
        a = {k: archive[k] for k in archive.files}
    models = [
        k
        for k in ["constant_velocity", "single", "social", "social_disabled"]
        if k in a
    ]
    scores = {k: errors(a[k], a["target"]) for k in models}
    labels = motion_categories(a["observed"], a["target"], float(a["dt"]))
    choices = {}
    for category in ["straight", "turning", "stopping", "starting"]:
        indices = np.flatnonzero(labels == category)
        if len(indices):
            choices[category] = int(
                indices[
                    np.argsort(scores["constant_velocity"][0][indices], kind="stable")[
                        len(indices) // 2
                    ]
                ]
            )
    difference = scores["social"][0] - scores["constant_velocity"][0]
    for name, mask in [
        ("social_win", difference < 0),
        ("social_failure", difference > 0),
    ]:
        indices = np.flatnonzero(mask)
        if len(indices):
            choices[name] = int(indices[np.argmax(np.abs(difference[indices]))])
    out.mkdir(parents=True, exist_ok=True)
    plt = pyplot()
    manifest = {}
    colors = {
        "constant_velocity": "royalblue",
        "single": "darkorange",
        "social": "purple",
        "social_disabled": "crimson",
    }
    for name, i in choices.items():
        fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
        for n, (xy, mask) in enumerate(zip(a["neighbors"][i], a["neighbor_mask"][i])):
            if mask.any():
                line = xy.copy()
                line[~mask] = np.nan
                ax.plot(
                    *line.T,
                    "o:",
                    color="gray",
                    alpha=0.65,
                    markersize=2,
                    label="Observed neighbors" if n == 0 else None,
                )
        ax.plot(*a["observed"][i].T, "ko-", markersize=3, label="Target observations")
        ax.plot(
            *np.vstack((a["observed"][i, -1], a["target"][i])).T,
            color="forestgreen",
            label="Target ground truth",
        )
        for model in models:
            ade, fde = scores[model]
            ax.plot(
                *np.vstack((a["observed"][i, -1], a[model][i])).T,
                "--",
                color=colors[model],
                label=f"{model}: ADE {ade[i]:.2f}, FDE {fde[i]:.2f} m",
            )
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.set(
            xlabel="World x (m)",
            ylabel="World y (m)",
            title=f"{a['scenes'][i]} | {name} | {labels[i]}\nObserved-only neighbors; future {a['target'].shape[1] * float(a['dt']):g} s",
        )
        ax.legend(fontsize=8)
        fig.savefig(out / f"{name}.png", dpi=140)
        plt.close(fig)
        manifest[name] = dict(
            key=str(a["keys"][i]),
            category=str(labels[i]),
            selection="largest absolute social-minus-CV ADE difference within win/loss"
            if name.startswith("social_")
            else "median CV ADE in category",
            metrics={
                model: dict(ade_m=float(s[0][i]), fde_m=float(s[1][i]))
                for model, s in scores.items()
            },
        )
    save_json(out / "selection.json", manifest)
    return manifest
