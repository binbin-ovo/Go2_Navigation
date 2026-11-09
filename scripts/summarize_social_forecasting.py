"""Aggregate seed variability without treating overlapping windows as independent samples."""

import argparse
import json
from pathlib import Path
import numpy as np
from go2_perception.forecasting.training import save_json

SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]
SEEDS = [42, 43, 44]
MODELS = ["constant_velocity", "single", "social", "social_disabled"]


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(
        mean=float(a.mean()), sample_std=float(a.std(ddof=1)), seed_values=a.tolist()
    )


def summary(root, phase4):
    runs = {
        str(seed): {
            scene: json.loads(
                (root / f"social_seed{seed}" / scene / "test/metrics.json").read_text()
            )
            for scene in SCENES
        }
        for seed in SEEDS
    }
    # Explicit audit of the paired sample population against the independently run Phase 4 evaluator.
    for seed in SEEDS:
        for scene in SCENES:
            oldroot = phase4 / "loso" if seed == 42 else root / f"single_seed{seed}"
            old = json.loads((oldroot / scene / "test/metrics.json").read_text())
            new = runs[str(seed)][scene]
            if (
                old["dataset"]["window_keys_sha256"]
                != new["dataset"]["window_keys_sha256"]
            ):
                raise ValueError("Unpaired target windows")
            with (
                np.load(oldroot / scene / "test/predictions.npz") as old_arrays,
                np.load(
                    root / f"social_seed{seed}" / scene / "test/predictions.npz"
                ) as new_arrays,
            ):
                for key in ["keys", "target", "observed"]:
                    np.testing.assert_array_equal(old_arrays[key], new_arrays[key])
                for old_key, new_key in [
                    ("transformer", "single"),
                    ("constant_velocity", "constant_velocity"),
                ]:
                    np.testing.assert_allclose(
                        old_arrays[old_key], new_arrays[new_key], rtol=0, atol=1e-7
                    )
            for key in ["ade_m", "fde_m"]:
                if not np.isclose(
                    old["transformer"][key],
                    new["models"]["single"]["transformer"][key],
                    rtol=0,
                    atol=1e-7,
                ):
                    raise ValueError("Original single-agent predictions changed")

    def metric(run, model):
        return (
            run["models"]["social"]["constant_velocity"]
            if model == "constant_velocity"
            else run["models"][model]["transformer"]
        )

    per_scene = {}
    aggregate = {}
    for scene in SCENES:
        per_scene[scene] = {
            model: {
                key: stats(
                    [metric(runs[str(seed)][scene], model)[key] for seed in SEEDS]
                )
                for key in ["ade_m", "fde_m"]
            }
            for model in MODELS
        }
        per_scene[scene]["count"] = metric(runs["42"][scene], "social")["count"]
    for model in MODELS:
        aggregate[model] = {}
        for key in ["ade_m", "fde_m"]:
            for scope in ["macro", "micro"]:
                values = []
                for seed in SEEDS:
                    vals = [metric(runs[str(seed)][scene], model) for scene in SCENES]
                    weights = (
                        np.ones(5)
                        if scope == "macro"
                        else np.array([v["count"] for v in vals])
                    )
                    values.append(
                        float(np.average([v[key] for v in vals], weights=weights))
                    )
                aggregate[model][f"{scope}_{key}"] = stats(values)
    strata = {}
    for kind in ["motion", "crowding"]:
        names = (
            ["straight", "turning", "stopping", "starting", "other"]
            if kind == "motion"
            else ["zero", "one", "two_plus"]
        )
        strata[kind] = {}
        for name in names:
            item = {}
            for model in MODELS:
                values = {key: [] for key in ["ade_m", "fde_m"]}
                for seed in SEEDS:
                    rows = []
                    for scene in SCENES:
                        r = runs[str(seed)][scene]
                        which = "social" if model == "constant_velocity" else model
                        entry = (
                            r["models"][which]["motion_strata"][name]
                            if kind == "motion"
                            else r["crowding"][name]
                        )
                        if not entry["count"]:
                            continue
                        m = entry if kind == "motion" else entry["models"][which]
                        rows.append(
                            m[
                                "constant_velocity"
                                if model == "constant_velocity"
                                else "transformer"
                            ]
                        )
                    for key in values:
                        values[key].append(
                            float(
                                np.average(
                                    [r[key] for r in rows],
                                    weights=[r["count"] for r in rows],
                                )
                            )
                        )
                item[model] = {key: stats(v) for key, v in values.items()}
            sample = runs["42"]
            item["count"] = sum(
                (
                    r["models"]["social"]["motion_strata"][name]
                    if kind == "motion"
                    else r["crowding"][name]
                )["count"]
                for r in sample.values()
            )
            strata[kind][name] = item
    robustness = {
        str(path.relative_to(root / "robustness").parent): json.loads(path.read_text())
        for path in sorted((root / "robustness").glob("*/*/metrics.json"))
    }
    return dict(
        seeds=SEEDS,
        scene_order=SCENES,
        aggregate=aggregate,
        per_scene=per_scene,
        strata=strata,
        runs=runs,
        validation_robustness=robustness,
        note="Sample standard deviation across 3 training seeds, not independent-window confidence intervals; scene-identifier held-out, no test tuning",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="data/outputs/social_forecasting")
    p.add_argument("--phase4-results", default="data/outputs/forecasting")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    r = summary(Path(a.results), Path(a.phase4_results))
    save_json(out, r)
    print(json.dumps(r["aggregate"], indent=2))
