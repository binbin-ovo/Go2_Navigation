"""Aggregate-only development report; no private trajectories or images are published."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", default="data/outputs/recovery")
    p.add_argument("--output", default="docs")
    a = p.parse_args()
    root = Path(a.inputs)
    out = Path(a.output)

    def load(p):
        return json.loads(p.read_text())

    variants = {
        k: load(root / (k + "_metrics.json"))
        for k in ["baseline03", "release01", "release03"]
    }
    old = load(Path("docs/STABLE_LIDAR_METRICS.json"))["real"]
    synthetic = load(root / "synthetic" / "metrics.json")
    results = dict(
        base_commit="89d901dc",
        baseline_provenance="Original ecfaae3 cached; 5A be4b9de; unchanged 5A 0.3 s counterfactual on this branch",
        recordings="A1/A2/B/C development only; annotations not human-verified",
        variants=variants,
        original_and_5a=old,
        synthetic=synthetic,
        freeze=load(root / "release_freeze.json"),
        optimization=load(root / "optimization_final.json"),
        self_region_audit=load(root / "self_region_audit.json"),
    )
    (out / "DETECTION_RECOVERY_METRICS.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )

    def f(v):
        return "—" if v is None else f"{v:.3f}"

    lines = [
        "# Detection recovery measured results",
        "",
        "Development diagnostics only. Current-direct and recent-evidence coverage are different estimands. Read DETECTION_RECOVERY_EXPERIMENTS.md before comparing the rows.",
        "",
        "| Recording/object | Method | Current coverage | Current MOVING recall | False MOVING | Recent ≤0.2 s coverage / recall | Jitter m | Velocity error m/s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, recording in variants["release03"].items():
        for key in recording["current_direct"]["objects"]:
            for method in ["original", "5A", "5A_0.3", "recovery_0.1", "recovery_0.3"]:
                if method in ["original", "5A"]:
                    r = old[name]["baseline" if method == "original" else "stable"][
                        "objects"
                    ][key]
                    recent = "—"
                else:
                    variant = {
                        "5A_0.3": "baseline03",
                        "recovery_0.1": "release01",
                        "recovery_0.3": "release03",
                    }[method]
                    r = variants[variant][name]["current_direct"]["objects"][key]
                    v = variants[variant][name]["recent_0p2s"]["objects"][key]
                    recent = f(v["coverage"]) + " / " + f(v["moving_recall"])
                lines.append(
                    f"| {name}/{key} | {method} | {f(r['coverage'])} | {f(r['moving_recall'])} | {f(r['false_moving_rate'])} | {recent} | {f(r['centroid_jitter_xy_m'])} | {f(r['velocity_error_mps'])} |"
                )
    lines += [
        "",
        "## Proposal quality, continuity and observation age",
        "",
        "| Recording | Window | Stale fraction | Self points excluded | Mean / p95 compute ms | Mean / p95 wall ms | Peak RSS MiB |",
        "|---|---|---|---|---|---|---|",
    ]
    for variant in ["baseline03", "release01", "release03"]:
        for name, r in variants[variant].items():
            run = r["runtime"]
            c = run["processing_ms"]
            w = run["wall_ms"]
            lines.append(
                f"| {name} | {variant} | {f(r['current_direct']['stale_fraction'])} | {r['self_excluded_points']} | {f(c['mean'])} / {f(c['p95'])} | {f(w['mean'])} / {f(w['p95'])} | {run['peak_rss_platform_units'] / 2**20:.1f} |"
            )
    lines += [
        "",
        "Full merge/split counts, duplicate frames, observation ages, event delays (including null failures), and geometry-matched ID-change motion reliability are in DETECTION_RECOVERY_METRICS.json. RSS is the process high-water mark, including imported dependencies; wall time excludes reader initialization and final summary, includes per-frame input load/process/JSON write.",
        "",
        "## Synthetic cases",
        "",
        "| Case | Method | Current coverage / recall | Recent coverage / recall | False MOVING | Dimension error m | Motion delays s (null = failure) |",
        "|---|---|---|---|---|---|---|",
    ]
    for case, methods in synthetic.items():
        for method, r in methods.items():
            c = r["current"]
            v = r["recent_0p2s"]
            delays = (
                ", ".join(f(x["moving_delay_s"]) for x in c["transition_events"]) or "—"
            )
            lines.append(
                f"| {case} | {method} | {f(c['coverage'])} / {f(c['moving_recall'])} | {f(v['coverage'])} / {f(v['moving_recall'])} | {f(c['stationary_false_moving_rate'])} | {f(c['dimension_error_mean_m'])} | {delays} |"
            )
    (out / "DETECTION_RECOVERY_RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
