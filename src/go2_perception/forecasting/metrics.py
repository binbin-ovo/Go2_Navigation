"""Euclidean distances in metres, not coordinate-wise MSE or best-of-K metrics."""

import numpy as np


def errors(predicted, target):
    p = np.asarray(predicted)
    y = np.asarray(target)
    if (
        p.shape != y.shape
        or p.ndim != 3
        or p.shape[-1] != 2
        or min(p.shape[:2]) < 1
        or not np.isfinite(p).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Matching finite nonempty B,T,2 trajectories required")
    distance = np.linalg.norm(p - y, axis=-1)
    return distance.mean(axis=1), distance[:, -1]


def summarize(predicted, target):
    ade, fde = errors(predicted, target)
    return dict(count=len(ade), ade_m=float(ade.mean()), fde_m=float(fde.mean()))


def motion_categories(observed, future, dt):
    """Evaluation-only strata, using GT future. Never called by model/inference adapter."""
    labels = []
    for x, y in zip(observed, future):
        vo = np.diff(x, axis=0)[-3:].mean(0) / dt
        vf = np.diff(np.vstack((x[-1], y)), axis=0)[-3:].mean(0) / dt
        a, b = np.linalg.norm(vo), np.linalg.norm(vf)
        angle = np.degrees(
            np.arccos(np.clip(np.dot(vo, vf) / max(a * b, 1e-12), -1, 1))
        )
        route = np.vstack((x[-1], y))
        length = np.linalg.norm(np.diff(route, axis=0), axis=1).sum()
        straightness = np.linalg.norm(y[-1] - x[-1]) / max(length, 1e-12)
        if a < 0.15 and b > 0.4:
            label = "starting"
        elif a > 0.4 and b < 0.15:
            label = "stopping"
        elif min(a, b) > 0.2 and angle >= 30:
            label = "turning"
        elif min(a, b) > 0.2 and angle <= 15 and straightness >= 0.95:
            label = "straight"
        else:
            label = "other"
        labels.append(label)
    return np.asarray(labels)


def paired_report(observed, target, learned, baseline, dt):
    report = dict(
        transformer=summarize(learned, target),
        constant_velocity=summarize(baseline, target),
    )
    a, f = errors(learned, target)
    b, g = errors(baseline, target)
    report["paired"] = dict(
        mean_delta_ade_m=float(np.mean(a - b)),
        mean_delta_fde_m=float(np.mean(f - g)),
        learned_lower_ade_fraction=float(np.mean(a < b)),
        note="Overlapping windows are correlated; no independent-window significance claim",
    )
    labels = motion_categories(observed, target, dt)
    report["motion_strata"] = {}
    for name in ("straight", "turning", "stopping", "starting", "other"):
        mask = labels == name
        report["motion_strata"][name] = dict(
            count=int(mask.sum()),
            transformer=summarize(learned[mask], target[mask]) if mask.any() else None,
            constant_velocity=summarize(baseline[mask], target[mask])
            if mask.any()
            else None,
        )
    report["horizons"] = {
        str(h): dict(
            seconds=h * dt,
            transformer=summarize(learned[:, :h], target[:, :h]),
            constant_velocity=summarize(baseline[:, :h], target[:, :h]),
        )
        for h in sorted(
            {min(4, len(target[0])), min(8, len(target[0])), len(target[0])}
        )
    }
    return report
