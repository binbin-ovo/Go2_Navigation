"""Strict metric trajectories, disjoint recording splits and gap-free common-cohort windows."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
from go2_perception.io.config import load_config


@dataclass
class Window:
    key: str
    scene: str
    recording: str
    agent_id: int
    times: np.ndarray
    observed: np.ndarray
    future: np.ndarray


def validate_trajectory(times, positions):
    t = np.asarray(times, dtype=float)
    p = np.asarray(positions, dtype=float)
    if (
        t.ndim != 1
        or p.shape != (len(t), 2)
        or len(t) < 2
        or not np.isfinite(t).all()
        or not np.isfinite(p).all()
    ):
        raise ValueError("Finite times and (N,2) metric positions required")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Timestamps must be strictly increasing")
    return t, p


def resample_observed(times, positions, target, *, max_gap):
    """Explicit linear interpolation only inside observed support, never extrapolation."""
    t, p = validate_trajectory(times, positions)
    target = np.asarray(target, dtype=float)
    if (
        target.ndim != 1
        or not len(target)
        or not np.isfinite(target).all()
        or np.any(np.diff(target) <= 0)
        or not np.isfinite(max_gap)
        or max_gap <= 0
    ):
        raise ValueError("Invalid target times or gap limit")
    if target[0] < t[0] - 1e-8 or target[-1] > t[-1] + 1e-8:
        raise ValueError("Extrapolation prohibited")
    out = []
    interpolated = []
    for q in target:
        exact = np.flatnonzero(np.isclose(t, q, rtol=0, atol=1e-8))
        if len(exact):
            out.append(p[exact[0]])
            interpolated.append(False)
            continue
        right = np.searchsorted(t, q)
        left = right - 1
        if right >= len(t) or left < 0 or t[right] - t[left] > max_gap + 1e-8:
            raise ValueError("Observation gap exceeds explicit interpolation limit")
        ratio = (q - t[left]) / (t[right] - t[left])
        out.append(p[left] * (1 - ratio) + p[right] * ratio)
        interpolated.append(True)
    return np.array(out), np.array(interpolated)


def load_recording(path, *, frame_stride, dt):
    a = np.loadtxt(path, ndmin=2)
    if a.ndim != 2 or a.shape[1] != 4 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Expected finite frame,id,x,y rows")
    if not np.isfinite([frame_stride, dt]).all() or frame_stride <= 0 or dt <= 0:
        raise ValueError("Invalid declared timing")
    if not np.equal(a[:, :2], np.rint(a[:, :2])).all():
        raise ValueError("Frame and identity must be integers")
    trajectories = {}
    # File may be frame-major; within each ID its original order must already be temporal.
    for identifier in np.unique(a[:, 1]).astype(int):
        rows = a[a[:, 1] == identifier]
        t = rows[:, 0] * dt / frame_stride
        validate_trajectory(t, rows[:, 2:]) if len(t) > 1 else None
        trajectories[identifier] = (t, rows[:, 2:].copy())
    return trajectories


def windows_for_recording(
    path,
    scene,
    recording,
    *,
    obs_steps,
    pred_steps,
    cohort_obs,
    cohort_pred,
    stride,
    dt,
    frame_stride,
):
    if (
        min(obs_steps, pred_steps, cohort_obs, cohort_pred, stride) < 1
        or obs_steps < 2
        or cohort_obs < obs_steps
        or cohort_pred < pred_steps
    ):
        raise ValueError("Invalid window/cohort lengths")
    windows = []
    for identifier, (t, p) in load_recording(
        path, frame_stride=frame_stride, dt=dt
    ).items():
        for end in range(cohort_obs - 1, len(t) - cohort_pred, stride):
            full = t[end - cohort_obs + 1 : end + cohort_pred + 1]
            if not np.allclose(np.diff(full), dt, atol=1e-7, rtol=0):
                continue
            windows.append(
                Window(
                    f"{recording}:{identifier}:{t[end]:.8f}",
                    scene,
                    recording,
                    identifier,
                    t[end - obs_steps + 1 : end + pred_steps + 1].copy(),
                    p[end - obs_steps + 1 : end + 1].copy(),
                    p[end + 1 : end + pred_steps + 1].copy(),
                )
            )
    return windows


def dataset(config, split):
    spec = load_config(config["manifest"])
    if spec["units"] != "m" or spec["coordinate_system"] != "metric_world_xy":
        raise ValueError("Metric world coordinates required")
    if not np.isclose(spec["step_seconds"], config["dt"], rtol=0, atol=1e-12):
        raise ValueError("Dataset sampling interval mismatch; resample explicitly")
    groups = [set(config[name]) for name in ("train", "validation", "test")]
    if any(not g for g in groups) or any(
        groups[i] & groups[j] for i in range(3) for j in range(i)
    ):
        raise ValueError("Scene splits must be nonempty and disjoint")
    files = {
        name: {p for scene in config[name] for p in spec["scenes"][scene]}
        for name in ("train", "validation", "test")
    }
    if any(
        files[a] & files[b]
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    ):
        raise ValueError("Recording leakage")
    root = Path(config["root"])
    output = []
    hashes = {}
    options = {
        k: config[k]
        for k in (
            "obs_steps",
            "pred_steps",
            "cohort_obs",
            "cohort_pred",
            "stride",
            "dt",
        )
    }
    for scene in config[split]:
        for name in spec["scenes"][scene]:
            p = root / name
            if not p.resolve().is_relative_to(root.resolve()):
                raise ValueError("Recording outside dataset root")
            hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
            if (
                name in spec.get("file_sha256", {})
                and hashes[name] != spec["file_sha256"][name]
            ):
                raise ValueError(
                    "Recording checksum differs from pinned dataset manifest"
                )
            output.extend(
                windows_for_recording(
                    p, scene, name, frame_stride=spec["frame_stride"], **options
                )
            )
    if not output:
        raise ValueError(f"No valid windows in {split}")
    return output, dict(
        split=split,
        scenes=config[split],
        recording_sha256=hashes,
        windows=len(output),
        window_keys_sha256=hashlib.sha256(
            "\n".join(w.key for w in output).encode()
        ).hexdigest(),
        units="m",
        dt=config["dt"],
    )
