"""Read-only offline adapter. Observed flags mean filter estimates supported by measurements."""

from dataclasses import dataclass, asdict
import numpy as np
import torch
from .data import validate_trajectory, resample_observed


@dataclass
class ForecastOutput:
    track_id: int
    timestamp: float
    future_timestamps: list[float]
    xy: list[list[float]]
    frame_id: str
    clock_domain: str
    units: str
    model_id: str
    model_version: str
    input_interpolation_mask: list[bool]
    semantic_source: str
    history_kind: str = "filtered_observation_supported"
    experimental: bool = True

    def to_dict(self):
        return asdict(self)


def track_history(
    track,
    obs_steps,
    dt,
    *,
    expected_frame="odom",
    units="m",
    resample=False,
    max_gap_s=0.15,
    assume_pedestrian=False,
):
    if (
        units != "m"
        or track.frame_id != expected_frame
        or track.velocity_reference != "odometry"
        or not expected_frame
        or not track.clock_domain
    ):
        raise ValueError("Explicit odometry frame, clock and metric units required")
    if (
        track.status != "Confirmed"
        or track.consecutive_misses
        or not np.isfinite(track.timestamp)
        or not np.isclose(
            track.last_observation_timestamp, track.timestamp, atol=1e-6, rtol=0
        )
    ):
        raise ValueError("Only currently observed confirmed tracks are eligible")
    if track.class_name != "person" and not assume_pedestrian:
        raise ValueError(
            "Pedestrian semantics not established; geometric clusters are not people"
        )
    if obs_steps < 2 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid forecast sampling")
    h = track.history
    if len(h) < 2:
        raise ValueError("Insufficient history")
    for entry in h:
        if type(entry.get("observed")) is not bool:
            raise ValueError("Explicit observed flag required")
        if (
            entry.get("frame_id", track.frame_id) != track.frame_id
            or entry.get("track_id", track.track_id) != track.track_id
        ):
            raise ValueError("History crossed a frame or identity boundary")
        if entry.get("clock_domain", track.clock_domain) != track.clock_domain:
            raise ValueError("History crossed a clock boundary")
        if (
            np.asarray(entry["position"]).shape != (3,)
            or not np.isfinite(entry["position"]).all()
        ):
            raise ValueError("Track history positions must be XYZ")
    times, positions = validate_trajectory(
        [v["timestamp"] for v in h], np.asarray([v["position"][:2] for v in h])
    )
    if (
        not np.isclose(times[-1], track.timestamp, atol=1e-6, rtol=0)
        or not h[-1]["observed"]
    ):
        raise ValueError("History must end at the current actual observation")
    mask = np.array([v["observed"] for v in h])
    times, positions = times[mask], positions[mask]
    targets = track.timestamp - np.arange(obs_steps - 1, -1, -1) * dt
    if resample:
        values, interpolated = resample_observed(
            times, positions, targets, max_gap=max_gap_s
        )
    else:
        indices = []
        for t in targets:
            matches = np.flatnonzero(np.isclose(times, t, atol=1e-6, rtol=0))
            if len(matches) != 1:
                raise ValueError(
                    "No actual observation on model time grid; explicit resampling required"
                )
            indices.append(matches[0])
        values = positions[indices]
        interpolated = np.zeros(obs_steps, dtype=bool)
    return values, targets, interpolated


def forecast_track(model, track, dt, *, model_id, **options):
    if not model_id:
        raise ValueError("Checkpoint/model identity required")
    if not np.isclose(dt, model.dt, rtol=0, atol=1e-12):
        raise ValueError("Track resampling interval must match the trained model")
    values, times, mask = track_history(track, model.obs_steps, dt, **options)
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        prediction = (
            model(torch.as_tensor(values[None], dtype=torch.float32, device=device))[0]
            .cpu()
            .numpy()
        )
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite forecast")
    return ForecastOutput(
        track.track_id,
        float(track.timestamp),
        (times[-1] + np.arange(1, model.pred_steps + 1) * dt).tolist(),
        prediction.tolist(),
        track.frame_id,
        track.clock_domain,
        "m",
        model_id,
        model.version,
        mask.tolist(),
        "track_person_label"
        if track.class_name == "person"
        else "explicit_experimental_override",
    )
