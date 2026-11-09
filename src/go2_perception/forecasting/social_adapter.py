"""Optional offline multi-track context with explicit external identity-quality attestation."""

from dataclasses import dataclass
import numpy as np
import torch
from .adapter import ForecastOutput, track_history
from .data import resample_observed
from .social_data import selection_options


@dataclass
class MultiTrackInput:
    target_id: int
    timestamp: float
    frame_id: str
    clock_domain: str
    units: str
    times: np.ndarray
    observed: np.ndarray
    target_interpolated: np.ndarray
    neighbor_ids: np.ndarray
    neighbors: np.ndarray
    neighbor_mask: np.ndarray
    neighbor_interpolated: np.ndarray
    rejected: dict


def multi_track_input(
    target,
    tracks,
    obs_steps,
    dt,
    *,
    validated_ids,
    expected_frame="odom",
    units="m",
    radius=3.0,
    max_neighbors=8,
    resample=False,
    max_gap_s=0.15,
):
    """validated_ids must come from external ID/semantic quality review, not tracker confidence."""
    selection_options(radius, max_neighbors)
    if target.track_id not in validated_ids:
        raise ValueError("Target identity has not been externally validated")
    identifiers = [t.track_id for t in tracks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate track IDs in simultaneous snapshot")
    if any(t.track_id == target.track_id and t is not target for t in tracks):
        raise ValueError("Ambiguous target snapshot")
    options = dict(
        expected_frame=expected_frame,
        units=units,
        resample=resample,
        max_gap_s=max_gap_s,
    )
    x, times, target_interp = track_history(target, obs_steps, dt, **options)
    rejected = {}
    candidates = []
    for neighbor in tracks:
        if neighbor.track_id == target.track_id:
            continue
        try:
            if neighbor.track_id not in validated_ids:
                raise ValueError("Identity not externally validated")
            if neighbor.clock_domain != target.clock_domain or not np.isclose(
                neighbor.timestamp, target.timestamp, rtol=0, atol=1e-6
            ):
                raise ValueError("Snapshot clock or timestamp mismatch")
            h = neighbor.history
            if any(v["timestamp"] > target.timestamp + 1e-8 for v in h):
                raise ValueError("Future history prohibited")
            observed = [v for v in h if v.get("observed") is True]
            if len(observed) < 2:
                raise ValueError("Neighbor needs two actual observations")
            # Reuse all existing frame/semantics/status/finite/history-boundary checks.
            pair_dt = observed[-1]["timestamp"] - observed[-2]["timestamp"]
            track_history(
                neighbor, 2, pair_dt, expected_frame=expected_frame, units=units
            )
            nt = np.asarray([v["timestamp"] for v in observed])
            xy = np.asarray([v["position"][:2] for v in observed])
            distance = float(np.linalg.norm(xy[-1] - x[-1]))
            if distance > radius:
                continue
            values = np.zeros_like(x)
            mask = np.zeros(obs_steps, bool)
            interpolated = np.zeros(obs_steps, bool)
            for k, q in enumerate(times):
                match = np.flatnonzero(np.isclose(nt, q, rtol=0, atol=1e-8))
                if len(match):
                    values[k] = xy[match[0]]
                    mask[k] = True
                elif resample:
                    try:
                        point, flag = resample_observed(nt, xy, [q], max_gap=max_gap_s)
                        values[k] = point[0]
                        mask[k] = True
                        interpolated[k] = flag[0]
                    except ValueError:
                        pass  # Missing neighbor support stays explicitly invalid, never padded as evidence.
            if not mask[-1]:
                raise ValueError("No neighbor observation at cutoff")
            candidates.append((distance, neighbor.track_id, values, mask, interpolated))
        except (ValueError, KeyError, TypeError) as e:
            rejected[str(neighbor.track_id)] = str(e)
    chosen = sorted(candidates, key=lambda v: (v[0], v[1]))[:max_neighbors]
    neighbors = np.zeros((max_neighbors, obs_steps, 2), np.float32)
    mask = np.zeros((max_neighbors, obs_steps), bool)
    interp = mask.copy()
    ids = np.full(max_neighbors, -1, dtype=np.int64)
    for i, (_, identifier, values, valid, flags) in enumerate(chosen):
        ids[i] = identifier
        neighbors[i] = values
        mask[i] = valid
        interp[i] = flags
    return MultiTrackInput(
        target.track_id,
        float(target.timestamp),
        target.frame_id,
        target.clock_domain,
        units,
        times,
        x.astype("float32"),
        target_interp,
        ids,
        neighbors,
        mask,
        interp,
        rejected,
    )


def forecast_multitrack(model, target, tracks, dt, *, model_id, **options):
    if not model_id or not np.isclose(dt, model.dt, rtol=0, atol=1e-12):
        raise ValueError("Model identity and trained sampling interval required")
    context = multi_track_input(target, tracks, model.obs_steps, dt, **options)
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        xy = (
            model(
                torch.as_tensor(context.observed[None], device=device),
                torch.as_tensor(context.neighbors[None], device=device),
                torch.as_tensor(context.neighbor_mask[None], device=device),
            )[0]
            .cpu()
            .numpy()
        )
    if not np.isfinite(xy).all():
        raise FloatingPointError("Nonfinite forecast")
    forecast = ForecastOutput(
        target.track_id,
        float(target.timestamp),
        (target.timestamp + np.arange(1, model.pred_steps + 1) * dt).tolist(),
        xy.tolist(),
        target.frame_id,
        target.clock_domain,
        "m",
        model_id,
        model.version,
        context.target_interpolated.tolist(),
        "track_person_label",
    )
    return forecast, context
