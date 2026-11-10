"""Observation-only synchronized neighbor context; target eligibility is unchanged."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .data import dataset, load_recording
from .training import arrays
from go2_perception.io.config import load_config


def selection_options(radius, max_neighbors):
    if (
        not np.isfinite(radius)
        or radius <= 0
        or type(max_neighbors) is not int
        or max_neighbors < 0
    ):
        raise ValueError(
            "Positive radius and nonnegative integer neighbor limit required"
        )


@dataclass
class SocialBatch:
    observed: np.ndarray  # B,T,2 global metric XY
    target_mask: np.ndarray  # B,T bool; final observation always valid
    neighbors: np.ndarray  # B,K,T,2 global metric XY; zero fill is not evidence
    neighbor_mask: np.ndarray  # B,K,T bool
    times: np.ndarray  # B,T shared observation timestamps, seconds
    neighbor_ids: np.ndarray  # B,K, -1 padding
    future: np.ndarray  # B,H,2 supervision only

    @property
    def relative_positions(self):
        return np.where(
            self.neighbor_mask[..., None],
            self.neighbors - self.observed[:, None, -1:, :],
            0,
        )

    @property
    def counts(self):
        return self.neighbor_mask.any(-1).sum(-1)


def context_arrays(windows, recordings, radius=3.0, max_neighbors=8):
    """recordings maps file -> {identity: (times, positions)}. No future reads for inputs."""
    selection_options(radius, max_neighbors)
    if not windows:
        raise ValueError("Nonempty target windows required")
    x, y = arrays(windows)
    b, t, _ = x.shape
    neighbors = np.zeros((b, max_neighbors, t, 2), dtype=np.float32)
    masks = np.zeros((b, max_neighbors, t), dtype=bool)
    ids = np.full((b, max_neighbors), -1, dtype=np.int64)
    # Round only for float timestamp lookup, well below recording precision (0.4 s).
    indices = {}
    at_time = {}
    for name, tracks in recordings.items():
        indices[name], at_time[name] = {}, {}
        for identity, (times, xy) in tracks.items():
            samples = {round(float(q), 8): p for q, p in zip(times, xy)}
            if len(samples) != len(times):
                raise ValueError("Duplicate timestamps after timestamp indexing")
            indices[name][identity] = samples
            for q, p in samples.items():
                at_time[name].setdefault(q, []).append((identity, p))
    times = np.stack([w.times[:t] for w in windows])
    for i, w in enumerate(windows):
        cutoff = round(float(times[i, -1]), 8)
        candidates = []
        for identity, position in at_time[w.recording].get(cutoff, []):
            distance = np.linalg.norm(position - w.observed[-1])
            if identity != w.agent_id and distance <= radius:
                candidates.append((distance, identity))
        for j, (_, identity) in enumerate(sorted(candidates)[:max_neighbors]):
            ids[i, j] = identity
            past = indices[w.recording][identity]
            for k, q in enumerate(times[i]):
                point = past.get(round(float(q), 8))
                if point is not None:
                    neighbors[i, j, k] = point
                    masks[i, j, k] = True
    return SocialBatch(x, np.ones((b, t), bool), neighbors, masks, times, ids, y)


def social_dataset(config, split):
    windows, meta = dataset(config["dataset"], split)
    d = config["dataset"]
    manifest = load_config(d["manifest"])
    recordings = {
        name: load_recording(
            Path(d["root"]) / name, frame_stride=manifest["frame_stride"], dt=d["dt"]
        )
        for name in meta["recording_sha256"]
    }
    batch = context_arrays(windows, recordings, **config["social"])
    meta = dict(
        meta,
        social=config["social"],
        neighbor_count_mean=float(batch.counts.mean()),
        zero_neighbor_windows=int((batch.counts == 0).sum()),
        partial_neighbor_histories=int(
            (batch.neighbor_mask.any(-1) & ~batch.neighbor_mask.all(-1)).sum()
        ),
    )
    return windows, batch, meta
