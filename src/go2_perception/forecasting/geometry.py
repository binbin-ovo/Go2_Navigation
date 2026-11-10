"""Observation-only normalization; row-vector rotation, no fitted statistics."""

import torch


def local_frame(observed, heading_align=False):
    if (
        observed.ndim != 3
        or observed.shape[0] < 1
        or observed.shape[-1] != 2
        or observed.shape[1] < 2
        or not torch.isfinite(observed).all()
    ):
        raise ValueError("Finite B,T,2 observations required")
    origin = observed[:, -1:, :]
    angle = torch.zeros(len(observed), device=observed.device, dtype=observed.dtype)
    if heading_align:
        delta = observed[:, -1] - observed[:, max(0, observed.shape[1] - 4)]
        angle = torch.where(
            torch.linalg.vector_norm(delta, dim=-1) > 1e-6,
            torch.atan2(delta[:, 1], delta[:, 0]),
            angle,
        )
    c, s = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack((c, -s, s, c), dim=-1).reshape(-1, 2, 2)
    return (observed - origin) @ rotation, origin, rotation


def to_global(local, origin, rotation):
    return local @ rotation.transpose(-1, -2) + origin
