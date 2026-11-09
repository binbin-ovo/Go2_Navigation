"""Shared temporal encoder and masked mean social context, deterministic direct forecasts."""

import torch
from torch import nn
from .model import TemporalTransformer
from .geometry import local_frame, to_global

SOCIAL_VERSION = "social-temporal-transformer-v1"


class SocialTransformer(TemporalTransformer):
    version = SOCIAL_VERSION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.embedding.out_features
        width = self.head[1].out_features
        self.relative_embedding = nn.Linear(2, d, bias=False)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, width),
            nn.GELU(),
            nn.Linear(width, self.pred_steps * 2),
        )

    def encode(self, local, mask):
        if self.representation == "local_position":
            features = local
        else:
            valid_pairs = mask[:, 1:] & mask[:, :-1]
            delta = torch.where(valid_pairs[..., None], torch.diff(local, dim=1), 0.0)
            features = torch.cat((torch.zeros_like(local[:, :1]), delta), dim=1)
        features = torch.where(mask[..., None], features, 0.0)
        hidden = self.encoder(
            self.embedding(features) + self.positional_encoding,
            src_key_padding_mask=~mask,
        )
        # Dataset and adapter require current observation, even for partial histories.
        return hidden[:, -1]

    def forward(self, observed, neighbors=None, neighbor_mask=None, target_mask=None):
        if (
            observed.ndim != 3
            or observed.shape[1:] != (self.obs_steps, 2)
            or len(observed) == 0
        ):
            raise ValueError("Expected nonempty B,T,2 target history")
        if target_mask is None:
            target_mask = torch.ones(
                observed.shape[:2], dtype=torch.bool, device=observed.device
            )
        if (
            target_mask.dtype != torch.bool
            or target_mask.shape != observed.shape[:2]
            or not target_mask[:, -1].all()
        ):
            raise ValueError(
                "Boolean target mask with valid current observation required"
            )
        if not torch.isfinite(observed[target_mask]).all():
            raise ValueError("Nonfinite valid target position")
        if self.heading_align and not target_mask.all():
            raise ValueError("Heading alignment requires complete target observations")
        clean = torch.where(target_mask[..., None], observed, observed[:, -1:, :])
        local, origin, rotation = local_frame(clean, self.heading_align)
        target = self.encode(local, target_mask)
        pooled = torch.zeros_like(target)
        if neighbors is not None:
            if (
                neighbors.ndim != 4
                or neighbors.shape[0] != len(observed)
                or neighbors.shape[2:] != (self.obs_steps, 2)
            ):
                raise ValueError("Expected B,K,T,2 neighbor histories")
            if (
                neighbor_mask is None
                or neighbor_mask.dtype != torch.bool
                or neighbor_mask.shape != neighbors.shape[:-1]
            ):
                raise ValueError("Explicit boolean neighbor observation masks required")
            active = neighbor_mask.any(-1)
            if (active & ~neighbor_mask[:, :, -1]).any() or not torch.isfinite(
                neighbors[neighbor_mask]
            ).all():
                raise ValueError(
                    "Each active neighbor needs finite observed positions and a current observation"
                )
            k = neighbors.shape[1]
            if k and active.any():
                safe = torch.where(neighbor_mask[..., None], neighbors, origin[:, None])
                relative = (safe - origin[:, None]) @ rotation[:, None]
                valid_local = relative[active]
                tokens = self.encode(valid_local, neighbor_mask[active])
                tokens = tokens + self.relative_embedding(valid_local[:, -1])
                encoded = target.new_zeros((len(observed), k, target.shape[-1]))
                encoded[active] = tokens
                pooled = encoded.sum(1) / active.sum(1).clamp_min(1)[:, None]
        elif neighbor_mask is not None:
            raise ValueError("Mask without neighbors")
        offsets = self.head(torch.cat((target, pooled), dim=-1)).reshape(
            -1, self.pred_steps, 2
        )
        return to_global(offsets, origin, rotation)
