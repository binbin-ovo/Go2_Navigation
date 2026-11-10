"""Direct multi-horizon offsets: all attention is within observed history."""

import math
import torch
from torch import nn
from .geometry import local_frame, to_global
from . import MODEL_VERSION


class TemporalTransformer(nn.Module):
    version = MODEL_VERSION

    def __init__(
        self,
        obs_steps=8,
        pred_steps=12,
        d_model=64,
        heads=4,
        layers=2,
        feedforward=128,
        dropout=0.1,
        representation="displacement",
        heading_align=False,
        dt=0.4,
    ):
        super().__init__()
        if (
            obs_steps < 2
            or pred_steps < 1
            or d_model < 2
            or d_model % 2
            or heads < 1
            or d_model % heads
            or layers < 1
            or feedforward < 1
            or not 0 <= dropout < 1
            or representation not in ("displacement", "local_position")
        ):
            raise ValueError("Invalid Transformer configuration")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid model sampling interval")
        self.dt = dt
        self.obs_steps, self.pred_steps = obs_steps, pred_steps
        self.representation, self.heading_align = representation, heading_align
        self.embedding = nn.Linear(2, d_model)
        pos = torch.arange(obs_steps).float()[:, None]
        freq = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(obs_steps, d_model)
        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq)
        self.register_buffer("positional_encoding", pe[None])
        layer = nn.TransformerEncoderLayer(
            d_model,
            heads,
            feedforward,
            dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, feedforward),
            nn.GELU(),
            nn.Linear(feedforward, pred_steps * 2),
        )

    def forward(self, observed):
        if observed.ndim != 3 or observed.shape[1:] != (self.obs_steps, 2):
            raise ValueError("Incorrect observation shape")
        local, origin, rotation = local_frame(observed, self.heading_align)
        features = (
            local
            if self.representation == "local_position"
            else torch.cat(
                (torch.zeros_like(local[:, :1]), torch.diff(local, dim=1)), dim=1
            )
        )
        hidden = self.encoder(self.embedding(features) + self.positional_encoding)
        offsets = self.head(hidden[:, -1]).reshape(-1, self.pred_steps, 2)
        return to_global(offsets, origin, rotation)
