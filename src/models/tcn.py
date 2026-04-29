"""Small temporal convolutional network for EMG windows."""

from __future__ import annotations

import torch
from torch import nn


class TemporalBlock(nn.Module):
    """Residual dilated convolution block."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual temporal convolutions."""
        output = self.net(x)
        if output.shape[-1] != x.shape[-1]:
            output = output[..., : x.shape[-1]]
        return x + output


class TCN(nn.Module):
    """Compact residual TCN classifier."""

    def __init__(
        self,
        input_channels: int = 8,
        num_classes: int = 7,
        hidden_channels: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            TemporalBlock(hidden_channels, dilation=1, dropout=dropout),
            TemporalBlock(hidden_channels, dilation=2, dropout=dropout),
            TemporalBlock(hidden_channels, dilation=4, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for input shaped (batch, channels, samples)."""
        features = self.input_projection(x)
        features = self.blocks(features)
        pooled = self.pool(features).squeeze(-1)
        return self.classifier(pooled)
