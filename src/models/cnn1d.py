"""Compact 1D CNN for EMG window classification."""

from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    """Small 1D convolutional classifier for EMG windows."""

    def __init__(
        self,
        input_channels: int = 8,
        num_classes: int = 7,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for input shaped (batch, channels, samples)."""
        features = self.features(x)
        pooled = self.pool(features).squeeze(-1)
        return self.classifier(pooled)
