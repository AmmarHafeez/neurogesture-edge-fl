"""Reproducibility helpers."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set common random seeds for local experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
