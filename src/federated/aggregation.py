"""Aggregation helpers for manual federated learning simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


StateDict = Mapping[str, torch.Tensor]
WeightedState = tuple[StateDict, int]


def fedavg_state_dict(updates: Sequence[WeightedState]) -> dict[str, torch.Tensor]:
    """Aggregate model state dictionaries with sample-weighted FedAvg.

    Floating tensors are averaged by local sample count. Non-floating tensors,
    such as BatchNorm counters, are copied from the first update to avoid
    invalid integer averaging.
    """
    if not updates:
        raise ValueError("At least one client update is required for FedAvg")

    total_samples = sum(int(sample_count) for _, sample_count in updates)
    if total_samples <= 0:
        raise ValueError("FedAvg requires a positive total sample count")

    reference_state = updates[0][0]
    aggregated: dict[str, torch.Tensor] = {}

    for key, reference_tensor in reference_state.items():
        if torch.is_floating_point(reference_tensor):
            accumulator = torch.zeros_like(reference_tensor, dtype=torch.float32)
            for state_dict, sample_count in updates:
                if key not in state_dict:
                    raise KeyError(f"Client update is missing state key: {key}")
                tensor = state_dict[key].detach().to(dtype=torch.float32)
                accumulator += tensor * (int(sample_count) / total_samples)
            aggregated[key] = accumulator.to(dtype=reference_tensor.dtype)
        else:
            aggregated[key] = reference_tensor.detach().clone()

    return aggregated
