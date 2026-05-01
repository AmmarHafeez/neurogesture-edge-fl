"""Federated client utilities for subject-local EMG training."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.train_deep import EMGWindowDataset


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClientUpdate:
    """Model update returned by one simulated federated client."""

    subject_id: str
    state_dict: dict[str, torch.Tensor]
    num_samples: int
    mean_loss: float


@dataclass
class FederatedClient:
    """One subject-local client for federated simulation."""

    subject_id: str
    X: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        self.subject_id = str(self.subject_id)
        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.int64)
        if self.X.ndim != 3:
            raise ValueError("Client X must have shape (n_windows, window_size, channels)")
        if len(self.X) != len(self.y):
            raise ValueError("Client X and y must have the same number of samples")
        if len(self.y) == 0:
            raise ValueError("Federated clients must contain at least one sample")

    @property
    def num_samples(self) -> int:
        """Return the number of local windows."""
        return int(len(self.y))

    def train(
        self,
        global_model: nn.Module,
        criterion: nn.Module,
        local_epochs: int,
        batch_size: int,
        learning_rate: float,
        device: torch.device,
        random_state: int = 42,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 5.0,
        fedprox_mu: float = 0.0,
    ) -> ClientUpdate:
        """Train a private local copy of the global model."""
        if local_epochs <= 0:
            raise ValueError("local_epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if fedprox_mu < 0:
            raise ValueError("fedprox_mu must be non-negative")

        torch.manual_seed(random_state)
        model = deepcopy(global_model).to(device)
        global_parameters = snapshot_model_parameters(model, device=device)
        dataset = EMGWindowDataset(self.X, self.y)
        generator = torch.Generator()
        generator.manual_seed(random_state)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        losses = []
        for _ in range(local_epochs):
            losses.append(
                train_one_epoch_local(
                    model=model,
                    dataloader=dataloader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    max_grad_norm=max_grad_norm,
                    fedprox_mu=fedprox_mu,
                    global_parameters=global_parameters,
                )
            )

        state_dict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        mean_loss = float(np.mean(losses))
        LOGGER.debug(
            "Client %s trained on %s samples with mean loss %.4f",
            self.subject_id,
            self.num_samples,
            mean_loss,
        )
        return ClientUpdate(
            subject_id=self.subject_id,
            state_dict=state_dict,
            num_samples=self.num_samples,
            mean_loss=mean_loss,
        )


def snapshot_model_parameters(
    model: nn.Module,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Copy model parameters for use as the FedProx reference point."""
    snapshot: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        value = parameter.detach().clone()
        if device is not None:
            value = value.to(device)
        snapshot[name] = value
    return snapshot


def fedprox_proximal_penalty(
    model: nn.Module,
    global_parameters: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return sum of squared distances from the round's global parameters."""
    try:
        first_parameter = next(model.parameters())
    except StopIteration as error:
        raise ValueError("Model has no parameters") from error

    penalty = torch.zeros((), device=first_parameter.device)
    for name, parameter in model.named_parameters():
        if name not in global_parameters:
            raise KeyError(f"Missing global parameter for FedProx: {name}")
        reference = global_parameters[name].to(device=parameter.device)
        penalty = penalty + torch.sum((parameter - reference) ** 2)
    return penalty


def train_one_epoch_local(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 5.0,
    fedprox_mu: float = 0.0,
    global_parameters: dict[str, torch.Tensor] | None = None,
) -> float:
    """Train one local epoch, optionally adding FedProx regularization."""
    if fedprox_mu < 0:
        raise ValueError("fedprox_mu must be non-negative")
    if fedprox_mu > 0 and global_parameters is None:
        raise ValueError("global_parameters are required when fedprox_mu > 0")

    model.train()
    total_loss = 0.0
    total_samples = 0
    for windows, labels in dataloader:
        windows = windows.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(windows)
        loss = criterion(logits, labels)
        if fedprox_mu > 0:
            loss = loss + (fedprox_mu / 2.0) * fedprox_proximal_penalty(
                model,
                global_parameters=global_parameters or {},
            )
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        batch_size = len(labels)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training dataloader is empty")
    return total_loss / total_samples


def build_subject_clients(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    train_idx: np.ndarray,
) -> dict[str, FederatedClient]:
    """Create one federated client per training subject."""
    values = np.asarray(X, dtype=np.float32)
    labels = np.asarray(y, dtype=np.int64)
    subjects = np.asarray(subject_ids).astype(str)
    indices = np.asarray(train_idx, dtype=int)
    if len(indices) == 0:
        raise ValueError("train_idx must contain at least one sample")

    clients: dict[str, FederatedClient] = {}
    for subject_id in sorted(np.unique(subjects[indices]).tolist()):
        subject_mask = subjects[indices] == subject_id
        local_indices = indices[subject_mask]
        clients[str(subject_id)] = FederatedClient(
            subject_id=str(subject_id),
            X=values[local_indices],
            y=labels[local_indices],
        )

    if not clients:
        raise ValueError("No federated clients were created")
    return clients
