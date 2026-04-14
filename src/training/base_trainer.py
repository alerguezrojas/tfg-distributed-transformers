from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """Abstract interface for all trainers.

    Defines the contract that every trainer must fulfill,
    regardless of whether it runs on a single GPU, multiple GPUs,
    or is wrapped by a decorator.
    """

    @abstractmethod
    def train_epoch(self, loader: DataLoader) -> dict:
        """Run one training epoch and return a dict with metrics."""

    @abstractmethod
    def eval_epoch(self, loader: DataLoader) -> dict:
        """Run one evaluation epoch and return a dict with metrics."""

    @abstractmethod
    def save_checkpoint(self, epoch: int, metrics: dict):
        """Save a model checkpoint."""

    @abstractmethod
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        """Full training loop for N epochs."""
