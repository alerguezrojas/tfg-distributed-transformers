"""OOP Decorator pattern for trainers.

All decorators extend TrainerDecorator which auto-delegates unknown
attributes to the wrapped trainer, so the full decorator chain is
transparent. Decorators can be stacked freely:

    trainer = TracingDecorator(
        LayerHooksDecorator(
            BatchMetricsDecorator(
                Trainer(...)
            )
        ),
        logger=logger,
    )
"""

import logging
import time

import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.base_trainer import BaseTrainer


def _eta_str(epoch_times: list[float], epochs_done: int, epochs_total: int) -> str:
    if not epoch_times:
        return "?"
    remaining_s = (epochs_total - epochs_done) * (sum(epoch_times) / len(epoch_times))
    h, m = int(remaining_s // 3600), int((remaining_s % 3600) // 60)
    return f"{h}h {m:02d}m"


class TrainerDecorator(BaseTrainer):
    """Base class for all trainer decorators.

    Automatically delegates any attribute not defined in the decorator
    to the wrapped trainer, traversing the full decorator chain.
    This avoids duplicating properties (model, optimizer, device, ...)
    in every decorator subclass.
    """

    def __init__(self, trainer: BaseTrainer):
        self._trainer = trainer

    def __getattr__(self, name: str):
        return getattr(self._trainer, name)


# ---------------------------------------------------------------------------
# Level 1 — Epoch-level console logging
# ---------------------------------------------------------------------------

class MetricsLoggerDecorator(TrainerDecorator):
    """Prints train/val metrics after each epoch with best F1 and ETA."""

    def train_epoch(self, loader: DataLoader) -> dict:
        return self._trainer.train_epoch(loader)

    def eval_epoch(self, loader: DataLoader) -> dict:
        return self._trainer.eval_epoch(loader)

    def save_checkpoint(self, epoch: int, metrics: dict):
        self._trainer.save_checkpoint(epoch, metrics)
        print(f"  Checkpoint guardado (epoch {epoch:03d})")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        best_f1 = 0.0
        epoch_times: list[float] = []

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_m = self.train_epoch(train_loader)
            val_m = self.eval_epoch(val_loader)
            epoch_times.append(time.time() - t0)

            if val_m["f1"] > best_f1:
                best_f1 = val_m["f1"]
                self.save_checkpoint(epoch, val_m)

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_m['loss']:.4f}  train_f1={train_m['f1']:.4f}  train_acc={train_m['accuracy']:.4f} | "
                f"val_loss={val_m['loss']:.4f}  val_f1={val_m['f1']:.4f}  best={best_f1:.4f}  val_acc={val_m['accuracy']:.4f} | "
                f"time={train_m['time']:.0f}s  ETA={_eta_str(epoch_times, epoch, epochs)}"
            )


# ---------------------------------------------------------------------------
# Level 2 — Batch-level progress bar
# ---------------------------------------------------------------------------

class BatchMetricsDecorator(TrainerDecorator):
    """Shows a real-time tqdm progress bar with loss updated per batch.

    Reimplements the training loop to intercept at batch level.
    This is a 'white-box' decorator: it needs direct access to model,
    optimizer, criterion and device from the wrapped trainer.
    TrainerDecorator.__getattr__ makes these available transparently.
    """

    def __init__(self, trainer: BaseTrainer, log_every: int = 50):
        super().__init__(trainer)
        self.log_every = log_every

    def train_epoch(self, loader: DataLoader) -> dict:
        model = self._trainer.model
        optimizer = self._trainer.optimizer
        criterion = self._trainer.criterion
        device = self._trainer.device
        scheduler = self._trainer.scheduler

        model.train()
        total_loss = 0.0
        start = time.time()

        progress = tqdm(loader, desc="  Train", leave=False, unit="batch")
        for batch_idx, (images, labels) in enumerate(progress, 1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{total_loss / batch_idx:.4f}")

        if scheduler:
            scheduler.step()

        return {
            "loss": total_loss / len(loader),
            "f1": 0.0,
            "accuracy": 0.0,
            "time": time.time() - start,
        }

    def eval_epoch(self, loader: DataLoader) -> dict:
        return self._trainer.eval_epoch(loader)

    def save_checkpoint(self, epoch: int, metrics: dict):
        self._trainer.save_checkpoint(epoch, metrics)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        self._trainer.fit(train_loader, val_loader, epochs)


# ---------------------------------------------------------------------------
# Level 3 — Layer-level activations via PyTorch forward hooks
# ---------------------------------------------------------------------------

class LayerHooksDecorator(TrainerDecorator):
    """Captures mean activations of Linear layers using PyTorch forward hooks.

    Hooks fire automatically on every forward pass without modifying the
    model. Every log_every_n_epochs epochs, prints a mini activation report
    showing which layers are active and how strongly.
    """

    def __init__(self, trainer: BaseTrainer, log_every_n_epochs: int = 5):
        super().__init__(trainer)
        self.log_every_n_epochs = log_every_n_epochs
        self._hooks: list = []
        self._activations: dict[str, float] = {}
        self._epoch: int = 0

    def _register_hooks(self):
        for name, module in self._trainer.model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(_module, _input, output):
            self._activations[name] = output.detach().abs().mean().item()
        return hook

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _print_activations(self, epoch: int):
        if not self._activations:
            return
        print(f"\n[hooks] Activaciones medias — epoch {epoch:03d}:")
        for name, val in list(self._activations.items())[:5]:
            bar = "█" * int(val * 20)
            print(f"  {name:<50} {val:.4f}  {bar}")

    def train_epoch(self, loader: DataLoader) -> dict:
        self._epoch += 1
        self._register_hooks()
        self._activations.clear()
        try:
            result = self._trainer.train_epoch(loader)
        finally:
            self._remove_hooks()
        if self._epoch % self.log_every_n_epochs == 0:
            self._print_activations(self._epoch)
        return result

    def eval_epoch(self, loader: DataLoader) -> dict:
        return self._trainer.eval_epoch(loader)

    def save_checkpoint(self, epoch: int, metrics: dict):
        self._trainer.save_checkpoint(epoch, metrics)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        self._trainer.fit(train_loader, val_loader, epochs)


# ---------------------------------------------------------------------------
# Level 4 — Structured logging with timestamps
# ---------------------------------------------------------------------------

class TracingDecorator(TrainerDecorator):
    """Structured epoch-level logging with timestamps to console and file.

    Emits INFO log lines per epoch: train + val metrics, best F1, ETA.
    Delegates train_epoch entirely to the inner trainer — no per-batch
    overhead, suitable for long training runs.

    Usage:
        logger = setup_logger("trainer", log_file="logs/train_20260101_120000.log")
        trainer = TracingDecorator(Trainer(...), logger=logger)
    """

    def __init__(self, trainer: BaseTrainer, logger: logging.Logger):
        super().__init__(trainer)
        self._logger = logger

    def train_epoch(self, loader: DataLoader) -> dict:
        return self._trainer.train_epoch(loader)

    def eval_epoch(self, loader: DataLoader) -> dict:
        return self._trainer.eval_epoch(loader)

    def save_checkpoint(self, epoch: int, metrics: dict):
        self._trainer.save_checkpoint(epoch, metrics)
        self._logger.info(f"[Epoch {epoch:03d}] Checkpoint guardado")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        self._logger.info(f"Iniciando entrenamiento — {epochs} epochs")
        best_f1 = 0.0
        epoch_times: list[float] = []

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self._logger.info(f"[Epoch {epoch:03d}/{epochs}] Entrenando...")
            train_m = self.train_epoch(train_loader)

            self._logger.info(f"[Epoch {epoch:03d}/{epochs}] Evaluando...")
            val_m = self.eval_epoch(val_loader)
            epoch_times.append(time.time() - t0)

            if val_m["f1"] > best_f1:
                best_f1 = val_m["f1"]
                self.save_checkpoint(epoch, val_m)

            self._logger.info(
                f"[Epoch {epoch:03d}/{epochs}] "
                f"train_loss={train_m['loss']:.4f}  train_f1={train_m['f1']:.4f}  train_acc={train_m['accuracy']:.4f} | "
                f"val_loss={val_m['loss']:.4f}  val_f1={val_m['f1']:.4f}  best={best_f1:.4f}  val_acc={val_m['accuracy']:.4f} | "
                f"time={train_m['time']:.0f}s  ETA={_eta_str(epoch_times, epoch, epochs)}"
            )

        self._logger.info(f"Entrenamiento completado — mejor Val F1: {best_f1:.4f}")
