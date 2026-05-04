import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.training.oop_decorators.base import TrainerDecorator


class LayerHooksDecorator(TrainerDecorator):
    """Captures mean activations of Linear layers via PyTorch forward hooks.

    Kept for didactic value: illustrates how hooks can observe a model
    without modifying it.
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
