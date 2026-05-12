"""LayerHooksDecorator — captures Linear layer activations via forward hooks."""

import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.training.decorators.base import TrainerDecorator


class LayerHooksDecorator(TrainerDecorator):
    """Aspect decorator that prints mean activations of Linear layers every N epochs.

    Registers forward hooks before each training epoch and removes them after.
    Hooks observe the model without modifying it.

    Stack this between the Trainer and the controller decorator:
        trainer = TracingDecorator(
            LayerHooksDecorator(Trainer(...), log_every_n_epochs=5)
        )
    """

    def __init__(self, trainer: BaseTrainer, log_every_n_epochs: int = 5):
        super().__init__(trainer)
        self.log_every_n_epochs = log_every_n_epochs
        self._hooks: list = []
        self._activations: dict[str, float] = {}
        self._epoch = 0

    def train_epoch(self, loader: DataLoader) -> dict:
        self._epoch += 1
        self._register()
        self._activations.clear()
        try:
            result = self._trainer.train_epoch(loader)
        finally:
            # always remove hooks even if train_epoch raises — leaked hooks accumulate memory
            self._remove()
        if self._epoch % self.log_every_n_epochs == 0:
            self._print()
        return result

    def _register(self):
        for name, module in self._trainer.model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)

    def _make_hook(self, name: str):
        # Returns a closure so each hook captures its own `name` by value,
        # not the loop variable (which would be the same for all hooks otherwise)
        def hook(_m, _i, output):
            self._activations[name] = output.detach().abs().mean().item()
        return hook

    def _remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _print(self):
        if not self._activations:
            return
        print(f"\n[hooks] Activaciones medias — epoch {self._epoch:03d}:")
        for name, val in list(self._activations.items())[:5]:
            print(f"  {name:<50} {val:.4f}  {'█' * int(val * 20)}")
