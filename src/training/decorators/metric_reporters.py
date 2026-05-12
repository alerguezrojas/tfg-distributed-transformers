"""Individual metric reporter decorators.

Each reporter is an independent aspect decorator (TrainerDecorator subclass)
that intercepts train_epoch / eval_epoch, reads one specific metric from the
result dict, and emits it.  Reporters are freely composable — stack only the
ones you need between the Trainer and the controller.

How it works
------------
1. train_epoch: the reporter delegates to the inner trainer and caches the
   metric value for its own key.
2. eval_epoch: the reporter delegates, then prints "train=X  val=Y" using
   the cached train value and the freshly computed val value.
3. The controller's _on_epoch_end is left free for structural info (ETA, best).

Print order = innermost reporter first.  Build the stack in the order you
want the output to appear, from inner to outer:

    inner = LossReporter(Trainer(...))      # prints first (loss)
    inner = F1Reporter(inner)               # prints second (f1)
    inner = AccuracyReporter(inner)         # prints third (accuracy)
    inner = PrecisionRecallReporter(inner)  # prints last (precision/recall)
    trainer = TracingDecorator(inner)

Available reporters
-------------------
LossReporter            — train_loss / val_loss
F1Reporter              — train_f1   / val_f1
AccuracyReporter        — train_acc  / val_acc
PrecisionRecallReporter — val_precision / val_recall  (no train equivalent)
"""

import logging

from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.training.decorators.base import TrainerDecorator


class _MetricReporter(TrainerDecorator):
    """Base for all metric reporters: handles console vs. logger output."""

    def __init__(self, trainer: BaseTrainer, logger: logging.Logger | None = None):
        super().__init__(trainer)
        self._logger = logger

    def _emit(self, msg: str):
        if self._logger:
            self._logger.info(msg)
        else:
            print(msg)


class _TrainValReporter(_MetricReporter):
    """Caches one named metric from train_epoch and prints train + val after eval_epoch.

    Subclasses only need to set _key (dict key) and _label (display label).
    """

    _key: str    # key in the result dict, e.g. "loss", "f1"
    _label: str  # display label, e.g. "loss", "f1"

    def train_epoch(self, loader: DataLoader) -> dict:
        result = self._trainer.train_epoch(loader)
        self._cached_train = result.get(self._key)
        return result

    def eval_epoch(self, loader: DataLoader) -> dict:
        result = self._trainer.eval_epoch(loader)
        val = result.get(self._key)
        if val is None:
            return result
        cached = getattr(self, "_cached_train", None)
        train_str = f"train={cached:.4f}  " if cached is not None else ""
        self._emit(f"  {self._label:<14} {train_str}val={val:.4f}")
        return result


class LossReporter(_TrainValReporter):
    """Reports train and val loss after each epoch."""

    _key = "loss"
    _label = "loss"


class F1Reporter(_TrainValReporter):
    """Reports train and val macro F1 after each epoch."""

    _key = "f1"
    _label = "f1"


class AccuracyReporter(_TrainValReporter):
    """Reports train and val sample-averaged accuracy after each epoch."""

    _key = "accuracy"
    _label = "accuracy"


class PrecisionRecallReporter(_MetricReporter):
    """Reports val precision and recall after each epoch.

    Precision and recall are only available from eval_epoch (not train_epoch).
    """

    def eval_epoch(self, loader: DataLoader) -> dict:
        result = self._trainer.eval_epoch(loader)
        p = result.get("precision")
        r = result.get("recall")
        if p is not None:
            self._emit(f"  {'precision':<14} val={p:.4f}")
        if r is not None:
            self._emit(f"  {'recall':<14} val={r:.4f}")
        return result
