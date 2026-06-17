"""TracingDecorator — epoch-level logging controller.

Works as console logger (logger=None) or structured file logger (logger=Logger).
Extends EpochController: only overrides the _on_* hooks, never the fit loop.
"""

import csv
import logging
import time
from pathlib import Path

from src.training.base_trainer import BaseTrainer
from src.training.metrics import eta_str
from src.training.decorators.base import EpochController


class TracingDecorator(EpochController):
    """Logs train/val metrics after each epoch to console and/or a file.

    Pass logger=None for console-only output (--trace off).
    Pass a Logger instance for structured file output (--trace simple).
    Pass epoch_csv=Path(...) to write per-epoch metrics to a structured CSV
    (consumed by the web dashboard instead of log parsing).

    Stack aspect decorators (PlottingDecorator, LayerHooksDecorator) between
    this controller and the Trainer — they are invoked transparently via the
    train_epoch / eval_epoch delegation chain.
    """

    _CSV_COLS = [
        "epoch", "train_loss", "val_loss", "train_f1", "val_f1",
        "train_acc", "val_acc", "val_prec", "val_rec",
        "epoch_time_s",
    ]

    def __init__(
        self,
        trainer: BaseTrainer,
        logger: logging.Logger | None = None,
        patience: int | None = None,
        epoch_csv: Path | None = None,
        select_metric: str = "f1",
    ):
        super().__init__(trainer, patience=patience, select_metric=select_metric)
        self._logger = logger
        self._epoch_csv = epoch_csv
        self._fit_start_time: float = 0.0

        if epoch_csv is not None:
            epoch_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(epoch_csv, "w", newline="") as f:
                csv.writer(f).writerow(self._CSV_COLS)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _emit(self, msg: str):
        # In DDP runs, rank is forwarded via __getattr__ from DDPTrainer.
        # Default to 0 (always emit) for single-GPU runs where rank is absent.
        if getattr(self, "rank", 0) != 0:
            return
        if self._logger:
            self._logger.info(msg)
        else:
            print(msg)

    # ── EpochController hooks ────────────────────────────────────────────────

    def _on_fit_start(self, epochs: int):
        self._fit_start_time = time.time()
        self._emit(f"Iniciando entrenamiento — {epochs} epochs")

    def _on_epoch_start(self, epoch: int, epochs: int):
        self._emit(f"\n── Epoch {epoch:03d}/{epochs:03d} " + "─" * 28)

    def _on_epoch_end(self, epoch, epochs, train_m, val_m, best_f1, epoch_times):
        opt_t = val_m.get("_optimal_threshold")
        opt_f1 = val_m.get("_f1_at_optimal_threshold")
        thresh_note = ""
        if opt_t is not None and opt_t != 0.5:
            thresh_note = f"  (threshold óptimo={opt_t:.2f}, F1={opt_f1:.4f})"
        self._emit(
            f"  ETA: {eta_str(epoch_times, epoch, epochs)}  "
            f"({train_m['time']:.0f}s/epoch, best_f1={best_f1:.4f})"
            + thresh_note
        )
        if self._epoch_csv is not None:
            row = {
                "epoch": epoch,
                "train_loss": round(train_m.get("loss", float("nan")), 6),
                "val_loss": round(val_m.get("loss", float("nan")), 6),
                "train_f1": round(train_m.get("f1", float("nan")), 6),
                "val_f1": round(val_m.get("f1", float("nan")), 6),
                "train_acc": round(train_m.get("accuracy", float("nan")), 6),
                "val_acc": round(val_m.get("accuracy", float("nan")), 6),
                "val_prec": round(val_m.get("precision", float("nan")), 6),
                "val_rec": round(val_m.get("recall", float("nan")), 6),
                "epoch_time_s": round(epoch_times[-1], 2) if epoch_times else float("nan"),
            }
            with open(self._epoch_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self._CSV_COLS).writerow(row)

    def _on_fit_end(self, best_f1: float):
        self._emit(f"Entrenamiento completado — mejor Val F1: {best_f1:.4f}")

    def _on_early_stop(self, epoch: int, best_f1: float):
        self._emit(
            f"[Early stopping] Sin mejora en {self._patience} epochs consecutivos. "
            f"Parado en epoch {epoch}. Mejor Val F1: {best_f1:.4f}"
        )

    def save_checkpoint(self, epoch: int, metrics: dict):
        self._trainer.save_checkpoint(epoch, metrics)
        self._emit(f"[Epoch {epoch:03d}] Checkpoint guardado")
