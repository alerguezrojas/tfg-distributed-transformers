"""BatchMonitorDecorator — batch-level loss monitoring with CSV export.

Registers a lightweight callback on the inner Trainer's batch hook system
instead of reimplementing train_epoch. This eliminates the DRY violation
that previously existed when duplicating the full training loop.

CSV format: epoch, batch, n_batches, running_loss
Output: logs/{env}/{mode}/{model}/batch_metrics_{timestamp}.csv
"""

from pathlib import Path

from src.training.decorators.base import TrainerDecorator
from src.training.base_trainer import BaseTrainer


class BatchMonitorDecorator(TrainerDecorator):
    """Aspect decorator that logs running train loss every N batches to a CSV.

    Activates with --layers batch-monitor.
    Compatible with all --trace modes and all other aspect decorators.
    """

    def __init__(
        self,
        trainer: BaseTrainer,
        log_every: int = 50,
        output_dir: str = "logs",
        timestamp: str = "",
    ):
        super().__init__(trainer)
        self._log_every = log_every
        csv_name = f"batch_metrics_{timestamp}.csv" if timestamp else "batch_metrics.csv"
        self._csv_path = Path(output_dir) / csv_name
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._csv_path, "w") as f:
            f.write("epoch,batch,n_batches,running_loss\n")

        # Register hook on the innermost Trainer — no loop duplication needed
        self._register_hook()

    def _register_hook(self) -> None:
        """Walk the decorator stack and register our callback on the core Trainer."""
        inner = self._trainer
        while hasattr(inner, "_trainer"):
            inner = inner._trainer
        if hasattr(inner, "register_batch_hook"):
            inner.register_batch_hook(self._on_batch)

    def _on_batch(self, epoch: int, batch_idx: int, n_batches: int, running_loss: float) -> None:
        if batch_idx % self._log_every == 0:
            with open(self._csv_path, "a") as f:
                f.write(f"{epoch},{batch_idx},{n_batches},{running_loss:.6f}\n")

    @property
    def batch_csv_path(self) -> Path:
        return self._csv_path
