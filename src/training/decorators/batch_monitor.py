"""BatchMonitorDecorator — monitorización a nivel de batch con exportación CSV.

Registra un callback en el sistema de hooks del Trainer en lugar de reimplementar
train_epoch, eliminando la duplicación del bucle.

Formato CSV: epoch, batch, n_batches, running_loss, batch_loss, lr
Salida: logs/{env}/{mode}/{model}/batch_metrics_{timestamp}.csv

Retrocompatibilidad: los CSVs generados antes de esta versión solo tienen
running_loss. batch_parser.py maneja ambos formatos.
"""

from pathlib import Path

from src.training.decorators.base import TrainerDecorator
from src.training.base_trainer import BaseTrainer


class BatchMonitorDecorator(TrainerDecorator):
    """Aspecto que registra métricas de entrenamiento cada N batches en un CSV.

    Activar con --layers batch-monitor.
    Compatible con todos los --trace modes y demás decoradores.

    Columnas del CSV:
        epoch        — número de epoch
        batch        — índice de batch dentro del epoch (1-based)
        n_batches    — total de batches en el epoch
        running_loss — loss media acumulada desde el inicio del epoch
        batch_loss   — loss instantánea de este batch específico
        lr           — learning rate actual (primer param group)
    """

    def __init__(
        self,
        trainer: BaseTrainer,
        log_every: int = 1,
        output_dir: str = "logs",
        timestamp: str = "",
    ):
        super().__init__(trainer)
        self._log_every = max(1, log_every)
        csv_name = f"batch_metrics_{timestamp}.csv" if timestamp else "batch_metrics.csv"
        self._csv_path = Path(output_dir) / csv_name
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._csv_path, "w") as f:
            f.write("epoch,batch,n_batches,running_loss,batch_loss,lr\n")

        self._register_hook()

    def _register_hook(self) -> None:
        """Recorre el stack de decoradores y registra el callback en el Trainer central."""
        inner = self._trainer
        while hasattr(inner, "_trainer"):
            inner = inner._trainer
        if hasattr(inner, "register_batch_hook"):
            inner.register_batch_hook(self._on_batch)

    def _on_batch(
        self,
        epoch: int,
        batch_idx: int,
        n_batches: int,
        running_loss: float,
        batch_loss: float = 0.0,
        lr: float = 0.0,
    ) -> None:
        if batch_idx % self._log_every == 0 or batch_idx == n_batches:
            with open(self._csv_path, "a") as f:
                f.write(
                    f"{epoch},{batch_idx},{n_batches},"
                    f"{running_loss:.6f},{batch_loss:.6f},{lr:.8f}\n"
                )

    @property
    def batch_csv_path(self) -> Path:
        return self._csv_path
