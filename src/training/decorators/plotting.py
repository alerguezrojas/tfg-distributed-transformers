"""PlottingDecorator — acumula en memoria el histórico de métricas train/val.

No escribe ningún fichero (ni PNGs ni CSVs): las curvas del dashboard web salen de
epoch_metrics_*.csv, que escribe el controlador de trazas. El histórico queda en
``_history`` (claves ``train_*`` / ``val_*``) y DeepTracingDecorator le propaga los
resultados de train cuando gestiona el bucle directamente.
"""

from collections import defaultdict

from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.training.decorators.base import TrainerDecorator


class PlottingDecorator(TrainerDecorator):
    """Aspecto: acumula en memoria las métricas numéricas de train y val.

    No escribe ficheros: el dashboard Streamlit dibuja las curvas a partir de
    epoch_metrics_*.csv.

    El parámetro output_path se mantiene por compatibilidad con el Builder
    pero no se usa.
    """

    def __init__(self, trainer: BaseTrainer, output_path: str = ""):
        super().__init__(trainer)
        self._history: dict[str, list[float]] = defaultdict(list)
        self._epoch = 0

    def train_epoch(self, loader: DataLoader) -> dict:
        result = self._trainer.train_epoch(loader)
        self._record_train_result(result)
        return result

    def _record_train_result(self, result: dict):
        """Registra métricas de train — llamado por DeepTracingDecorator
        cuando gestiona el bucle directamente."""
        for k, v in result.items():
            if isinstance(v, (int, float)):
                self._history[f"train_{k}"].append(v)

    def eval_epoch(self, loader: DataLoader) -> dict:
        result = self._trainer.eval_epoch(loader)
        self._epoch += 1
        for k, v in result.items():
            if isinstance(v, (int, float)):
                self._history[f"val_{k}"].append(v)
        return result
