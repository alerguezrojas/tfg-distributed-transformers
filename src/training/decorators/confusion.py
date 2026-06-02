"""ConfusionMatrixDecorator — métricas por clase para clasificación multi-label.

Genera por epoch dos CSVs (sin PNGs — el dashboard web genera las gráficas de forma
interactiva con Plotly desde estos CSVs):
  - perclass_metrics_TIMESTAMP.csv  — F1, precision, recall por clase
  - confusion_matrix_TIMESTAMP.csv  — matriz 19×19 normalizada

CSV columns: epoch, true_class, pred_class, value
Celda (i, j) = P(predice j | verdadero es i). Diagonal = recall por clase.

Trainer.eval_epoch debe incluir '_preds' y '_labels' en su dict de retorno.
"""

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.training.decorators.base import TrainerDecorator
from src.data.dataset import CLASSES


class ConfusionMatrixDecorator(TrainerDecorator):
    """Aspect decorator that saves per-class metric plots after each eval epoch.

    Activates with --layers confusion. Compatible with all trace modes and other
    aspect decorators. Positions between Trainer and metric reporters in the stack.

    The plot is saved as '{output_dir}/perclass_{timestamp}_epoch{N:03d}.png'.
    If csv_dir is provided, a row is appended to
    '{csv_dir}/perclass_metrics_{timestamp}.csv' after each epoch.
    """

    def __init__(
        self,
        trainer,
        output_dir: str = "plots",
        timestamp: str = "",
        every_n_epochs: int = 1,
        csv_dir: str | None = None,
    ):
        super().__init__(trainer)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._timestamp = timestamp
        self._every_n = every_n_epochs
        self._epoch = 0

        self._csv_path: Path | None = None
        self._confusion_csv_path: Path | None = None
        if csv_dir is not None:
            csv_base = Path(csv_dir)
            csv_base.mkdir(parents=True, exist_ok=True)
            self._csv_path = csv_base / f"perclass_metrics_{timestamp}.csv"
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "class_name", "class_idx", "f1", "precision", "recall"])
            self._confusion_csv_path = csv_base / f"confusion_matrix_{timestamp}.csv"
            with open(self._confusion_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "true_class", "pred_class", "value"])

    def eval_epoch(self, loader: DataLoader) -> dict:
        result = self._trainer.eval_epoch(loader)
        self._epoch += 1

        preds = result.pop("_preds", None)
        labels = result.pop("_labels", None)

        if preds is not None and labels is not None and self._epoch % self._every_n == 0:
            if self._csv_path is not None:
                self._write_csv(preds, labels, self._epoch)
            if self._confusion_csv_path is not None:
                self._write_confusion_matrix_csv(preds, labels, self._epoch)

        return result

    def _write_confusion_matrix_csv(self, preds: torch.Tensor, labels: torch.Tensor, epoch: int):
        import numpy as np
        preds_b = preds.bool().numpy().astype(float)
        labels_b = labels.bool().numpy().astype(float)
        co = labels_b.T @ preds_b
        row_sums = labels_b.sum(axis=0)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        matrix = co / row_sums[:, None]
        with open(self._confusion_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for i, true_name in enumerate(CLASSES):
                for j, pred_name in enumerate(CLASSES):
                    writer.writerow([epoch, true_name, pred_name, round(float(matrix[i, j]), 6)])

    def _write_csv(self, preds: torch.Tensor, labels: torch.Tensor, epoch: int):
        preds = preds.bool()
        labels = labels.bool()
        tp = (preds & labels).float().sum(0)
        fp = (preds & ~labels).float().sum(0)
        fn = (~preds & labels).float().sum(0)
        per_prec = (tp / (tp + fp + 1e-8)).tolist()
        per_rec = (tp / (tp + fn + 1e-8)).tolist()
        per_f1 = [2 * p * r / (p + r + 1e-8) for p, r in zip(per_prec, per_rec)]
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for i, name in enumerate(CLASSES):
                writer.writerow([epoch, name, i,
                                  round(per_f1[i], 6),
                                  round(per_prec[i], 6),
                                  round(per_rec[i], 6)])

