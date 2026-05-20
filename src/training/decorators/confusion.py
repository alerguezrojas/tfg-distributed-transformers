"""ConfusionMatrixDecorator — per-class metrics visualization for multi-label tasks.

For multi-label classification (19 classes) a classic N×N confusion matrix does not
apply. Instead this decorator generates two complementary visualizations:
  1. Horizontal bar chart: per-class F1 score (sorted descending)
  2. Grouped bar chart: per-class precision, recall and F1 side-by-side

Optionally writes a structured CSV alongside each PNG for interactive web display.

Trainer.eval_epoch must include '_preds' and '_labels' in its return dict
(which the base Trainer already does). This decorator extracts and removes
those tensors before returning, so upstream metric reporters receive a clean dict.
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
        if csv_dir is not None:
            self._csv_path = Path(csv_dir) / f"perclass_metrics_{timestamp}.csv"
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "class_name", "class_idx", "f1", "precision", "recall"])

    def eval_epoch(self, loader: DataLoader) -> dict:
        result = self._trainer.eval_epoch(loader)
        self._epoch += 1

        preds = result.pop("_preds", None)
        labels = result.pop("_labels", None)

        if preds is not None and labels is not None and self._epoch % self._every_n == 0:
            self._save_plot(preds, labels, self._epoch)
            if self._csv_path is not None:
                self._write_csv(preds, labels, self._epoch)

        return result

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

    def _save_plot(self, preds: torch.Tensor, labels: torch.Tensor, epoch: int):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        preds = preds.bool()
        labels = labels.bool()

        tp = (preds & labels).float().sum(0)
        fp = (preds & ~labels).float().sum(0)
        fn = (~preds & labels).float().sum(0)

        per_prec = (tp / (tp + fp + 1e-8)).numpy()
        per_rec  = (tp / (tp + fn + 1e-8)).numpy()
        per_f1   = (2 * per_prec * per_rec / (per_prec + per_rec + 1e-8))

        # Sort by F1 descending
        order = np.argsort(per_f1)[::-1]
        names = [CLASSES[i] for i in order]
        f1_s  = per_f1[order]
        prec_s = per_prec[order]
        rec_s  = per_rec[order]

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Left: F1 horizontal bars
        ax = axes[0]
        colors = ["steelblue" if v >= 0.5 else "salmon" for v in f1_s]
        y = np.arange(len(names))
        ax.barh(y, f1_s, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("F1 Score")
        ax.set_title(f"F1 por clase — epoch {epoch}")
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()

        # Right: grouped bars precision / recall / F1
        ax2 = axes[1]
        x = np.arange(len(names))
        w = 0.27
        ax2.bar(x - w, prec_s, w, label="Precision", color="steelblue", alpha=0.8)
        ax2.bar(x,      rec_s,  w, label="Recall",    color="darkorange", alpha=0.8)
        ax2.bar(x + w,  f1_s,   w, label="F1",        color="seagreen",   alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax2.set_ylabel("Score")
        ax2.set_title(f"Precision / Recall / F1 por clase — epoch {epoch}")
        ax2.legend()
        ax2.set_ylim(0, 1)
        ax2.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        path = self._output_dir / f"perclass_{self._timestamp}_epoch{epoch:03d}.png"
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
