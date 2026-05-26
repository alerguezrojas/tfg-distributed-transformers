"""ConfusionMatrixDecorator — per-class metrics visualization for multi-label tasks.

Generates per epoch:
  - perclass_TIMESTAMP_epochNNN.png      — bar charts: F1, precision, recall per class
  - confusion_matrix_TIMESTAMP_epochNNN.png — 19×19 normalized heatmap (static, for reports)
  - confusion_matrix_TIMESTAMP.csv       — 19×19 matrix data for interactive web display

CSV columns: epoch, true_class, pred_class, value
Cell (i, j) = P(predict j | true is i). Diagonal = per-class recall.

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
            self._save_plot(preds, labels, self._epoch)
            self._save_confusion_heatmap(preds, labels, self._epoch)
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

    def _save_confusion_heatmap(self, preds: torch.Tensor, labels: torch.Tensor, epoch: int):
        """Save a 19×19 normalized confusion heatmap.

        Cell (i, j) = P(model predicts class j | true label is class i).
        Diagonal = per-class recall. Off-diagonal = inter-class confusion.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        preds_b = preds.bool().numpy().astype(float)   # (N, 19)
        labels_b = labels.bool().numpy().astype(float) # (N, 19)

        n_classes = labels_b.shape[1]
        # matrix[i, j] = sum over samples of (true==i AND pred==j) / sum(true==i)
        co = labels_b.T @ preds_b          # (19, 19): raw co-occurrence counts
        row_sums = labels_b.sum(axis=0)    # (19,): how many times each class is true
        row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid /0
        matrix = co / row_sums[:, None]    # normalize by row (true class frequency)

        short_names = [c.replace("_", " ").replace(" and ", "/").replace(" coniferous", " con.").replace(" broadleaf", " brd.").replace(" transitional", " trans.") for c in CLASSES]

        fig, ax = plt.subplots(figsize=(13, 11))
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="P(pred j | true i)")

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short_names, fontsize=7)
        ax.set_xlabel("Clase predicha", fontsize=9)
        ax.set_ylabel("Clase verdadera", fontsize=9)
        ax.set_title(f"Matriz de confusión normalizada — epoch {epoch}\n"
                     f"(diagonal = recall por clase, fuera de diagonal = confusiones)", fontsize=10)

        # Annotate cells with value if they exceed a threshold (avoid clutter)
        for i in range(n_classes):
            for j in range(n_classes):
                val = matrix[i, j]
                if val >= 0.1:
                    color = "white" if val > 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=5.5, color=color)

        fig.tight_layout()
        path = self._output_dir / f"confusion_matrix_{self._timestamp}_epoch{epoch:03d}.png"
        plt.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
