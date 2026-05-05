"""PlottingDecorator — saves train/val curves to PNG after each epoch."""

from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.training.decorators.base import TrainerDecorator


class PlottingDecorator(TrainerDecorator):
    """Aspect decorator that accumulates metrics and saves a PNG plot after each epoch.

    Wraps train_epoch and eval_epoch. The plot is updated after every eval call
    so you can watch training progress in real time without opening TensorBoard.

    Stack this between the Trainer and the controller decorator:
        trainer = TracingDecorator(
            PlottingDecorator(Trainer(...), output_path="plots/run.png")
        )
    """

    def __init__(self, trainer: BaseTrainer, output_path: str = "plots/training.png"):
        super().__init__(trainer)
        self._history: dict[str, list[float]] = defaultdict(list)
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._epoch = 0

    def train_epoch(self, loader: DataLoader) -> dict:
        result = self._trainer.train_epoch(loader)
        for k, v in result.items():
            if isinstance(v, (int, float)):
                self._history[f"train_{k}"].append(v)
        return result

    def eval_epoch(self, loader: DataLoader) -> dict:
        result = self._trainer.eval_epoch(loader)
        self._epoch += 1
        for k, v in result.items():
            if isinstance(v, (int, float)):
                self._history[f"val_{k}"].append(v)
        self._save_plot()
        return result

    def _save_plot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = range(1, self._epoch + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        self._subplot(axes[0], epochs, "loss", "Loss")
        self._subplot(axes[1], epochs, "f1", "F1 Score (macro)")

        fig.suptitle(f"Entrenamiento — epoch {self._epoch}", fontsize=11)
        plt.tight_layout()
        plt.savefig(self._output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _subplot(self, ax, epochs, key: str, title: str):
        train_vals = self._history.get(f"train_{key}", [])
        val_vals = self._history.get(f"val_{key}", [])
        n = len(train_vals)
        if train_vals:
            ax.plot(range(1, n + 1), train_vals, label="train", color="steelblue")
        if val_vals:
            ax.plot(epochs, val_vals, label="val", color="darkorange")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
