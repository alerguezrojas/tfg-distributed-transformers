import functools
from collections import defaultdict
from pathlib import Path


class PlotMetrics:
    """Stateful decorator that accumulates metrics and plots training curves.

    Apply separately to train_epoch and eval_epoch using the same instance
    so both curves appear on the same plot. A new figure is saved to disk
    after every eval call.

    Example:
        plotter = PlotMetrics(output_path="plots/training.png")
        trainer.train_epoch = plotter.wrap(trainer.train_epoch, tag="train")
        trainer.eval_epoch  = plotter.wrap(trainer.eval_epoch,  tag="val")

    The plot shows two subplots: Loss and F1, each with train/val curves.
    It is saved as a PNG after every evaluation so you can watch progress live.
    """

    def __init__(self, output_path: str = "plots/training.png"):
        self._history: dict[str, list[float]] = defaultdict(list)
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._eval_calls = 0

    def wrap(self, fn, tag: str = ""):
        """Return a wrapped version of fn that records its metrics dict."""
        prefix = f"{tag}_" if tag else ""

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, (int, float)):
                        self._history[f"{prefix}{k}"].append(v)
                if tag == "val":
                    self._eval_calls += 1
                    self._save_plot()
            return result

        return wrapper

    def _save_plot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = range(1, self._eval_calls + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        self._plot_metric(axes[0], epochs, "loss", "Loss")
        self._plot_metric(axes[1], epochs, "f1", "F1 Score (macro)")

        fig.suptitle(f"Entrenamiento — epoch {self._eval_calls}", fontsize=12)
        plt.tight_layout()
        plt.savefig(self._output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _plot_metric(self, ax, epochs, key: str, title: str):
        train_key = f"train_{key}"
        val_key = f"val_{key}"

        if train_key in self._history:
            n = len(self._history[train_key])
            ax.plot(range(1, n + 1), self._history[train_key], label="train", color="steelblue")
        if val_key in self._history:
            ax.plot(epochs, self._history[val_key], label="val", color="darkorange")

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
