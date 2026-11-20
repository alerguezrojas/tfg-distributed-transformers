"""What the web can show about a run's dataset from ``logs/`` alone.

For BigEarthNet the web has its own story (``dataset_stats``, needs the dataset
mounted). For any other dataset it only uses what the run left in its folder:

* the run manifest (name, task, classes, train/val sizes, providers),
* the inspection files ``inspect_<ts>/`` written by the monitor: the validation targets
  of the last epoch (``epoch_NNN.npz``, the class distribution of the whole validation
  set) and the fixed panel of the monitor set with its thumbnails (examples).

Nothing is invented: when a run has no inspection directory there is no distribution and
no thumbnails, and the caller says so. Pure (NumPy): no Streamlit, no torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.analysis.io import list_epochs, load_epoch, load_monitor_set


def inspect_dir_of(run) -> Path | None:
    """``inspect_<ts>/`` next to the run's log, if the run was monitored and it is here."""
    d = Path(run.log_path).parent / f"inspect_{run.timestamp}"
    return d if (d / "monitor_set.npz").is_file() else None


def val_class_counts(inspect_dir: Path | None) -> tuple[list[str], np.ndarray] | None:
    """(class names, validation samples per class) from the last inspected epoch, or None.

    Counts each (sample, class) positive, so multi-label datasets add up to more than the
    number of samples.
    """
    if inspect_dir is None:
        return None
    try:
        classes = list(load_monitor_set(inspect_dir).classes)
        for epoch in reversed(list_epochs(inspect_dir)):
            targets = load_epoch(inspect_dir, epoch).val_targets
            if targets is not None and targets.shape[1] == len(classes):
                return classes, np.asarray(targets).sum(axis=0).astype(int)
    except Exception:       # half-written or unreadable files: the view just does not appear
        return None
    return None


@dataclass(frozen=True)
class Example:
    thumb: np.ndarray       # (t, t, 3) uint8
    class_name: str         # dominant label of the sample


def panel_examples(inspect_dir: Path | None, limit: int = 10) -> list[Example]:
    """Thumbnails of the monitor's fixed panel (it cycles through the classes), or []."""
    if inspect_dir is None:
        return []
    try:
        mon = load_monitor_set(inspect_dir)
        if mon.thumbs is None:
            return []
        positions = mon.panel if mon.panel is not None and len(mon.panel) else \
            np.arange(min(limit, len(mon.thumbs)))
        return [Example(np.asarray(mon.thumbs[int(p)]),
                        mon.classes[int(np.argmax(mon.targets[int(p)]))])
                for p in list(positions)[:limit]]
    except Exception:
        return []
