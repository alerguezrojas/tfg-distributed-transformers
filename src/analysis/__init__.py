"""Pure-NumPy analysis of the inspection data (see ``docs/inspeccion.md``).

No torch, no Streamlit, no GPU, no I/O beyond ``io``: the web reads the ``.npz`` files with
``io`` and calls these functions on CPU.
"""
from src.analysis.calibration import calibration
from src.analysis.confusion import confusion
from src.analysis.curves import per_class_curves, pr_curve, roc_curve
from src.analysis.dynamics import (cartography, flips, prob_of_true, sample_correct,
                                   worst_errors)
from src.analysis.io import (EpochData, MonitorSet, list_epochs, load_epoch,
                             load_monitor_set)
from src.analysis.projection import project_2d
from src.analysis.thresholds import threshold_sweep

__all__ = [
    "EpochData", "MonitorSet", "calibration", "cartography", "confusion", "flips",
    "list_epochs", "load_epoch", "load_monitor_set", "per_class_curves", "pr_curve",
    "prob_of_true", "project_2d", "roc_curve", "sample_correct", "threshold_sweep",
    "worst_errors",
]
