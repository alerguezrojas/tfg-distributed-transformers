"""Pure-NumPy analysis of the inspection data (see ``docs/inspeccion.md``).

No torch, no Streamlit, no GPU, no I/O beyond ``io``: the web reads the ``.npz`` files with
``io`` and calls these functions on CPU.
"""
from src.analysis.bootstrap import bootstrap_ci, metrics_with_ci
from src.analysis.calibration import (adaptive_ece, calibration, calibration_curve_stats,
                                      per_class_calibration)
from src.analysis.confusion import confusion
from src.analysis.curves import per_class_curves, pr_curve, roc_curve
from src.analysis.dynamics import (cartography, flips, prob_of_true, sample_correct,
                                   worst_errors)
from src.analysis.io import (EpochData, MonitorSet, list_epochs, load_epoch,
                             load_monitor_set)
from src.analysis.operating_point import (operating_point, operating_points,
                                          predictive_values, threshold_for_sensitivity,
                                          youden_threshold)
from src.analysis.projection import project_2d
from src.analysis.subgroups import subgroup_metrics
from src.analysis.thresholds import threshold_sweep

__all__ = [
    "EpochData", "MonitorSet", "adaptive_ece", "bootstrap_ci", "calibration",
    "calibration_curve_stats", "cartography", "confusion", "flips", "list_epochs",
    "load_epoch", "load_monitor_set", "metrics_with_ci", "operating_point",
    "operating_points", "per_class_calibration", "per_class_curves", "pr_curve",
    "predictive_values", "prob_of_true", "project_2d", "roc_curve", "sample_correct",
    "subgroup_metrics", "threshold_for_sensitivity", "threshold_sweep", "worst_errors",
    "youden_threshold",
]
