"""PerformancePredictor — empirical-prior Val-F1 estimate. Thin wrapper over
the single quality engine in src/performance_model.py."""
from __future__ import annotations

from src.performance_model import N_FULL_TRAIN, predict_quality
from src.feasibility.value_objects import PerformancePrediction


class PerformancePredictor:
    """Empirical-prior Val-F1 estimate — a thin wrapper over the single quality
    engine in ``src/performance_model.py`` (``predict_quality``).

    Until now this held a hardcoded per-family lookup that ignored the dataset
    size (it returned ~0.68 for vit_base whether the run used the full 237k
    images or the 5 000-image subset, where the real result was ~0.55). It now
    delegates to the data-scaling-aware quality model so the F1 estimate moves
    with the dataset, and so there is ONE prediction engine (time + memory +
    cost + quality) instead of two."""

    def predict(
        self,
        model_name: str,
        n_epochs: int,
        has_llrd: bool = True,
        has_label_smoothing: bool = True,
        dataset_size: int = N_FULL_TRAIN,
    ) -> PerformancePrediction:
        q = predict_quality(model_name, dataset_size=dataset_size, epochs=n_epochs)
        return PerformancePrediction(
            model_name=model_name,
            predicted_best_f1=q.expected_best_f1,
            predicted_best_epoch=q.best_epoch,
            predicted_early_stop_epoch=q.early_stop_epoch,
            confidence=q.confidence,
            curve_epochs=q.curve_epochs,
            curve_f1_train=q.curve_train_f1,
            curve_f1_val=q.curve_val_f1,
            notes=" | ".join(q.notes),
        )
