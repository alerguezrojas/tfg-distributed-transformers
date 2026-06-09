"""Selectable speedup models for the feasibility predictor.

The feasibility checker computes a single speedup estimate with its own
compute/IO-aware model. For the dashboard we expose several *analytic* scaling
laws so the user can compare them against that estimate and against the real
measurements, and pick the one that best matches their setup.

Every model is a pure function ``speedup(n_gpus, serial_fraction) -> float`` with
a human-readable name and formula, so they are unit-testable without any GPU.

Models
------
- **linear**     ideal scaling, ``S(n) = n`` (upper bound, never reached).
- **amdahl**     fixed problem size: ``S(n) = 1 / (s + (1 - s)/n)`` where ``s`` is
                 the serial (non-parallelizable) fraction. Plateaus at ``1/s``.
- **gustafson**  scaled problem size: ``S(n) = n - s*(n - 1)``. Optimistic;
                 grows ~linearly because the workload grows with ``n``.

The feasibility's own curve (compute/IO-aware, including the NFS bottleneck and
gradient-sync overhead) is read from the report CSV and shown alongside these.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


def linear(n_gpus: int, serial_fraction: float = 0.0) -> float:
    """Ideal linear scaling — the theoretical upper bound."""
    return float(n_gpus)


def amdahl(n_gpus: int, serial_fraction: float = 0.05) -> float:
    """Amdahl's law — fixed problem size, bounded by the serial fraction."""
    s = max(0.0, min(1.0, serial_fraction))
    if n_gpus <= 0:
        return 0.0
    return 1.0 / (s + (1.0 - s) / n_gpus)


def gustafson(n_gpus: int, serial_fraction: float = 0.05) -> float:
    """Gustafson's law — scaled problem size, near-linear growth."""
    s = max(0.0, min(1.0, serial_fraction))
    return n_gpus - s * (n_gpus - 1)


@dataclass(frozen=True)
class SpeedupModel:
    key: str
    name: str
    formula: str
    description: str
    fn: Callable[..., float]

    def speedup(self, n_gpus: int, serial_fraction: float = 0.05) -> float:
        return self.fn(n_gpus, serial_fraction)


SPEEDUP_MODELS: dict[str, SpeedupModel] = {
    "linear": SpeedupModel(
        key="linear",
        name="Linear (ideal)",
        formula="S(n) = n",
        description="Perfect scaling: every GPU adds full throughput. Upper bound, never reached in practice.",
        fn=linear,
    ),
    "amdahl": SpeedupModel(
        key="amdahl",
        name="Amdahl's law",
        formula="S(n) = 1 / (s + (1 - s)/n)",
        description="Fixed problem size. A serial fraction s caps the speedup at 1/s no matter how many GPUs.",
        fn=amdahl,
    ),
    "gustafson": SpeedupModel(
        key="gustafson",
        name="Gustafson's law",
        formula="S(n) = n - s·(n - 1)",
        description="Scaled problem size (bigger global batch). Grows almost linearly; optimistic bound.",
        fn=gustafson,
    ),
}


def speedup_curve(model_key: str, n_gpus_list: list[int], serial_fraction: float = 0.05) -> list[float]:
    """Speedup values for a model over a list of GPU counts."""
    model = SPEEDUP_MODELS[model_key]
    return [model.speedup(int(n), serial_fraction) for n in n_gpus_list]
