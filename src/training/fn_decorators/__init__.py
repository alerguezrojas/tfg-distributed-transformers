"""Python @ function decorator library for training cross-cutting concerns.

Each decorator is independent and does exactly one thing.
They can be freely combined by stacking or using compose().

    from src.training.fn_decorators import timed, measure_energy, log_call, compose, PlotMetrics

    # Stack manually:
    trainer.train_epoch = measure_energy(timed(trainer.train_epoch))

    # Or use compose (applied left to right, outermost first):
    trainer.train_epoch = compose(measure_energy, timed)(trainer.train_epoch)

    # Plotting (stateful — use the same instance for train and eval):
    plotter = PlotMetrics(output_path="plots/training.png")
    trainer.train_epoch = plotter.wrap(trainer.train_epoch, tag="train")
    trainer.eval_epoch  = plotter.wrap(trainer.eval_epoch,  tag="val")
"""

from src.training.fn_decorators.timing import timed
from src.training.fn_decorators.logging_ import log_call
from src.training.fn_decorators.resilience import retry_on_cuda_oom
from src.training.fn_decorators.energy import measure_energy
from src.training.fn_decorators.plotting import PlotMetrics


def compose(*decorators):
    """Apply decorators left to right (outermost first).

    compose(f, g)(fn)  ≡  f(g(fn))

    Example:
        compose(measure_energy, timed)(trainer.train_epoch)
        # execution order: measure_energy wraps timed wraps original
    """
    def apply(fn):
        for dec in reversed(decorators):
            fn = dec(fn)
        return fn
    return apply


__all__ = [
    "timed",
    "log_call",
    "retry_on_cuda_oom",
    "measure_energy",
    "PlotMetrics",
    "compose",
]
