"""StepTimer con un reloj simulado: reparto exacto entre espera de datos y paso."""
from types import SimpleNamespace

import pytest

from src.training import throughput
from src.training.throughput import StepTimer


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(throughput, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    return now


def _iteration(timer, clock, wait, step, hooks, n=8):
    clock[0] += wait
    timer.batch_ready()
    clock[0] += step
    out = timer.step_done(n)
    clock[0] += hooks
    timer.iteration_end()
    return out


def test_a_batch_splits_wait_and_step(clock):
    t = StepTimer()
    t.start()
    out = _iteration(t, clock, wait=0.25, step=0.75, hooks=0.0)
    assert out["data_time_s"] == pytest.approx(0.25)
    assert out["step_time_s"] == pytest.approx(0.75)
    assert out["samples_per_s"] == pytest.approx(8.0)          # 8 muestras en 1 s


def test_hooks_between_iterations_are_not_data_wait(clock):
    t = StepTimer()
    t.start()
    _iteration(t, clock, 0.1, 0.4, hooks=10.0)                 # un hook muy lento
    out = _iteration(t, clock, 0.1, 0.4, hooks=0.0)
    assert out["data_time_s"] == pytest.approx(0.1)


def test_epoch_stats_aggregate_all_batches(clock):
    t = StepTimer()
    t.start()
    _iteration(t, clock, 1.0, 3.0, 0.0, n=10)
    _iteration(t, clock, 1.0, 3.0, 0.0, n=6)
    stats = t.epoch_stats()
    assert stats["data_wait_frac"] == pytest.approx(0.25)
    assert stats["samples_per_s"] == pytest.approx(16 / 8.0)


def test_first_wait_counts_from_start(clock):
    t = StepTimer()
    clock[0] += 50.0                                           # tiempo antes de empezar: no cuenta
    t.start()
    out = _iteration(t, clock, 0.5, 0.5, 0.0)
    assert out["data_time_s"] == pytest.approx(0.5)


def test_empty_epoch_has_defined_stats(clock):
    assert StepTimer().epoch_stats() == {"samples_per_s": 0.0, "data_wait_frac": 0.0}


def test_zero_time_batch_does_not_divide_by_zero(clock):
    t = StepTimer()
    t.start()
    out = _iteration(t, clock, 0.0, 0.0, 0.0)
    assert out["samples_per_s"] == 0.0
