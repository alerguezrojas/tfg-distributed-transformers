"""LayerStatsMonitor: a NaN/inf gradient must not cost the `layers` event (review of alerts)."""
import torch
import torch.nn as nn

from src.training.monitor import LayerStatsMonitor


class Sink:
    def __init__(self):
        self.events = []

    def emit(self, event_type, **fields):
        self.events.append((event_type, fields))


class Holder:
    def __init__(self):
        self.model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)


def _backward(holder):
    holder.optimizer.zero_grad()
    holder.model(torch.randn(3, 4)).sum().backward()


def test_non_finite_gradients_keep_the_layers_event_and_skip_the_histogram():
    holder, sink = Holder(), Sink()
    mon = LayerStatsMonitor(holder, sink, log_every=1)
    mon.on_epoch_start(1, 1)
    _backward(holder)
    holder.model[0].weight.grad[0, 0] = float("nan")
    mon.on_batch(1, 2, 2, {})
    kinds = [t for t, _ in sink.events]
    assert kinds == ["layers"]                                  # emitted before the histogram
    mon.on_epoch_end(1, 1, {}, {}, 0.0, False, 0.0)
    hists = [(f["kind"], [g["name"] for g in f["groups"]]) for t, f in sink.events
             if t == "histograms"]
    assert hists == [("weights", ["0", "1"]), ("gradients", ["1"])]   # NaN group skipped


def test_gradient_histogram_cover_the_last_batch_only():
    holder, sink = Holder(), Sink()
    mon = LayerStatsMonitor(holder, sink, log_every=1)
    mon.on_epoch_start(1, 1)
    _backward(holder)
    mon.on_batch(1, 1, 2, {})                                    # not the last batch
    assert mon._grad_hists is None
    _backward(holder)
    mon.on_batch(1, 2, 2, {})
    assert {g["name"] for g in mon._grad_hists} == {"0", "1"}

