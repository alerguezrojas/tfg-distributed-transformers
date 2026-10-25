"""Alert rules (pure) and the AlertMonitor state machine (docs/inspeccion.md)."""
import math

import pytest

from src.training import alerts as A
from src.training.alerts import AlertMonitor
from src.training.config_validator import _ALERT_KEYS, ConfigError, validate_config
from src.training.events import JsonlEventSink
from tests.unit.test_config_validator import _base_cfg


class Sink:
    def __init__(self):
        self.events = []

    def emit(self, event_type, **fields):
        self.events.append((event_type, fields))

    def alerts(self):
        return [f for t, f in self.events if t == "alert"]


def _epoch(mon, epoch, train=1.0, val=1.0, **train_extra):
    mon.on_epoch_start(epoch, 10)
    mon.on_epoch_end(epoch, 10, {"loss": train, **train_extra}, {"loss": val}, 0.0, False, 1.0)


# ── Rules ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value, fires", [
    (float("nan"), True), (float("inf"), True), (-math.inf, True),
    (0.0, False), (3.2, False), (None, False)])
def test_nan_loss(value, fires):
    f = A.check_nan_loss(value)
    assert (f is not None) == fires
    if fires:
        assert f.level == "critical" and f.code == "nan_loss"


def test_loss_divergence_boundaries():
    assert A.check_loss_divergence([1.0]) is None                  # nothing to compare to
    assert A.check_loss_divergence([1.0, 2.0]) is None             # exactly 2x: not above
    assert A.check_loss_divergence([1.0, 2.01]).code == "loss_divergence"
    assert A.check_loss_divergence([1.0, 0.5, 1.01]).value == pytest.approx(1.01)  # best=0.5
    assert A.check_loss_divergence([0.0, 5.0]) is None             # best 0: undefined ratio
    assert A.check_loss_divergence([1.0, 3.0], factor=4) is None


def test_overfitting_needs_val_up_and_train_down_n_epochs():
    train = [1.0, 0.9, 0.8, 0.7]
    val = [0.5, 0.6, 0.7, 0.8]
    assert A.check_overfitting(train, val, 3).code == "overfitting"
    assert A.check_overfitting(train[:3], val[:3], 3) is None      # only 2 steps
    assert A.check_overfitting(train, [0.5, 0.6, 0.6, 0.8], 3) is None   # not strict
    assert A.check_overfitting([1.0, 0.9, 0.95, 0.7], val, 3) is None    # train not falling
    assert A.check_overfitting(train, val, 2).code == "overfitting"
    assert A.check_overfitting(train, [0.5, 0.6, float("nan"), 0.8], 3) is None


def test_grad_explosion():
    hist = [1.0] * 5
    assert A.check_grad_explosion("g", 100.0, hist) is None        # exactly 100x
    f = A.check_grad_explosion("g", 100.5, hist)
    assert f.code == "grad_explosion" and f.group == "g" and f.threshold == pytest.approx(100.0)
    assert A.check_grad_explosion("g", 1e9, [1.0] * 4) is None     # too little history
    assert A.check_grad_explosion("g", 1e9, [0.0] * 8) is None     # zero median
    assert A.check_grad_explosion("g", float("inf"), hist).value is None


def test_update_ratio_needs_persistence():
    ok, low, high = 1e-3, 1e-8, 0.5
    assert A.check_update_ratio("g", [low, low], patience=3) is None
    assert A.check_update_ratio("g", [low, ok, low], patience=3) is None
    assert A.check_update_ratio("g", [low, low, low], patience=3).threshold == 1e-6
    assert A.check_update_ratio("g", [ok, high, high, high], patience=3).threshold == 0.1
    assert A.check_update_ratio("g", [1e-6, 1e-1, 1e-6], patience=3) is None   # bounds inclusive
    assert A.check_update_ratio("g", [None, None, None]) is None


def test_dataloader_gpu_and_throughput_thresholds():
    assert A.check_dataloader_bound(0.3) is None and A.check_dataloader_bound(0.31)
    assert A.check_dataloader_bound(None) is None
    assert A.check_gpu_memory(0, 950, 1000) is None and A.check_gpu_memory(0, 951, 1000)
    assert A.check_gpu_memory(0, None, 1000) is None and A.check_gpu_memory(0, 1, 0) is None
    assert A.check_gpu_temp(1, 85) is None and A.check_gpu_temp(1, 86).group == "gpu1"
    assert A.check_gpu_temp(1, None) is None
    assert A.check_throughput_drop([100.0], 10.0) is None          # needs two epochs
    assert A.check_throughput_drop([100.0, 100.0], 50.0) is None   # exactly 50 %
    assert A.check_throughput_drop([100.0, 100.0], 49.9).code == "throughput_drop"
    assert A.check_throughput_drop([100.0, 100.0], None) is None


# ── Monitor ───────────────────────────────────────────────────────────────────

def test_epoch_alerts_fire_once_and_rearm():
    sink = Sink()
    mon = AlertMonitor(sink)
    _epoch(mon, 1, val=1.0)
    _epoch(mon, 2, val=2.5)                    # divergence
    _epoch(mon, 3, val=3.0)                    # still diverged: no repeat
    assert [a["code"] for a in sink.alerts()] == ["loss_divergence"]
    a = sink.alerts()[0]
    assert a["level"] == "warning" and a["epoch"] == 2 and a["batch"] is None
    assert a["value"] == pytest.approx(2.5) and a["threshold"] == pytest.approx(2.0)
    _epoch(mon, 4, val=1.1)                    # cleared: re-armed
    _epoch(mon, 5, val=2.4)
    assert [x["code"] for x in sink.alerts()] == ["loss_divergence"] * 2


def test_nan_loss_is_critical_from_epoch_and_batch():
    sink = Sink()
    mon = AlertMonitor(sink)
    mon.on_epoch_start(1, 3)
    mon.on_batch(1, 4, 10, {"batch_loss": float("nan")})
    mon.on_batch(1, 5, 10, {"batch_loss": float("nan")})        # same condition: once
    mon.on_epoch_end(1, 3, {"loss": float("nan")}, {"loss": 0.7}, 0.0, False, 1.0)
    (a,) = sink.alerts()
    assert a["level"] == "critical" and a["code"] == "nan_loss"
    assert a["epoch"] == 1 and a["batch"] == 4 and a["group"] == "train"
    _epoch(mon, 2)                                              # finite again: re-armed
    _epoch(mon, 3, train=float("inf"))
    assert len(sink.alerts()) == 2 and sink.alerts()[1]["batch"] is None


def test_overfitting_dataloader_and_throughput_from_epochs():
    sink = Sink()
    mon = AlertMonitor(sink)
    rows = [(1.0, 0.5, 100), (0.9, 0.6, 100), (0.8, 0.7, 100), (0.7, 0.8, 40)]
    for i, (t, v, sps) in enumerate(rows, 1):
        _epoch(mon, i, train=t, val=v, samples_per_s=sps,
               data_wait_frac=0.5 if i == 4 else 0.1)
    assert {a["code"] for a in sink.alerts()} == {"overfitting", "dataloader_bound",
                                                  "throughput_drop"}


def test_config_overrides_thresholds():
    sink = Sink()
    mon = AlertMonitor(sink, {"loss_divergence_factor": 10})
    _epoch(mon, 1, val=1.0)
    _epoch(mon, 2, val=5.0)
    assert not sink.alerts()


def test_system_alerts_per_gpu_and_rearm():
    sink = Sink()
    mon = AlertMonitor(sink)
    mon.on_epoch_start(2, 5)

    def sample(mem0, temp1):
        mon.on_event("system", {"gpus": [
            {"index": 0, "mem_used_mb": mem0, "mem_total_mb": 1000.0, "temp_c": 60},
            {"index": 1, "mem_used_mb": 10.0, "mem_total_mb": 1000.0, "temp_c": temp1}]})

    sample(990.0, 90)
    sample(990.0, 90)                          # no repeats
    assert sorted((a["code"], a["group"]) for a in sink.alerts()) == [
        ("gpu_memory_high", "gpu0"), ("gpu_temp_high", "gpu1")]
    assert all(a["epoch"] == 2 for a in sink.alerts())
    sample(100.0, 50)
    sample(990.0, 50)
    assert [a["code"] for a in sink.alerts()].count("gpu_memory_high") == 2
    mon.on_event("system", {"gpus": []})       # CPU-only: nothing, no crash


def _layers(norm, ratio=1e-3, epoch=1, batch=1):
    return {"epoch": epoch, "batch": batch, "groups": [
        {"name": "head", "grad_norm": norm, "weight_norm": 1.0, "update_ratio": ratio}]}


def test_layer_alerts_need_layer_events():
    sink = Sink()
    mon = AlertMonitor(sink)
    _epoch(mon, 1)                               # no `layers` events: rules never evaluated
    assert not sink.alerts()
    for i in range(6):
        mon.on_event("layers", _layers(1.0 + 0.01 * i, batch=i))
    mon.on_event("layers", _layers(500.0, batch=7))
    mon.on_event("layers", _layers(600.0, batch=8))              # still exploded: once
    (a,) = sink.alerts()
    assert a["code"] == "grad_explosion" and a["group"] == "head" and a["batch"] == 7
    for i in range(3):
        mon.on_event("layers", _layers(1.0, ratio=1e-9, batch=10 + i))
    assert [x["code"] for x in sink.alerts()] == ["grad_explosion", "update_ratio_out_of_range"]


def test_a_failing_sink_only_warns():
    class Broken(Sink):
        def emit(self, *a, **k):
            raise OSError("disk full")

    mon = AlertMonitor(Broken())
    with pytest.warns(RuntimeWarning, match="alert monitor"):
        _epoch(mon, 1, train=float("nan"))
    mon.on_batch(1, 1, 5, {"batch_loss": float("nan")})          # silent after the first
    mon.on_event("system", {"gpus": "bad"})


def test_sink_subscribers_see_emitted_events_only(tmp_path):
    sink = JsonlEventSink(tmp_path / "e.jsonl", {})
    seen = []
    sink.subscribe(lambda t, f: seen.append(t))
    sink.subscribe(lambda t, f: 1 / 0)                           # a broken observer
    sink.emit("system", gpus=[])
    sink.on_epoch_start(1, 1)                                    # lifecycle: not delivered
    assert seen == ["system"]


# ── Config ────────────────────────────────────────────────────────────────────

def test_alerts_config_defaults_and_validation():
    assert A.alerts_config({})["enabled"] is True
    assert A.alerts_config({"monitor": {"alerts": {"gpu_temp_c": 80.0}}})["gpu_temp_c"] == 80.0
    assert set(_ALERT_KEYS) == set(A.ALERT_DEFAULTS)
    cfg = _base_cfg()
    cfg["monitor"] = {"alerts": {"enabled": False, "gpu_temp_c": 80, "overfit_epochs": 4}}
    validate_config(cfg)


@pytest.mark.parametrize("alerts, message", [
    ({"gpu_temp": 80}, "not a known key"),
    ({"enabled": 1}, "must be bool"),
    ({"overfit_epochs": 2.5}, "must be int"),
    ({"data_wait_frac": -0.1}, "must be >= 0"),
    ([1], "mapping"),
])
def test_invalid_alerts_config(alerts, message):
    cfg = _base_cfg()
    cfg["monitor"] = {"alerts": alerts}
    with pytest.raises(ConfigError, match=message):
        validate_config(cfg)
