"""Training events: the Observer hooks of EpochController and the JSONL sink."""
import json
import math

import pytest
import torch

from src.training.base_trainer import BaseTrainer
from src.training.decorators.base import EpochController
from src.training.events import (
    SCHEMA_VERSION,
    JsonlEventSink,
    TrainingListener,
    environment_fingerprint,
    public_metrics,
)


class _Scripted(BaseTrainer):
    """Scripted val F1 per epoch; optionally raises in the train step of one epoch."""

    def __init__(self, val_f1, fail_at=None):
        self.val_f1, self.fail_at, self._epoch = val_f1, fail_at, 0

    def train_epoch(self, loader):
        self._epoch += 1
        if self._epoch == self.fail_at:
            raise RuntimeError("CUDA out of memory")
        return {"loss": 1.0, "f1": 0.1, "time": 0.0}

    def eval_epoch(self, loader):
        return {"loss": 0.5, "f1": self.val_f1[self._epoch - 1], "_preds": torch.zeros(2, 2)}

    def save_checkpoint(self, epoch, metrics):
        pass

    def fit(self, *a):
        pass


class _Recorder(TrainingListener):
    def __init__(self):
        self.calls = []

    def on_fit_start(self, epochs):
        self.calls.append(("fit_start", epochs))

    def on_epoch_start(self, epoch, epochs):
        self.calls.append(("epoch_start", epoch))

    def on_epoch_end(self, epoch, epochs, train_m, val_m, best, is_best, epoch_time_s):
        self.calls.append(("epoch_end", epoch, best, is_best))

    def on_early_stop(self, epoch, best):
        self.calls.append(("early_stop", epoch, best))

    def on_fit_end(self, best, epochs_run, early_stopped):
        self.calls.append(("fit_end", best, epochs_run, early_stopped))

    def on_fit_error(self, error, epochs_run, best):
        self.calls.append(("fit_error", type(error).__name__, epochs_run, best))


def _fit(trainer, epochs, patience=None, listener=None):
    ctrl = EpochController(trainer, patience=patience)
    listener = listener or _Recorder()
    ctrl.add_listener(listener)
    ctrl.fit(None, None, epochs=epochs)
    return listener


# ── EpochController → listeners ───────────────────────────────────────────────

def test_listener_sees_the_whole_lifecycle_in_order():
    rec = _fit(_Scripted([0.5, 0.4, 0.6]), epochs=3)
    assert rec.calls == [
        ("fit_start", 3),
        ("epoch_start", 1), ("epoch_end", 1, 0.5, True),
        ("epoch_start", 2), ("epoch_end", 2, 0.5, False),
        ("epoch_start", 3), ("epoch_end", 3, 0.6, True),
        ("fit_end", 0.6, 3, False),
    ]


def test_early_stop_is_reported_and_ends_the_run():
    rec = _fit(_Scripted([0.5, 0.4, 0.3, 0.2]), epochs=4, patience=2)
    assert ("early_stop", 3, 0.5) in rec.calls
    assert rec.calls[-1] == ("fit_end", 0.5, 3, True)


def test_a_failing_run_is_reported_and_the_error_propagates():
    rec = _Recorder()
    with pytest.raises(RuntimeError, match="out of memory"):
        _fit(_Scripted([0.5, 0.6], fail_at=2), epochs=2, listener=rec)
    assert rec.calls[-1] == ("fit_error", "RuntimeError", 1, 0.5)


def test_a_broken_listener_never_stops_training():
    class Broken(TrainingListener):
        def on_epoch_end(self, *a, **k):
            raise OSError("disk full")

    trainer = _Scripted([0.5, 0.6])
    ctrl = EpochController(trainer)
    ctrl.add_listener(Broken())
    rec = _Recorder()
    ctrl.add_listener(rec)
    with pytest.warns(RuntimeWarning, match="disk full"):
        ctrl.fit(None, None, epochs=2)
    assert rec.calls[-1] == ("fit_end", 0.6, 2, False)


def test_without_listeners_the_loop_is_unchanged():
    ctrl = EpochController(_Scripted([0.5, 0.6]))
    ctrl.fit(None, None, epochs=2)        # no listener: nothing to notify, no error


# ── JSONL sink ────────────────────────────────────────────────────────────────

def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_sink_writes_one_valid_json_object_per_event(tmp_path):
    path = tmp_path / "events_t.jsonl"
    sink = JsonlEventSink(path, run_info={"run_id": "t", "dataset": "toy"})
    _fit(_Scripted([0.5, 0.6]), epochs=2, listener=sink)

    events = _read(path)
    assert [e["type"] for e in events] == [
        "run_start", "epoch_start", "epoch_end", "epoch_start", "epoch_end", "run_end"]
    assert all(e["v"] == SCHEMA_VERSION and isinstance(e["time"], float) for e in events)
    start, end = events[0], events[-1]
    assert start["run_id"] == "t" and start["dataset"] == "toy" and start["epochs"] == 2
    assert end["status"] == "completed" and end["epochs_run"] == 2 and end["best"] == 0.6
    assert end["total_time_s"] >= 0
    first = events[2]
    assert first["val"] == {"loss": 0.5, "f1": 0.5}          # private tensors stripped
    assert first["is_best"] is True and first["epoch_time_s"] >= 0


def test_sink_reports_failure_and_early_stop(tmp_path):
    path = tmp_path / "events_fail.jsonl"
    with pytest.raises(RuntimeError):
        _fit(_Scripted([0.5], fail_at=1), epochs=3,
             listener=JsonlEventSink(path, run_info={}))
    end = _read(path)[-1]
    assert end["status"] == "failed" and end["epochs_run"] == 0
    assert end["best"] is None                                 # no epoch finished
    assert end["error"] == "RuntimeError: CUDA out of memory"

    path = tmp_path / "events_stop.jsonl"
    _fit(_Scripted([0.5, 0.4, 0.3]), epochs=3, patience=1,
         listener=JsonlEventSink(path, run_info={}))
    types = [e["type"] for e in _read(path)]
    assert types[-2:] == ["early_stop", "run_end"]
    assert _read(path)[-1]["status"] == "early_stopped"


def test_public_metrics_are_plain_json():
    m = public_metrics({"loss": torch.tensor(0.25), "auc": float("nan"),
                        "_optimal_threshold": 0.35, "_probs": torch.zeros(4, 2),
                        "per_class": torch.ones(3)})
    assert m == {"loss": 0.25, "auc": None, "optimal_threshold": 0.35}
    json.dumps(m, allow_nan=False)


def test_batch_events_follow_the_log_cadence(tmp_path):
    path = tmp_path / "events_b.jsonl"
    sink = JsonlEventSink(path, run_info={}, log_every=4)
    for b in range(1, 11):
        sink.on_batch(1, b, 10, {"batch_loss": 1.0 / b, "lr": 1e-4})
    batches = [e["batch"] for e in _read(path)]
    assert batches == [4, 8, 10]                                # every 4th and the last


def test_sink_starts_a_fresh_file(tmp_path):
    path = tmp_path / "events_x.jsonl"
    path.write_text('{"stale": true}\n')
    JsonlEventSink(path, run_info={})
    assert path.read_text() == ""


def test_environment_fingerprint_has_the_schema_keys():
    env = environment_fingerprint()
    assert set(env) == {"host", "python", "torch", "timm", "git_commit", "devices"}
    assert env["devices"] and all(isinstance(d, str) for d in env["devices"])


def test_nan_best_is_never_written_as_invalid_json(tmp_path):
    path = tmp_path / "events_nan.jsonl"
    sink = JsonlEventSink(path, run_info={})
    sink.on_epoch_end(1, 1, {"loss": math.inf}, {"f1": float("nan")}, 0.0, True, 1.0)
    event = _read(path)[0]
    assert event["train"]["loss"] is None and event["val"]["f1"] is None


def test_a_failing_batch_write_is_only_a_warning(tmp_path):
    sink = JsonlEventSink(tmp_path / "events_w.jsonl", run_info={})

    def boom(*a, **k):
        raise OSError("No space left on device")

    sink._write = boom
    with pytest.warns(RuntimeWarning, match="No space left"):
        sink.on_batch(1, 1, 1, {"batch_loss": 1.0})
    sink.on_batch(1, 1, 1, {"batch_loss": 1.0})       # silenced after the first


def test_batch_events_use_the_loop_epoch(tmp_path):
    # After --resume the trainer's own counter continues from the checkpoint (e.g. 6)
    # while the loop restarts at 1: batch events must match epoch_start/epoch_end.
    path = tmp_path / "events_r.jsonl"
    sink = JsonlEventSink(path, run_info={})
    sink.on_epoch_start(1, 3)
    sink.on_batch(6, 1, 1, {"batch_loss": 1.0})
    assert _read(path)[-1]["epoch"] == 1


def test_heartbeats_while_the_run_is_alive(tmp_path):
    import time
    path = tmp_path / "events_h.jsonl"
    sink = JsonlEventSink(path, run_info={}, heartbeat_s=0.05)
    sink.on_fit_start(1)
    time.sleep(0.3)
    sink.on_fit_end(0.5, 1, False)
    n = len(_read(path))
    time.sleep(0.15)
    events = _read(path)
    assert len(events) == n                                    # stopped with the run
    beats = [e for e in events if e["type"] == "heartbeat"]
    assert beats and all(e["elapsed_s"] >= 0 for e in beats)
    assert events[-1]["type"] == "run_end"


def test_ctrl_c_is_reported_as_interrupted(tmp_path):
    path = tmp_path / "events_i.jsonl"
    sink = JsonlEventSink(path, run_info={}, heartbeat_s=0)
    sink.on_fit_start(2)
    sink.on_fit_error(KeyboardInterrupt(), 0, -1.0)
    assert _read(path)[-1]["status"] == "interrupted"


def test_the_optimal_threshold_f1_is_public():
    m = public_metrics({"f1": 0.4, "_optimal_threshold": 0.3, "_f1_at_optimal_threshold": 0.6})
    assert m == {"f1": 0.4, "optimal_threshold": 0.3, "f1_at_optimal_threshold": 0.6}


def test_emit_appends_a_custom_event(tmp_path):
    path = tmp_path / "events_e.jsonl"
    sink = JsonlEventSink(path, run_info={}, heartbeat_s=0)
    sink.emit("system", gpus=[], cpu_pct=12.5)
    event = _read(path)[0]
    assert event["type"] == "system" and event["cpu_pct"] == 12.5 and event["v"] == SCHEMA_VERSION
