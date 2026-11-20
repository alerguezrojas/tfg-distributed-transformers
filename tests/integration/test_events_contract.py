"""Contract: what the trainer's JsonlEventSink writes, the web's EventsFollower reads."""
import pytest

from src.training.events import JsonlEventSink
from src.web.events_reader import COMPLETED, FAILED, INTERRUPTED, RUNNING, EventsFollower


def _run(path, end):
    sink = JsonlEventSink(path, run_info={"run_id": "t", "dataset": "toy", "task": "binary",
                                          "classes": ["tumour"], "select_metric": "auc"},
                          log_every=2, heartbeat_s=0)
    sink.on_fit_start(3)
    for epoch in (1, 2):
        sink.on_epoch_start(epoch, 3)
        for b in (1, 2, 3):
            sink.on_batch(epoch, b, 3, {"batch_loss": 1.0 / b, "running_loss": 0.9, "lr": 1e-4})
        sink.on_epoch_end(epoch, 3, {"loss": 0.5, "time": 9.0},
                          {"loss": 0.4, "f1": 0.5, "auc": 0.8 + epoch / 100,
                           "_optimal_threshold": 0.3, "_probs": None},
                          0.8 + epoch / 100, True, 10.0)
    end(sink)
    return sink


def test_a_finished_run_reads_back_completely(tmp_path):
    path = tmp_path / "events_t.jsonl"
    _run(path, lambda s: s.on_fit_end(0.82, 2, False))
    f = EventsFollower(path)
    f.poll()
    st = f.state
    assert st.n_ignored == 0
    assert st.status() == COMPLETED
    assert (st.epochs_done, st.planned_epochs, st.select_metric) == (2, 3, "auc")
    assert st.best == pytest.approx(0.82)
    assert st.last_val["optimal_threshold"] == pytest.approx(0.3)
    assert [r["batch"] for r in st.batch_rows()] == [2, 3, 2, 3]


@pytest.mark.parametrize("error, status", [(RuntimeError("OOM"), FAILED),
                                           (KeyboardInterrupt(), INTERRUPTED)])
def test_a_failed_or_interrupted_run_is_final(tmp_path, error, status):
    path = tmp_path / "events_t.jsonl"
    _run(path, lambda s: s.on_fit_error(error, 2, 0.82))
    f = EventsFollower(path)
    f.poll()
    assert f.state.status() == status


def test_a_run_in_progress_is_followed_incrementally(tmp_path):
    path = tmp_path / "events_t.jsonl"
    sink = JsonlEventSink(path, run_info={"run_id": "t"}, heartbeat_s=0)
    f = EventsFollower(path)
    sink.on_fit_start(2)
    sink.on_epoch_start(1, 2)
    f.poll()
    assert f.state.status() == RUNNING and f.state.epochs_done == 0
    sink.on_epoch_end(1, 2, {"loss": 1.0}, {"loss": 1.0, "f1": 0.1}, 0.1, True, 5.0)
    assert f.poll() == 1 and f.state.epochs_done == 1
