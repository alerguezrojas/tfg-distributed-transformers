"""Lector incremental del flujo de eventos (events_*.jsonl) que alimenta la pestaña Live."""
import json
import os
import shutil
from pathlib import Path

import pytest

from src.web.events_reader import (
    COMPLETED, EARLY_STOPPED, FAILED, RUNNING, STALLED, WAITING,
    EventsFollower, RunState, discover_event_files, has_run_end, parse_event,
)
from src.web.run_registry import discover_runs, timestamp_sort_key

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "events_17102026_120000.jsonl"


def _line(**kw) -> str:
    return json.dumps({"v": 1, "time": 1000.0, **kw}) + "\n"


def _epoch_end(epoch, epochs=5, t=40.0, best=0.7, **kw):
    return _line(type="epoch_end", epoch=epoch, epochs=epochs, train={"loss": 0.3},
                 val={"loss": 0.2, "f1": 0.4, "auc": 0.7}, epoch_time_s=t, best=best,
                 is_best=True, **kw)


# ── parse_event ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", ["", "{not json", "[1, 2]", '{"v": 1}', '{"type": 3}',
                                 '{"v": 2, "type": "batch"}', '{"v": true, "type": "batch"}'])
def test_parse_event_rejects_unusable_lines(raw):
    assert parse_event(raw) is None


def test_parse_event_accepts_v1_and_missing_version():
    assert parse_event('{"v": 1, "type": "batch"}')["type"] == "batch"
    assert parse_event(b'{"type": "batch"}')["type"] == "batch"


# ── Incremental reading ──────────────────────────────────────────────────────────

def test_follower_reads_fixture_and_folds_state():
    f = EventsFollower(FIXTURE)
    assert f.poll() == 15
    s = f.state
    assert s.run_start["dataset"] == "medmnist_chest"
    assert s.planned_epochs == 5 and s.epochs_done == 2 and s.current_epoch == 3
    assert len(s.batches) == 9
    assert s.best == pytest.approx(0.78) and s.select_metric == "auc"
    assert s.last_val["auc"] == pytest.approx(0.78)
    assert f.poll() == 0                          # nothing new → nothing re-read


def test_follower_only_reads_appended_bytes(tmp_path):
    p = tmp_path / "events_17102026_120000.jsonl"
    p.write_text(_line(type="run_start", epochs=3) + _epoch_end(1, epochs=3), encoding="utf-8")
    f = EventsFollower(p)
    assert f.poll() == 2
    offset = f.offset
    assert offset == p.stat().st_size
    with p.open("a", encoding="utf-8") as fh:
        fh.write(_epoch_end(2, epochs=3))
    assert f.poll() == 1
    assert f.offset > offset and f.state.epochs_done == 2


def test_half_written_last_line_is_retried_next_poll(tmp_path):
    p = tmp_path / "events_17102026_120000.jsonl"
    full = _epoch_end(2, epochs=3)
    p.write_text(_line(type="run_start", epochs=3) + full[:25], encoding="utf-8")
    f = EventsFollower(p)
    assert f.poll() == 1                          # only run_start; the stump is held back
    assert f.state.epochs_done == 0 and f.state.n_ignored == 0
    with p.open("a", encoding="utf-8") as fh:
        fh.write(full[25:])
    assert f.poll() == 1
    assert f.state.epochs_done == 1


def test_complete_last_line_without_newline_is_accepted_once(tmp_path):
    p = tmp_path / "events_17102026_120000.jsonl"
    p.write_text(_line(type="run_start", epochs=3) + _epoch_end(1, epochs=3).rstrip("\n"),
                 encoding="utf-8")
    f = EventsFollower(p)
    assert f.poll() == 2
    with p.open("a", encoding="utf-8") as fh:
        fh.write("\n" + _epoch_end(2, epochs=3))      # the pending newline arrives
    assert f.poll() == 1
    assert f.state.epochs_done == 2                   # epoch 1 was not counted twice


def test_corrupt_complete_line_is_skipped_and_counted(tmp_path):
    p = tmp_path / "events_17102026_120000.jsonl"
    p.write_text(_line(type="run_start", epochs=2) + "garbage{\n" + _epoch_end(1, epochs=2),
                 encoding="utf-8")
    f = EventsFollower(p)
    assert f.poll() == 2
    assert f.state.n_ignored == 1 and f.state.epochs_done == 1


def test_unknown_types_and_newer_versions_are_ignored(tmp_path):
    p = tmp_path / "events_17102026_120000.jsonl"
    p.write_text(
        _line(type="run_start", epochs=2)
        + _line(type="gpu_sample", util=0.9)                       # unknown type
        + json.dumps({"v": 2, "type": "epoch_end", "epoch": 9}) + "\n"   # newer schema
        + _epoch_end(1, epochs=2), encoding="utf-8")
    f = EventsFollower(p)
    f.poll()
    assert f.state.epochs_done == 1 and f.state.n_events == 2
    assert f.state.n_ignored == 2


def test_missing_file_and_truncation(tmp_path):
    p = tmp_path / "events_17102026_120000.jsonl"
    f = EventsFollower(p)
    assert f.poll() == 0                              # not created yet
    p.write_text(_line(type="run_start", epochs=2) + _epoch_end(1, epochs=2), encoding="utf-8")
    f.poll()
    assert f.state.epochs_done == 1
    p.write_text(_line(type="run_start", epochs=4), encoding="utf-8")   # file replaced (shorter)
    assert f.poll() == 1
    assert f.state.epochs_done == 0 and f.state.planned_epochs == 4


def test_batches_are_bounded(tmp_path):
    p = tmp_path / "events_17102026_120000.jsonl"
    from src.web import events_reader
    n = events_reader.MAX_BATCHES + 50
    p.write_text("".join(_line(type="batch", epoch=1, batch=i, n_batches=n, metrics={})
                         for i in range(n)), encoding="utf-8")
    f = EventsFollower(p)
    f.poll()
    assert len(f.state.batches) == events_reader.MAX_BATCHES
    assert f.state.batches[-1]["batch"] == n - 1


def test_repeated_epoch_end_keeps_the_latest(tmp_path):
    s = RunState()
    s.apply(json.loads(_epoch_end(1, t=10.0)))
    s.apply(json.loads(_epoch_end(1, t=20.0)))
    assert s.epochs_done == 1 and s.epoch_ends[0]["epoch_time_s"] == 20.0


# ── Derived state ────────────────────────────────────────────────────────────────

def _state(*lines: str, mtime=None) -> RunState:
    s = RunState()
    for ln in lines:
        s.apply(json.loads(ln))
    s.mtime = mtime
    return s


def test_status_running_then_stalled_with_simulated_mtime():
    s = _state(_line(type="run_start", epochs=3), mtime=5000.0)
    assert s.status(now=5100.0, stalled_after_s=600) == RUNNING
    assert s.status(now=5700.0, stalled_after_s=600) == STALLED
    assert s.status(now=5700.0, stalled_after_s=1200) == RUNNING     # configurable


def test_status_uses_last_event_time_when_mtime_unknown():
    s = _state(_line(type="run_start", epochs=3))                    # time = 1000.0
    assert s.status(now=1100.0) == RUNNING
    assert s.status(now=1000.0 + 601) == STALLED


def test_status_waiting_when_no_events():
    assert RunState().status(now=0.0) == WAITING


@pytest.mark.parametrize("end, expected", [("completed", COMPLETED),
                                           ("early_stopped", EARLY_STOPPED),
                                           ("failed", FAILED), ("weird", COMPLETED)])
def test_status_from_run_end_ignores_staleness(end, expected):
    s = _state(_line(type="run_start", epochs=3), _line(type="run_end", status=end),
               mtime=1.0)
    assert s.status(now=10**9) == expected


def test_early_stop_without_run_end_is_still_running():
    s = _state(_line(type="run_start", epochs=9), _line(type="early_stop", epoch=4, best=0.7),
               mtime=1000.0)
    assert s.status(now=1001.0) == RUNNING and s.early_stop["epoch"] == 4


def test_progress_eta_and_best():
    s = _state(_line(type="run_start", epochs=5, select_metric="f1"),
               _epoch_end(1, t=40.0, best=0.5), _epoch_end(2, t=60.0, best=0.6),
               mtime=1000.0)
    assert s.epochs_done == 2 and s.progress == pytest.approx(0.4)
    assert s.mean_epoch_time_s == pytest.approx(50.0)
    assert s.eta_s(now=1001.0) == pytest.approx(150.0)               # 50 s × 3 epochs left
    assert s.best == pytest.approx(0.6)


def test_eta_is_none_when_not_computable_or_finished():
    assert _state(_line(type="run_start", epochs=5), mtime=1.0).eta_s(now=2.0) is None
    done = _state(_line(type="run_start", epochs=2), _epoch_end(1, epochs=2),
                  _line(type="run_end", status="completed", best=0.7), mtime=1.0)
    assert done.eta_s(now=2.0) is None and done.best == pytest.approx(0.7)
    stalled = _state(_line(type="run_start", epochs=5), _epoch_end(1), mtime=1.0)
    assert stalled.eta_s(now=10**6) is None


def test_epoch_and_batch_rows_are_flat():
    s = _state(_line(type="run_start", epochs=5), _epoch_end(1),
               _line(type="batch", epoch=2, batch=50, n_batches=200,
                     metrics={"batch_loss": 0.3, "lr": 1e-4}))
    row = s.epoch_rows()[0]
    assert row["train_loss"] == 0.3 and row["val_auc"] == 0.7 and row["epoch_time_s"] == 40.0
    b = s.batch_rows()[0]
    assert b["batch_loss"] == 0.3 and b["progress"] == pytest.approx(1.25)


# ── Discovery and run registry ────────────────────────────────────────────────────

def _make_run(root: Path, ts: str, *, finished: bool, env="local",
              sub="single/vit_tiny_patch16_224", mtime=None) -> Path:
    folder = root / "logs" / env / sub
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"train_{ts}.log").write_text("2026 [INFO ] Configuración: modelo=x\n",
                                            encoding="utf-8")
    ev = folder / f"events_{ts}.jsonl"
    text = _line(type="run_start", epochs=2)
    if finished:
        text += _line(type="run_end", status="completed")
    ev.write_text(text, encoding="utf-8")
    if mtime is not None:
        os.utime(ev, (mtime, mtime))
    return ev


def test_has_run_end_reads_the_tail(tmp_path):
    a = _make_run(tmp_path, "17102026_100000", finished=True)
    b = _make_run(tmp_path, "17102026_110000", finished=False)
    assert has_run_end(a) and not has_run_end(b)
    assert not has_run_end(tmp_path / "nope.jsonl")


def test_discovery_puts_active_runs_first_then_newest(tmp_path):
    now = 10_000_000.0
    _make_run(tmp_path, "17102026_090000", finished=True, mtime=now - 10)
    _make_run(tmp_path, "17102026_100000", finished=False, mtime=now - 5)          # active, older
    _make_run(tmp_path, "18102026_080000", finished=False, mtime=now - 99_999)     # abandoned
    _make_run(tmp_path, "17102026_110000", finished=False, mtime=now - 1,
              env="kaggle", sub="medmnist_chest/ddp/vit_tiny_patch16_224")         # active, newest
    found = discover_event_files(tmp_path, now=now)
    assert [f.timestamp for f in found] == [
        "17102026_110000", "17102026_100000", "18102026_080000", "17102026_090000"]
    assert [f.active for f in found] == [True, True, False, False]
    first = found[0]
    assert (first.env, first.dataset, first.mode) == ("kaggle", "medmnist_chest", "ddp")
    assert "[kaggle]" in first.label and "[ddp]" in first.label
    assert discover_event_files(tmp_path / "empty") == []


def test_discover_runs_links_events_by_timestamp(tmp_path):
    ev = _make_run(tmp_path, "17102026_100000", finished=False)
    folder = ev.parent
    (folder / "train_17102026_110000.log").write_text("x\n", encoding="utf-8")   # no events
    by_ts = {r.timestamp: r for r in discover_runs(tmp_path)}
    assert by_ts["17102026_100000"].events_path == ev
    assert by_ts["17102026_110000"].events_path is None


def test_sort_key_helper_matches_run_property():
    assert timestamp_sort_key("27052026_101010") < timestamp_sort_key("02062026_000000")
    assert timestamp_sort_key("20260527_101010") == "20260527_101010"


def test_fixture_copy_roundtrip(tmp_path):
    dst = tmp_path / FIXTURE.name
    shutil.copy(FIXTURE, dst)
    f = EventsFollower(dst)
    f.poll()
    last = f.state.last_event_time
    os.utime(dst, (last, last))               # the file was last written at the last event
    f.poll()                                  # (size unchanged: only the mtime is refreshed)
    assert f.state.status(now=last + 5) == RUNNING
    assert f.state.status(now=last + 5000) == STALLED


def test_heartbeat_is_a_known_event_that_keeps_the_run_alive():
    s = _state(_line(type="run_start", epochs=5), _line(type="heartbeat", epoch=1, elapsed_s=30.0))
    assert s.n_ignored == 0 and s.n_events == 2
    assert s.status(now=s.last_event_time + 60, stalled_after_s=180) == RUNNING


def test_interrupted_run_is_final():
    from src.web.events_reader import INTERRUPTED, FINAL_STATES
    s = _state(_line(type="run_start", epochs=5),
               _line(type="run_end", status="interrupted", epochs_run=0, best=None,
                     error="KeyboardInterrupt: "))
    assert s.status() == INTERRUPTED and INTERRUPTED in FINAL_STATES
