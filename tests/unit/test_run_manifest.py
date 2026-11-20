"""Run manifest (run_<ts>.json): schema, atomic write, the sink that writes it and the
backfill script for older runs."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.run_manifest import (
    MANIFEST_VERSION, build_manifest, loss_name, manifest_from_log, manifest_path,
    parse_config_line, read_manifest, read_run_start, write_manifest)
from src.training.events import JsonlEventSink

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "backfill_script", ROOT / "scripts" / "backfill_run_manifests.py")
backfill_script = importlib.util.module_from_spec(_spec)
sys.modules["backfill_script"] = backfill_script
_spec.loader.exec_module(backfill_script)

RUN_START = {
    "v": 1, "type": "run_start", "time": 1.0, "run_id": "19112026_191255", "dataset": "panda",
    "task": "multiclass", "classes": ["ISUP 0", "ISUP 1"], "model": "vit_small_patch16_224",
    "mode": "single", "world_size": 1, "select_metric": "qwk", "epochs": 15,
    "config": {"data": {"name": "panda", "providers": {"train": ["karolinska"]}},
               "training": {"seed": 42, "loss": "ce_emd", "select_by": "qwk"}},
    "env": {"host": "pc"}}

CONFIG_LINE = ("2026-10-17 16:57:24 [INFO ] Configuración: modelo=vit_tiny_patch16_224 | "
               "batch=128/GPU (global=128) | epochs=5 | lr=0.0001 | precision=fp32 | "
               "loss=ce_emd(w=1.0) | train=89996 | val=10004 | dataset=medmnist_path | seed=43\n")


# ── Manifest ─────────────────────────────────────────────────────────────────

def test_build_manifest_keeps_run_start_fields_and_drops_stream_bookkeeping():
    m = build_manifest(RUN_START, command="train_single_gpu.py --epochs 15")
    assert m["type"] == "run_manifest" and m["v"] == MANIFEST_VERSION
    assert m["dataset"] == "panda" and m["select_metric"] == "qwk"
    assert m["config"]["training"]["loss"] == "ce_emd"
    assert m["command"] == "train_single_gpu.py --epochs 15" and m["source"] == "training"
    assert "time" not in m and "type" in m and m["type"] != "run_start"


def test_manifest_round_trip_and_atomic_write(tmp_path):
    path = write_manifest(manifest_path(tmp_path, "19112026_191255"), build_manifest(RUN_START))
    assert path.name == "run_19112026_191255.json"
    assert not list(tmp_path.glob("*.tmp"))              # temp file renamed away
    assert read_manifest(path)["classes"] == ["ISUP 0", "ISUP 1"]


@pytest.mark.parametrize("content", ["not json", '{"type": "other"}', "[]",
                                     '{"type": "run_manifest", "v": 99}'])
def test_read_manifest_rejects_what_is_not_a_known_manifest(tmp_path, content):
    p = tmp_path / "run_x.json"
    p.write_text(content)
    assert read_manifest(p) is None
    assert read_manifest(tmp_path / "missing.json") is None and read_manifest(None) is None


def test_read_run_start_takes_the_first_line_only(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text(json.dumps(RUN_START) + "\n" + json.dumps({"type": "run_start", "run_id": "z"}))
    assert read_run_start(p)["run_id"] == "19112026_191255"
    p.write_text(json.dumps({"type": "heartbeat"}) + "\n")
    assert read_run_start(p) is None and read_run_start(tmp_path / "none") is None


def test_parse_config_line_and_loss_name():
    cfg = parse_config_line("noise\n" + CONFIG_LINE)
    assert cfg["seed"] == "43" and cfg["dataset"] == "medmnist_path"
    assert cfg["batch"] == "128/GPU (global=128)"
    assert parse_config_line("Configuración recomendada: 1 GPU") == {}
    assert loss_name("ce_emd(w=1.0)") == "ce_emd" and loss_name(None) is None


def test_manifest_from_log_records_only_what_the_line_has(tmp_path):
    log = tmp_path / "train_17102026_165723.log"
    log.write_text(CONFIG_LINE)
    m = manifest_from_log(log, dataset="medmnist_path", model="vit_tiny_patch16_224",
                          mode="single", run_id="17102026_165723", env="local")
    t = m["config"]["training"]
    assert m["source"] == "log" and m["n_train"] == 89996 and m["n_val"] == 10004
    assert t == {"epochs": 5, "lr": 0.0001, "seed": 43, "precision": "fp32",
                 "batch_size": 128, "loss": "ce_emd"}
    assert "select_metric" not in m and "task" not in m       # not in the log: not invented
    # a log without the line still yields the layout facts
    bare = tmp_path / "train_01012026_000000.log"
    bare.write_text("nothing here\n")
    m2 = manifest_from_log(bare, dataset="bigearthnet", model="m", mode="ddp", run_id="r")
    assert m2["dataset"] == "bigearthnet" and m2["mode"] == "ddp" and "n_train" not in m2


# ── Sink ─────────────────────────────────────────────────────────────────────

def test_sink_writes_manifest_when_the_run_starts(tmp_path):
    mp = tmp_path / "run_t.json"
    sink = JsonlEventSink(tmp_path / "events_t.jsonl", run_info={"run_id": "t", "dataset": "toy"},
                          heartbeat_s=0, manifest_path=mp, command="train.py --x 1")
    assert not mp.exists()                    # only once the run starts
    sink.on_fit_start(3)
    m = read_manifest(mp)
    assert m["run_id"] == "t" and m["dataset"] == "toy" and m["epochs"] == 3
    assert m["command"] == "train.py --x 1"


def test_sink_without_manifest_path_writes_none(tmp_path):
    sink = JsonlEventSink(tmp_path / "events_t.jsonl", run_info={"run_id": "t"}, heartbeat_s=0)
    sink.on_fit_start(1)
    assert not list(tmp_path.glob("run_*.json"))


def test_unwritable_manifest_warns_but_never_stops_training(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder")
    sink = JsonlEventSink(tmp_path / "events_t.jsonl", run_info={"run_id": "t"}, heartbeat_s=0,
                          manifest_path=blocker / "run_t.json")
    with pytest.warns(RuntimeWarning, match="manifest not written"):
        sink.on_fit_start(1)
    assert (tmp_path / "events_t.jsonl").read_text().count("run_start") == 1


# ── Backfill ─────────────────────────────────────────────────────────────────

def _tree(tmp_path):
    logs = tmp_path / "logs"
    d = logs / "local" / "panda" / "single" / "vit_small_patch16_224"
    d.mkdir(parents=True)
    line = CONFIG_LINE.replace("medmnist_path", "panda")
    (d / "train_19112026_191255.log").write_text(line)                  # has events elsewhere
    (d / "train_18112026_100000.log").write_text(line)                  # log only
    other = tmp_path / "other" / "deep"
    other.mkdir(parents=True)
    (other / "events_19112026_191255.jsonl").write_text(json.dumps(RUN_START) + "\n")
    return logs, d, tmp_path / "other"


def test_backfill_uses_events_when_found_and_the_log_otherwise(tmp_path):
    logs, d, other = _tree(tmp_path)
    done = backfill_script.backfill(logs, [other])
    assert len(done["events"]) == 1 and len(done["log"]) == 1
    from_events = read_manifest(d / "run_19112026_191255.json")
    assert from_events["source"] == "events" and from_events["select_metric"] == "qwk"
    from_log = read_manifest(d / "run_18112026_100000.json")
    assert from_log["source"] == "log" and from_log["dataset"] == "panda"
    assert from_log["config"]["output"]["env"] == "local"       # taken from the folder layout
    assert from_log["config"]["training"]["seed"] == 43


def test_backfill_skips_existing_unless_forced_and_dry_run_writes_nothing(tmp_path):
    logs, d, other = _tree(tmp_path)
    dry = backfill_script.backfill(logs, [other], dry_run=True)
    assert len(dry["events"]) + len(dry["log"]) == 2 and not list(d.glob("run_*.json"))
    backfill_script.backfill(logs, [other])
    again = backfill_script.backfill(logs, [other])
    assert len(again["skipped"]) == 2 and not again["events"] and not again["log"]
    (d / "run_18112026_100000.json").write_text("{}")
    forced = backfill_script.backfill(logs, [other], force=True)
    assert len(forced["events"]) + len(forced["log"]) == 2
    assert read_manifest(d / "run_18112026_100000.json") is not None
