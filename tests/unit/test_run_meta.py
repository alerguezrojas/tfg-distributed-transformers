"""What a run is, for the web: metadata sources, main metric, loss and label tags."""
import json
from pathlib import Path

import pandas as pd

from src.run_manifest import build_manifest, write_manifest
from src.web.run_meta import (
    AUC, F1, QWK, metric_info, resolve_meta, resolve_metric)
from src.web.run_registry import RunInfo, discover_runs

CFG = ("2026-11-19 19:12:56 [INFO ] Configuración: modelo=vit_small_patch16_224 | "
       "batch=16/GPU (global=16) | epochs=15 | lr=0.0001 | precision=amp | "
       "loss=ce_emd(w=1.0) | train=3821 | val=819 | dataset=panda | seed=43\n")

PANDA_START = {
    "run_id": "x", "dataset": "panda", "task": "multiclass", "ordinal": True,
    "classes": ["ISUP 0", "ISUP 1"], "select_metric": "qwk", "epochs": 15,
    "config": {"data": {"name": "panda", "image_size": 512, "color_jitter": 0.2,
                        "providers": {"train": ["karolinska"], "val": ["karolinska"],
                                      "test": ["radboud"]}},
               "model": {"img_size": 512},
               "training": {"seed": 44, "loss": "ce_emd", "emd_weight": 1.0,
                            "batch_size": 16, "lr": 0.0001, "precision": "amp"}}}


def _log(tmp_path, text=CFG):
    p = tmp_path / "train_19112026_191255.log"
    p.write_text(text)
    return p


# ── Sources and their order ──────────────────────────────────────────────────

def test_manifest_has_priority_and_the_log_only_fills_gaps(tmp_path):
    manifest = build_manifest(PANDA_START)
    meta = resolve_meta(manifest=manifest, run_start={"dataset": "other"},
                        log_path=_log(tmp_path), dataset="panda")
    assert meta.source == "manifest" and meta.dataset == "panda"
    assert meta.seed == 44                       # manifest wins over the log's seed=43
    assert (meta.n_train, meta.n_val) == (3821, 819)     # only the log knows the sizes
    assert meta.providers_train == ("karolinska",) and meta.providers_test == ("radboud",)
    assert meta.color_jitter == 0.2 and meta.img_size == 512 and meta.ordinal is True


def test_events_are_used_when_there_is_no_manifest(tmp_path):
    meta = resolve_meta(manifest=None, run_start=PANDA_START, log_path=_log(tmp_path),
                        dataset="panda")
    assert meta.source == "events" and meta.select == "qwk" and meta.classes == ("ISUP 0", "ISUP 1")


def test_log_line_and_then_folder_layout_as_last_resorts(tmp_path):
    meta = resolve_meta(manifest=None, run_start=None, log_path=_log(tmp_path), dataset="panda")
    assert meta.source == "log" and meta.seed == 43 and meta.loss == "ce_emd"
    assert meta.precision == "amp" and meta.batch_size == 16 and meta.epochs == 15
    assert meta.task == "multiclass" and meta.ordinal is True    # known from the dataset name
    bare = resolve_meta(manifest=None, run_start=None, log_path=_log(tmp_path, "no line\n"),
                        dataset="medmnist_derma")
    assert bare.source == "path" and bare.dataset == "medmnist_derma" and bare.seed is None
    assert bare.task is None                                      # nothing invented


def test_seed_none_in_the_log_means_unknown(tmp_path):
    meta = resolve_meta(manifest=None, run_start=None, dataset="bigearthnet",
                        log_path=_log(tmp_path, CFG.replace("seed=43", "seed=None")))
    assert meta.seed is None


# ── Main metric and loss ─────────────────────────────────────────────────────

def test_main_metric_follows_select_by_then_the_dataset():
    assert metric_info("qwk") is QWK and metric_info("AUC") is AUC and metric_info("f1") is F1
    assert metric_info(None, "panda") is QWK
    assert metric_info(None, "medmnist_path") is AUC
    assert metric_info(None, "bigearthnet") is F1
    assert metric_info("whatever", "medmnist_chest") is AUC       # unknown key → dataset default
    assert metric_info("f1_optimal").column == "f1_at_threshold"


def test_resolve_metric_falls_back_to_f1_when_the_column_was_not_logged():
    with_qwk = pd.DataFrame({"epoch": [1], "val_f1": [0.5], "val_qwk": [0.7]})
    without = pd.DataFrame({"epoch": [1], "val_f1": [0.5]})
    assert resolve_metric(QWK, with_qwk) is QWK
    assert resolve_metric(QWK, without) is F1
    assert resolve_metric(AUC, pd.DataFrame({"val_auc": [None]})) is F1


def test_loss_label_uses_the_task_default_when_unset():
    ce = resolve_meta(manifest=None, run_start={"dataset": "panda", "task": "multiclass"},
                      log_path=None, dataset="panda")
    assert ce.loss is None and ce.effective_loss == "ce" and ce.loss_label == "CE"
    emd = resolve_meta(manifest=build_manifest(PANDA_START), run_start=None, log_path=None,
                       dataset="panda")
    assert emd.loss_label == "CE + EMD"
    ben = resolve_meta(manifest=None, run_start=None, log_path=None, dataset="bigearthnet")
    assert ben.loss_label == "BCE"
    unknown = resolve_meta(manifest=None, run_start=None, log_path=None, dataset="medmnist_x")
    assert unknown.loss_label == "Loss"


# ── Label tags ───────────────────────────────────────────────────────────────

def test_tags_show_only_what_is_not_the_default():
    meta = resolve_meta(manifest=build_manifest(PANDA_START), run_start=None, log_path=None,
                        dataset="panda")
    assert meta.tags() == ["ce_emd", "seed 44", "train: karolinska", "color 0.2", "512px"]
    assert "seed 44" not in meta.tags(with_seed=False)
    plain = resolve_meta(manifest=None, run_start=None, log_path=None, dataset="bigearthnet")
    assert plain.tags() == []
    base = {**PANDA_START, "config": {"training": {"seed": 42, "loss": "ce"},
                                       "model": {"img_size": 224}}}
    assert resolve_meta(manifest=build_manifest(base), run_start=None, log_path=None,
                        dataset="panda").tags() == []


def test_providers_label_mentions_val_only_when_it_differs():
    start = {**PANDA_START, "config": {"data": {"providers": {"train": ["radboud"],
                                                               "val": ["karolinska"]}}}}
    meta = resolve_meta(manifest=build_manifest(start), run_start=None, log_path=None,
                        dataset="panda")
    assert meta.providers_label == "train: radboud · val: karolinska"


# ── RunInfo integration ──────────────────────────────────────────────────────

def _make_run(root: Path, ts: str, seed: int | None, manifest: bool = True) -> None:
    d = root / "logs" / "local" / "panda" / "single" / "vit_small_patch16_224"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"train_{ts}.log").write_text(CFG.replace("seed=43", f"seed={seed}"))
    if manifest:
        start = json.loads(json.dumps(PANDA_START))
        start["config"]["training"]["seed"] = seed
        write_manifest(d / f"run_{ts}.json", build_manifest(start))


def test_discover_runs_loads_meta_and_labels_runs_by_what_tells_them_apart(tmp_path):
    _make_run(tmp_path, "19112026_191255", 43)
    _make_run(tmp_path, "19112026_183927", 44)
    runs = discover_runs(tmp_path)
    assert [r.meta.seed for r in runs] == [43, 44] and runs[0].meta.source == "manifest"
    assert runs[0].label.endswith("· ce_emd · seed 43 · train: karolinska · color 0.2 · 512px")
    assert "[panda]" in runs[0].label
    assert runs[0].label != runs[1].label
    assert runs[0].main_metric is QWK
    assert runs[0].group_label == "[local] [panda] vit_small [amp] · ce_emd · " \
                                  "train: karolinska · color 0.2 · 512px"


def test_runs_that_differ_only_in_the_seed_share_a_group_key(tmp_path):
    _make_run(tmp_path, "19112026_191255", 43)
    _make_run(tmp_path, "19112026_183927", 44)
    a, b = discover_runs(tmp_path)
    assert a.seed_group_key == b.seed_group_key
    other = RunInfo(timestamp="19112026_100000", log_path=a.log_path, trace_mode="simple",
                    env="local", mode="single", model=a.model, dataset="panda",
                    precision="amp", meta=a.meta.__class__(dataset="panda", seed=43, loss="ce"))
    assert other.seed_group_key != a.seed_group_key           # another loss: not a sibling


def test_a_run_without_seed_never_joins_a_group(tmp_path):
    d = tmp_path / "logs" / "local" / "single" / "vit_tiny_patch16_224"
    d.mkdir(parents=True)
    for ts in ("17102026_100000", "17102026_110000"):
        (d / f"train_{ts}.log").write_text("no config line\n")
    a, b = discover_runs(tmp_path)
    assert a.seed_group_key != b.seed_group_key


def test_identical_labels_get_their_seconds(tmp_path):
    d = tmp_path / "logs" / "local" / "single" / "vit_tiny_patch16_224"
    d.mkdir(parents=True)
    for ts in ("17102026_100001", "17102026_100045"):         # same minute
        (d / f"train_{ts}.log").write_text("x\n")
    a, b = discover_runs(tmp_path)
    assert a.label != b.label and a.label.split()[1].count(":") == 2


def test_fmt_span_switches_from_seconds_to_minutes():
    from src.web.ui.helpers import _fmt_span
    assert _fmt_span(1.4) == "1.4 s" and _fmt_span(59.9) == "59.9 s"
    assert _fmt_span(90) == "1.5 min"
