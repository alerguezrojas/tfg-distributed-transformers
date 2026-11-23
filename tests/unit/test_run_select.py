"""Lógica pura de la web: selector de runs, enlaces directos, test por dataset, GPUs, clases."""
from pathlib import Path

import yaml

from src.run_manifest import build_manifest, write_manifest
from src.web.data_paths import bigearthnet_paths
from src.web.run_registry import RunInfo
from src.web.run_select import (
    ALL, filter_options, filter_runs_by, gpu_count, order_classes, query_params_for,
    query_target, run_classes, run_rows, best_test_by_dataset)

NAV = ["overview", "run", "compare"]


def _run(ts, env="local", dataset="bigearthnet", model="vit_tiny_patch16_224", mode="single",
         log_path=None,
         **kw):
    return RunInfo(timestamp=ts, log_path=log_path or Path(f"/nonexistent/train_{ts}.log"),
                   trace_mode="simple", env=env, dataset=dataset, model=model, mode=mode, **kw)


RUNS = [_run("17102026_100000"),
        _run("17102026_110000", dataset="panda", model="vit_small_patch16_224", mode="ddp"),
        _run("17102026_120000", env="kaggle", dataset="panda", model="vit_small_patch16_224")]


def test_filters_combine_and_all_means_no_filter():
    assert filter_runs_by(RUNS, dataset=ALL, env=ALL) == RUNS
    assert filter_runs_by(RUNS, dataset="panda") == RUNS[1:]
    assert filter_runs_by(RUNS, dataset="panda", mode="ddp") == [RUNS[1]]
    assert filter_runs_by(RUNS, env="verode") == []
    assert filter_runs_by(RUNS, model="vit_tiny_patch16_224") == [RUNS[0]]


def test_search_matches_every_word_in_any_field():
    assert filter_runs_by(RUNS, "kaggle panda") == [RUNS[2]]
    assert filter_runs_by(RUNS, "VIT_SMALL ddp") == [RUNS[1]]
    assert filter_runs_by(RUNS, "17102026_12") == [RUNS[2]]
    assert filter_runs_by(RUNS, "nada") == []


def test_options_and_rows():
    assert filter_options(RUNS, "dataset") == [ALL, "bigearthnet", "panda"]
    rows = run_rows(RUNS)
    assert len(rows) == 3 and rows[1]["Strategy"] == "ddp" and rows[1]["Run"] == RUNS[1].label


def test_query_target_and_back():
    label, tab = query_target({"run": "17102026_110000", "tab": "compare"}, RUNS, NAV)
    assert label == RUNS[1].label and tab == "compare"
    assert query_params_for(RUNS[1], "compare") == {"run": "17102026_110000", "tab": "compare"}
    assert query_params_for(None, "overview") == {"tab": "overview"}


def test_query_target_ignores_unknown_values():
    assert query_target({"run": "x", "tab": "nope"}, RUNS, NAV) == (None, None)
    assert query_target({}, RUNS, NAV) == (None, None)


def test_test_results_use_each_datasets_own_metric_and_pick_the_best():
    runs = [_run("17102026_100000", test_csv_paths=[Path("a")]),
            _run("17102026_110000", test_csv_paths=[Path("b")]),
            _run("17102026_120000", dataset="panda", test_csv_paths=[Path("c"), Path("d")]),
            _run("17102026_130000", dataset="panda")]          # never evaluated
    aggs = {"a": {"f1_t05": 0.40, "qwk": 0.99}, "b": {"f1_t05": 0.55},
            "c": {"f1_t05": 0.9, "qwk": 0.70}, "d": {"qwk": 0.80}}
    rows = best_test_by_dataset(runs, parse=lambda p: (None, aggs[str(p)]))
    by_ds = {r["Dataset"]: r for r in rows}
    assert by_ds["bigearthnet"]["Metric"] == "F1" and by_ds["bigearthnet"]["Test"] == 0.55
    assert by_ds["bigearthnet"]["Evaluated"] == 2
    assert by_ds["panda"]["Metric"] == "QWK" and by_ds["panda"]["Test"] == 0.80
    assert by_ds["panda"]["Evaluated"] == 1


def test_test_results_skip_csvs_without_the_main_metric():
    runs = [_run("17102026_120000", dataset="panda", test_csv_paths=[Path("c")])]
    assert best_test_by_dataset(runs, parse=lambda p: (None, {"f1_t05": 0.9})) == []


def _log(tmp_path, ts, batch):
    p = tmp_path / f"train_{ts}.log"
    p.write_text(f"2026-11-19 [INFO ] Configuración: modelo=x | batch={batch} | epochs=1\n")
    return p


def test_gpu_count_from_config_line_manifest_and_mode(tmp_path):
    ddp = _run("17102026_100000", mode="ddp", log_path=_log(tmp_path, "a", "16/GPU (global=32, 2 GPUs)"))
    assert gpu_count(ddp) == 2
    single = RunInfo(timestamp="b", log_path=_log(tmp_path, "b", "32/GPU (global=32)"),
                     trace_mode="simple", mode="single")
    assert gpu_count(single) == 1
    unknown = RunInfo(timestamp="c", log_path=_log(tmp_path, "c", "8/GPU"), trace_mode="simple",
                      mode="heterogeneous")
    assert gpu_count(unknown) is None
    man = write_manifest(tmp_path / "run_c.json", build_manifest({"world_size": 4}))
    unknown.manifest_path = man
    assert gpu_count(unknown) == 4


def test_gpu_count_model_parallel_devices(tmp_path):
    p = tmp_path / "train_d.log"
    p.write_text("Configuración: modelo=x | devices=cuda:0+cuda:1 | batch=8 (global)\n")
    run = RunInfo(timestamp="d", log_path=p, trace_mode="simple", mode="model_parallel")
    assert gpu_count(run) == 2


def test_run_classes_prefers_manifest_then_bigearthnet_then_empty():
    from src.web.run_meta import RunMeta
    panda = _run("17102026_110000", dataset="panda",
                 meta=RunMeta(dataset="panda", classes=("ISUP 0", "ISUP 1")))
    assert run_classes(panda) == ("ISUP 0", "ISUP 1")
    assert len(run_classes(RUNS[0])) == 19
    assert run_classes(RUNS[2]) == ()


def test_order_classes_follows_dataset_order_or_keeps_given():
    from src.web.run_meta import RunMeta
    panda = _run("17102026_110000", dataset="panda",
                 meta=RunMeta(dataset="panda", classes=("b", "a", "c")))
    assert order_classes(panda, ["a", "b", "c"]) == ["b", "a", "c"]
    assert order_classes(panda, ["a", "z"]) == ["a", "z"]


def _profiles(tmp_path, root, meta_dir):
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(
        {"env": "p", "data_roots": {"bigearthnet": {"root": str(root), "metadata_dir": str(meta_dir)}}}))
    return tmp_path


def test_bigearthnet_paths_come_from_profiles(tmp_path):
    root, meta = tmp_path / "S2", tmp_path / "meta"
    root.mkdir(), meta.mkdir()
    (meta / "metadata.parquet").write_bytes(b"")
    machines = _profiles(tmp_path, root, meta)
    paths = bigearthnet_paths(environ={"PARAVIT_MACHINE": "p"}, machines_dir=machines)
    assert paths.root == str(root) and paths.metadata == str(meta / "metadata.parquet")


def test_bigearthnet_paths_missing_data_gives_none(tmp_path):
    machines = _profiles(tmp_path, tmp_path / "no", tmp_path / "no")
    paths = bigearthnet_paths(environ={}, machines_dir=machines)
    assert paths.root is None and paths.metadata is None
    assert bigearthnet_paths(environ={"PARAVIT_MACHINE": "ghost"}, machines_dir=machines).root is None
