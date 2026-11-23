"""Aggregation of repeated (seeded) runs: only seeded runs with a test CSV count."""
import pytest

from src.results_summary import collect, summarize


def _run(folder, ts, seed, auc, acc):
    seed_part = f" | seed={seed}" if seed is not None else ""
    (folder / f"train_{ts}.log").write_text(
        f"2026-10-17 [INFO ] Configuración: modelo=vit | dataset=x{seed_part}\n")
    (folder / f"test_{ts}.csv").write_text(
        "class_idx,class_name,f1,precision,recall\n0,a,0.5,0.5,0.5\n\n"
        f"# aggregate,loss=0.3,f1_t05=0.5,accuracy={acc},auc={auc}\n")


def test_seeded_runs_are_aggregated(tmp_path):
    _run(tmp_path, "17102026_100000", 42, 0.90, 0.80)
    _run(tmp_path, "17102026_110000", 43, 0.92, 0.84)
    _run(tmp_path, "17102026_120000", None, 0.10, 0.10)       # no seed → ignored
    (tmp_path / "train_17102026_130000.log").write_text("Configuración: seed=44\n")  # no test CSV
    runs = collect(tmp_path)
    assert [r.seed for r in runs] == [42, 43]
    stats = summarize(runs)
    mean, std, n = stats["auc"]
    assert mean == pytest.approx(0.91)
    assert std == pytest.approx(0.0141421, rel=1e-4)        # sample std
    assert n == 2
    assert "f1_opt" not in stats                             # not reported → left out


def test_single_run_has_zero_std(tmp_path):
    _run(tmp_path, "17102026_100000", 42, 0.9, 0.8)
    assert summarize(collect(tmp_path))["accuracy"] == (pytest.approx(0.8), 0.0, 1)


def test_qwk_is_aggregated_for_ordinal_runs(tmp_path):
    for ts, seed, qwk in (("20112026_100000", 42, 0.70), ("20112026_110000", 43, 0.80)):
        (tmp_path / f"train_{ts}.log").write_text(f"Configuración: modelo=vit | seed={seed}\n")
        (tmp_path / f"test_{ts}.csv").write_text(
            "class_idx,class_name,f1,precision,recall\n0,a,0.5,0.5,0.5\n\n"
            f"# aggregate,loss=0.3,f1_t05=0.5,accuracy=0.8,auc=0.9,qwk={qwk}\n")
    mean, _, n = summarize(collect(tmp_path))["qwk"]
    assert mean == pytest.approx(0.75) and n == 2


def test_seed_falls_back_to_the_manifest(tmp_path):
    # DDP logs written before the seed joined the Configuración: line: the seed is in run_<ts>.json
    ts = "20112026_163243"
    (tmp_path / f"train_{ts}.log").write_text("Configuración: modelo=vit | dataset=panda\n")
    (tmp_path / f"run_{ts}.json").write_text('{"config": {"training": {"seed": 42}}}')
    (tmp_path / f"test_{ts}.csv").write_text(
        "class_idx,class_name,f1,precision,recall\n0,a,0.5,0.5,0.5\n\n# aggregate,auc=0.9\n")
    assert [r.seed for r in collect(tmp_path)] == [42]


def test_no_seed_in_log_nor_manifest_is_ignored(tmp_path):
    ts = "20112026_163243"
    (tmp_path / f"train_{ts}.log").write_text("Configuración: modelo=vit | seed=None\n")
    (tmp_path / f"run_{ts}.json").write_text('{"config": {"training": {"seed": null}}}')
    (tmp_path / f"test_{ts}.csv").write_text(
        "class_idx,class_name,f1,precision,recall\n0,a,0.5,0.5,0.5\n\n# aggregate,auc=0.9\n")
    assert collect(tmp_path) == []
