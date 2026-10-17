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
