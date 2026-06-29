"""Unit tests for src/web/benchmark_comparison.py — the THREE-way comparison
(analytic vs benchmark vs real run) that powers the 'Estimate vs Benchmark vs Run' tab.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.web.benchmark_comparison import build_comparison, ComparisonRow, _sum_opt


def _meta(**over):
    m = {
        "model_name": "vit_base_patch16_224",
        "hardware_name": "Tesla T4",
        "n_train": 5000,
        "n_val": 1500,
        "nfs_factor": 1.0,
        "flops_mflops": 17600.0,
        "total_params_M": 85.8,
        "total_static_mb": 1500.0,
        "activation_mb_per_image": 95.0,
        "prediction": {"predicted_best_f1": 0.55},
        "precision_cmp": {"speedup": 4.0},
    }
    m.update(over)
    return m


def _feas(batch=96):
    return pd.DataFrame([{
        "batch_size": batch, "trace_mode": "simple",
        "est_train_min_per_epoch": 3.2, "est_eval_min_per_epoch": 0.4,
        "est_total_min_per_epoch": 3.6, "peak_vram_gb": 3.9,
        "s_per_batch_train": 1.8, "s_per_batch_eval": 0.6, "imgs_per_s_train": 26.4,
        "est_energy_train_wh_per_epoch": 3.3, "est_energy_eval_wh_per_epoch": 0.2,
        "avg_power_w": 64.0, "optimizer_steps_per_epoch": 53,
    }])


def _actual(energy=True):
    df = pd.DataFrame({
        "epoch": [1, 2, 3],
        "epoch_time": [190.0, 188.0, 192.0],     # seconds → 3.17 min
        "time_train_s": [170.0, 168.0, 172.0],
        "time_eval_s": [20.0, 20.0, 20.0],
        "val_f1": [0.50, 0.55, 0.56],
    })
    if energy:
        df["energy_train_j"] = [11600.0, 11700.0, 11500.0]   # ~3.23 Wh
        df["energy_eval_wh"] = [0.3, 0.3, 0.3]
    return df


def _row(cmp, name):
    return next(r for r in cmp.rows if r.metric == name)


def test_three_sources_present_for_single_fp32():
    cmp = build_comparison(_meta(), _feas(), _actual(), batch_size=96,
                           strategy="single", gpu_name="Tesla T4", n_gpus=1, precision="fp32")
    assert cmp is not None
    tt = _row(cmp, "Total time / epoch")
    assert tt.analytic is not None and tt.estimated is not None and tt.actual is not None
    et = _row(cmp, "Energy total / epoch")
    assert et.analytic is not None and et.estimated is not None and et.actual is not None


def test_actual_train_energy_derived_from_joules():
    cmp = build_comparison(_meta(), _feas(), _actual(), batch_size=96,
                           strategy="single", gpu_name="Tesla T4", n_gpus=1, precision="fp32")
    e_train = _row(cmp, "Energy train / epoch")
    # mean(11600,11700,11500)/3600 ≈ 3.22 Wh
    assert e_train.actual == pytest.approx(11600 / 3600, abs=0.05)


def test_f1_row_three_ways():
    cmp = build_comparison(_meta(), _feas(), _actual(), batch_size=96,
                           strategy="single", gpu_name="Tesla T4", n_gpus=1, precision="fp32")
    f1 = _row(cmp, "Best Val F1")
    assert f1.estimated == pytest.approx(0.55)      # benchmark prior
    assert f1.actual == pytest.approx(0.56)         # real run max
    assert f1.analytic is not None                  # analytic prior


def test_no_analytic_without_gpu():
    cmp = build_comparison(_meta(), _feas(), _actual(), batch_size=96, gpu_name=None)
    assert _row(cmp, "Total time / epoch").analytic is None


def test_precision_speedup_corrects_benchmark_time_and_energy():
    """An AMP run: the fp32 benchmark time/energy are divided by the measured speedup."""
    base = build_comparison(_meta(), _feas(), _actual(), batch_size=96,
                            strategy="single", gpu_name="Tesla T4", precision="fp32")
    amp = build_comparison(_meta(), _feas(), _actual(), batch_size=96,
                           strategy="single", gpu_name="Tesla T4", precision="amp",
                           precision_speedup=4.0)
    assert _row(amp, "Total time / epoch").estimated == pytest.approx(
        _row(base, "Total time / epoch").estimated / 4.0, rel=1e-6)
    assert _row(amp, "Energy total / epoch").estimated == pytest.approx(
        _row(base, "Energy total / epoch").estimated / 4.0, rel=1e-6)


def test_ddp_divides_time_but_not_total_energy():
    """DDP: wall-clock ÷ speedup, but TOTAL energy is conserved across GPUs."""
    single = build_comparison(_meta(), _feas(), _actual(), batch_size=96,
                              strategy="single", gpu_name="Tesla T4", precision="fp32")
    ddp = build_comparison(_meta(), _feas(), _actual(), batch_size=96,
                           strategy="ddp", gpu_name="Tesla T4", n_gpus=2, precision="fp32",
                           ddp_speedup=2.0)
    assert _row(ddp, "Total time / epoch").estimated == pytest.approx(
        _row(single, "Total time / epoch").estimated / 2.0, rel=1e-6)
    # energy unchanged by GPU count
    assert _row(ddp, "Energy total / epoch").estimated == pytest.approx(
        _row(single, "Energy total / epoch").estimated, rel=1e-6)


def test_to_dataframe_has_three_columns():
    cmp = build_comparison(_meta(), _feas(), _actual(), batch_size=96,
                           strategy="single", gpu_name="Tesla T4", precision="fp32")
    df = cmp.to_dataframe()
    for col in ("Analytic", "Benchmark", "Real", "Δ analytic %", "Δ benchmark %"):
        assert col in df.columns


def test_error_pct_helpers():
    r = ComparisonRow("m", "f", estimated=11.0, actual=10.0, analytic=9.0)
    assert r.error_pct == pytest.approx(10.0)
    assert r.analytic_error_pct == pytest.approx(-10.0)


def test_sum_opt():
    assert _sum_opt(None, None) is None
    assert _sum_opt(1.0, None) == 1.0
    assert _sum_opt(1.0, 2.0) == 3.0


def test_ddp_predict_receives_global_batch(monkeypatch):
    """build_comparison must hand predict() the GLOBAL batch (per-GPU × n_gpus) for DDP,
    since predict() re-splits it; the feas_df matching still uses the per-GPU batch."""
    import src.performance_model as pm
    captured = {}
    real_predict = pm.predict

    def spy(strategy, model_name, gpu_name, **kw):
        captured["batch"] = kw.get("batch")
        return real_predict(strategy, model_name, gpu_name, **kw)

    monkeypatch.setattr(pm, "predict", spy)
    build_comparison(_meta(), _feas(batch=48), _actual(), batch_size=48,
                     strategy="ddp", gpu_name="Tesla T4", n_gpus=2, precision="fp32",
                     ddp_speedup=2.0)
    assert captured["batch"] == 96   # 48 per-GPU × 2 GPUs


def test_run_dataset_size_overrides_benchmark_size():
    """The comparison uses the RUN's dataset size, recomputing the benchmark per-epoch
    estimate from its (size-independent) s/batch — so a run and a benchmark report of
    different sizes stay comparable, and the estimate scales with the run's N."""
    small = build_comparison(_meta(n_train=999999), _feas(), _actual(), batch_size=96,
                             strategy="single", gpu_name="Tesla T4", precision="fp32",
                             run_n_train=5000, run_n_val=1500)
    big = build_comparison(_meta(n_train=999999), _feas(), _actual(), batch_size=96,
                           strategy="single", gpu_name="Tesla T4", precision="fp32",
                           run_n_train=10000, run_n_val=3000)
    ts = _row(small, "Total time / epoch").estimated
    tb = _row(big, "Total time / epoch").estimated
    assert ts is not None and tb is not None
    assert tb == pytest.approx(2 * ts, rel=0.05)   # ~linear in dataset size, not 999999


def test_perf_strategy_maps_ddp_hetero():
    from src.web.tabs.benchmark.validate import _perf_strategy
    assert _perf_strategy("ddp_hetero") == "heterogeneous"
    for m in ("single", "ddp", "model_parallel"):
        assert _perf_strategy(m) == m


def test_select_benchmark_prefers_report_with_batch_and_sizes():
    from src.web.tabs.benchmark.validate import _select_benchmark
    df_no48 = pd.DataFrame({"batch_size": [32, 64]})
    df_48 = pd.DataFrame({"batch_size": [48, 96]})
    parsed = [
        ("kaggle", "vit_base", {"n_train": None}, df_no48),
        ("kaggle", "vit_base", {"n_train": 5000}, df_48),     # has bs 48 AND #sizes
    ]
    m, df = _select_benchmark(parsed, "kaggle", "vit_base", 48)
    assert m["n_train"] == 5000 and (df["batch_size"] == 48).any()
    # No candidate of that model → (None, None)
    assert _select_benchmark(parsed, "kaggle", "resnet50", 48) == (None, None)


def test_run_sizes_unknown_returns_none(tmp_path):
    """A run with no config line matched to a benchmark with no #sizes → unknown size,
    NOT a fabricated 5000 default (which would make the estimate wildly off)."""
    from types import SimpleNamespace
    from src.web.tabs.benchmark.validate import _run_sizes
    log = tmp_path / "train_old.log"
    log.write_text("2026-05-01 [INFO] some old run with no Configuración line\n")
    r = SimpleNamespace(log_path=log)
    assert _run_sizes(r, {"n_train": None, "n_val": None}) == (None, None)
    # With a #sizes benchmark, it uses that.
    assert _run_sizes(r, {"n_train": 5000, "n_val": 1500}) == (5000, 1500)


def test_single_predict_receives_run_batch(monkeypatch):
    import src.performance_model as pm
    captured = {}
    real_predict = pm.predict
    monkeypatch.setattr(pm, "predict",
                        lambda *a, **k: (captured.update(batch=k.get("batch")) or real_predict(*a, **k)))
    build_comparison(_meta(), _feas(batch=96), _actual(), batch_size=96,
                     strategy="single", gpu_name="Tesla T4", n_gpus=1, precision="fp32")
    assert captured["batch"] == 96
