"""Tests para check_feasibility.py v3 — perfilado de sistema, DDP, predicción."""

import sys
import tempfile
import math
from pathlib import Path
from dataclasses import dataclass

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ── CPUInfo / HardwareProbe ───────────────────────────────────────────────────


def test_hardware_probe_cpu_returns_cpu_info():
    from scripts.check_feasibility import HardwareProbe
    probe = HardwareProbe()
    cpu = probe.probe_cpu()
    assert cpu.logical_cores >= 1
    assert cpu.physical_cores >= 1
    assert cpu.ram_total_gb > 0


def test_hardware_probe_gpu_returns_hardware_info():
    from scripts.check_feasibility import HardwareProbe
    probe = HardwareProbe()
    gpu = probe.probe_gpu()
    assert gpu.device_name != ""
    # En CPU debe devolver is_cuda=False con valores 0
    # En GPU debe devolver is_cuda=True con VRAM > 0


def test_disk_probe_nonexistent_path_returns_zeroes():
    from scripts.check_feasibility import DiskProbe
    probe = DiskProbe()
    disk = probe.probe("/nonexistent/path/that/cannot/exist")
    assert disk.read_mb_per_s == 0.0
    assert disk.files_per_second == 0.0


def test_disk_probe_none_path():
    from scripts.check_feasibility import DiskProbe
    probe = DiskProbe()
    disk = probe.probe(None)
    assert disk.dataset_path == ""
    assert disk.read_mb_per_s == 0.0


def test_disk_probe_existing_path():
    from scripts.check_feasibility import DiskProbe
    with tempfile.TemporaryDirectory() as tmp:
        disk = DiskProbe().probe(tmp)
        # Debe devolver algo, sin crashear
        assert disk.disk_type in ("SSD", "HDD", "NFS", "Unknown")


# ── DatasetProfiler ───────────────────────────────────────────────────────────


def test_dataset_profiler_no_dataset():
    from scripts.check_feasibility import DatasetProfiler
    dp = DatasetProfiler(None, None).profile(0.5, 32)
    assert dp.n_files_total_est == 0
    assert dp.io_bottleneck_ratio == 0.0


def test_dataset_profiler_compute_bottleneck():
    from scripts.check_feasibility import DatasetProfiler, DiskInfo
    # Si files_per_second es muy alto, el ratio debe ser < 1 (compute-bound)
    fast_disk = DiskInfo(
        dataset_path="/fake",
        is_nfs=False, disk_type="SSD",
        read_mb_per_s=5000.0,
        files_per_second=10000.0,  # muy rápido
    )
    with tempfile.TemporaryDirectory() as tmp:
        dp = DatasetProfiler(tmp, fast_disk).profile(0.5, 32)
        # io_time = 32*3/10000 = 0.0096s; compute = 0.5s → ratio ≈ 0.02 (compute-bound)
        assert dp.io_bottleneck_ratio < 1.0


def test_dataset_profiler_io_bottleneck():
    from scripts.check_feasibility import DatasetProfiler, DiskInfo
    # Si files_per_second es muy bajo, el ratio debe ser > 1 (I/O-bound)
    slow_disk = DiskInfo(
        dataset_path="/fake",
        is_nfs=True, disk_type="NFS",
        read_mb_per_s=10.0,
        files_per_second=5.0,  # muy lento
    )
    with tempfile.TemporaryDirectory() as tmp:
        dp = DatasetProfiler(tmp, slow_disk).profile(0.1, 32)
        # io_time = 32*3/5 = 19.2s; compute = 0.1s → ratio = 192 (I/O-bound)
        assert dp.io_bottleneck_ratio > 1.0


# ── PerformancePredictor ──────────────────────────────────────────────────────


def test_performance_predictor_vit_base_v3():
    from scripts.check_feasibility import PerformancePredictor
    pred = PerformancePredictor().predict("vit_base_patch16_224", n_epochs=30,
                                          has_llrd=True, has_label_smoothing=True)
    assert pred.predicted_best_f1 > 0.6, "ViT-Base debería alcanzar F1 > 0.6"
    assert 1 <= pred.predicted_best_epoch <= 30
    assert pred.predicted_early_stop_epoch > pred.predicted_best_epoch
    assert pred.confidence in ("alta", "media", "baja")
    assert len(pred.curve_f1_val) == 30
    assert len(pred.curve_f1_train) == 30


def test_performance_predictor_vit_tiny():
    from scripts.check_feasibility import PerformancePredictor
    pred = PerformancePredictor().predict("vit_tiny_patch16_224", n_epochs=20)
    # ViT-Tiny tiene menor techo que ViT-Base
    assert pred.predicted_best_f1 < 0.65


def test_performance_predictor_curve_monotone_increasing_initially():
    from scripts.check_feasibility import PerformancePredictor
    pred = PerformancePredictor().predict("vit_base_patch16_224", n_epochs=15,
                                          has_llrd=True, has_label_smoothing=True)
    # Los primeros 5 epochs deben mostrar crecimiento
    early = pred.curve_f1_val[:5]
    assert early[-1] > early[0], "La curva debe crecer en los primeros epochs"


def test_performance_predictor_train_above_val():
    from scripts.check_feasibility import PerformancePredictor
    pred = PerformancePredictor().predict("vit_base_patch16_224", n_epochs=10)
    # Train F1 siempre debe ser ≥ Val F1 (efecto overfitting)
    for tr, va in zip(pred.curve_f1_train[3:], pred.curve_f1_val[3:]):
        assert tr >= va - 0.05, f"Train {tr:.3f} < Val {va:.3f}"


def test_performance_predictor_unknown_model():
    from scripts.check_feasibility import PerformancePredictor
    # No debe fallar con modelos desconocidos
    pred = PerformancePredictor().predict("some_unknown_model_xyz", n_epochs=10)
    assert pred.predicted_best_f1 > 0


# ── DDPOptimizer ──────────────────────────────────────────────────────────────


def _make_ddp_optimizer():
    from scripts.check_feasibility import (
        DDPOptimizer, ModelInfo, HardwareInfo, CPUInfo, DiskInfo, BenchmarkResult,
    )
    model_info = ModelInfo(
        name="vit_base", total_params=86_000_000, trainable_params=86_000_000,
        flops_per_image_mflops=17000, weight_mb=344, gradient_mb=344, optimizer_mb=688,
        activation_mb_per_image=120,
    )
    hardware_info = HardwareInfo(
        device_name="Tesla V100", total_vram_gb=32.0, free_vram_gb=30.0,
        is_cuda=True, compute_capability="7.0",
    )
    cpu_info = CPUInfo(logical_cores=16, physical_cores=8, freq_mhz=2400,
                       ram_total_gb=112, ram_free_gb=80, platform="linux")
    disk_info = DiskInfo(dataset_path="/nfs/data", is_nfs=True, disk_type="NFS",
                         read_mb_per_s=50, files_per_second=100)
    benchmark = [
        BenchmarkResult(
            batch_size=64, trace_mode="off",
            seconds_per_batch_train=0.65, seconds_per_batch_eval=0.25,
            images_per_second_train=98.5, images_per_second_eval=256,
            peak_vram_gb=12.5, avg_power_w=200,
        )
    ]
    return DDPOptimizer(model_info, hardware_info, cpu_info, disk_info,
                        benchmark, dataset_train=237871, dataset_val=122342, nfs_factor=1.3)


def test_ddp_optimizer_compute_scenarios():
    optimizer = _make_ddp_optimizer()
    scenarios = optimizer.compute_scenarios(n_epochs=17)
    assert len(scenarios) >= 4  # 1, 2, 4, 8 GPUs
    for s in scenarios:
        assert s.n_gpus >= 1
        assert s.batch_per_gpu > 0
        assert 0 <= s.scaling_efficiency <= 100


def test_ddp_optimizer_single_gpu_efficiency_100():
    from scripts.check_feasibility import DDPOptimizer
    optimizer = _make_ddp_optimizer()
    scenarios = optimizer.compute_scenarios(n_epochs=5)
    single = next(s for s in scenarios if s.n_gpus == 1)
    assert single.scaling_efficiency == pytest.approx(100.0)
    assert single.estimated_speedup == pytest.approx(1.0, abs=0.01)


def test_ddp_optimizer_more_gpus_faster():
    optimizer = _make_ddp_optimizer()
    scenarios = optimizer.compute_scenarios(n_epochs=17)
    times = {s.n_gpus: s.time_total_s for s in scenarios}
    # Más GPUs → menos tiempo total
    assert times[2] < times[1]
    assert times[4] < times[2]


def test_ddp_optimizer_nfs_bottleneck_detection():
    optimizer = _make_ddp_optimizer()
    scenarios = optimizer.compute_scenarios(n_epochs=5)
    # Con NFS lento y dataset grande, debería detectar bottleneck en io o sync
    bottlenecks = {s.bottleneck for s in scenarios}
    assert bottlenecks.issubset({"compute", "io", "sync"})


def test_ddp_optimizer_recommend_config():
    optimizer = _make_ddp_optimizer()
    rec = optimizer.recommend_config()
    assert "n_gpus" in rec
    assert rec["n_gpus"] >= 1
    assert rec["batch_per_gpu"] > 0
    assert 0 < rec["efficiency_pct"] <= 100


# ── FeasibilityParser v3 ──────────────────────────────────────────────────────


def test_feasibility_parser_v3_reads_new_blocks():
    import csv, tempfile
    from src.web.feasibility_parser import parse_feasibility_csv

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        w = csv.writer(f)
        w.writerow(["#meta", "model_name", "total_params_M", "flops_mflops",
                    "hardware_name", "total_vram_gb", "free_vram_gb"])
        w.writerow(["#meta", "vit_base", "85.8", "17000", "V100", "32.0", "30.0"])
        w.writerow(["#cpu", "logical_cores", "physical_cores", "freq_mhz", "ram_total_gb", "ram_free_gb"])
        w.writerow(["#cpu", "16", "8", "2400", "112.0", "80.0"])
        w.writerow(["#disk", "type", "is_nfs", "read_mb_per_s", "files_per_second"])
        w.writerow(["#disk", "NFS", "yes", "50.0", "100.0"])
        w.writerow(["#prediction", "predicted_best_f1", "predicted_best_epoch",
                    "predicted_early_stop_epoch", "confidence"])
        w.writerow(["#prediction", "0.68", "7", "17", "alta"])
        w.writerow(["#curve_val_f1", "0.35", "0.55", "0.65", "0.68", "0.67"])
        w.writerow(["#curve_epochs", "1", "2", "3", "4", "5"])
        w.writerow(["#ddp", "n_gpus", "batch_per_gpu", "global_batch", "workers_per_gpu",
                    "speedup", "efficiency_pct", "sync_overhead_pct", "bottleneck",
                    "time_train_epoch_min", "time_total_h"])
        w.writerow(["#ddp", "1", "64", "64", "8", "1.0", "100.0", "0.0", "compute", "45.0", "19.0"])
        w.writerow(["#ddp", "2", "64", "128", "4", "1.7", "85.0", "5.0", "compute", "27.0", "11.5"])
        w.writerow(["batch_size", "trace_mode", "s_per_batch_train", "imgs_per_s_train",
                    "s_per_batch_eval", "imgs_per_s_eval", "peak_vram_gb", "avg_power_w", "oom"])
        w.writerow(["64", "off", "0.65", "98.5", "0.25", "256", "12.5", "200", "no"])
        tmp_path = Path(f.name)

    meta, df = parse_feasibility_csv(tmp_path)
    tmp_path.unlink()

    assert meta["model_name"] == "vit_base"
    assert "cpu" in meta
    assert meta["cpu"]["logical_cores"] == "16"
    assert "disk" in meta
    assert meta["disk"]["is_nfs"] == "yes"
    assert "prediction" in meta
    assert float(meta["prediction"]["predicted_best_f1"]) == pytest.approx(0.68)
    assert "curve_val_f1" in meta
    assert len(meta["curve_val_f1"]) == 5
    assert "ddp_scenarios" in meta
    assert len(meta["ddp_scenarios"]) == 2
    assert not df.empty


def test_feasibility_parser_parse_ddp_scenarios():
    import csv, tempfile
    from src.web.feasibility_parser import parse_feasibility_csv, parse_ddp_scenarios

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        w = csv.writer(f)
        w.writerow(["#meta", "model_name"])
        w.writerow(["#meta", "vit_tiny"])
        w.writerow(["#ddp", "n_gpus", "batch_per_gpu", "global_batch", "workers_per_gpu",
                    "speedup", "efficiency_pct", "sync_overhead_pct", "bottleneck",
                    "time_train_epoch_min", "time_total_h"])
        w.writerow(["#ddp", "1", "32", "32", "8", "1.0", "100.0", "0.0", "compute", "20.0", "10.0"])
        w.writerow(["#ddp", "2", "32", "64", "4", "1.8", "90.0", "3.0", "compute", "12.0", "6.0"])
        w.writerow(["batch_size", "trace_mode", "oom"])
        w.writerow(["32", "off", "no"])
        tmp_path = Path(f.name)

    meta, _ = parse_feasibility_csv(tmp_path)
    ddp_df = parse_ddp_scenarios(meta)
    tmp_path.unlink()

    assert len(ddp_df) == 2
    assert "n_gpus" in ddp_df.columns
    assert "speedup" in ddp_df.columns
