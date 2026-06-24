"""Tests del parsing de los bloques de estudio empírico en el CSV de viabilidad."""

import csv
import tempfile
from pathlib import Path

import pytest

from src.web.benchmark_parser import parse_benchmark_csv


def _write_csv_with_study(tmp: Path) -> Path:
    path = tmp / "benchmark_study.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#meta", "model_name", "total_params_M", "flops_mflops",
                    "hardware_name", "total_vram_gb", "free_vram_gb"])
        w.writerow(["#meta", "vit_tiny", "5.5", "34", "RTX 3060 Ti", "8.0", "8.0"])
        # Bloques de estudio
        w.writerow(["#study_lr", "suggested_lr", "min_loss_lr", "diverged_lr"])
        w.writerow(["#study_lr", "1.000e-04", "1.000e-02", "1.000e-01"])
        w.writerow(["#study_lr_curve_lrs", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2"])
        w.writerow(["#study_lr_curve_losses", "0.7", "0.65", "0.5", "0.4", "0.6"])
        w.writerow(["#study_conv", "fit_a", "fit_b", "fit_c", "r_squared",
                    "loss_1ep", "loss_final", "best_f1", "epochs_to_plateau",
                    "measured_imgs_per_s"])
        w.writerow(["#study_conv", "2.0", "0.5", "0.15", "0.97",
                    "0.21", "0.17", "0.64", "8", "95.0"])
        w.writerow(["#study_conv_steps", "1", "2", "3", "4", "5"])
        w.writerow(["#study_conv_losses", "0.5", "0.45", "0.42", "0.40", "0.38"])
        w.writerow(["#study_conv_f1s", "0.3", "0.35", "0.40", "0.43", "0.46"])
        w.writerow(["#study_grad", "grad_norm_mean", "grad_norm_std",
                    "noise_scale", "suggested_batch_size", "cv"])
        w.writerow(["#study_grad", "1.5", "0.3", "12.5", "64", "0.2"])
        # Benchmark
        w.writerow(["batch_size", "trace_mode", "oom"])
        w.writerow(["32", "off", "no"])
    return path


def test_parser_reads_study_block():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_csv_with_study(Path(tmp))
        meta, df = parse_benchmark_csv(path)
        assert "study" in meta
        study = meta["study"]
        assert "lr" in study
        assert "conv" in study
        assert "grad" in study


def test_parser_study_lr_values():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_csv_with_study(Path(tmp))
        meta, _ = parse_benchmark_csv(path)
        lr = meta["study"]["lr"]
        assert lr["suggested_lr"] == "1.000e-04"
        assert meta["study"]["lr_curve_lrs"] == [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        assert meta["study"]["lr_curve_losses"] == [0.7, 0.65, 0.5, 0.4, 0.6]


def test_parser_study_convergence_curve():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_csv_with_study(Path(tmp))
        meta, _ = parse_benchmark_csv(path)
        study = meta["study"]
        assert study["conv"]["best_f1"] == "0.64"
        assert study["conv"]["r_squared"] == "0.97"
        assert study["conv_steps"] == [1, 2, 3, 4, 5]
        assert study["conv_losses"] == [0.5, 0.45, 0.42, 0.40, 0.38]
        assert study["conv_f1s"] == [0.3, 0.35, 0.40, 0.43, 0.46]


def test_parser_study_gradient_noise():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_csv_with_study(Path(tmp))
        meta, _ = parse_benchmark_csv(path)
        grad = meta["study"]["grad"]
        assert grad["suggested_batch_size"] == "64"
        assert grad["grad_norm_mean"] == "1.5"


def test_parser_no_study_block_when_absent():
    """Un CSV sin bloques de estudio no debe tener la clave 'study'."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "no_study.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["#meta", "model_name"])
            w.writerow(["#meta", "vit_base"])
            w.writerow(["batch_size", "trace_mode", "oom"])
            w.writerow(["64", "off", "no"])
        meta, _ = parse_benchmark_csv(path)
        assert "study" not in meta


def test_parser_benchmark_still_parsed_with_study():
    """El benchmark debe parsearse aunque haya bloques de estudio."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_csv_with_study(Path(tmp))
        _, df = parse_benchmark_csv(path)
        assert not df.empty
        assert "batch_size" in df.columns
