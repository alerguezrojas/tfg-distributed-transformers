"""Unit tests for src/precision.py — the Tensor-core (precision) selector logic."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.precision import (
    ALL_PRECISIONS, available_precisions, needs_scaler, uses_tensor_cores,
)


def test_fp32_does_not_use_tensor_cores():
    assert uses_tensor_cores("fp32") is False
    assert uses_tensor_cores("tf32") is True
    assert uses_tensor_cores("amp") is True
    assert uses_tensor_cores("bf16") is True


def test_only_amp_needs_scaler():
    assert needs_scaler("amp") is True
    for p in ("fp32", "tf32", "bf16"):
        assert needs_scaler(p) is False


def test_cpu_only_fp32():
    assert available_precisions("8.6", is_cuda=False) == ["fp32"]


def test_volta_turing_support_fp16_not_tf32_bf16():
    # V100 (7.0) and T4 (7.5): Tensor cores via FP16, but no TF32/BF16.
    for cc in ("7.0", "7.5"):
        avail = available_precisions(cc)
        assert "amp" in avail
        assert "tf32" not in avail
        assert "bf16" not in avail
        assert avail[0] == "fp32"


def test_ampere_supports_all():
    # RTX 3060 Ti (8.6) / A100 (8.0): fp32, tf32, amp, bf16.
    avail = available_precisions("8.6")
    assert avail == ["fp32", "tf32", "amp", "bf16"]


def test_pascal_only_fp32():
    # GTX 10-series (6.1): no Tensor cores at all.
    assert available_precisions("6.1") == ["fp32"]


def test_unknown_cc_falls_back_to_fp32_only():
    assert available_precisions(None) == ["fp32"]
    assert available_precisions("") == ["fp32"]


def test_order_is_canonical():
    avail = available_precisions("9.0")  # Hopper
    assert avail == [p for p in ALL_PRECISIONS if p in avail]


def test_benchmark_parser_reads_precision_blocks(tmp_path):
    """The parser must round-trip the #precision and #precision_cmp blocks."""
    from src.web.benchmark_parser import parse_benchmark_csv

    csv = tmp_path / "benchmark_prec.csv"
    csv.write_text(
        "#meta,model_name,total_params_M,flops_mflops,hardware_name,total_vram_gb,free_vram_gb\n"
        "#meta,vit_base_patch16_224,85.8,200.8,RTX 3060 Ti,8.2,8.2\n"
        "#precision,mode\n"
        "#precision,fp32\n"
        "#precision_cmp,batch_size,tc_precision,fp32_imgs_s,tc_imgs_s,speedup,fp32_vram_gb,tc_vram_gb\n"
        "#precision_cmp,32,amp,68.1,172.7,2.54,5.63,4.15\n"
        "batch_size,trace_mode,oom\n"
        "32,off,no\n"
    )
    meta, _ = parse_benchmark_csv(csv)
    assert meta.get("precision") == "fp32"
    cmp = meta.get("precision_cmp")
    assert cmp is not None
    assert cmp["tc_precision"] == "amp"
    assert cmp["fp32_imgs_s"] == 68.1
    assert cmp["tc_imgs_s"] == 172.7
    assert cmp["speedup"] == 2.54


def test_trainer_accepts_precision_on_cpu():
    """Trainer must accept a precision arg and stay fp32 on CPU (no crash)."""
    import torch
    from src.models.vit import build_model
    from src.training.trainer import Trainer
    model = build_model("vit_tiny_patch16_224", pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tr = Trainer(model, opt, None, torch.device("cpu"), precision="amp")
    # On CPU we force fp32 (autocast/scaler are CUDA-only).
    assert tr.precision == "fp32"
    assert tr._use_amp is False
