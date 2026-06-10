"""Numeric precision modes — the practical switch for using Tensor cores.

You cannot address CUDA cores vs Tensor cores directly: the GPU scheduler and
cuDNN/cuBLAS route each operation automatically. What you *can* choose is the
**numeric precision**, and that decides which units do the heavy matrix math:

    fp32  full precision        -> mostly CUDA cores            (no tensor cores)
    tf32  TensorFloat-32        -> Tensor cores (Ampere+)       matmul only
    amp   mixed precision fp16  -> Tensor cores (Volta+)        autocast + GradScaler
    bf16  mixed precision bf16  -> Tensor cores (Ampere+)       autocast, no scaler

So "use the Tensor cores" really means "train in tf32/fp16/bf16". This module is
the single source of truth for: which modes a GPU supports (from its compute
capability), whether a mode engages the Tensor cores, the autocast dtype, and
whether a GradScaler is needed. The lookup is pure (no GPU) and unit-tested.
"""
from __future__ import annotations

ALL_PRECISIONS = ["fp32", "tf32", "amp", "bf16"]

_LABELS = {
    "fp32": "FP32 (full precision)",
    "tf32": "TF32 (Tensor cores)",
    "amp": "FP16 AMP (Tensor cores)",
    "bf16": "BF16 (Tensor cores)",
}

_DESCRIPTIONS = {
    "fp32": "Full 32-bit precision. Runs on the conventional CUDA cores. Baseline.",
    "tf32": "TensorFloat-32: matmuls go through the Tensor cores in reduced mantissa. "
            "Ampere+ only. Faster than FP32 with negligible accuracy change.",
    "amp": "Automatic mixed precision (float16): the heavy matmuls/convs run on the "
           "Tensor cores, the delicate parts stay FP32. Volta+ (incl. T4/V100). "
           "Needs a GradScaler. Fastest and uses less VRAM.",
    "bf16": "Mixed precision with bfloat16: like AMP but a wider exponent, so no "
            "GradScaler needed. Ampere+ only.",
}


def _parse_cc(compute_capability: str | None) -> tuple[int, int]:
    if not compute_capability:
        return (0, 0)
    try:
        major, minor = compute_capability.split(".")
        return int(major), int(minor)
    except (ValueError, AttributeError):
        return (0, 0)


def uses_tensor_cores(precision: str) -> bool:
    """True for the modes that engage the Tensor cores."""
    return precision in ("tf32", "amp", "bf16")


def label(precision: str) -> str:
    return _LABELS.get(precision, precision)


def description(precision: str) -> str:
    return _DESCRIPTIONS.get(precision, "")


def autocast_dtype(precision: str):
    """torch dtype for autocast, or None when autocast is not used (fp32/tf32)."""
    import torch
    return {"amp": torch.float16, "bf16": torch.bfloat16}.get(precision)


def needs_scaler(precision: str) -> bool:
    """Only float16 AMP needs gradient scaling."""
    return precision == "amp"


def available_precisions(compute_capability: str | None, is_cuda: bool = True) -> list[str]:
    """Precision modes a GPU can use, given its compute capability.

    - fp32: always.
    - amp (fp16): Volta and newer (cc >= 7.0) — first GPUs with Tensor cores.
    - tf32 / bf16: Ampere and newer (cc >= 8.0).
    """
    if not is_cuda:
        return ["fp32"]
    major, minor = _parse_cc(compute_capability)
    out = ["fp32"]
    if (major, minor) >= (8, 0):
        out.append("tf32")
    if major >= 7:
        out.append("amp")
    if (major, minor) >= (8, 0):
        out.append("bf16")
    # Keep canonical order.
    return [p for p in ALL_PRECISIONS if p in out]


def apply_backend_flags(precision: str) -> None:
    """Enable/disable TF32 matmul/cudnn paths to match the chosen precision.

    TF32 is a global backend toggle (it is what makes plain FP32 matmuls use the
    Tensor cores). We turn it on only for ``tf32`` and off otherwise so the modes
    are clean and comparable in benchmarks.
    """
    try:
        import torch
    except Exception:
        return
    allow = precision == "tf32"
    try:
        torch.backends.cuda.matmul.allow_tf32 = allow
        torch.backends.cudnn.allow_tf32 = allow
    except Exception:
        pass
