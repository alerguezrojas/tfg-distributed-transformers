"""Predict the monetary cost of training the model on cloud providers.

The benchmark checker already estimates the **total training time** on the
benchmarked GPU. From there the cost on a cloud provider is simply:

    cost = estimated_hours_on_that_gpu × provider_price_per_hour

The estimated hours are scaled across GPUs by their relative FP16 (Tensor-core)
throughput, since the benchmark ran on one specific GPU:

    hours_target = hours_reference × (tflops_reference / tflops_target)

so a faster GPU finishes sooner (fewer billed hours). This is a rough,
compute-bound estimate — real training also pays I/O, so treat it as an upper
bound on the speed advantage of the bigger GPUs. Prices are approximate
on-demand single-GPU rates (2024–2025) and are meant to be edited as they change.

The tables and ``estimate_costs`` are pure (no network, no GPU) and unit-tested.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CloudOption:
    provider: str
    gpu: str                 # GPU model on that instance
    usd_per_hour: float      # approximate on-demand, single GPU
    note: str = ""


# Approximate on-demand single-GPU prices (USD/hour). Editable — they drift.
CLOUD_OPTIONS: list[CloudOption] = [
    CloudOption("Kaggle", "Tesla T4 ×2", 0.0, "Free, 30 h/week, 2 GPUs"),
    CloudOption("Google Colab", "Tesla T4", 0.0, "Free tier (or ~0.1–0.2 $/h Pro)"),
    CloudOption("Vast.ai", "RTX 3090", 0.20, "Community marketplace, variable"),
    CloudOption("Vast.ai", "RTX 4090", 0.35, "Community marketplace, variable"),
    CloudOption("GCP", "Tesla T4", 0.35, "+ instance cost"),
    CloudOption("Azure", "Tesla T4", 0.53, "NCasT4_v3"),
    CloudOption("AWS", "Tesla T4", 0.526, "g4dn.xlarge"),
    CloudOption("GCP", "L4", 0.71, "g2 instance"),
    CloudOption("RunPod", "RTX 4090", 0.69, "Secure cloud"),
    CloudOption("Lambda Labs", "RTX 6000 Ada", 0.80, ""),
    CloudOption("AWS", "A10G", 1.006, "g5.xlarge"),
    CloudOption("Lambda Labs", "A100 40GB", 1.10, ""),
    CloudOption("RunPod", "A100 80GB", 1.19, ""),
    CloudOption("GCP", "V100", 2.48, ""),
    CloudOption("Lambda Labs", "H100", 2.49, ""),
    CloudOption("AWS", "Tesla V100", 3.06, "p3.2xlarge"),
    CloudOption("Azure", "Tesla V100", 3.06, "NC6s_v3"),
    CloudOption("GCP", "A100 40GB", 3.67, ""),
    CloudOption("AWS", "A100 40GB", 4.10, "p4d (per-GPU share)"),
]

# Approximate FP16 / Tensor-core throughput (dense TFLOPS) — used only to scale
# the training time across GPUs (relative, not an absolute benchmark).
GPU_TFLOPS_FP16: dict[str, float] = {
    "P100": 19.0,
    "Tesla T4": 65.0, "T4": 65.0,
    "RTX 3060 Ti": 35.0,
    "RTX 3090": 142.0,
    "RTX 4090": 330.0,
    "RTX 6000 Ada": 91.0,
    "A10G": 125.0,
    "L4": 121.0,
    "Tesla V100": 112.0, "V100": 112.0,
    "A100 40GB": 312.0, "A100 80GB": 312.0, "A100": 312.0,
    "H100": 990.0,
}


def gpu_tflops(name: str | None) -> float | None:
    """Approximate FP16 TFLOPS for a GPU name (fuzzy substring match)."""
    if not name:
        return None
    n = name.replace("NVIDIA", "").replace("GeForce", "").strip()
    # Exact first, then longest-key substring match (so 'A100 80GB' beats 'A100').
    if n in GPU_TFLOPS_FP16:
        return GPU_TFLOPS_FP16[n]
    best = None
    for key, tf in sorted(GPU_TFLOPS_FP16.items(), key=lambda kv: -len(kv[0])):
        if key.lower() in n.lower():
            best = tf
            break
    return best


def estimate_costs(
    total_hours_ref: float,
    ref_gpu_name: str | None,
    options: list[CloudOption] | None = None,
) -> list[dict]:
    """Estimate cost per cloud option for a training that takes
    ``total_hours_ref`` on the reference (benchmarked) GPU.

    Returns a list of dicts sorted by cost ascending:
        {provider, gpu, usd_per_hour, est_hours, cost_usd, scaled, note}
    ``scaled`` is False when the relative throughput could not be derived
    (then the reference time is used as-is).
    """
    options = options if options is not None else CLOUD_OPTIONS
    ref_tf = gpu_tflops(ref_gpu_name)
    rows: list[dict] = []
    for opt in options:
        tgt_tf = gpu_tflops(opt.gpu)
        if ref_tf and tgt_tf:
            hours = total_hours_ref * (ref_tf / tgt_tf)
            scaled = True
        else:
            hours = total_hours_ref
            scaled = False
        rows.append({
            "provider": opt.provider,
            "gpu": opt.gpu,
            "usd_per_hour": opt.usd_per_hour,
            "est_hours": round(hours, 2),
            "cost_usd": round(hours * opt.usd_per_hour, 2),
            "scaled": scaled,
            "note": opt.note,
        })
    rows.sort(key=lambda r: (r["cost_usd"], r["usd_per_hour"]))
    return rows
