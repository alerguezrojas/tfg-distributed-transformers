r"""Analytic performance model — predict training time/speedup/memory/cost for
ANY (strategy, model, GPU, n_gpus, dataset, batch, precision) WITHOUT running a
benchmark, from curated hardware/model specs.

This generalizes the feasibility checker (which until now needed a real
benchmark on the target machine) into a closed-form predictor, as requested by
the tutor. The formulas are derived in docs/performance_model.md; the master
equation for a DDP train epoch is:

    T(n, π) = φ · [ max( N/(π·r_c·n) , N/r_io )  +  (8P/β)·⌈N/(b·n)⌉ ]
              \_______ compute ∥ disk (the slower) ______/   \__ gradient sync __/

with parameters estimated from specs:
    r_c  = MFU · TFLOPS_fp32(gpu) / FLOPs_train(model)     (compute throughput)
    π    = Tensor-core speedup for the precision (fp32 → 1)
    r_io = disk throughput (img/s), from a disk-type table or a real probe
    β    = interconnect bandwidth (NVLink/PCIe/Ethernet)
    φ    = NFS penalty (~1.3 on the cluster, 1.0 on local disk)

Pure module: no torch, no GPU, no I/O — fully unit-testable. If a real measured
r_c is available it can be passed in to CALIBRATE (overrides the estimate).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ── Calibration constants (fitted to the real Kaggle 2×T4 measurements) ──────────
# vit_base on a T4, fp32, gave r_c ≈ 26 img/s. With FLOPs_train = 3 × 17.6 GFLOP
# and the T4's 8.1 fp32 TFLOPS that pins MFU = 26·52.8e9/8.1e12 ≈ 0.17.
MFU: float = 0.17                 # model-FLOPs utilization (fraction of peak)
TRAIN_FLOPS_FACTOR: float = 3.0   # forward+backward ≈ 3× forward
BYTES_PER_PARAM_GRAD_OPT: int = 16  # fp32 Adam: 4 (w) + 4 (grad) + 8 (m,v)
CUDA_OVERHEAD_GB: float = 0.6     # context + cudnn workspaces
# Activation memory ≈ ACT_BYTES_PER_PARAM_IMG bytes per parameter per image,
# fitted to vit_large = 13.78 GB at batch 32 on a T4 (the precisely measured OOM).
ACT_BYTES_PER_PARAM_IMG: float = 0.86e-9  # GB per (param · image)


# ── Hardware spec tables ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GpuSpec:
    name: str
    fp32_tflops: float       # dense FP32 peak
    tensor_speedup: float    # measured r_c(amp)/r_c(fp32) — Tensor-core gain
    vram_gb: float
    interconnect: str        # "nvlink" | "pcie" (datacenter GPUs prefer NVLink)


# fp32 TFLOPS and the measured/typical Tensor-core speedup π per GPU.
GPU_TABLE: dict[str, GpuSpec] = {
    "Tesla T4":      GpuSpec("Tesla T4", 8.1, 3.8, 16.0, "pcie"),
    "Tesla V100":    GpuSpec("Tesla V100", 14.0, 3.5, 32.0, "nvlink"),
    "V100-PCIE":     GpuSpec("V100-PCIE", 14.0, 3.5, 32.0, "nvlink"),
    "RTX 3060 Ti":   GpuSpec("RTX 3060 Ti", 16.2, 2.54, 8.0, "pcie"),
    "RTX 3090":      GpuSpec("RTX 3090", 35.6, 3.5, 24.0, "pcie"),
    "RTX 4090":      GpuSpec("RTX 4090", 82.6, 4.0, 24.0, "pcie"),
    "A100 40GB":     GpuSpec("A100 40GB", 19.5, 4.0, 40.0, "nvlink"),
    "A100 80GB":     GpuSpec("A100 80GB", 19.5, 4.0, 80.0, "nvlink"),
    "P100":          GpuSpec("P100", 9.3, 1.0, 16.0, "pcie"),  # no usable Tensor cores
    "H100":          GpuSpec("H100", 67.0, 5.0, 80.0, "nvlink"),
}

# Disk throughput in IMAGES/second (includes TIFF decode + transform), calibrated
# from the real I/O probes: Kaggle SSD ≈ 130 img/s. Override with a real measure.
DISK_RIO: dict[str, float] = {
    "nvme": 220.0, "ssd": 130.0, "hdd": 45.0, "nfs": 100.0, "unknown": 130.0,
}

# Interconnect bandwidth in bytes/s for the gradient all-reduce.
INTERCONNECT_BW: dict[str, float] = {
    "nvlink": 300e9, "pcie": 16e9, "ethernet": 0.125e9,  # 1 GbE
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_m: float          # millions of parameters
    gflops_fwd: float        # forward GFLOPs per 224×224 image


# Curated from timm / the literature (forward GFLOPs at 224²).
MODEL_TABLE: dict[str, ModelSpec] = {
    "vit_tiny_patch16_224":  ModelSpec("vit_tiny_patch16_224", 5.7, 1.3),
    "vit_small_patch16_224": ModelSpec("vit_small_patch16_224", 22.0, 4.6),
    "vit_base_patch16_224":  ModelSpec("vit_base_patch16_224", 85.8, 17.6),
    "vit_large_patch16_224": ModelSpec("vit_large_patch16_224", 303.0, 61.6),
    "resnet50":              ModelSpec("resnet50", 25.6, 4.1),
    "efficientnet_b0":       ModelSpec("efficientnet_b0", 5.3, 0.4),
}

# Bytes per parameter when only storing the model (no optimizer), by precision.
_WEIGHT_BYTES = {"fp32": 4, "tf32": 4, "amp": 6, "bf16": 6}  # amp/bf16 keep an fp32 master + fp16 copy


# ── Spec lookup with fuzzy matching ──────────────────────────────────────────────

def _fuzzy(table: dict, name: str | None):
    if not name:
        return None
    n = name.replace("NVIDIA", "").replace("GeForce", "").strip()
    if n in table:
        return table[n]
    # longest-key substring match (so 'A100 80GB' beats 'A100')
    cands = [k for k in table if k.lower() in n.lower() or n.lower() in k.lower()]
    if cands:
        return table[max(cands, key=len)]
    return None


def gpu_spec(name: str | None) -> GpuSpec | None:
    return _fuzzy(GPU_TABLE, name)


def model_spec(name: str | None) -> ModelSpec | None:
    return _fuzzy(MODEL_TABLE, name)


# ── Parameter estimators ─────────────────────────────────────────────────────────

def estimate_rc(model: ModelSpec, gpu: GpuSpec, precision: str = "fp32",
                mfu: float = MFU) -> float:
    """Compute throughput r_c (img/s) on ONE GPU, estimated from specs.

    r_c(fp32) = MFU · TFLOPS_fp32 / FLOPs_train ; Tensor precisions multiply by π.
    """
    flops_train = TRAIN_FLOPS_FACTOR * model.gflops_fwd * 1e9
    rc_fp32 = mfu * gpu.fp32_tflops * 1e12 / flops_train
    pi = 1.0 if precision == "fp32" else gpu.tensor_speedup
    return rc_fp32 * pi


def precision_factor(gpu: GpuSpec, precision: str) -> float:
    return 1.0 if precision == "fp32" else gpu.tensor_speedup


def estimate_rio(disk_type: str = "ssd", nfs: bool = False,
                 measured: float | None = None) -> float:
    """Disk throughput r_io (img/s). A measured probe wins; otherwise the table."""
    if measured and measured > 0:
        return measured
    key = "nfs" if nfs else disk_type
    return DISK_RIO.get(key, DISK_RIO["unknown"])


# ── Memory model ─────────────────────────────────────────────────────────────────

def estimate_vram_gb(model: ModelSpec, batch: int, precision: str = "fp32") -> float:
    """Peak VRAM (GB) for training: weights+grad+optimizer + activations + overhead."""
    p = model.params_m * 1e6
    weights_opt_gb = BYTES_PER_PARAM_GRAD_OPT * p / 1e9
    if precision in ("amp", "bf16"):
        weights_opt_gb += 2 * p / 1e9          # extra fp16 copy of weights+grad
    act_gb = ACT_BYTES_PER_PARAM_IMG * p * batch
    if precision in ("amp", "bf16"):
        act_gb *= 0.6                          # half-precision activations
    return weights_opt_gb + act_gb + CUDA_OVERHEAD_GB


def fits_in_memory(model: ModelSpec, gpu: GpuSpec, batch: int,
                   precision: str = "fp32") -> bool:
    return estimate_vram_gb(model, batch, precision) <= gpu.vram_gb


def max_batch(model: ModelSpec, gpu: GpuSpec, precision: str = "fp32",
              cap: int = 512) -> int:
    """Largest power-of-two-ish batch that fits in this GPU's VRAM (0 if none)."""
    b = 1
    best = 0
    while b <= cap:
        if fits_in_memory(model, gpu, b, precision):
            best = b
        else:
            break
        b *= 2
    return best


# ── The master time formula, per strategy ────────────────────────────────────────

def _sync_time(params_m: float, beta: float, n_batches: int) -> float:
    """All-reduce time: 8·P bytes (2·P fp32 grads) per batch / interconnect bw."""
    p = params_m * 1e6
    return (8 * p / beta) * n_batches


@dataclass
class EpochPrediction:
    time_train_s: float
    speedup: float
    efficiency: float        # speedup / n_gpus
    bottleneck: str          # "compute" | "io" | "sync"
    t_compute_s: float
    t_io_s: float
    t_sync_s: float


def predict_epoch(strategy: str, model: ModelSpec, gpu: GpuSpec, n_gpus: int,
                  n_train: int, batch_per_gpu: int, precision: str,
                  disk_type: str = "ssd", nfs: bool = False,
                  rc_measured: float | None = None,
                  rio_measured: float | None = None) -> EpochPrediction:
    """Train-epoch time + speedup for one strategy, from the master formula."""
    phi = 1.3 if nfs else 1.0
    pi = precision_factor(gpu, precision)
    rc = rc_measured if (rc_measured and rc_measured > 0) else estimate_rc(model, gpu, precision="fp32")
    rio = estimate_rio(disk_type, nfs, rio_measured)
    beta = INTERCONNECT_BW["nvlink" if gpu.interconnect == "nvlink" else "pcie"]

    def single_epoch(n: int) -> tuple[float, float, float, float]:
        t_compute = n_train / (pi * rc * n)
        t_io = n_train / rio
        n_batches = math.ceil(n_train / (batch_per_gpu * n))
        t_sync = _sync_time(model.params_m, beta, n_batches) if n > 1 else 0.0
        return phi * (max(t_compute, t_io) + t_sync), t_compute, t_io, t_sync

    t1, *_ = single_epoch(1)

    if strategy == "single":
        t, tc, tio, tsync = single_epoch(1)
    elif strategy == "ddp":
        t, tc, tio, tsync = single_epoch(n_gpus)
    elif strategy == "model_parallel":
        # Naive pipeline: stages serialize, no data-parallel split. Compute is the
        # whole model on the full data (split across devices but sequential), plus
        # a small inter-stage activation transfer ≈ one extra all-reduce-ish term.
        t_compute = n_train / (pi * rc)
        t_io = n_train / rio
        transfer = _sync_time(model.params_m, beta, math.ceil(n_train / batch_per_gpu)) * 0.25
        t = phi * (max(t_compute, t_io) + transfer)
        tc, tio, tsync = t_compute, t_io, transfer
    elif strategy == "heterogeneous":
        # Synchronous DDP at the pace of the slowest rank (GPU + a ~50× slower CPU
        # worker). Throughput sum ≈ rc·(1 + 1/50); barrier dominates → ~no gain.
        rc_sum = pi * rc * (1 + 1 / 50)
        t_compute = n_train / rc_sum
        t_io = n_train / rio
        n_batches = math.ceil(n_train / (batch_per_gpu * n_gpus))
        t_sync = _sync_time(model.params_m, INTERCONNECT_BW["ethernet"], n_batches)
        t = phi * (max(t_compute, t_io) + t_sync) * 2.0  # gloo/barrier overhead
        tc, tio, tsync = t_compute, t_io, t_sync
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    speedup = t1 / t if t > 0 else 0.0
    eff = speedup / n_gpus if n_gpus else speedup
    # Bottleneck of the parallel part.
    if tsync >= max(tc / max(n_gpus, 1), tio) and strategy in ("ddp", "heterogeneous"):
        bottleneck = "sync"
    elif tio >= tc / max(n_gpus, 1):
        bottleneck = "io"
    else:
        bottleneck = "compute"
    return EpochPrediction(t, speedup, eff, bottleneck, tc, tio, tsync)


# ── Unified prediction API (what the tutor asked for) ────────────────────────────

@dataclass
class Prediction:
    strategy: str
    model_name: str
    gpu_name: str
    n_gpus: int
    precision: str
    time_per_epoch_train_s: float
    time_total_train_s: float
    speedup: float
    efficiency: float
    bottleneck: str
    vram_per_gpu_gb: float
    fits_in_memory: bool
    recommended_batch: int
    calibrated: bool          # True if a measured r_c/r_io was used
    notes: list[str] = field(default_factory=list)


def predict(strategy: str, model_name: str, gpu_name: str, n_gpus: int = 1,
            dataset_size: int = 5000, batch: int = 96, precision: str = "fp32",
            epochs: int = 15, disk_type: str = "ssd", nfs: bool = False,
            rc_measured: float | None = None,
            rio_measured: float | None = None) -> Prediction | None:
    """Closed-form prediction for any combination. Returns None for unknown specs.

    ``batch`` is the GLOBAL batch; it is split across the n GPUs for DDP.
    Pass ``rc_measured`` / ``rio_measured`` to calibrate against a real benchmark.
    """
    model = model_spec(model_name)
    gpu = gpu_spec(gpu_name)
    if model is None or gpu is None:
        return None

    batch_per_gpu = max(1, batch // n_gpus) if strategy in ("ddp", "heterogeneous") else batch
    ep = predict_epoch(strategy, model, gpu, n_gpus, dataset_size, batch_per_gpu,
                       precision, disk_type, nfs, rc_measured, rio_measured)

    vram = estimate_vram_gb(model, batch_per_gpu, precision)
    fits = vram <= gpu.vram_gb
    rec_b = max_batch(model, gpu, precision)

    notes: list[str] = []
    if strategy == "model_parallel":
        notes.append("Model parallelism does not accelerate (≈1×); it exists to fit "
                     "models too big for one GPU.")
    if strategy == "heterogeneous":
        notes.append("Synchronous DDP runs at the pace of the slowest worker (the CPU); "
                     "imbalanced hardware penalizes.")
    if not fits:
        notes.append(f"OOM: needs ~{vram:.1f} GB but the GPU has {gpu.vram_gb:.0f} GB. "
                     f"Largest batch that fits: {rec_b}.")

    return Prediction(
        strategy=strategy, model_name=model.name, gpu_name=gpu.name, n_gpus=n_gpus,
        precision=precision,
        time_per_epoch_train_s=ep.time_train_s,
        time_total_train_s=ep.time_train_s * epochs,
        speedup=ep.speedup, efficiency=ep.efficiency, bottleneck=ep.bottleneck,
        vram_per_gpu_gb=vram, fits_in_memory=fits, recommended_batch=rec_b,
        calibrated=bool(rc_measured or rio_measured), notes=notes,
    )
