r"""Analytic performance model — predict training time/speedup/memory/cost for
ANY (strategy, model, GPU, n_gpus, dataset, batch, precision) WITHOUT running a
benchmark, from curated hardware/model specs.

This generalizes the benchmark checker (which until now needed a real
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
# Fraction of VRAM actually usable for training: PyTorch's caching allocator
# reserves context + workspaces + fragmentation, so you OOM before 100%.
USABLE_VRAM_FRACTION: float = 0.92
# Activation memory is per-model (GB per image), NOT a single per-parameter
# constant: it scales with tokens × hidden_dim × layers, which is not the
# parameter count (a vit_base "weighs" proportionally more in activations than a
# vit_large). Calibrated from two real points — vit_base = 4.95 GB @ batch 32 on
# an RTX 3060 Ti, and vit_large = 13.78 GB @ batch 32 on a T4 — and extended to
# the rest by hidden_dim×layers. See ModelSpec.act_gb_per_img.

# ── Energy calibration ────────────────────────────────────────────────────────────
# Energy is just power × the time we already predict. The only missing term is the
# average power the GPU draws while training; instead of the nameplate TDP (a
# ceiling the GPU rarely reaches), we calibrate the EFFECTIVE average power from the
# project's own `--fn energy` runs (the "potencia media XX W" lines in the logs),
# exactly as MFU was calibrated from real throughput. Two measured facts shape the
# model:
#   1. Power is almost precision-independent (T4 fp32 ≈ 64 W ≈ AMP ≈ 63 W). AMP saves
#      energy by finishing ~4× faster — through TIME, which the model already predicts
#      — not by drawing less power. So power here does NOT vary with precision.
#   2. Eval draws slightly less power than train: see EVAL_POWER_FRACTION.
# Effective power lives on GpuSpec.train_power_w (W per GPU); GpuSpec.power_calibrated
# flags whether it comes from a real measurement (True) or a TDP fallback (False).
# The measured eval/train power ratio VARIES by GPU (T4 ≈ 0.98, V100 ≈ 0.7–1.0,
# RTX 3060 Ti ≈ 0.67); 0.9 is a single conservative approximation. Eval is short
# (forward-only over the small val split), so it is a few % of per-epoch energy.
EVAL_POWER_FRACTION: float = 0.9
# Fraction of nameplate TDP used as the effective-power fallback for GPUs we never
# measured here (datacenter cards on NFS run well below TDP; ~0.65 is a fair middle).
POWER_TDP_FALLBACK_FRACTION: float = 0.65

# Nameplate TDP (W) for GPUs whose effective power we did NOT measure here; the
# fallback power is POWER_TDP_FALLBACK_FRACTION × this (so editing the fraction
# propagates instead of hardcoding each value).
_TDP_W: dict[str, float] = {
    "RTX 3090": 350.0, "RTX 4090": 450.0, "A100 40GB": 400.0, "A100 80GB": 400.0,
    "P100": 250.0, "H100": 700.0,
}


def _tdp_fallback_power(name: str) -> float:
    return round(_TDP_W[name] * POWER_TDP_FALLBACK_FRACTION)


# ── Hardware spec tables ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GpuSpec:
    name: str
    fp32_tflops: float       # dense FP32 peak
    tensor_speedup: float    # measured r_c(amp)/r_c(fp32) — Tensor-core gain
    vram_gb: float
    interconnect: str        # "nvlink" | "pcie" (datacenter GPUs prefer NVLink)
    train_power_w: float = 0.0     # effective avg power per GPU during train (W)
    power_calibrated: bool = False  # True = measured here; False = TDP fallback


# fp32 TFLOPS, the measured/typical Tensor-core speedup π, and the effective average
# training power per GPU (W). The power of T4 / V100 / RTX 3060 Ti is CALIBRATED from
# the project's own `--fn energy` logs (potencia media); the rest are TDP-fallback
# estimates (≈0.65 × nameplate TDP) and flagged power_calibrated=False.
GPU_TABLE: dict[str, GpuSpec] = {
    "Tesla T4":      GpuSpec("Tesla T4", 8.1, 3.8, 16.0, "pcie", train_power_w=64.0, power_calibrated=True),    # Kaggle: 64 W fp32 ≈ 63 W amp
    "Tesla V100":    GpuSpec("Tesla V100", 14.0, 3.5, 32.0, "nvlink", train_power_w=103.0, power_calibrated=True),  # Verode: ~103 W avg (eval-phase readings; train pynvml sparse)
    "V100-PCIE":     GpuSpec("V100-PCIE", 14.0, 3.5, 32.0, "nvlink", train_power_w=103.0, power_calibrated=True),
    "RTX 3060 Ti":   GpuSpec("RTX 3060 Ti", 16.2, 2.54, 8.0, "pcie", train_power_w=180.0, power_calibrated=True),   # local: ~180 W (SSD, compute-bound)
    "RTX 3090":      GpuSpec("RTX 3090", 35.6, 3.5, 24.0, "pcie", train_power_w=_tdp_fallback_power("RTX 3090")),
    "RTX 4090":      GpuSpec("RTX 4090", 82.6, 4.0, 24.0, "pcie", train_power_w=_tdp_fallback_power("RTX 4090")),
    "A100 40GB":     GpuSpec("A100 40GB", 19.5, 4.0, 40.0, "nvlink", train_power_w=_tdp_fallback_power("A100 40GB")),
    "A100 80GB":     GpuSpec("A100 80GB", 19.5, 4.0, 80.0, "nvlink", train_power_w=_tdp_fallback_power("A100 80GB")),
    "P100":          GpuSpec("P100", 9.3, 1.0, 16.0, "pcie", train_power_w=_tdp_fallback_power("P100")),  # no usable Tensor cores
    "H100":          GpuSpec("H100", 67.0, 5.0, 80.0, "nvlink", train_power_w=_tdp_fallback_power("H100")),
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
    act_gb_per_img: float    # training activation memory (GB) per image


# Curated from timm / the literature (forward GFLOPs at 224²).
# act_gb_per_img: ViT family calibrated/scaled from the two measured points
# (base 0.093 from 4.95 GB @ b32 on 3060 Ti; large 0.260 from 13.78 GB @ b32 on
# T4), the others scaled by hidden_dim×layers; resnet/effnet are estimates.
MODEL_TABLE: dict[str, ModelSpec] = {
    "vit_tiny_patch16_224":  ModelSpec("vit_tiny_patch16_224", 5.7, 1.3, 0.024),
    "vit_small_patch16_224": ModelSpec("vit_small_patch16_224", 22.0, 4.6, 0.047),
    "vit_base_patch16_224":  ModelSpec("vit_base_patch16_224", 85.8, 17.6, 0.093),
    "vit_large_patch16_224": ModelSpec("vit_large_patch16_224", 303.0, 61.6, 0.260),
    "resnet50":              ModelSpec("resnet50", 25.6, 4.1, 0.100),
    "efficientnet_b0":       ModelSpec("efficientnet_b0", 5.3, 0.4, 0.060),
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


def estimate_power(gpu: GpuSpec, phase: str = "train") -> float:
    """Effective average power (W) for ONE GPU in the given phase ("train"|"eval").

    Calibrated from the project's logged `potencia media` (GpuSpec.train_power_w);
    eval draws ≈EVAL_POWER_FRACTION of train. Power is precision-independent (the
    AMP saving is in TIME, not watts), so precision is not a parameter here."""
    base = gpu.train_power_w if gpu.train_power_w and gpu.train_power_w > 0 else 0.0
    return base * (EVAL_POWER_FRACTION if phase == "eval" else 1.0)


def power_working_gpus(strategy: str, n_gpus: int) -> int:
    """How many GPUs draw power for a strategy. ddp/model-parallel power all the GPUs
    in the run (naive model-parallel keeps both GPUs powered even though the stages
    serialize). single and heterogeneous draw on ONE GPU: in this project's
    heterogeneous strategy the second worker is a CPU, not a GPU, so the GPU-power
    calibration applies to a single GPU (the CPU's wall power is negligible)."""
    if strategy in ("ddp", "model_parallel"):
        return max(1, n_gpus)
    return 1


def estimate_eval_time_s(model: ModelSpec, gpu: GpuSpec, n_val: int, strategy: str,
                         n_gpus: int, precision: str, disk_type: str = "ssd",
                         nfs: bool = False, rc_measured: float | None = None,
                         rio_measured: float | None = None) -> float:
    """Eval-epoch time (s): a forward-only pass over the val split.

    Eval has no backward, so its compute throughput is ≈TRAIN_FLOPS_FACTOR× the train
    r_c (forward is ~1/3 of forward+backward). It still reads the val set from disk, so
    it is max(compute, I/O), and it shards across the GPUs for ddp/heterogeneous."""
    phi = 1.3 if nfs else 1.0
    pi = precision_factor(gpu, precision)
    rc_train = rc_measured if (rc_measured and rc_measured > 0) else estimate_rc(
        model, gpu, precision="fp32")
    rc_fwd = rc_train * TRAIN_FLOPS_FACTOR     # forward-only is ~3× the train rate
    rio = estimate_rio(disk_type, nfs, rio_measured)
    n_eff = n_gpus if strategy in ("ddp", "heterogeneous") else 1
    t_compute = n_val / (pi * rc_fwd * max(1, n_eff))
    t_io = n_val / rio
    return phi * max(t_compute, t_io)


# ── Memory model ─────────────────────────────────────────────────────────────────

def estimate_vram_gb(model: ModelSpec, batch: int, precision: str = "fp32") -> float:
    """Peak VRAM (GB) for training: weights+grad+optimizer + activations + overhead.

    Activations use the model's per-image figure (``act_gb_per_img``), not a
    single per-parameter constant — that is what makes the per-model max batch
    realistic (e.g. vit_base on an 8 GB card tops out at batch 32, not 64)."""
    p = model.params_m * 1e6
    weights_opt_gb = BYTES_PER_PARAM_GRAD_OPT * p / 1e9
    if precision in ("amp", "bf16"):
        weights_opt_gb += 2 * p / 1e9          # extra fp16 copy of weights+grad
    act_gb = model.act_gb_per_img * batch
    if precision in ("amp", "bf16"):
        act_gb *= 0.6                          # half-precision activations
    return weights_opt_gb + act_gb + CUDA_OVERHEAD_GB


def fits_in_memory(model: ModelSpec, gpu: GpuSpec, batch: int,
                   precision: str = "fp32") -> bool:
    """Fits if the estimate is within the *usable* VRAM (not 100% of it)."""
    return estimate_vram_gb(model, batch, precision) <= gpu.vram_gb * USABLE_VRAM_FRACTION


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
    # Per-epoch time breakdown (seconds), so callers can show the formula behind it:
    # time/epoch = max(compute, io) + sync. Bottleneck = the dominant term.
    t_compute_s: float = 0.0
    t_io_s: float = 0.0
    t_sync_s: float = 0.0
    batch_per_gpu: int = 0
    # Eval + total time (seconds). time_total covers train+eval, matching what a run
    # and the benchmark report per epoch.
    time_per_epoch_eval_s: float = 0.0
    time_per_epoch_total_s: float = 0.0
    time_total_s: float = 0.0          # total over all epochs (train+eval)
    # Energy = effective power × the time above. avg_power_w is per GPU; power_total_w
    # multiplies by the GPUs actually working. All energies in Wh.
    avg_power_w: float = 0.0           # effective power per GPU during train
    power_total_w: float = 0.0         # × n working GPUs
    energy_train_wh: float = 0.0       # per epoch
    energy_eval_wh: float = 0.0        # per epoch
    energy_per_epoch_wh: float = 0.0   # train + eval, per epoch
    energy_total_wh: float = 0.0       # × epochs (train + eval)
    power_calibrated: bool = False     # power from a real measurement vs TDP fallback


def predict(strategy: str, model_name: str, gpu_name: str, n_gpus: int = 1,
            dataset_size: int = 5000, batch: int = 96, precision: str = "fp32",
            epochs: int = 15, disk_type: str = "ssd", nfs: bool = False,
            rc_measured: float | None = None,
            rio_measured: float | None = None,
            val_size: int | None = None) -> Prediction | None:
    """Closed-form prediction for any combination. Returns None for unknown specs.

    ``batch`` is the GLOBAL batch; it is split across the n GPUs for DDP.
    ``dataset_size`` is the train split; ``val_size`` is the val split used for the
    eval-epoch time/energy (defaults to ~0.51× the train split, the BigEarthNet ratio).
    Pass ``rc_measured`` / ``rio_measured`` to calibrate against a real benchmark.
    """
    model = model_spec(model_name)
    gpu = gpu_spec(gpu_name)
    if model is None or gpu is None:
        return None

    batch_per_gpu = max(1, batch // n_gpus) if strategy in ("ddp", "heterogeneous") else batch
    ep = predict_epoch(strategy, model, gpu, n_gpus, dataset_size, batch_per_gpu,
                       precision, disk_type, nfs, rc_measured, rio_measured)

    # Eval-epoch time (forward-only over the val split) → total epoch = train + eval.
    n_val = int(val_size) if val_size and val_size > 0 else round(dataset_size * 0.51)
    t_eval = estimate_eval_time_s(model, gpu, n_val, strategy, n_gpus, precision,
                                  disk_type, nfs, rc_measured, rio_measured)
    t_total_epoch = ep.time_train_s + t_eval

    # Energy = effective power × time. Power per GPU is calibrated (or a TDP fallback);
    # the working-GPU count scales it for distributed strategies.
    n_work = power_working_gpus(strategy, n_gpus)
    p_train = estimate_power(gpu, "train")
    p_eval = estimate_power(gpu, "eval")
    power_total = p_train * n_work
    e_train_wh = power_total * ep.time_train_s / 3600.0
    e_eval_wh = (p_eval * n_work) * t_eval / 3600.0
    e_epoch_wh = e_train_wh + e_eval_wh

    vram = estimate_vram_gb(model, batch_per_gpu, precision)
    # Use the SAME usable-VRAM threshold (0.92) as max_batch/fits_in_memory, so the
    # `fits` flag, the OOM note and the recommended batch never contradict each other.
    fits = fits_in_memory(model, gpu, batch_per_gpu, precision)
    rec_b = max_batch(model, gpu, precision)

    notes: list[str] = []
    if strategy == "model_parallel":
        notes.append("Model parallelism does not accelerate (≈1×); it exists to fit "
                     "models too big for one GPU.")
    if strategy == "heterogeneous":
        notes.append("Synchronous DDP runs at the pace of the slowest worker (the CPU); "
                     "imbalanced hardware penalizes.")
    if not fits:
        usable = gpu.vram_gb * USABLE_VRAM_FRACTION
        notes.append(f"OOM: needs ~{vram:.1f} GB but only ~{usable:.1f} GB of the "
                     f"{gpu.vram_gb:.0f} GB is usable (PyTorch reserves ~8%). "
                     f"Largest batch that fits: {rec_b}.")
    if not gpu.power_calibrated:
        notes.append(f"Energy uses a TDP-fallback power for {gpu.name} (≈"
                     f"{gpu.train_power_w:.0f} W/GPU, not measured here) — treat it as "
                     f"a rough estimate.")

    return Prediction(
        strategy=strategy, model_name=model.name, gpu_name=gpu.name, n_gpus=n_gpus,
        precision=precision,
        time_per_epoch_train_s=ep.time_train_s,
        time_total_train_s=ep.time_train_s * epochs,
        speedup=ep.speedup, efficiency=ep.efficiency, bottleneck=ep.bottleneck,
        vram_per_gpu_gb=vram, fits_in_memory=fits, recommended_batch=rec_b,
        calibrated=bool(rc_measured or rio_measured), notes=notes,
        t_compute_s=ep.t_compute_s, t_io_s=ep.t_io_s, t_sync_s=ep.t_sync_s,
        batch_per_gpu=batch_per_gpu,
        time_per_epoch_eval_s=t_eval,
        time_per_epoch_total_s=t_total_epoch,
        time_total_s=t_total_epoch * epochs,
        avg_power_w=p_train, power_total_w=power_total,
        energy_train_wh=e_train_wh, energy_eval_wh=e_eval_wh,
        energy_per_epoch_wh=e_epoch_wh, energy_total_wh=e_epoch_wh * epochs,
        power_calibrated=gpu.power_calibrated,
    )


# ── Quality model: expected best-F1 vs dataset size (empirical prior) ─────────────
# This is the honest counterpart to the time model. It does NOT pretend to predict
# an exact F1 from first principles — that would require training. Instead it is an
# EMPIRICAL PRIOR: anchored to our documented BigEarthNet-S2 runs and extended to
# any dataset fraction with the standard log-linear data-scaling law
#
#     F1_inf(N) = F1_full − k · log10(N_full / N)
#
# (more data → higher F1 with diminishing returns; the same shape used by the
# Hestness/Kaplan data-scaling studies). It is calibrated against TWO real points
# for vit_base — full dataset = 0.68, the 5 000-image subset = 0.55 — which pins
# k ≈ 0.078. Smaller families lose more F1 on little data (steeper k). For families
# we have not run to convergence the confidence is flagged low. The MEASURED
# alternative is the convergence study (LR range test + real mini-training); this
# prior is for planning a run before spending the GPU hours.

N_FULL_TRAIN: int = 237_871   # BigEarthNet-S2 train split (the anchor for N_full)
N_SUBSET_TRAIN: int = 5_000   # the demo subset used across the comparative study

# family → (F1_full, best_epoch, early_stop_epoch, data_sensitivity_k, confidence)
_QUALITY_ANCHORS: dict[str, tuple[float, int, int, float, str]] = {
    "vit_base":     (0.68, 7, 17, 0.078, "high"),    # v1–v4, full ↔ subset both measured
    "vit_small":    (0.64, 8, 18, 0.110, "medium"),
    "vit_tiny":     (0.59, 8, 18, 0.190, "medium"),  # subset 5k ≈ 0.27 measured (Kaggle)
    "vit_large":    (0.69, 7, 16, 0.070, "low"),     # not run to convergence
    "resnet50":     (0.60, 10, 22, 0.130, "low"),
    "efficientnet": (0.56, 9, 20, 0.140, "low"),
}
_CONF_BAND = {"high": 0.020, "medium": 0.035, "low": 0.050}  # ± F1 by confidence


def _quality_family(name: str) -> str:
    n = (name or "").lower()
    for fam in ("vit_base", "vit_small", "vit_tiny", "vit_large", "resnet", "efficientnet"):
        if fam in n:
            return "resnet50" if fam == "resnet" else fam
    if "deit" in n:
        return "vit_base"          # DeiT behaves like ViT
    return "vit_base"


@dataclass
class QualityPrediction:
    model_name: str
    dataset_size: int
    expected_best_f1: float
    best_epoch: int
    early_stop_epoch: int
    confidence: str               # "high" | "medium" | "low"
    band: float                   # ± F1 uncertainty (widens as confidence drops)
    method: str                   # always "empirical-prior" here (vs the study's "measured")
    curve_epochs: list[int] = field(default_factory=list)
    curve_val_f1: list[float] = field(default_factory=list)
    curve_train_f1: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def expected_best_f1(model_name: str, dataset_size: int = N_FULL_TRAIN) -> tuple[float, str, float]:
    """Empirical-prior best Val F1 for a model on N training images.

    Returns ``(f1, confidence, band)``. Uses the log-linear data-scaling law
    anchored to the documented full-dataset plateau of the model family.
    """
    fam = _quality_family(model_name)
    f_full, _, _, k, conf = _QUALITY_ANCHORS[fam]
    n = max(1, int(dataset_size))
    drop = k * math.log10(max(1.0, N_FULL_TRAIN / n))
    f1 = max(0.0, min(f_full, f_full - drop))
    return round(f1, 3), conf, _CONF_BAND[conf]


def _learning_curve(f_inf: float, best_epoch: int, epochs: list[int]
                    ) -> tuple[list[float], list[float]]:
    """Standard saturating learning curve, normalised to peak f_inf at best_epoch.

    Val rises as f_inf·(1−e^(−t/τ)) (renormalised so val(best_epoch)=f_inf), then
    decays slightly from overfitting. Train rises higher and faster (the gap)."""
    tau = max(1.5, best_epoch / 2.2)
    denom = 1.0 - math.exp(-best_epoch / tau)
    ceil_train = min(0.97, f_inf + 0.22)
    tau_train = tau * 0.7
    val, train = [], []
    for e in epochs:
        if e <= best_epoch:
            v = f_inf * (1.0 - math.exp(-e / tau)) / denom
        else:
            v = max(f_inf * 0.96, f_inf - min(0.03, (e - best_epoch) * 0.0015))
        val.append(round(v, 3))
        train.append(round(ceil_train * (1.0 - math.exp(-e / tau_train)), 3))
    return val, train


def predict_quality(model_name: str, dataset_size: int = N_FULL_TRAIN,
                    epochs: int = 30) -> QualityPrediction | None:
    """Empirical-prior quality estimate (best Val F1 + a learning curve).

    The honest planning counterpart to ``predict``: anchored to documented runs,
    extended to any dataset size by the data-scaling law. Not a measurement — for
    that, run the convergence study. Returns ``None`` for an unknown family only
    in the sense that it always falls back to the ViT-Base prior (so never None
    here, but kept Optional for symmetry with ``predict``)."""
    fam = _quality_family(model_name)
    f_full, best_ep, stop_ep, _k, _conf = _QUALITY_ANCHORS[fam]
    f1, conf, band = expected_best_f1(model_name, dataset_size)

    n_show = max(best_ep + 2, min(int(epochs), 40))
    curve_epochs = list(range(1, n_show + 1))
    val, train = _learning_curve(f1, best_ep, curve_epochs)
    stop = min(stop_ep, n_show) if epochs >= stop_ep else min(epochs, n_show)

    frac = dataset_size / N_FULL_TRAIN
    notes = [
        f"Empirical prior anchored to documented {fam} runs on BigEarthNet-S2; "
        f"confidence {conf}.",
        f"Dataset = {int(dataset_size):,} train images "
        f"({frac*100:.0f}% of the full {N_FULL_TRAIN:,}).",
    ]
    if frac < 0.9:
        notes.append(f"Smaller dataset lowers the expected F1 (full-set plateau ≈ {f_full:.2f}).")
    notes.append("This is a planning prior, not a measurement — run the convergence "
                 "study for a measured estimate.")

    return QualityPrediction(
        model_name=model_name, dataset_size=int(dataset_size),
        expected_best_f1=f1, best_epoch=best_ep,
        early_stop_epoch=stop, confidence=conf, band=band,
        method="empirical-prior",
        curve_epochs=curve_epochs, curve_val_f1=val, curve_train_f1=train,
        notes=notes,
    )
