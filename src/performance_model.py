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
# Fraction of VRAM actually usable for training: PyTorch's caching allocator
# reserves context + workspaces + fragmentation, so you OOM before 100%.
USABLE_VRAM_FRACTION: float = 0.92
# Activation memory is per-model (GB per image), NOT a single per-parameter
# constant: it scales with tokens × hidden_dim × layers, which is not the
# parameter count (a vit_base "weighs" proportionally more in activations than a
# vit_large). Calibrated from two real points — vit_base = 4.95 GB @ batch 32 on
# an RTX 3060 Ti, and vit_large = 13.78 GB @ batch 32 on a T4 — and extended to
# the rest by hidden_dim×layers. See ModelSpec.act_gb_per_img.


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
        t_compute_s=ep.t_compute_s, t_io_s=ep.t_io_s, t_sync_s=ep.t_sync_s,
        batch_per_gpu=batch_per_gpu,
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
