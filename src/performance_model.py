r"""Analytic performance model — predict training time/speedup/memory/cost for
ANY (strategy, model, GPU, n_gpus, dataset, batch, precision) WITHOUT running a
benchmark, from curated hardware/model specs.

This generalizes the benchmark checker (which until now needed a real
benchmark on the target machine) into a closed-form predictor, as requested by
the tutor. The formulas are derived in docs/performance_model.md; the master
equation for a DDP train epoch is:

    T(N, n) = φ · [ max( N/(r_c·n) , N/r_io )  +  (2(n−1)/n · 4P/β)·⌈N/(b·n)⌉ ]
              \____ compute ∥ disk (the slower) ____/   \_____ gradient sync ______/

with parameters estimated from specs:
    r_c  = training throughput of one GPU at the precision: MFU · TFLOPS_fp32 / FLOPs_train
           × π (Tensor-core speedup) in fp32/bf16; three terms (compute + memory traffic +
           fixed cost per step) with AMP
    r_io = disk throughput (img/s), from a disk-type table or a real probe
    β    = all-reduce bus bandwidth (measured per GPU, else derated NVLink/PCIe nominal)
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
# The measured eval/train power ratio VARIES by GPU (T4 ≈ 0.98, V100 0.65, RTX 3060 Ti
# ≈ 0.67); 0.9 is the default for GPUs without their own GpuSpec.eval_power_fraction. Eval
# is short (forward-only over the small val split), so it is a few % of per-epoch energy.
EVAL_POWER_FRACTION: float = 0.9
# Fraction of nameplate TDP used as the effective-power fallback for GPUs we never
# measured here (datacenter cards on NFS run well below TDP; ~0.65 is a fair middle).
POWER_TDP_FALLBACK_FRACTION: float = 0.65
# Effective per-GPU power of the strategies that split the model, as a fraction of
# train_power_w: linear in the fraction of the step each GPU is busy, busy = m/(m+p−1) (naive
# model parallelism is m = 1). Least-squares fit to the 4 Kaggle 2×T4 runs of 24/11/2026 (ViT-B,
# 512 px, batch 16: model parallel 109.7 W, pipeline m = 2/4/8 118.8/125.0/129.2 W for the 2 GPUs;
# logs/kaggle/panda/{model_parallel,pipeline}/vit_base_patch16_224.augreg2_in21k_ft_in1k/), errors
# ≤0.5 %. Validated out of sample on the pipeline_micro1 runs (m = 16 at batch 16: +0.2 %; m = 8 at
# batch 8: −2.0 %). Above 1 when busy → 1: a half-loaded T4 stays under its 70 W cap and clocks
# higher than the single-GPU reference (64 W). Calibrated on the T4 only.
SPLIT_POWER_BASE: float = 0.664
SPLIT_POWER_SLOPE: float = 0.390
# Naive model parallelism on 2 GPUs (busy 1/2): 0.86 (an older 2×T4 run gave 0.83).
MODEL_PARALLEL_POWER_FRACTION: float = SPLIT_POWER_BASE + SPLIT_POWER_SLOPE / 2
# GPipe: cost of each extra micro-batch in a step (s), on top of the fixed cost per step d of
# the three-term model (which is paid once per step: optimizer, zero_grad, scaler). Least-squares
# fit to the epoch time (train + val) of the pipeline m = 2/4/8 runs above (+4.9/+1.5/−1.5 %);
# validated out of sample on pipeline_micro1 (1-image micro-batches: m = 16 at batch 16, −5.2 %;
# m = 8 at batch 8, −3.3 %), where charging d to every micro-batch erred +19 % and charging it
# once per step −10/−8 %. The VALUE of c is out of sample; the FORM (a cost per micro-batch) was
# chosen after seeing that m = 16 runs slower than m = 8 there. logs/kaggle/panda/pipeline/vit_base_patch16_224.augreg2_in21k_ft_in1k/.
# Calibrated on the T4.
PIPELINE_MICRO_BATCH_COST_S: float = 3.4e-3

# GPU names shared by the TDP and spec tables below.
_T4 = "Tesla T4"
_V100 = "Tesla V100"
_RTX_3060_TI = "RTX 3060 Ti"
_RTX_3090 = "RTX 3090"
_RTX_4090 = "RTX 4090"
_A100_40GB = "A100 40GB"
_A100_80GB = "A100 80GB"

# Nameplate TDP (W) for GPUs whose effective power we did NOT measure here; the
# fallback power is POWER_TDP_FALLBACK_FRACTION × this (so editing the fraction
# propagates instead of hardcoding each value).
_TDP_W: dict[str, float] = {
    _RTX_3090: 350.0, _RTX_4090: 450.0, _A100_40GB: 400.0, _A100_80GB: 400.0,
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
    mem_bw_gbs: float = 0.0        # DRAM bandwidth from the datasheet (GB/s); 0 = unknown
    # MEASURED all-reduce bus bandwidth between the GPUs of one machine (GB/s, nccl-tests
    # convention, scripts/measure_allreduce.py); 0 = not measured: INTERCONNECT_BW is used.
    allreduce_busbw_gbs: float = 0.0
    # MEASURED eval/train power ratio of this GPU; 0 = not measured: EVAL_POWER_FRACTION is used.
    eval_power_fraction: float = 0.0


# fp32 TFLOPS, the measured/typical Tensor-core speedup π, and the effective average
# training power per GPU (W). The power of T4 / V100 / RTX 3060 Ti is CALIBRATED from
# the project's own `--fn energy` logs (potencia media); the rest are TDP-fallback
# estimates (≈0.65 × nameplate TDP) and flagged power_calibrated=False.
# mem_bw_gbs: DRAM bandwidth of the NVIDIA datasheets (GB/s); the SXM figure where the key
# covers both forms (A100 80GB, H100), consistent with the fp32_tflops of the same row.
GPU_TABLE: dict[str, GpuSpec] = {
    # T4 all-reduce: 2 × T4 on Kaggle, 4.05 GB/s with 64-512 MB messages and 4.01 GB/s with
    # 16 MB, close to DDP's 25 MB buckets (strategies_512 batch, 23/11/2026; docs/research/datos/
    # allreduce_t4_kaggle.csv) — a quarter of the PCIe nominal. Measured with 2 GPUs only.
    _T4:             GpuSpec(_T4, 8.1, 3.8, 16.0, "pcie", train_power_w=64.0, power_calibrated=True, mem_bw_gbs=320.0,
                             allreduce_busbw_gbs=4.05),    # Kaggle: 64 W fp32 ≈ 63 W amp
    # V100 power: median of the train-phase `potencia media` of 8 PANDA trainings at 512 px with AMP
    # on verode21 (26/11/2026, logs/verode/panda/single/*/train_26112026_*; 142-218 W by model,
    # eval/train 0.65, median of the same runs). The former 103 W came from sparse eval-phase
    # readings (May 2026) and put the energy of every V100 prediction ~45 % low.
    _V100:           GpuSpec(_V100, 14.0, 3.5, 32.0, "nvlink", train_power_w=181.0, power_calibrated=True, mem_bw_gbs=900.0, eval_power_fraction=0.65),
    "V100-PCIE":     GpuSpec("V100-PCIE", 14.0, 3.5, 32.0, "nvlink", train_power_w=181.0, power_calibrated=True, mem_bw_gbs=900.0, eval_power_fraction=0.65),
    _RTX_3060_TI:    GpuSpec(_RTX_3060_TI, 16.2, 2.54, 8.0, "pcie", train_power_w=180.0, power_calibrated=True, mem_bw_gbs=448.0),   # local: ~180 W (SSD, compute-bound)
    _RTX_3090:      GpuSpec(_RTX_3090, 35.6, 3.5, 24.0, "pcie", train_power_w=_tdp_fallback_power(_RTX_3090), mem_bw_gbs=936.0),
    _RTX_4090:      GpuSpec(_RTX_4090, 82.6, 4.0, 24.0, "pcie", train_power_w=_tdp_fallback_power(_RTX_4090), mem_bw_gbs=1008.0),
    _A100_40GB:     GpuSpec(_A100_40GB, 19.5, 4.0, 40.0, "nvlink", train_power_w=_tdp_fallback_power(_A100_40GB), mem_bw_gbs=1555.0),
    _A100_80GB:     GpuSpec(_A100_80GB, 19.5, 4.0, 80.0, "nvlink", train_power_w=_tdp_fallback_power(_A100_80GB), mem_bw_gbs=2039.0),
    "P100":          GpuSpec("P100", 9.3, 1.0, 16.0, "pcie", train_power_w=_tdp_fallback_power("P100"), mem_bw_gbs=732.0),  # no usable Tensor cores
    "H100":          GpuSpec("H100", 67.0, 5.0, 80.0, "nvlink", train_power_w=_tdp_fallback_power("H100"), mem_bw_gbs=3350.0),
}

# Disk throughput in IMAGES/second (includes TIFF decode + transform), calibrated
# from the real I/O probes: Kaggle SSD ≈ 130 img/s. Override with a real measure.
DISK_RIO: dict[str, float] = {
    "nvme": 220.0, "ssd": 130.0, "hdd": 45.0, "nfs": 100.0, "unknown": 130.0,
}

# Nominal interconnect bandwidth in bytes/s, used as the all-reduce bus bandwidth when a GPU
# has no measured one (GpuSpec.allreduce_busbw_gbs). Datasheet figures: the measured T4 pair on
# Kaggle sustained a quarter of the PCIe value, so these are optimistic.
INTERCONNECT_BW: dict[str, float] = {
    "nvlink": 300e9, "pcie": 16e9, "ethernet": 0.125e9,  # 1 GbE
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_m: float          # millions of parameters
    gflops_fwd: float        # forward GFLOPs per 224×224 image (multiply-add = 1, see below)
    act_gb_per_img: float    # training activation memory (GB) per image
    calibrated: bool = True  # False = profiled automatically, activations by rule
    # GB that autograd saves for the backward pass of one image (fp32, parameters excluded;
    # src/model_profile.saved_activation_gb). It is the memory-TRAFFIC feature of the
    # three-term throughput model, not a VRAM figure (that is act_gb_per_img). None = not
    # measured: the throughput falls back to the FLOPs-only MFU estimate.
    saved_gb_per_img: float | None = None


# Curated from timm / the literature (forward GFLOPs at 224²).
# Unit: these are what timm reports as "GMACs" (a multiply-add counts as ONE
# operation; vit_base = 17.6). torch's FlopCounterMode counts a multiply-add as TWO
# (vit_base = 35.1), so src/model_profile.py halves its count to stay in this unit.
# MFU was calibrated against these numbers, so it absorbs the factor 2.
# act_gb_per_img: ViT family calibrated/scaled from the two measured points
# (base 0.093 from 4.95 GB @ b32 on 3060 Ti; large 0.260 from 13.78 GB @ b32 on
# T4), the others scaled by hidden_dim×layers; resnet/effnet are estimates.
# saved_gb_per_img: measured with src/model_profile.saved_activation_gb(name, 3, 19, 224)
# on 23/11/2026 (CPU, flash attention, parameters excluded).
MODEL_TABLE: dict[str, ModelSpec] = {
    "vit_tiny_patch16_224":  ModelSpec("vit_tiny_patch16_224", 5.7, 1.3, 0.024, saved_gb_per_img=0.0300),
    "vit_small_patch16_224": ModelSpec("vit_small_patch16_224", 22.0, 4.6, 0.047, saved_gb_per_img=0.0594),
    "vit_base_patch16_224":  ModelSpec("vit_base_patch16_224", 85.8, 17.6, 0.093, saved_gb_per_img=0.1182),
    "vit_large_patch16_224": ModelSpec("vit_large_patch16_224", 303.0, 61.6, 0.260, saved_gb_per_img=0.3124),
    "resnet50":              ModelSpec("resnet50", 25.6, 4.1, 0.100, saved_gb_per_img=0.0863),
    "efficientnet_b0":       ModelSpec("efficientnet_b0", 5.3, 0.4, 0.060, saved_gb_per_img=0.0885),
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


def _gpu_key(name: str) -> str:
    """Comparison key of a GPU name: no vendor words, case, spaces or punctuation."""
    n = name.lower().replace("nvidia", "").replace("geforce", "")
    return "".join(c for c in n if c.isalnum())


# Names the driver reports (torch.cuda.get_device_name) that are NOT the table key: one
# explicit entry per real variant. Anything not listed or equal to a key is unknown, so
# 'RTX 3060' (12 GB) is never taken for 'RTX 3060 Ti' (8 GB) nor 'A100' for a given A100.
_GPU_ALIASES: dict[str, str] = {
    "Tesla V100-PCIE-16GB": _V100, "Tesla V100-PCIE-32GB": _V100,
    "Tesla V100-SXM2-16GB": _V100, "Tesla V100-SXM2-32GB": _V100,
    "Tesla P100": "P100", "Tesla P100-PCIE-16GB": "P100",
    "A100-SXM4-40GB": _A100_40GB, "A100-PCIE-40GB": _A100_40GB,
    "A100-SXM4-80GB": _A100_80GB, "A100-PCIE-80GB": _A100_80GB, "A100 80GB PCIe": _A100_80GB,
    "H100 PCIe": "H100", "H100 80GB HBM3": "H100", "H100 SXM5 80GB": "H100",
}
_GPU_INDEX: dict[str, GpuSpec] = {
    **{_gpu_key(k): v for k, v in GPU_TABLE.items()},
    **{_gpu_key(a): GPU_TABLE[k] for a, k in _GPU_ALIASES.items()},
}


def gpu_spec(name: str | None) -> GpuSpec | None:
    """Spec of a GPU by its name (exact after normalizing, or an explicit alias); ``None`` if
    it is not in the table — the caller must say so instead of guessing a similar card."""
    return _GPU_INDEX.get(_gpu_key(name)) if name else None


def model_spec(name: str | None) -> ModelSpec | None:
    return _fuzzy(MODEL_TABLE, name)


# ── Models outside MODEL_TABLE: spec from an automatic profile ───────────────────
# src/model_profile.py measures params and forward FLOPs of any timm model with
# torch's FLOP counter; this pure part turns that profile into a ModelSpec. What a
# FLOP count cannot give is the activation memory per image, so it is ESTIMATED by
# a rule anchored to the two calibrated ViT points (MODEL_TABLE: vit_base 0.093 GB
# @ 17.6 GFLOPs, vit_large 0.260 GB @ 61.6 GFLOPs):
#   * above vit_base: the power law through both points (exponent ≈ 0.82);
#   * below vit_base: exponent 0.5 — at fixed depth a ViT's activations grow with
#     the width d while its FLOPs grow with d², which is the hidden_dim×layers
#     scaling the table uses for vit_tiny/vit_small (the rule reproduces them
#     within 3%). The flatter exponent errs on the side of MORE memory for small
#     models (a straight per-FLOP rule would give vit_tiny 0.007 instead of 0.024);
#   * convolutional backbones keep full-resolution feature maps (+ BatchNorm), so
#     they carry far more activation per FLOP: × the factor that maps the rule onto
#     the resnet50 entry (≈2.2). That entry over-predicts the measured resnet50
#     peaks (logs/*/benchmark, ×1.2–1.5), so the factor is on the safe side.
# Uncalibrated by construction: specs built here carry calibrated=False and
# predict() adds a note saying so.
_ACT_SMALL_MODEL_EXPONENT: float = 0.5
_VIT_BASE = MODEL_TABLE["vit_base_patch16_224"]
_VIT_LARGE = MODEL_TABLE["vit_large_patch16_224"]
_ACT_LARGE_MODEL_EXPONENT: float = (
    math.log(_VIT_LARGE.act_gb_per_img / _VIT_BASE.act_gb_per_img)
    / math.log(_VIT_LARGE.gflops_fwd / _VIT_BASE.gflops_fwd))


def _vit_act_rule(gflops_fwd: float) -> float:
    exp = _ACT_SMALL_MODEL_EXPONENT if gflops_fwd <= _VIT_BASE.gflops_fwd else _ACT_LARGE_MODEL_EXPONENT
    return _VIT_BASE.act_gb_per_img * (gflops_fwd / _VIT_BASE.gflops_fwd) ** exp


CONV_ACTIVATION_FACTOR: float = (MODEL_TABLE["resnet50"].act_gb_per_img
                                 / _vit_act_rule(MODEL_TABLE["resnet50"].gflops_fwd))


def estimate_act_gb_per_img(gflops_fwd: float, convolutional: bool = False) -> float:
    """Activation memory (GB/image) for a model NOT in MODEL_TABLE, by rule.

    See the comment above: power law through the calibrated ViT points, flatter
    (exponent 0.5) below vit_base, × CONV_ACTIVATION_FACTOR for CNN backbones."""
    act = _vit_act_rule(max(gflops_fwd, 1e-6))
    return act * (CONV_ACTIVATION_FACTOR if convolutional else 1.0)


def spec_from_profile(name: str, params_m: float, gflops_fwd: float,
                      convolutional: bool = False,
                      saved_gb_per_img: float | None = None) -> ModelSpec:
    """ModelSpec for a profiled model: measured params/FLOPs (and, when given, saved
    activation GB for the throughput model) + estimated activation memory."""
    return ModelSpec(name, params_m, gflops_fwd,
                     estimate_act_gb_per_img(gflops_fwd, convolutional), calibrated=False,
                     saved_gb_per_img=saved_gb_per_img)


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


# ── Three-term throughput model (mixed precision) ────────────────────────────────
# Seconds per training image on ONE GPU (docs/performance_model.md, section 8):
#
#     t = a·F + b·A + d/B
#
# F = forward GFLOPs per image (the MODEL_TABLE unit), A = GB autograd saves per image
# (ModelSpec.saved_gb_per_img), B = images per step on that GPU. a is the inverse of an
# effective compute rate, b of an effective memory rate and d a fixed cost per step
# (kernel launches, optimizer step, Python). The FLOPs-only form (MFU) misses the memory
# term: 193 % mean error across 10 architectures against 10 % with it (leave-one-out).
#
# Fitted with src/analysis/throughput_model.fit_with_overhead on synthetic batches
# (`paravit benchmark`, no I/O), AMP, data in docs/research/datos/throughput_arquitecturas_amp.csv:
#   * RTX 3060 Ti — 10 architectures at 224/384/512 px with batch 16 plus 5 of them at 224 px with
#     batch 8/32/64 (45 points; logs/local/benchmark/benchmark_23112026_{10,16,223}*). The batch
#     sweep pins d: the per-step intercepts measured per model are 12-20 ms, the fit gives 18 ms;
#   * Tesla T4    — 13 architectures at 512 px with batch 16 plus 5 of them at 224 px with
#     batch 8/16/32/64 (33 points; logs/kaggle/benchmark/throughput_t4_{23,24}112026.csv). Per-model
#     intercepts 15-40 ms, the fit gives 20 ms. The former fit (512 px only, d = 24 ms) predicted
#     the 20 new points with 13 % mean error before they were added (out-of-sample);
#   * Tesla V100  — the same 33-point design on the V100-PCIE-32GB of VERODE (verode21, 26/11/2026;
#     logs/verode/benchmark/throughput_v100_26112026.csv): 13 % mean error in-sample, 15 % leaving
#     each model out. d = 34 ms, almost twice the others (hypothesis, not measured: d is host-side and
#     the node's CPU is older). Before this fit, the datasheet transfer of the 3060 Ti fit predicted these
#     measurements with 19 % mean error at 512 px (median ×1.09 too optimistic at batch 16; ×1.46 on the T4)
#     and 36 % at 224 px with batch 8-64, where the 3060 Ti's d (18 ms) falls short.
# Only measured with AMP (fp16 autocast), so it is applied to amp only; fp32 keeps MFU and bf16
# (not measured; the T4 and V100 have no native bf16) keeps MFU × π.
MIXED_PRECISIONS: tuple[str, ...] = ("amp",)


@dataclass(frozen=True)
class ThroughputFit:
    a_s_per_gflop: float      # compute term
    b_s_per_gb: float         # memory-traffic term
    d_s_per_step: float       # fixed cost per optimizer step


THROUGHPUT_FITS: dict[str, ThroughputFit] = {
    _RTX_3060_TI: ThroughputFit(2.0453e-4, 1.2468e-2, 17.77e-3),
    _T4:          ThroughputFit(3.3620e-4, 3.1241e-2, 20.08e-3),
    _V100:        ThroughputFit(1.2392e-4, 8.7489e-3, 33.98e-3),
}
THROUGHPUT_FITS["V100-PCIE"] = THROUGHPUT_FITS[_V100]      # the same card, by its other table name
# GPUs without a fit get the reference fit scaled by the datasheets: a by TFLOPS·π, b by DRAM
# bandwidth (d is host-side and kept). On the T4 this transfer was ×1.5 too optimistic (median
# ratio of the 13 models, 1.46), mostly in the memory term, while the relative cost of the
# architectures held within 11 %; so transferred predictions are flagged as uncalibrated.
THROUGHPUT_REFERENCE_GPU: str = _RTX_3060_TI


def throughput_fit(gpu: GpuSpec) -> tuple[ThroughputFit, bool] | None:
    """``(fit, calibrated)`` for a GPU: its own measured fit, or the reference one
    transferred by the datasheets (``calibrated=False``); ``None`` if it lacks a bandwidth."""
    own = THROUGHPUT_FITS.get(gpu.name)
    if own is not None:
        return own, True
    ref_gpu = GPU_TABLE[THROUGHPUT_REFERENCE_GPU]
    if gpu.mem_bw_gbs <= 0 or gpu.fp32_tflops <= 0:
        return None
    ref = THROUGHPUT_FITS[THROUGHPUT_REFERENCE_GPU]
    k_c = (ref_gpu.fp32_tflops * ref_gpu.tensor_speedup) / (gpu.fp32_tflops * gpu.tensor_speedup)
    k_m = ref_gpu.mem_bw_gbs / gpu.mem_bw_gbs
    return ThroughputFit(ref.a_s_per_gflop * k_c, ref.b_s_per_gb * k_m, ref.d_s_per_step), False


def three_term_ips(model: ModelSpec, gpu: GpuSpec, batch_per_gpu: int
                   ) -> tuple[float, bool] | None:
    """``(train img/s on one GPU, calibrated GPU)`` from the three-term model, or ``None``
    when it does not apply (no saved-activation figure for the model, or no fit/bandwidth
    for the GPU). Mixed precision only: the caller checks the precision."""
    if model.saved_gb_per_img is None or model.saved_gb_per_img <= 0:
        return None
    found = throughput_fit(gpu)
    if found is None:
        return None
    fit, calibrated = found
    t = (fit.a_s_per_gflop * model.gflops_fwd + fit.b_s_per_gb * model.saved_gb_per_img
         + fit.d_s_per_step / max(1, batch_per_gpu))
    return 1.0 / t, calibrated


THROUGHPUT_MEASURED = "measured"
THROUGHPUT_THREE_TERM = "three-term"
THROUGHPUT_MFU = "mfu"


def train_throughput(model: ModelSpec, gpu: GpuSpec, precision: str, batch_per_gpu: int,
                     rc_measured: float | None = None) -> tuple[float, str, bool]:
    """Training throughput of ONE GPU at ``precision`` (img/s), with how it was obtained
    (``"measured"`` | ``"three-term"`` | ``"mfu"``) and whether that source is calibrated
    for this GPU. A measured r_c (fp32) wins; mixed precision uses the three-term model
    when it applies; everything else is the FLOPs-only MFU estimate × π."""
    pi = precision_factor(gpu, precision)
    if rc_measured and rc_measured > 0:
        return pi * rc_measured, THROUGHPUT_MEASURED, True
    if precision in MIXED_PRECISIONS:
        tt = three_term_ips(model, gpu, batch_per_gpu)
        if tt is not None:
            return tt[0], THROUGHPUT_THREE_TERM, tt[1]
    return pi * estimate_rc(model, gpu, precision="fp32"), THROUGHPUT_MFU, model.calibrated


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
    eval draws :func:`eval_power_fraction` of train. Power is precision-independent (the
    AMP saving is in TIME, not watts), so precision is not a parameter here."""
    base = gpu.train_power_w if gpu.train_power_w and gpu.train_power_w > 0 else 0.0
    return base * (eval_power_fraction(gpu) if phase == "eval" else 1.0)


def eval_power_fraction(gpu: GpuSpec) -> float:
    """Eval/train power ratio of a GPU: its measured one, or EVAL_POWER_FRACTION."""
    return gpu.eval_power_fraction if gpu.eval_power_fraction > 0 else EVAL_POWER_FRACTION


def split_power_fraction(strategy: str, n_work: int, micro_batches: int = 1) -> float:
    """Average power per GPU, as a fraction of ``train_power_w``, for the strategies that split
    the model: linear in the fraction of the step each GPU is busy, m/(m+p−1), with naive model
    parallelism as m = 1 (calibrated on 2 × T4; see SPLIT_POWER_BASE)."""
    if strategy not in SPLIT_STRATEGIES or n_work <= 1:
        return 1.0
    m = max(1, micro_batches) if strategy == "pipeline" else 1
    busy = m / (m + n_work - 1)
    return SPLIT_POWER_BASE + SPLIT_POWER_SLOPE * busy


def power_working_gpus(strategy: str, n_gpus: int) -> int:
    """How many GPUs draw power for a strategy. ddp/model-parallel power all the GPUs
    in the run (naive model-parallel keeps both GPUs powered even though the stages
    serialize). single and heterogeneous draw on ONE GPU: in this project's
    heterogeneous strategy the second worker is a CPU, not a GPU, so the GPU-power
    calibration applies to a single GPU (the CPU's wall power is negligible)."""
    if strategy in ("ddp", *SPLIT_STRATEGIES):
        return max(1, n_gpus)
    return 1


def estimate_eval_time_s(model: ModelSpec, gpu: GpuSpec, n_val: int, strategy: str,
                         n_gpus: int, precision: str, disk_type: str = "ssd",
                         nfs: bool = False, rc_measured: float | None = None,
                         rio_measured: float | None = None,
                         rc_at_precision: float | None = None) -> float:
    """Eval-epoch time (s): a forward-only pass over the val split.

    Eval has no backward, so its compute throughput is ≈TRAIN_FLOPS_FACTOR× the train
    r_c (forward is ~1/3 of forward+backward). It still reads the val set from disk, so
    it is max(compute, I/O), and it shards across the GPUs for ddp/heterogeneous.
    ``rc_at_precision`` is the train throughput already at ``precision`` (see
    :func:`train_throughput`); without it, the fp32 r_c × π is used."""
    phi = 1.3 if nfs else 1.0
    if rc_at_precision and rc_at_precision > 0:
        rc_train_p = rc_at_precision
    else:
        rc_train = rc_measured if (rc_measured and rc_measured > 0) else estimate_rc(
            model, gpu, precision="fp32")
        rc_train_p = precision_factor(gpu, precision) * rc_train
    rc_fwd = rc_train_p * TRAIN_FLOPS_FACTOR   # forward-only is ~3× the train rate
    rio = estimate_rio(disk_type, nfs, rio_measured)
    # Only DDP sticks the eval set across the GPUs; for heterogeneous the GPU does the
    # eval (the CPU worker's contribution is negligible), so it does not divide.
    n_eff = n_gpus if strategy == "ddp" else 1
    t_compute = n_val / (rc_fwd * max(1, n_eff))
    t_io = n_val / rio
    return phi * max(t_compute, t_io)


# ── Memory model ─────────────────────────────────────────────────────────────────

def estimate_vram_gb(model: ModelSpec, batch: int, precision: str = "fp32",
                     stages: int = 1) -> float:
    """Peak VRAM (GB) for training: weights+grad+optimizer + activations + overhead.

    Activations use the model's per-image figure (``act_gb_per_img``), not a
    single per-parameter constant — that is what makes the per-model max batch
    realistic (e.g. vit_base on an 8 GB card tops out at batch 32, not 64).
    ``stages > 1`` (model parallel / pipeline): each GPU holds ~1/stages of the blocks,
    so of the weights, optimizer state and the activations of the batch (GPipe keeps those
    of every micro-batch until the backward); the CUDA overhead is per GPU."""
    p = model.params_m * 1e6
    weights_opt_gb = BYTES_PER_PARAM_GRAD_OPT * p / 1e9
    if precision in ("amp", "bf16"):
        weights_opt_gb += 2 * p / 1e9          # extra fp16 copy of weights+grad
    act_gb = model.act_gb_per_img * batch
    if precision in ("amp", "bf16"):
        act_gb *= 0.6                          # half-precision activations
    return (weights_opt_gb + act_gb) / max(1, stages) + CUDA_OVERHEAD_GB


def fits_in_memory(model: ModelSpec, gpu: GpuSpec, batch: int,
                   precision: str = "fp32", stages: int = 1) -> bool:
    """Fits if the estimate is within the *usable* VRAM (not 100% of it)."""
    return estimate_vram_gb(model, batch, precision, stages) <= gpu.vram_gb * USABLE_VRAM_FRACTION


def max_batch(model: ModelSpec, gpu: GpuSpec, precision: str = "fp32",
              cap: int = 512, stages: int = 1) -> int:
    """Largest power-of-two-ish batch that fits in this GPU's VRAM (0 if none)."""
    b = 1
    best = 0
    while b <= cap:
        if fits_in_memory(model, gpu, b, precision, stages):
            best = b
        else:
            break
        b *= 2
    return best


# ── The master time formula, per strategy ────────────────────────────────────────

# DDP all-reduces the fp32 gradients (also under AMP: the parameters and their gradients stay
# fp32). A ring all-reduce over n GPUs moves 2(n−1)/n of those bytes through each link, which is
# what the bus bandwidth measures, so the time per step is 2(n−1)/n · 4P / busbw. The overlap of
# the all-reduce with the backward pass (DDP buckets) is not modelled: this is an upper bound.
GRAD_BYTES_PER_PARAM: int = 4


# Without a measurement the nominal bandwidth is derated by half, which keeps the sync term of
# unmeasured GPUs where it was before 24/11/2026 (8P/β equals a ring of 2 GPUs at β/2). It is
# still optimistic: the measured T4 pair on Kaggle sustains a quarter of the PCIe nominal; such
# predictions carry a note (see predict()).
NOMINAL_ALLREDUCE_DERATE: float = 0.5


def allreduce_bandwidth(gpu: GpuSpec) -> tuple[float, bool]:
    """``(bus bandwidth in bytes/s, measured)`` of an all-reduce between GPUs of this kind in
    one machine: the measured one if any, else the derated nominal one of its interconnect."""
    if gpu.allreduce_busbw_gbs > 0:
        return gpu.allreduce_busbw_gbs * 1e9, True
    nominal = INTERCONNECT_BW["nvlink" if gpu.interconnect == "nvlink" else "pcie"]
    return nominal * NOMINAL_ALLREDUCE_DERATE, False


# Strategies the predictor models (names of src/training/strategies.py) and those that split the
# model across GPUs (each GPU holds 1/n of it).
STRATEGIES: tuple[str, ...] = ("single", "ddp", "model_parallel", "pipeline", "heterogeneous")
SPLIT_STRATEGIES: tuple[str, ...] = ("model_parallel", "pipeline")
DEFAULT_MICRO_BATCHES: int = 4         # --micro-batches default of the pipeline strategy

# How each strategy's sync/transfer term is computed, for the CLI and the web to display.
SYNC_FORMULA: dict[str, str] = {
    "single": "— (one GPU)",
    "ddp": "2(n−1)/n · 4P / β · n_batches  (ring all-reduce)",
    "model_parallel": "0.25 · 8P / β · N/b  (activations between stages)",
    "pipeline": "0.25 · 8P / β · N/b  (activations between stages)",
    "heterogeneous": "8P / β_ethernet · n_batches",
}


def allreduce_time(params_m: float, n_gpus: int, busbw: float, n_batches: int) -> float:
    """Ring all-reduce time of the gradients over ``n_batches`` steps (0 with one GPU)."""
    if n_gpus <= 1:
        return 0.0
    grad_bytes = GRAD_BYTES_PER_PARAM * params_m * 1e6
    return 2 * (n_gpus - 1) / n_gpus * grad_bytes / busbw * n_batches


def _sync_time(params_m: float, beta: float, n_batches: int) -> float:
    """Coarse transfer term of the model-parallel and heterogeneous strategies: 8·P bytes per
    batch / bandwidth (their calibration was made with it; DDP uses :func:`allreduce_time`)."""
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
                  rio_measured: float | None = None,
                  rc_at_precision: float | None = None,
                  micro_batches: int = 1,
                  rc_single: float | None = None) -> EpochPrediction:
    """Train-epoch time + speedup for one strategy, from the master formula.

    ``rc_at_precision`` is one GPU's train throughput already at ``precision`` (see
    :func:`train_throughput`); without it, the fp32 r_c (measured or MFU) × π is used."""
    phi = 1.3 if nfs else 1.0
    if rc_at_precision and rc_at_precision > 0:
        rc_p = rc_at_precision
    else:
        rc = rc_measured if (rc_measured and rc_measured > 0) else estimate_rc(model, gpu, precision="fp32")
        rc_p = precision_factor(gpu, precision) * rc
    rio = estimate_rio(disk_type, nfs, rio_measured)
    beta = INTERCONNECT_BW["nvlink" if gpu.interconnect == "nvlink" else "pcie"]
    busbw, _ = allreduce_bandwidth(gpu)

    def single_epoch(n: int) -> tuple[float, float, float, float]:
        t_compute = n_train / (rc_p * n)
        t_io = n_train / rio
        n_batches = math.ceil(n_train / (batch_per_gpu * n))
        t_sync = allreduce_time(model.params_m, n, busbw, n_batches)
        return phi * (max(t_compute, t_io) + t_sync), t_compute, t_io, t_sync

    t1, *_ = single_epoch(1)
    if rc_single and rc_single > 0:            # the 1-GPU reference at the full batch (pipeline)
        t1 = phi * (max(n_train / rc_single, n_train / rio))

    if strategy == "single":
        t, tc, tio, tsync = single_epoch(1)
    elif strategy == "ddp":
        t, tc, tio, tsync = single_epoch(n_gpus)
    elif strategy == "model_parallel":
        # Naive pipeline: stages serialize, no data-parallel split. Compute is the
        # whole model on the full data (split across devices but sequential), plus
        # a small inter-stage activation transfer ≈ one extra all-reduce-ish term.
        t_compute = n_train / rc_p
        t_io = n_train / rio
        transfer = _sync_time(model.params_m, beta, math.ceil(n_train / batch_per_gpu)) * 0.25
        t = phi * (max(t_compute, t_io) + transfer)
        tc, tio, tsync = t_compute, t_io, transfer
    elif strategy == "pipeline":
        # GPipe with p = n_gpus balanced stages and m micro-batches: each GPU computes 1/p of
        # the model and idles a fraction (p−1)/(m+p−1) of the step (the bubble), so the
        # compute time is the single-GPU one × (m+p−1)/(m·p). rc_p already carries the cost
        # of the extra micro-batches (predict()). Same coarse activation-transfer term as above.
        p_st = max(1, n_gpus)
        m = max(1, micro_batches)
        t_compute = n_train / rc_p * (m + p_st - 1) / (m * p_st)
        t_io = n_train / rio
        transfer = _sync_time(model.params_m, beta, math.ceil(n_train / batch_per_gpu)) * 0.25
        t = phi * (max(t_compute, t_io) + transfer)
        tc, tio, tsync = t_compute, t_io, transfer
    elif strategy == "heterogeneous":
        # Synchronous DDP at the pace of the slowest rank (GPU + a ~50× slower CPU
        # worker). Throughput sum ≈ rc·(1 + 1/50); barrier dominates → ~no gain.
        rc_sum = rc_p * (1 + 1 / 50)
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
    # Bottleneck of the parallel part. tc is already wall-clock time in every branch (the
    # data-parallel ones divide by n inside), so it is compared as it is (until 24/11/2026 it was
    # divided by n again, which flagged I/O where compute set the pace).
    if tsync >= max(tc, tio) and strategy in ("ddp", "heterogeneous"):
        bottleneck = "sync"
    elif tio >= tc:
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
    # Training throughput of one GPU at this precision and where it comes from:
    # "measured" (rc_measured), "three-term" (mixed precision, section 8) or "mfu".
    train_ips_per_gpu: float = 0.0
    throughput_model: str = THROUGHPUT_MFU


def predict(strategy: str, model_name: str, gpu_name: str, n_gpus: int = 1,
            dataset_size: int = 5000, batch: int = 96, precision: str = "fp32",
            epochs: int = 15, disk_type: str = "ssd", nfs: bool = False,
            rc_measured: float | None = None,
            rio_measured: float | None = None,
            val_size: int | None = None,
            model: ModelSpec | None = None,
            micro_batches: int = DEFAULT_MICRO_BATCHES) -> Prediction | None:
    """Closed-form prediction for any combination. Returns None for unknown specs.

    ``batch`` is the GLOBAL batch; it is split across the n GPUs for DDP.
    ``dataset_size`` is the train split; ``val_size`` is the val split used for the
    eval-epoch time/energy (defaults to ~0.51× the train split, the BigEarthNet ratio).
    Pass ``rc_measured`` / ``rio_measured`` to calibrate against a real benchmark.
    ``model`` overrides the MODEL_TABLE lookup by name — e.g. a spec built from an
    automatic profile (src/model_profile.py) for a model outside the table.
    """
    if model is None:
        model = model_spec(model_name)
    gpu = gpu_spec(gpu_name)
    if model is None or gpu is None:
        return None
    if strategy not in STRATEGIES:
        return None

    batch_per_gpu = max(1, batch // n_gpus) if strategy in ("ddp", "heterogeneous") else batch
    # GPipe runs the model on micro-batches: the throughput is that of the full batch (the fixed
    # cost per step d is paid once per step) plus a small cost per extra micro-batch.
    # ModelParallelViT never makes more micro-batches than images in the batch.
    micro = min(max(1, micro_batches), batch_per_gpu) if strategy == "pipeline" else 1
    rc_p, tp_model, tp_calibrated = train_throughput(model, gpu, precision, batch_per_gpu,
                                                     rc_measured)
    rc_full = None
    if micro > 1:
        rc_full = rc_p
        rc_p = 1 / (1 / rc_full + PIPELINE_MICRO_BATCH_COST_S * (micro - 1) / batch_per_gpu)
    ep = predict_epoch(strategy, model, gpu, n_gpus, dataset_size, batch_per_gpu,
                       precision, disk_type, nfs, rc_measured, rio_measured,
                       rc_at_precision=rc_p, micro_batches=micro, rc_single=rc_full)

    # Eval-epoch time (forward-only over the val split) → total epoch = train + eval.
    n_val = int(val_size) if val_size and val_size > 0 else round(dataset_size * 0.51)
    t_eval = estimate_eval_time_s(model, gpu, n_val, strategy, n_gpus, precision,
                                  disk_type, nfs, rc_measured, rio_measured,
                                  rc_at_precision=rc_p)
    t_total_epoch = ep.time_train_s + t_eval

    # Energy = effective power × time. Power per GPU is calibrated (or a TDP fallback);
    # the working-GPU count scales it for distributed strategies. Naive model parallelism
    # keeps every GPU powered but idling between stages, so each draws less (split_power_fraction).
    n_work = power_working_gpus(strategy, n_gpus)
    mp = split_power_fraction(strategy, n_work, micro)
    p_train = estimate_power(gpu, "train") * mp
    p_eval = estimate_power(gpu, "eval") * mp
    power_total = p_train * n_work
    e_train_wh = power_total * ep.time_train_s / 3600.0
    e_eval_wh = (p_eval * n_work) * t_eval / 3600.0
    e_epoch_wh = e_train_wh + e_eval_wh

    stages = n_gpus if strategy in SPLIT_STRATEGIES else 1
    vram = estimate_vram_gb(model, batch_per_gpu, precision, stages)
    # Use the SAME usable-VRAM threshold (0.92) as max_batch/fits_in_memory, so the
    # `fits` flag, the OOM note and the recommended batch never contradict each other.
    fits = fits_in_memory(model, gpu, batch_per_gpu, precision, stages)
    rec_b = max_batch(model, gpu, precision, stages=stages)

    notes: list[str] = []
    if strategy == "model_parallel":
        notes.append("Model parallelism does not accelerate (≈1×); it exists to fit "
                     "models too big for one GPU.")
    if strategy in SPLIT_STRATEGIES and n_gpus > 2:
        notes.append(f"The implementation (ModelParallelViT) splits the model into 2 stages; "
                     f"{n_gpus} stages is the formula only, not something it can run.")
    if strategy == "pipeline":
        notes.append(f"GPipe with {n_gpus} stages and {micro} micro-batches: each GPU idles "
                     f"{(n_gpus - 1) / (micro + n_gpus - 1):.0%} of the step (bubble). Balanced "
                     f"stages assumed; time and power calibrated on 2 × T4 only.")
    if strategy in SPLIT_STRATEGIES and n_gpus > 1 and (gpu.name != _T4 or precision != "amp"):
        notes.append("The split strategies' power (and the pipeline's cost per micro-batch) are "
                     "calibrated on 2 × Tesla T4 with AMP only; here they are extrapolated.")
    if strategy == "heterogeneous":
        notes.append("Synchronous DDP runs at the pace of the slowest worker (the CPU); "
                     "imbalanced hardware penalizes.")
    if not fits:
        usable = gpu.vram_gb * USABLE_VRAM_FRACTION
        notes.append(f"OOM: needs ~{vram:.1f} GB but only ~{usable:.1f} GB of the "
                     f"{gpu.vram_gb:.0f} GB is usable (PyTorch reserves ~8%). "
                     f"Largest batch that fits: {rec_b}.")
    if not model.calibrated:
        notes.append(f"Model profiled automatically ({model.params_m:.1f}M params, "
                     f"{model.gflops_fwd:.2f} GFLOPs measured); activation memory "
                     f"estimated, not calibrated — treat VRAM/OOM as a rough estimate.")
    if tp_model == THROUGHPUT_MFU and (not model.calibrated or precision in MIXED_PRECISIONS):
        notes.append(f"Compute throughput from FLOPs only (MFU {MFU}, calibrated on vit_base): "
                     f"architectures that move much memory per FLOP (Swin, CNNs, "
                     f"EfficientNet) run slower than this, so time and energy are rough.")
    if tp_model == THROUGHPUT_THREE_TERM and not tp_calibrated:
        notes.append(f"Throughput from the three-term model transferred to {gpu.name} by its "
                     f"datasheet (TFLOPS, bandwidth), not measured on it: the relative cost of "
                     f"architectures holds (~11 % on a T4), the absolute speed may be "
                     f"optimistic (×1.5 on a T4).")
    if strategy == "ddp" and n_gpus > 1:
        _, bw_measured = allreduce_bandwidth(gpu)
        if not bw_measured:
            notes.append(f"Gradient sync uses a nominal interconnect bandwidth for {gpu.name} "
                         f"(not measured here, derated ×{NOMINAL_ALLREDUCE_DERATE}); the measured "
                         f"T4 pair on Kaggle sustained a quarter of its nominal, so the sync "
                         f"time may be optimistic. Measure it with scripts/measure_allreduce.py.")
    if not gpu.power_calibrated:
        notes.append(f"Energy uses a TDP-fallback power for {gpu.name} (≈"
                     f"{gpu.train_power_w:.0f} W/GPU, not measured here) — treat it as "
                     f"a rough estimate.")
    # When a Tensor-core precision shrinks compute below the disk floor, the epoch goes
    # I/O-bound: time stops dropping and the per-GPU energy stays high. On a RAM-cached
    # dataset (e.g. the demo subset) the real run stays compute-bound and uses roughly
    # half this — flag it so the time/energy are read as a conservative upper bound.
    if precision != "fp32" and ep.bottleneck == "io":
        notes.append("I/O floor active under this precision: if the dataset is RAM-cached "
                     "the real run stays compute-bound and uses ~half this time/energy — "
                     "pass a measured throughput (rc/rio) to calibrate.")

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
        train_ips_per_gpu=rc_p, throughput_model=tp_model,
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


def expected_best_f1(model_name: str, dataset_size: int = N_FULL_TRAIN,
                     calibrated: bool = True) -> tuple[float, str, float]:
    """Empirical-prior best Val F1 for a model on N training images.

    Returns ``(f1, confidence, band)``. Uses the log-linear data-scaling law
    anchored to the documented full-dataset plateau of the model family.
    ``calibrated=False`` (a model profiled automatically, outside MODEL_TABLE) forces
    confidence "low": the family prior (falling back to vit_base) was never run for
    that architecture."""
    fam = _quality_family(model_name)
    f_full, _, _, k, conf = _QUALITY_ANCHORS[fam]
    if not calibrated:
        conf = "low"
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
                    epochs: int = 30, calibrated: bool = True) -> QualityPrediction | None:
    """Empirical-prior quality estimate (best Val F1 + a learning curve).

    The honest planning counterpart to ``predict``: anchored to documented runs,
    extended to any dataset size by the data-scaling law. Not a measurement — for
    that, run the convergence study. Returns ``None`` for an unknown family only
    in the sense that it always falls back to the ViT-Base prior (so never None
    here, but kept Optional for symmetry with ``predict``).

    Pass ``calibrated=model_spec.calibrated``: for a model profiled automatically the
    prior is that of its nearest family (vit_base when none matches) and the
    confidence is forced to "low" with a note saying so."""
    fam = _quality_family(model_name)
    f_full, best_ep, stop_ep, _k, _conf = _QUALITY_ANCHORS[fam]
    f1, conf, band = expected_best_f1(model_name, dataset_size, calibrated)

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
    if not calibrated:
        notes.append(f"Model outside the curated table: the {fam} prior is the nearest "
                     f"family, not a run of this architecture — confidence forced to low.")
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
