"""Compare three sources of truth for a training configuration.

Given a benchmark CSV and an epoch_metrics CSV (or parsed log DataFrame),
produces a structured comparison table with, per metric:
  - analytic  — the closed-form prediction from ``src.performance_model`` (no GPU),
  - benchmark — the empirical estimate from a real ``paravit benchmark`` run,
  - actual    — what the training run actually did,
plus the error % of each estimate against the real run and the formula behind it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.performance_model import EVAL_POWER_FRACTION

# Fallback to the full BigEarthNet-S2 if the benchmark CSV does not carry the
# real size (old CSVs without a #sizes block). New CSVs include n_train/n_val
# from the metadata actually used (subset or full) → valid comparison.
_N_TRAIN_DEFAULT = 237_871
_N_VAL_DEFAULT = 122_342


@dataclass
class ComparisonRow:
    metric: str
    formula: str
    estimated: float | None              # benchmark (empirical) estimate
    actual: float | None                 # what the run actually did
    unit: str = ""
    analytic: float | None = None        # closed-form prediction (performance_model)

    @staticmethod
    def _pct(pred: float | None, actual: float | None) -> float | None:
        if pred is None or actual is None or actual == 0:
            return None
        return (pred - actual) / abs(actual) * 100

    @property
    def error_pct(self) -> float | None:
        """Benchmark estimate vs the real run."""
        return self._pct(self.estimated, self.actual)

    @property
    def analytic_error_pct(self) -> float | None:
        """Analytic prediction vs the real run."""
        return self._pct(self.analytic, self.actual)


@dataclass
class BenchmarkComparison:
    """Three-way comparison: analytic vs benchmark vs the actual training run."""

    model_name: str
    batch_size: int
    trace_mode: str
    nfs_factor: float
    rows: list[ComparisonRow] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        records = []
        for r in self.rows:
            def _fmt(v):
                return f"{v:.2f}" if v is not None else "—"
            err = r.error_pct
            aerr = r.analytic_error_pct
            records.append({
                "Metric": r.metric,
                "Unit": r.unit,
                "Analytic": _fmt(r.analytic),
                "Benchmark": _fmt(r.estimated),
                "Real": _fmt(r.actual),
                "Δ analytic %": f"{aerr:+.1f}%" if aerr is not None else "—",
                "Δ benchmark %": f"{err:+.1f}%" if err is not None else "—",
                "Formula": r.formula,
            })
        return pd.DataFrame(records)


def build_comparison(
    meta: dict,
    feas_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    batch_size: int,
    trace_mode: str = "simple",
    nfs_factor: float = 1.0,
    strategy: str = "single",
    gpu_name: str | None = None,
    n_gpus: int = 1,
    precision: str = "fp32",
    precision_speedup: float | None = None,
    ddp_speedup: float | None = None,
    run_n_train: int | None = None,
    run_n_val: int | None = None,
) -> BenchmarkComparison | None:
    """Build a BenchmarkComparison from parsed benchmark data and actual metrics.

    Parameters
    ----------
    meta:        metadata dict from parse_benchmark_csv()
    feas_df:     benchmark DataFrame from parse_benchmark_csv()
    actual_df:   epoch_metrics DataFrame (from CSV or log_parser)
    batch_size:  batch size to use for matching
    trace_mode:  trace mode to match in feas_df
    nfs_factor:  NFS correction factor used in benchmark run
    strategy/gpu_name/n_gpus/precision: the run's configuration, used to fill the
        analytic column from src.performance_model.predict() (the third source). When
        gpu_name is None the analytic column is left empty (benchmark-vs-run only).
    precision_speedup: measured fp32→Tensor-core speedup (from the report's
        --compare-precision block); when the run uses a Tensor-core precision the
        benchmark's fp32 time/energy are divided by it (throughput multiplied) so the
        benchmark column is comparable to the run. None → no correction.
    ddp_speedup: predicted N-GPU speedup; for a distributed run the single-GPU
        benchmark time/energy-train are divided by it so the benchmark column reflects
        the run's GPU count. None → no correction.
    run_n_train/run_n_val: the RUN's actual dataset size. When given, the analytic AND
        the benchmark per-epoch estimates are computed for THIS size (the benchmark's
        per-batch throughput is dataset-size-independent, so a run and a benchmark report
        of different sizes stay comparable). Defaults to the benchmark report's #sizes.
    """
    if feas_df.empty or actual_df.empty or "batch_size" not in feas_df.columns:
        return None

    model_name = meta.get("model_name", "unknown")

    # Real dataset size (from the #sizes CSV); fallback to the full set.
    def _int(v, default):
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return default
    # Prefer the RUN's dataset size; fall back to the benchmark report's #sizes, then
    # the full set. Using the run's size keeps the comparison valid even when the
    # matched benchmark report profiled a different dataset size.
    n_train = _int(run_n_train, None) or _int(meta.get("n_train"), _N_TRAIN_DEFAULT)
    n_val = _int(run_n_val, None) or _int(meta.get("n_val"), _N_VAL_DEFAULT)

    # ── Analytic prediction (third source) — closed-form, no GPU ─────────────────
    pred = None
    pred_f1 = None
    if gpu_name:
        try:
            from src.performance_model import predict, expected_best_f1
            # batch_size here is the run's PER-GPU batch (it matches the single-GPU
            # benchmark row); predict() expects the GLOBAL batch and re-splits it, so
            # scale back up for distributed strategies.
            global_batch = (batch_size * max(1, n_gpus)
                            if strategy in ("ddp", "heterogeneous") else batch_size)
            pred = predict(strategy, model_name, gpu_name, n_gpus=max(1, n_gpus),
                           dataset_size=n_train, batch=global_batch, precision=precision,
                           epochs=1, nfs=(nfs_factor > 1.05), val_size=n_val)
            pred_f1 = expected_best_f1(model_name, n_train)[0]
        except Exception:
            pred = None
    _a_train_min = pred.time_per_epoch_train_s / 60 if pred else None
    _a_eval_min = pred.time_per_epoch_eval_s / 60 if pred else None
    _a_total_min = pred.time_per_epoch_total_s / 60 if pred else None
    _a_energy_train = pred.energy_train_wh if pred else None
    _a_energy_eval = pred.energy_eval_wh if pred else None

    # Find matching benchmark row
    mask = (feas_df["batch_size"] == batch_size)
    if "trace_mode" in feas_df.columns:
        mask &= (feas_df["trace_mode"] == trace_mode)
    row_feas = feas_df[mask]
    if row_feas.empty:
        # Relax trace_mode filter
        row_feas = feas_df[feas_df["batch_size"] == batch_size]
    if row_feas.empty:
        return None
    frow = row_feas.iloc[0]

    # The benchmark runs single-GPU and fp32. Correct its estimates to the run's actual
    # configuration so the benchmark column is comparable:
    #   - Tensor-core precision: same power, less time → divide TIME and ENERGY by the
    #     measured precision speedup (throughput multiplied).
    #   - DDP n GPUs: faster wall-clock → divide TIME by the predicted DDP speedup. The
    #     benchmark ENERGY is left at the single-GPU value: that equals the DDP total
    #     ONLY when DDP is compute-bound (n GPUs do 1/n of the work each). For an
    #     I/O-bound DDP run the wall-clock does not drop while n GPUs draw full power, so
    #     the true total is ~n× — there the ANALYTIC column (which keeps the fixed I/O
    #     time) is the correct distributed-energy estimate, not this one.
    _tensor = precision in ("amp", "fp16", "tf32", "bf16")
    _psp = precision_speedup if (_tensor and precision_speedup and precision_speedup > 0) else 1.0
    _dsp = ddp_speedup if (strategy in ("ddp", "heterogeneous") and ddp_speedup and ddp_speedup > 0) else 1.0
    _time_factor = _psp * _dsp        # divides time, multiplies throughput
    _energy_factor = _psp             # only precision changes total energy

    def _scale(v, factor):
        return (v / factor) if (v is not None and factor) else v

    est_vram = _safe_float(frow.get("peak_vram_gb"))
    s_per_batch_train = _safe_float(frow.get("s_per_batch_train") or frow.get("s_per_batch"))
    s_per_batch_eval = _safe_float(frow.get("s_per_batch_eval") or frow.get("s_per_batch"))
    imgs_per_s_train = _safe_float(frow.get("imgs_per_s_train") or frow.get("imgs_per_s"))
    if imgs_per_s_train is not None and _time_factor:
        imgs_per_s_train *= _time_factor
    avg_power = _safe_float(frow.get("avg_power_w"))

    n_train_batches = math.ceil(n_train / batch_size) if batch_size > 0 else None
    n_val_batches = math.ceil(n_val / batch_size) if batch_size > 0 else None

    # Recompute the benchmark per-epoch time/energy for the RUN's dataset size from the
    # measured per-batch throughput (which is dataset-size-independent), so the benchmark
    # column stays comparable even when the matched report profiled a different N. Falls
    # back to the CSV per-epoch totals only if the per-batch figure is missing.
    def _bench_sec(n_batches, s_per_batch, fallback_min):
        if n_batches and s_per_batch:
            return n_batches * s_per_batch * nfs_factor
        return fallback_min * 60 if fallback_min is not None else None

    sec_train = _bench_sec(n_train_batches, s_per_batch_train,
                           _safe_float(frow.get("est_train_min_per_epoch")))
    sec_eval = _bench_sec(n_val_batches, s_per_batch_eval,
                          _safe_float(frow.get("est_eval_min_per_epoch")))
    # Benchmark estimates (single-GPU fp32 → corrected to the run's config).
    est_train_min = _scale(sec_train / 60 if sec_train is not None else None, _time_factor)
    est_eval_min = _scale(sec_eval / 60 if sec_eval is not None else None, _time_factor)
    est_total_min = _sum_opt(est_train_min, est_eval_min)
    # Energy = the benchmark's measured train power × the recomputed time (eval at
    # EVAL_POWER_FRACTION), so both energy estimates use one consistent eval factor.
    _bench_e_train = avg_power * sec_train / 3600 if (avg_power and sec_train is not None) else None
    _bench_e_eval = (avg_power * EVAL_POWER_FRACTION * sec_eval / 3600
                     if (avg_power and sec_eval is not None) else None)

    # Actual values from training
    act_epoch_time_s = (
        actual_df["epoch_time"].mean() if "epoch_time" in actual_df.columns
        and actual_df["epoch_time"].notna().any() else None
    )
    act_train_time_s = (
        actual_df["time_train_s"].mean() if "time_train_s" in actual_df.columns
        and actual_df["time_train_s"].notna().any() else None
    )
    act_eval_time_s = (
        actual_df["time_eval_s"].mean() if "time_eval_s" in actual_df.columns
        and actual_df["time_eval_s"].notna().any() else None
    )
    act_total_min = act_epoch_time_s / 60 if act_epoch_time_s is not None else None
    act_train_min = act_train_time_s / 60 if act_train_time_s is not None else None
    act_eval_min = act_eval_time_s / 60 if act_eval_time_s is not None else None

    flops = _safe_float(meta.get("flops_mflops"))
    params_m = _safe_float(meta.get("total_params_M"))
    static_mb = _safe_float(meta.get("total_static_mb"))
    act_mb = _safe_float(meta.get("activation_mb_per_image"))

    rows: list[ComparisonRow] = []

    # ── Time estimates ────────────────────────────────────────────────────────
    if n_train_batches and s_per_batch_train:
        formula_train = (
            f"⌈{n_train}/{batch_size}⌉ × {s_per_batch_train:.3f}s × {nfs_factor:.1f}(NFS) / 60"
        )
    else:
        formula_train = "n_batches × s/batch × nfs_factor / 60"

    rows.append(ComparisonRow(
        metric="Train time / epoch",
        formula=formula_train,
        estimated=est_train_min,
        actual=act_train_min,
        unit="min",
        analytic=_a_train_min,
    ))

    if n_val_batches and s_per_batch_eval:
        formula_eval = f"⌈{n_val}/{batch_size}⌉ × {s_per_batch_eval:.3f}s / 60"
    else:
        formula_eval = "n_val_batches × s/batch_eval / 60"

    rows.append(ComparisonRow(
        metric="Eval time / epoch",
        formula=formula_eval,
        estimated=est_eval_min,
        actual=act_eval_min,
        unit="min",
        analytic=_a_eval_min,
    ))

    rows.append(ComparisonRow(
        metric="Total time / epoch",
        formula="train_time + eval_time",
        estimated=est_total_min,
        actual=act_total_min,
        unit="min",
        analytic=_a_total_min,
    ))

    # ── Throughput ───────────────────────────────────────────────────────────
    act_throughput = None
    if act_train_time_s and act_train_time_s > 0:
        act_throughput = (n_train / act_train_time_s)

    a_throughput = (n_train / pred.time_per_epoch_train_s) if (
        pred and pred.time_per_epoch_train_s) else None
    rows.append(ComparisonRow(
        metric="Train throughput",
        formula=f"batch_size / s_per_batch = {batch_size} / {s_per_batch_train:.3f}s" if s_per_batch_train else "batch_size / s_per_batch",
        estimated=imgs_per_s_train,
        actual=act_throughput,
        unit="imgs/s",
        analytic=a_throughput,
    ))

    # ── VRAM ─────────────────────────────────────────────────────────────────
    if static_mb and act_mb:
        vram_formula = (
            f"({static_mb:.0f}MB static + {batch_size} × {act_mb:.1f}MB/img) / 1024"
        )
        vram_est = (static_mb + batch_size * act_mb) / 1024
    else:
        vram_formula = "(static_mem + batch_size × activation_mb_per_img) / 1024"
        vram_est = est_vram

    rows.append(ComparisonRow(
        metric="Peak VRAM",
        formula=vram_formula,
        estimated=vram_est,
        actual=est_vram,  # benchmark peak_vram is measured, not estimated
        unit="GB",
    ))

    # ── Energy ───────────────────────────────────────────────────────────────
    # Recomputed for the run's dataset size (above) and corrected for precision.
    est_energy_train = _scale(_bench_e_train, _energy_factor)
    est_energy_eval_raw = _scale(_bench_e_eval, _energy_factor)
    # Actual train energy is in the log as Joules (energy_train_j); the eval energy is
    # already in Wh (energy_eval_wh). Derive train Wh = J / 3600 when present.
    act_energy_train_wh = None
    if "energy_train_j" in actual_df.columns and actual_df["energy_train_j"].notna().any():
        act_energy_train_wh = float(actual_df["energy_train_j"].mean()) / 3600.0
    act_energy_eval_wh = (
        actual_df["energy_eval_wh"].mean() if "energy_eval_wh" in actual_df.columns
        and actual_df["energy_eval_wh"].notna().any() else None
    )
    if (est_energy_train is not None or act_energy_train_wh is not None
            or _a_energy_train is not None):
        formula_energy = (
            f"{avg_power:.0f}W × train_time_h"
            if avg_power else "avg_power_w × train_time_h"
        )
        rows.append(ComparisonRow(
            metric="Energy train / epoch",
            formula=formula_energy,
            estimated=est_energy_train,
            actual=act_energy_train_wh,
            unit="Wh",
            analytic=_a_energy_train,
        ))
    if act_energy_eval_wh is not None or _a_energy_eval is not None or est_energy_eval_raw is not None:
        _evf = f"{avg_power:.0f}W × {EVAL_POWER_FRACTION:g} × eval_time_h" if avg_power else "power × eval_time_h"
        rows.append(ComparisonRow(
            metric="Energy eval / epoch",
            formula=_evf,
            estimated=est_energy_eval_raw,
            actual=act_energy_eval_wh,
            unit="Wh",
            analytic=_a_energy_eval,
        ))
    # Total energy / epoch (train + eval) — the headline 3-way energy number. Gated on
    # its own check (not the train guard) so eval-only data still yields a total.
    _e_bench = _sum_opt(est_energy_train, est_energy_eval_raw)
    _e_real = _sum_opt(act_energy_train_wh, act_energy_eval_wh)
    _e_analytic = pred.energy_per_epoch_wh if pred else None
    if _e_bench is not None or _e_real is not None or _e_analytic is not None:
        rows.append(ComparisonRow(
            metric="Energy total / epoch",
            formula="energy_train + energy_eval",
            estimated=_e_bench,
            actual=_e_real,
            unit="Wh",
            analytic=_e_analytic,
        ))

    # ── Optimizer steps ──────────────────────────────────────────────────────
    est_steps = _safe_float(frow.get("optimizer_steps_per_epoch"))
    if est_steps is not None and n_train_batches:
        rows.append(ComparisonRow(
            metric="Optimizer steps / epoch",
            formula=f"⌈{n_train}/{batch_size}⌉ = {n_train_batches}",
            estimated=est_steps,
            actual=float(n_train_batches),
            unit="steps",
        ))

    # DDP scaling is intentionally NOT shown here: the compute/IO/sync-aware
    # DDPOptimizer (#ddp block, parsed by parse_ddp_scenarios) is the single source
    # of truth for distributed scaling. The old flat 0.85-efficiency projection was
    # removed because it contradicted those precise scenarios.

    # ── Complexity / FLOPs ───────────────────────────────────────────────────
    if flops and n_train_batches and batch_size:
        total_gflops = flops * n_train / 1000  # MFLOPs × N / 1000 → GFLOPs
        rows.append(ComparisonRow(
            metric="FLOPs / train epoch",
            formula=f"{flops:.0f} MFLOPs/img × {n_train} imgs / 1000",
            estimated=total_gflops,
            actual=None,
            unit="GFLOPs",
        ))

    if params_m:
        rows.append(ComparisonRow(
            metric="Model parameters",
            formula="Σ parameters across all layers",
            estimated=params_m,
            actual=params_m,
            unit="M",
        ))

    # ── Best Val F1 ───────────────────────────────────────────────────────────
    # Analytic and benchmark F1 use the SAME empirical-prior engine (expected_best_f1),
    # but the analytic column recomputes it for the RUN's dataset size while the benchmark
    # column carries the value stored in the report at ITS profiled size — so the two
    # coincide only when the run and the matched report share a dataset size, and differ
    # otherwise. Only the run is a real measurement.
    bench_f1 = None
    try:
        bench_f1 = float((meta.get("prediction") or {}).get("predicted_best_f1") or 0) or None
    except (TypeError, ValueError):
        bench_f1 = None
    real_f1 = None
    if "val_f1" in actual_df.columns and actual_df["val_f1"].notna().any():
        real_f1 = float(actual_df["val_f1"].max())
    if pred_f1 is not None or bench_f1 is not None or real_f1 is not None:
        rows.append(ComparisonRow(
            metric="Best Val F1",
            formula="empirical prior F1_inf(N) = F1_full − k·log10(N_full/N)",
            estimated=bench_f1,
            actual=real_f1,
            unit="F1",
            analytic=pred_f1,
        ))

    return BenchmarkComparison(
        model_name=model_name,
        batch_size=batch_size,
        trace_mode=trace_mode,
        nfs_factor=nfs_factor,
        rows=rows,
    )


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _sum_opt(a: float | None, b: float | None) -> float | None:
    """Sum two optionals, ignoring None; returns None only if BOTH are None."""
    if a is None and b is None:
        return None
    return (a or 0.0) + (b or 0.0)
