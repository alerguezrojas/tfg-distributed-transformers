"""Compare feasibility estimates against actual training results.

Given a feasibility CSV and an epoch_metrics CSV (or parsed log DataFrame),
produces a structured comparison table with: estimated value, actual value,
error %, and the formula/model used for each estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Fallback to the full BigEarthNet-S2 if the feasibility CSV does not carry the
# real size (old CSVs without a #sizes block). New CSVs include n_train/n_val
# from the metadata actually used (subset or full) → valid comparison.
_N_TRAIN_DEFAULT = 237_871
_N_VAL_DEFAULT = 122_342


@dataclass
class ComparisonRow:
    metric: str
    formula: str
    estimated: float | None
    actual: float | None
    unit: str = ""

    @property
    def error_pct(self) -> float | None:
        if self.estimated is None or self.actual is None:
            return None
        if self.actual == 0:
            return None
        return (self.estimated - self.actual) / abs(self.actual) * 100


@dataclass
class FeasibilityComparison:
    """Comparison between feasibility estimates and actual training results."""

    model_name: str
    batch_size: int
    trace_mode: str
    nfs_factor: float
    rows: list[ComparisonRow] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        records = []
        for r in self.rows:
            err = r.error_pct
            records.append({
                "Metric": r.metric,
                "Formula": r.formula,
                f"Estimated ({r.unit})": (
                    f"{r.estimated:.2f}" if r.estimated is not None else "—"
                ),
                f"Actual ({r.unit})": (
                    f"{r.actual:.2f}" if r.actual is not None else "—"
                ),
                "Error %": f"{err:+.1f}%" if err is not None else "—",
            })
        return pd.DataFrame(records)


def build_comparison(
    meta: dict,
    feas_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    batch_size: int,
    trace_mode: str = "simple",
    nfs_factor: float = 1.0,
) -> FeasibilityComparison | None:
    """Build a FeasibilityComparison from parsed feasibility data and actual metrics.

    Parameters
    ----------
    meta:        metadata dict from parse_feasibility_csv()
    feas_df:     benchmark DataFrame from parse_feasibility_csv()
    actual_df:   epoch_metrics DataFrame (from CSV or log_parser)
    batch_size:  batch size to use for matching
    trace_mode:  trace mode to match in feas_df
    nfs_factor:  NFS correction factor used in feasibility run
    """
    if feas_df.empty or actual_df.empty:
        return None

    model_name = meta.get("model_name", "unknown")

    # Real dataset size (from the #sizes CSV); fallback to the full set.
    def _int(v, default):
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return default
    n_train = _int(meta.get("n_train"), _N_TRAIN_DEFAULT)
    n_val = _int(meta.get("n_val"), _N_VAL_DEFAULT)

    # Find matching feasibility row
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

    # Feasibility estimates (convert min → min for display)
    est_train_min = _safe_float(frow.get("est_train_min_per_epoch"))
    est_eval_min = _safe_float(frow.get("est_eval_min_per_epoch"))
    est_total_min = _safe_float(frow.get("est_total_min_per_epoch"))
    est_vram = _safe_float(frow.get("peak_vram_gb"))
    s_per_batch_train = _safe_float(frow.get("s_per_batch_train") or frow.get("s_per_batch"))
    s_per_batch_eval = _safe_float(frow.get("s_per_batch_eval") or frow.get("s_per_batch"))
    imgs_per_s_train = _safe_float(frow.get("imgs_per_s_train") or frow.get("imgs_per_s"))

    n_train_batches = math.ceil(n_train / batch_size) if batch_size > 0 else None
    n_val_batches = math.ceil(n_val / batch_size) if batch_size > 0 else None

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
    ))

    rows.append(ComparisonRow(
        metric="Total time / epoch",
        formula="train_time + eval_time",
        estimated=est_total_min,
        actual=act_total_min,
        unit="min",
    ))

    # ── Throughput ───────────────────────────────────────────────────────────
    act_throughput = None
    if act_train_time_s and act_train_time_s > 0:
        act_throughput = (n_train / act_train_time_s)

    rows.append(ComparisonRow(
        metric="Train throughput",
        formula=f"batch_size / s_per_batch = {batch_size} / {s_per_batch_train:.3f}s" if s_per_batch_train else "batch_size / s_per_batch",
        estimated=imgs_per_s_train,
        actual=act_throughput,
        unit="imgs/s",
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
        actual=est_vram,  # feasibility peak_vram is measured, not estimated
        unit="GB",
    ))

    # ── Energy ───────────────────────────────────────────────────────────────
    est_energy_train = _safe_float(frow.get("est_energy_train_wh_per_epoch"))
    act_energy_eval_wh = (
        actual_df["energy_eval_wh"].mean() if "energy_eval_wh" in actual_df.columns
        and actual_df["energy_eval_wh"].notna().any() else None
    )
    avg_power = _safe_float(frow.get("avg_power_w"))
    if est_energy_train is not None or act_energy_eval_wh is not None:
        formula_energy = (
            f"{avg_power:.0f}W × train_time_h × 1Wh"
            if avg_power else "avg_power_w × time_h"
        )
        rows.append(ComparisonRow(
            metric="Energy train / epoch",
            formula=formula_energy,
            estimated=est_energy_train,
            actual=None,  # training logs have eval energy, not train
            unit="Wh",
        ))
        if act_energy_eval_wh is not None:
            est_energy_eval = _safe_float(frow.get("est_energy_eval_wh_per_epoch"))
            rows.append(ComparisonRow(
                metric="Energy eval / epoch",
                formula=f"{avg_power:.0f}W × 0.4 × eval_time_h" if avg_power else "power × eval_time_h",
                estimated=est_energy_eval,
                actual=act_energy_eval_wh,
                unit="Wh",
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

    # ── DDP projection ───────────────────────────────────────────────────────
    est_total_h = _safe_float(frow.get(next(
        (c for c in (frow.index if hasattr(frow, 'index') else [])
         if c.startswith("est_total_h_")), None
    ))) if hasattr(frow, 'index') else None
    est_ddp2 = _safe_float(frow.get(next(
        (c for c in (frow.index if hasattr(frow, 'index') else [])
         if c.startswith("est_ddp_2gpu_h_")), None
    ))) if hasattr(frow, 'index') else None
    if est_ddp2 is not None:
        rows.append(ComparisonRow(
            metric="DDP ×2 GPU total (est.)",
            formula="total_time / (2 × 0.85_efficiency)",
            estimated=est_ddp2,
            actual=None,
            unit="h",
        ))

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

    return FeasibilityComparison(
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
