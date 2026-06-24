"""Compare — speedup."""
from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.benchmark_parser import (parse_ddp_scenarios, parse_benchmark_csv)
from src.web.run_registry import RunInfo
from src.web.ui import theme
from src.web.ui.charts import (COLORS, _base_layout, _show)
from src.web.ui.helpers import (_get_benchmark_csvs)
from src.web.tabs.comparison._common import (_has, _prec)


def _predicted_2gpu_speedup(env: str, model: str) -> float | None:
    """The benchmark's predicted 2-GPU speedup for this env/model, if any."""
    for p in _get_benchmark_csvs():
        try:
            if p.parent.parent.name != env:
                continue
            meta, _ = parse_benchmark_csv(p)
            if meta.get("model_name") != model:
                continue
            scen = parse_ddp_scenarios(meta)
            if not scen.empty and {"n_gpus", "speedup"}.issubset(scen.columns):
                r2 = scen[scen["n_gpus"] == 2]
                if not r2.empty:
                    return float(r2.iloc[0]["speedup"])
        except Exception:
            pass
    return None


def _speedup_section(sel: list[tuple[str, RunInfo]], df_by_label: dict[str, pd.DataFrame]) -> None:
    """Every selected run against one baseline (the generalized pair analysis)."""
    st.markdown("### Speedup analysis")
    st.caption(
        "All the runs selected at the top are compared against ONE of them — the "
        "baseline, which counts as 1.00×. To compare more runs, add them to the "
        "selector at the top of the page."
    )

    timed: list[tuple[str, RunInfo, float]] = []
    for lbl, r in sel:
        d = df_by_label[lbl]
        if _has(d, "epoch_time"):
            timed.append((lbl, r, float(d["epoch_time"].dropna().mean())))
    if len(timed) < 2:
        st.caption("Speedup needs at least 2 selected runs with epoch timing.")
        return
    if len(timed) < len(sel):
        st.caption(f"{len(sel) - len(timed)} selected run(s) have no epoch timing "
                   "and are excluded from the speedup table.")

    # Baseline default: a single-GPU fp32 simple-trace run (the natural 1.00x
    # reference); ties broken by recency.
    def _baseline_rank(t):
        _, r, _ = t
        return (r.mode == "single", _prec(r) == "fp32", r.trace_mode != "deep", r.sort_key)

    timed_labels = [lbl for lbl, _, _ in timed]
    default_lbl = max(timed, key=_baseline_rank)[0]
    # No fixed key: the widget is re-created when the selection changes, so the
    # smart default re-applies (a fixed key would freeze the first-render pick).
    base_lbl = st.selectbox(
        "Baseline run (= 1.00×) — every other selected run is measured against it",
        timed_labels, index=timed_labels.index(default_lbl),
    )
    _, base_r, base_t = next(t for t in timed if t[0] == base_lbl)

    rows, bar_lbls, bar_vals, bar_colors = [], [], [], []
    for lbl, r, t in timed:
        sp = base_t / t if t > 0 else float("nan")
        notes = []
        if r.model != base_r.model:
            notes.append("different model — not directly comparable")
        if _prec(r) != _prec(base_r):
            notes.append(f"precision {_prec(r)} vs {_prec(base_r)} (Tensor cores ~3-4×)")
        if r.mode == "ddp":
            # % of ideal only makes sense vs a same-precision baseline —
            # otherwise the Tensor-core effect pollutes the DDP efficiency.
            if _prec(r) == _prec(base_r):
                notes.append(f"{sp / 2 * 100:.0f}% of ideal 2×")
        elif r.mode == "ddp_hetero":
            notes.append("synchronous + imbalanced hardware")
        elif r.mode == "model_parallel":
            notes.append("naive pipeline — ≈1× expected")
        if r.trace_mode == "deep" and base_r.trace_mode != "deep":
            notes.append("deep trace ~20% overhead")
        rows.append({
            "Run": lbl, "Mode": r.mode, "Precision": _prec(r),
            "Avg epoch (min)": round(t / 60, 2),
            "Speedup": "baseline" if lbl == base_lbl else f"{sp:.2f}×",
            "Notes": "; ".join(notes) if lbl != base_lbl and notes else "—",
        })
        bar_lbls.append(lbl)
        bar_vals.append(sp)
        bar_colors.append("#94a3b8" if lbl == base_lbl else COLORS[0])

    st.dataframe(pd.DataFrame(rows).set_index("Run"), use_container_width=True)

    # Ranked bar: sort by speedup (fastest on top) and colour by faster / slower
    # than the baseline, so "scales up" vs "penalises" reads instantly.
    order = sorted(range(len(bar_vals)), key=lambda i: bar_vals[i])
    s_lbls = [bar_lbls[i] for i in order]
    s_vals = [bar_vals[i] for i in order]
    s_cols = ["#94A3B8" if abs(v - 1.0) < 0.05 else (theme.GOOD if v > 1.0 else theme.BAD)
              for v in s_vals]
    fig_sp = go.Figure(go.Bar(
        y=s_lbls, x=s_vals, orientation="h", marker_color=s_cols,
        text=[f"{v:.2f}×" for v in s_vals], textposition="outside",
        cliponaxis=False,
    ))
    fig_sp.update_layout(
        **_base_layout(150 + 40 * len(s_lbls), "Speedup vs baseline",
                       margin=dict(l=10, r=64, t=48, b=40)),
        xaxis_title="× speedup  (green = faster than baseline, red = slower)",
        showlegend=False,
    )
    fig_sp.update_yaxes(automargin=True)
    fig_sp.add_vline(x=1.0, line_dash="dash", line_color="#475569",
                     annotation_text="baseline 1.0×", annotation_position="top")
    _show(fig_sp, "compare_speedup")

    # One pedagogical banner per special mode present in the selection.
    others = [(lbl, r, base_t / t) for lbl, r, t in timed if lbl != base_lbl]
    if any(r.mode == "model_parallel" for _, r, _ in others):
        st.info(
            "**Model parallelism is not expected to accelerate training (≈1×).** "
            "The naive pipeline runs the stages sequentially — one GPU is idle while the "
            "other computes — so the theoretical ceiling is ≈1×. Its purpose is to **train "
            "models that do not fit on a single GPU**: vit_large exceeds the memory of one "
            "T4 but trains when split 12/24 across both."
        )
    if any(r.mode == "ddp_hetero" and sp < 1 for _, r, sp in others):
        st.warning(
            "**Heterogeneous DDP is slower than the GPU alone** — the expected outcome of "
            "**synchronous** DDP on imbalanced hardware (V100 + CPU): on every batch the "
            "GPU stalls on the CPU (~50× slower), so the system runs at the pace of the "
            "slowest worker. An example of when distribution is not beneficial."
        )

    # Benchmark validation: predicted vs measured for the first homogeneous
    # DDP run comparable with a single-GPU baseline (data-parallel prediction).
    if base_r.mode == "single":
        for lbl, r, sp in others:
            if r.mode == "ddp" and r.model == base_r.model and r.env == base_r.env \
                    and _prec(r) == _prec(base_r):
                pred_sp = _predicted_2gpu_speedup(r.env, r.model)
                if pred_sp:
                    err = (pred_sp - sp) / sp * 100
                    pp1, pp2, pp3 = st.columns(3)
                    pp1.metric("Predicted speedup (benchmark)", f"{pred_sp:.2f}×")
                    pp2.metric(f"Measured ({lbl})", f"{sp:.2f}×")
                    pp3.metric("Prediction error", f"{err:+.0f}%")
                    ok = abs(err) <= 15
                    (st.success if ok else st.info)(
                        f"The benchmark predicts the speedup from a **1-GPU** benchmark; "
                        f"here it is validated against the real multi-GPU run "
                        f"({'accurate' if ok else 'off'}: predicted {pred_sp:.2f}× vs "
                        f"measured {sp:.2f}×)."
                    )
                break

    # Theoretical (data-parallel) scaling — distributed runs as points at 2
    # workers vs the perfect-scaling line from the baseline.
    dist_timed = [(lbl, r, t) for lbl, r, t in timed
                  if r.mode in ("ddp", "ddp_hetero", "model_parallel")]
    if base_r.mode == "single" and dist_timed:
        st.markdown("### Theoretical vs real scaling")
        world_sizes = [1, 2, 4, 8]
        fig_scale = go.Figure()
        fig_scale.add_trace(go.Scatter(
            x=world_sizes, y=[base_t / ws / 60 for ws in world_sizes],
            name="Theoretical (100% efficiency)",
            line=dict(color=COLORS[4], width=2, dash="dash"), mode="lines+markers",
        ))
        for i, (lbl, r, t) in enumerate(dist_timed):
            fig_scale.add_trace(go.Scatter(
                x=[2], y=[t / 60], name=lbl, mode="markers",
                marker=dict(color=COLORS[i % len(COLORS)], size=14, symbol="star"),
            ))
        _n = len(dist_timed)
        fig_scale.update_layout(
            **_base_layout(320 + 20 * _n, "Epoch time vs number of workers",
                           margin=dict(l=50, r=16, t=48, b=70 + 20 * _n)),
            xaxis_title="Number of workers (processes)", yaxis_title="Minutes per epoch",
        )
        fig_scale.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.22,
                                            xanchor="left", x=0, font=dict(size=11)))
        fig_scale.update_xaxes(tickvals=world_sizes)
        _show(fig_scale, "ddp_scaling")
        st.caption(
            "The theoretical line assumes adding workers IDENTICAL to the baseline "
            "(perfect linear scaling, only valid for DATA parallelism). Real points fall "
            "below it due to communication overhead, the I/O bottleneck, imbalanced "
            "hardware (V100+CPU) or — for model parallelism — stage serialization."
        )


# ── Radar ───────────────────────────────────────────────────────────────────────

