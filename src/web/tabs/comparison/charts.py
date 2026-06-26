"""Compare — charts."""
from __future__ import annotations


import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.ui.charts import (COLORS, _base_layout, _overlay_fig, _show)
from src.web.ui.helpers import (_safe_val_at_best)
from src.web.tabs.comparison._common import (_has)


def _radar_section(compare_dfs: list[tuple[str, pd.DataFrame]]) -> None:
    st.markdown("#### Metric radar at the best epoch")
    radar_metrics = ["val_f1", "train_f1", "val_acc", "val_prec", "val_rec"]
    radar_fig = go.Figure()
    for i, (lbl, cdf) in enumerate(compare_dfs):
        vals = [
            float(v) if (v := _safe_val_at_best(cdf, "val_f1", m)) is not None else 0.0
            for m in radar_metrics
        ]
        vals_closed = vals + [vals[0]]
        radar_fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=radar_metrics + [radar_metrics[0]],
            fill="toself", name=lbl,
            line=dict(color=COLORS[i % len(COLORS)]), opacity=0.6,
        ))
    # Full-label legend below the radar: one row per run stays readable
    # even with 7-8 runs selected.
    _n_radar = len(compare_dfs)
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=380 + 20 * _n_radar,
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0),
        margin=dict(l=60, r=60, t=40, b=40 + 20 * _n_radar),
        title=dict(text="Metrics at the best Val F1 epoch", font=dict(size=13)),
    )
    _show(radar_fig, "radar_comparison")


# ── Energy ──────────────────────────────────────────────────────────────────────

def _energy_section(compare_dfs: list[tuple[str, pd.DataFrame]]) -> None:
    energy_rows = []
    for lbl, cdf in compare_dfs:
        t_wh = cdf["energy_train_wh"].dropna().sum() if _has(cdf, "energy_train_wh") else 0.0
        e_wh = cdf["energy_eval_wh"].dropna().sum() if _has(cdf, "energy_eval_wh") else 0.0
        if t_wh or e_wh:
            energy_rows.append((lbl, t_wh, e_wh))
    if not energy_rows:
        return

    st.markdown("---")
    st.markdown("#### Energy consumption")
    st.caption(
        "Total energy over the whole run (Wh), as measured by pynvml on the "
        "logging GPU. Runs without energy measurement (no `--fn energy`, e.g. "
        "model-parallel) are not shown."
    )
    fig_energy = go.Figure()
    _lbls = [l for l, _, _ in energy_rows]
    fig_energy.add_trace(go.Bar(
        y=_lbls, x=[t for _, t, _ in energy_rows], name="Train",
        orientation="h", marker_color=COLORS[0],
    ))
    fig_energy.add_trace(go.Bar(
        y=_lbls, x=[e for _, _, e in energy_rows], name="Eval",
        orientation="h", marker_color=COLORS[1],
    ))
    fig_energy.update_layout(
        **_base_layout(160 + 44 * len(energy_rows), "Total energy per run (Wh)",
                       margin=dict(l=10, r=16, t=48, b=40)),
        barmode="stack", xaxis_title="Wh",
    )
    fig_energy.update_yaxes(autorange="reversed", automargin=True)
    # Outside the plot: the inside-top-left default would cover the first bar.
    fig_energy.update_layout(legend=dict(
        orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
        bgcolor="rgba(0,0,0,0)",
    ))
    _show(fig_energy, "compare_energy_total")

    _n_eff = [(l, (t + e)) for l, t, e in energy_rows]
    _best = min(_n_eff, key=lambda x: x[1])
    _worst = max(_n_eff, key=lambda x: x[1])
    if _worst[1] > 0 and _best != _worst:
        st.caption(
            f"Most efficient: **{_best[0]}** ({_best[1]:.1f} Wh) — "
            f"{_worst[1]/_best[1]:.1f}× less energy than **{_worst[0]}** "
            f"({_worst[1]:.1f} Wh)."
        )
    st.markdown("---")


# ── Per-epoch overlays ──────────────────────────────────────────────────────────

def _overlay_charts(compare_dfs: list[tuple[str, pd.DataFrame]]) -> None:
    has_energy = any(_has(d, "energy_train_wh") or _has(d, "energy_eval_wh")
                     for _, d in compare_dfs)
    _energy_opts = (["energy_train_wh", "energy_eval_wh", "power_train_w"]
                    if has_energy else [])
    metrics_to_compare = st.multiselect(
        "Metrics to overlay",
        ["val_f1", "val_loss", "train_f1", "train_loss", "val_prec", "val_rec",
         "epoch_time"] + _energy_opts,
        default=["val_f1", "val_loss"],
    )
    cols = st.columns(2)
    for idx, col_name in enumerate(metrics_to_compare):
        fig = _overlay_fig(compare_dfs, col=col_name,
                           title=col_name.replace("_", " "), y_label=col_name)
        with cols[idx % 2]:
            _show(fig, f"compare_{col_name}")
