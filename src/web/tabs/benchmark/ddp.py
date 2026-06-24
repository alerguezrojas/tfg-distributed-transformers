"""Benchmark — distributed-scaling prediction (lean: table + speedup + total time)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.benchmark_parser import (parse_ddp_scenarios)
from src.web.ui.charts import (COLORS, _base_layout, _dl_csv, _show)


def render_ddp_analysis(meta, benchmark_csvs) -> None:
    if not benchmark_csvs:
        st.info("Run the benchmark analysis first.")
        return
    ddp_df = parse_ddp_scenarios(meta)
    if ddp_df.empty:
        st.caption("No distributed-scaling prediction in this report.")
        return

    st.caption("Predicted speedup, efficiency and bottleneck from 1 to 8 GPUs "
               "(data parallel). The I/O total is fixed across GPUs, so efficiency "
               "drops once it dominates the per-GPU compute.")

    # ── Scenario table (bottleneck colour + speedup gradient) ───────────────────
    def _color_bottleneck(val: str) -> str:
        if val == "io":
            return "background-color: #fee2e2; color: #991b1b"
        if val == "sync":
            return "background-color: #fef3c7; color: #92400e"
        return "background-color: #d1fae5; color: #065f46"

    styled = ddp_df.style
    if "bottleneck" in ddp_df.columns:
        styled = styled.map(_color_bottleneck, subset=["bottleneck"])
    if "speedup" in ddp_df.columns:
        styled = styled.background_gradient(
            subset=["speedup"], cmap="Greens",
            vmin=1.0, vmax=float(ddp_df["n_gpus"].max() or 8))
    st.dataframe(styled, use_container_width=True, hide_index=True)
    _dl_csv(ddp_df, "ddp_scenarios.csv", "Download DDP scenarios")

    # ── Two compact charts: speedup vs GPUs + total time ────────────────────────
    if {"n_gpus", "speedup"}.issubset(ddp_df.columns):
        d = ddp_df.copy()
        for c in ("n_gpus", "speedup"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["n_gpus", "speedup"])
        ns = d["n_gpus"].tolist()
        g1, g2 = st.columns(2)
        with g1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ns, y=ns, name="Ideal (linear)",
                                     line=dict(color="#94a3b8", dash="dash")))
            fig.add_trace(go.Scatter(x=ns, y=d["speedup"], name="Predicted",
                                     mode="lines+markers",
                                     line=dict(color=COLORS[2], width=3), marker=dict(size=9)))
            fig.update_layout(**_base_layout(300, "Speedup vs number of GPUs"),
                              xaxis_title="GPUs", yaxis_title="Speedup ×")
            fig.update_xaxes(tickvals=ns)
            _show(fig, "ddp_speedup")
        with g2:
            if "time_total_h" in d.columns:
                d["t"] = pd.to_numeric(d["time_total_h"], errors="coerce")
                fig_t = go.Figure(go.Bar(
                    x=[f"{int(n)} GPU" for n in ns], y=d["t"], marker_color=COLORS[0],
                    text=[f"{v:.2f}h" for v in d["t"]], textposition="outside"))
                fig_t.update_layout(**_base_layout(300, "Predicted total time"),
                                    xaxis_title="", yaxis_title="Hours")
                _show(fig_t, "ddp_total_time")
