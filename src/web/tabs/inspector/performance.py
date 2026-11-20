"""Performance sub-view: throughput, DataLoader wait and step time."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.web.inspector_data import (
    DATA_WAIT_IO_BOUND, batch_performance, diagnose_data_wait, epoch_performance,
)
from src.web.tabs.inspector._common import BATCH_HINT, InspectorCtx, fmt, lines
from src.web.ui.charts import _show

_DIAG = {"bad": st.error, "warn": st.warning, "ok": st.success, "none": st.info}


def render(c: InspectorCtx) -> None:
    if c.state is None:
        st.info("Performance comes from the run's event stream (`events_*.jsonl`), which this "
                "run does not have.")
        return
    batches = batch_performance(c.state.batch_rows())
    epochs = epoch_performance(c.state.epoch_ends)
    if batches.empty and epochs.empty:
        st.info("No throughput data in this run. " + BATCH_HINT)
        return

    # Diagnosis: the latest epoch's data_wait_frac; per-batch estimate as a fallback.
    frac = None
    if not epochs.empty and epochs["data_wait_frac"].notna().any():
        frac = float(epochs["data_wait_frac"].dropna().iloc[-1])
    elif "data_frac" in batches.columns and batches["data_frac"].notna().any():
        frac = float(batches["data_frac"].mean())
    level, text = diagnose_data_wait(frac)
    _DIAG[level](text)

    k = st.columns(4)
    sps = batches["samples_per_s"] if "samples_per_s" in batches.columns else epochs["samples_per_s"]
    k[0].metric("Mean samples/s", fmt(sps.mean(), "{:,.0f}"))
    k[1].metric("Data wait", fmt(None if frac is None else 100 * frac, "{:.0f} %"))
    if "step_time_s" in batches.columns:
        k[2].metric("Median step time", fmt(1000 * batches["step_time_s"].median(), "{:.0f} ms"))
        k[3].metric("p95 step time", fmt(1000 * batches["step_time_s"].quantile(0.95), "{:.0f} ms"))

    left, right = st.columns(2)
    with left:
        if "samples_per_s" in batches.columns:
            lines(batches, "progress", "samples_per_s", None, "Samples/s per batch",
                  "samples/s", "perf_batch_sps")
        else:
            st.caption("Per-batch throughput not logged. " + BATCH_HINT)
    with right:
        if {"data_time_s", "step_time_s"} <= set(batches.columns):
            fig = go.Figure()
            for col, name in (("step_time_s", "Step"), ("data_time_s", "DataLoader wait")):
                fig.add_trace(go.Scatter(x=batches["progress"], y=batches[col], name=name,
                                         mode="lines", stackgroup="t"))
            fig.update_layout(height=260, title={"text": "Time per batch"}, hovermode="x unified",
                              yaxis_title="seconds")
            _show(fig, "perf_time_split")
        else:
            st.caption("Step / data time per batch not logged. " + BATCH_HINT)

    if not epochs.empty:
        e1, e2 = st.columns(2)
        with e1:
            if epochs["samples_per_s"].notna().any():
                fig = go.Figure(go.Bar(x=epochs["epoch"], y=epochs["samples_per_s"]))
                fig.update_layout(height=260, title={"text": "Samples/s per epoch"},
                                  xaxis_title="epoch", yaxis_title="samples/s")
                _show(fig, "perf_epoch_sps")
        with e2:
            if epochs["data_wait_frac"].notna().any():
                fig = go.Figure(go.Bar(x=epochs["epoch"], y=epochs["data_wait_frac"]))
                fig.add_hline(y=DATA_WAIT_IO_BOUND, line_dash="dash", line_color="#B0413E",
                              annotation_text=f"I/O-bound > {DATA_WAIT_IO_BOUND:.1f}")
                fig.update_layout(height=260, title={"text": "DataLoader wait (fraction of train time)"},
                                  xaxis_title="epoch", yaxis_title="fraction",
                                  yaxis_range=[0, max(1.0, float(epochs["data_wait_frac"].max()))])
                _show(fig, "perf_epoch_wait")
    st.caption(f"Rule of thumb: waiting more than {100 * DATA_WAIT_IO_BOUND:.0f}% of the training "
               "time for the DataLoader means the run is limited by input, not by compute.")
