"""Alerts sub-view: chronological list of the trainer's health alerts (NaN loss, divergence,
overfitting, exploding gradients, I/O-bound loader, hot or full GPU...)."""
from __future__ import annotations

import streamlit as st

from src.web.events_reader import MAX_ALERTS
from src.web.inspector_data import CRITICAL, WARNING, alert_frame
from src.web.tabs.inspector._common import InspectorCtx

ALERTS_HINT = ("The trainer emits an `alert` event when one of its health rules fires "
               "(codes: nan_loss, loss_divergence, overfitting, grad_explosion, "
               "update_ratio_out_of_range, dataloader_bound, gpu_memory_high, gpu_temp_high, "
               "throughput_drop). A run without any simply had none, or predates them.")


def render(c: InspectorCtx) -> None:
    if c.state is None:
        st.info("Alerts come from the run's event stream (`events_*.jsonl`), which this run "
                "does not have.")
        return
    df = alert_frame(c.state.alerts)
    if df.empty:
        st.success("No alerts in this run.")
        st.caption(ALERTS_HINT)
        return
    n_crit = int((df["level"] == CRITICAL).sum())
    k = st.columns(3)
    k[0].metric("Alerts", len(df))
    k[1].metric("Critical", n_crit)
    k[2].metric("Warnings", len(df) - n_crit)
    levels = st.multiselect("Level", [CRITICAL, WARNING], default=[CRITICAL, WARNING],
                            key="insp_alert_level")
    codes = sorted(df["code"].unique())
    shown_codes = st.multiselect("Code", codes, default=codes, key="insp_alert_code")
    view = df[df["level"].isin(levels) & df["code"].isin(shown_codes)]
    st.dataframe(
        view.assign(time=view["time"].dt.strftime("%Y-%m-%d %H:%M:%S")),
        hide_index=True, width="stretch",
        column_config={"epoch": st.column_config.NumberColumn(format="%d"),
                       "batch": st.column_config.NumberColumn(format="%d"),
                       "value": st.column_config.NumberColumn(format="%.4g"),
                       "threshold": st.column_config.NumberColumn(format="%.4g")})
    st.caption(f"Chronological, oldest first (the most recent {MAX_ALERTS} are kept). "
               "Times are UTC.")
