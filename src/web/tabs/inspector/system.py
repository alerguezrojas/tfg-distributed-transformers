"""System sub-view: GPU / CPU / RAM time series from the ``system`` events."""
from __future__ import annotations

import streamlit as st

from src.web.inspector_data import energy_wh, system_frames, system_summary
from src.web.tabs.inspector._common import SYSTEM_HINT, InspectorCtx, fmt, lines

_GPU_CHARTS = (
    ("util_pct", "GPU utilisation", "%"), ("mem_used_mb", "GPU memory used", "MB"),
    ("power_w", "GPU power", "W"), ("temp_c", "GPU temperature", "°C"),
    ("sm_clock_mhz", "SM clock", "MHz"),
)
_HOST_CHARTS = (
    ("cpu_pct", "CPU (whole machine)", "%"), ("ram_used_pct", "RAM used (machine)", "%"),
    ("proc_cpu_pct", "Training process CPU", "% of one core"),
    ("proc_rss_mb", "Training process memory (RSS)", "MB"),
)


def render(c: InspectorCtx) -> None:
    if c.state is None or not c.state.systems:
        st.info("No system samples in this run. " + SYSTEM_HINT)
        return
    gpu, host = system_frames(c.state.systems)

    if gpu.empty:
        st.caption("No GPU readings (NVML unavailable on the training machine): "
                   "showing CPU and memory only.")
    else:
        summary = system_summary(gpu)
        energy = energy_wh(gpu)
        top = st.columns(4)
        top[0].metric("Mean GPU utilisation", fmt(gpu["util_pct"].mean(), "{:.0f} %"))
        top[1].metric("Peak GPU memory", fmt(gpu["mem_used_mb"].max(), "{:,.0f} MB"))
        top[2].metric("Mean power (all GPUs)",
                      fmt(gpu.groupby("t_s")["power_w"].sum().mean(), "{:.0f} W"))
        top[3].metric("Energy (∫ power dt)", fmt(sum(energy.values()), "{:.2f} Wh") if energy else "—",
                      help="Trapezoidal integral of the sampled power, summed over GPUs. An "
                           "approximation limited by the sampling interval, not a metered value.")
        st.dataframe(summary.round(1), hide_index=True, width="stretch")

        gpu = gpu.assign(device=gpu["gpu"].map(lambda g: f"GPU {g}"))
        for i in range(0, len(_GPU_CHARTS), 2):
            cols = st.columns(2)
            for col, (field, title, unit) in zip(cols, _GPU_CHARTS[i:i + 2]):
                with col:
                    lines(gpu, "t_s", field, "device", title, unit, f"sys_{field}",
                          xtitle="seconds since start")
        st.caption("Time axis: seconds since the first sample.")

    st.markdown("**Host**")
    for i in range(0, len(_HOST_CHARTS), 2):
        cols = st.columns(2)
        for col, (field, title, unit) in zip(cols, _HOST_CHARTS[i:i + 2]):
            if host[field].notna().any():
                with col:
                    lines(host, "t_s", field, None, title, unit, f"sys_{field}",
                          xtitle="seconds since start")
