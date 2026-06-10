"""Tab render module — see src/web/app.py for the orchestrator."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.web.confusion_matrix_parser import get_matrix_for_epoch, parse_confusion_matrix_csv
from src.web.dataset_stats import (
    CLASS_NAMES, SPLIT_SIZES,
    class_distribution_approximate, class_distribution_from_parquet,
    get_country_distribution, find_example_patches, load_rgb_image,
)
from src.web.feasibility_comparison import build_comparison
from src.web.feasibility_parser import parse_feasibility_csv, parse_ddp_scenarios
from src.web.model_explorer import ALL_FAMILIES, CURATED_MODELS, compare_models
from src.web.perclass_parser import parse_perclass_csv
from src.web.run_registry import RunInfo
from src.web.system_monitor import get_snapshot

from src.web.ui.charts import (
    COLORS, _show, _dl_csv, _base_layout, _metric_fig, _overlay_fig,
    _CLASS_GROUPS, _CLASS_GROUP_COLOR,
)
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import (
    ROOT, _load_df, _load_batch, _load_perclass, _get_runs, _get_feasibility_csvs,
    _feas_label, _run_config, _load_class_distribution, _load_example_images,
    _safe_max, _safe_idxmax, _safe_val_at_best, _throughput_col, _dur_str,
    _get_configs, _detect_anomalies, _read_log_tail, _parse_log_progress,
    _gpu_usage, _launch_process, _color_f1_cell,
)


def render(ctx: DashboardContext) -> None:
    st.markdown("## System")
    st.caption("Live hardware metrics, active-run monitor and training launcher.")
    sub = st.tabs(["Monitor", "Live", "Launcher"])
    with sub[0]:
        _monitor(ctx)
    with sub[1]:
        _live(ctx)
    with sub[2]:
        _launcher(ctx)


def _monitor(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    run = ctx.run
    refresh_interval = ctx.refresh_interval
    ref_int = st.sidebar.slider("System refresh (s)", 2, 30, 5, key="sys_ref_int")

    @st.fragment(run_every=ref_int)
    def _system_panel():
        snap = get_snapshot(disk_paths=["/", "/home", "/media"])

        st.markdown("### CPU")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Usage", f"{snap.cpu.usage_pct:.1f}%")
        sc2.metric("Logical cores", snap.cpu.count_logical)
        sc3.metric("Physical cores", snap.cpu.count_physical)
        sc4.metric("Frequency", f"{snap.cpu.freq_mhz:.0f} MHz" if snap.cpu.freq_mhz else "—")
        st.progress(snap.cpu.usage_pct / 100)

        st.markdown("### RAM")
        sr1, sr2, sr3, sr4 = st.columns(4)
        sr1.metric("Used", f"{snap.ram.used_gb:.1f} GB")
        sr2.metric("Total", f"{snap.ram.total_gb:.1f} GB")
        sr3.metric("Available", f"{snap.ram.available_gb:.1f} GB")
        sr4.metric("Usage %", f"{snap.ram.percent:.1f}%")
        st.progress(snap.ram.percent / 100)
        if snap.ram.swap_total_gb > 0:
            st.caption(f"Swap: {snap.ram.swap_used_gb:.1f} / {snap.ram.swap_total_gb:.1f} GB")

        st.markdown("### GPU")
        if snap.gpus:
            for gpu in snap.gpus:
                mem_pct = gpu.mem_used_mb / gpu.mem_total_mb * 100 if gpu.mem_total_mb else 0
                g1, g2, g3, g4, g5 = st.columns(5)
                g1.metric(f"GPU {gpu.index}", gpu.name[:28])
                g2.metric("VRAM used", f"{gpu.mem_used_mb / 1024:.1f} GB")
                g3.metric("VRAM total", f"{gpu.mem_total_mb / 1024:.1f} GB")
                g4.metric("Utilization", f"{gpu.util_pct}%")
                g5.metric("Temperature", f"{gpu.temp_c}°C")
                st.progress(mem_pct / 100,
                            text=f"VRAM {gpu.mem_used_mb}/{gpu.mem_total_mb} MB ({mem_pct:.1f}%)")
                if gpu.power_w is not None:
                    limit_str = f" / {gpu.power_limit_w:.0f} W" if gpu.power_limit_w else ""
                    st.caption(f"Power draw: {gpu.power_w:.1f} W{limit_str}")
                # Derived hardware specs (compute capability × SM count)
                if gpu.cuda_cores:
                    h1, h2, h3, h4, h5 = st.columns(5)
                    h1.metric("Architecture", gpu.architecture or "—")
                    h2.metric("Compute cap.", gpu.compute_capability or "—")
                    h3.metric("SMs", gpu.sm_count or "—")
                    h4.metric("CUDA cores", f"{gpu.cuda_cores:,}")
                    h5.metric("Tensor cores", f"{gpu.tensor_cores:,}" if gpu.tensor_cores else "0")
        else:
            st.info("No GPU detected (nvidia-smi unavailable).")

        st.markdown("### Disk")
        if snap.disks:
            disk_cols = st.columns(len(snap.disks))
            for col, disk in zip(disk_cols, snap.disks):
                col.metric(disk.path, f"{disk.free_gb:.1f} GB free")
                col.progress(disk.percent / 100,
                             text=f"{disk.used_gb:.1f} / {disk.total_gb:.1f} GB ({disk.percent:.1f}%)")
        else:
            st.info("Could not read disk usage.")

        st.markdown("### Network (cumulative since boot)")
        nn1, nn2 = st.columns(2)
        nn1.metric("Sent", f"{snap.network.bytes_sent_mb / 1024:.2f} GB")
        nn2.metric("Received", f"{snap.network.bytes_recv_mb / 1024:.2f} GB")

    _system_panel()



def _live(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    run = ctx.run
    refresh_interval = ctx.refresh_interval
    st.subheader("Live monitor")

    now_ts = time.time()
    recent_runs = [
        r for r in runs
        if r.log_path.exists() and (now_ts - r.log_path.stat().st_mtime) < 1800
    ]

    if not recent_runs:
        st.info(
            "No active runs (no log modified in the last 30 min). "
            "Launch a training from the Launcher tab."
        )
    else:
        live_labels = {r.label: r for r in recent_runs}
        lc1, lc2 = st.columns([3, 1])
        with lc1:
            live_sel = st.selectbox("Active run", list(live_labels.keys()), key="live_run_sel")
        with lc2:
            refresh_interval = st.slider("Refresh (s)", 5, 60, refresh_interval, key="live_ref_int")
        live_run = live_labels[live_sel]

        @st.fragment(run_every=refresh_interval)
        def _live_panel(run: RunInfo):
            _load_df.clear()

            gpu = _gpu_usage()
            if gpu:
                g1, g2, g3, g4 = st.columns(4)
                g1.metric("GPU", gpu["name"])
                g2.metric("VRAM", f"{gpu['mem_used_mb']/1024:.1f} / {gpu['mem_total_mb']/1024:.1f} GB")
                g3.metric("Utilization", f"{gpu['util_pct']}%")
                g4.metric("Temperature", f"{gpu['temp_c']} °C")
            else:
                st.caption("GPU info unavailable (nvidia-smi not found).")

            progress = _parse_log_progress(run.log_path)
            if progress["epochs"] > 0:
                pct = progress["epoch"] / progress["epochs"]
                st.progress(pct, text=f"Epoch {progress['epoch']} / {progress['epochs']}")

            if progress["last_val_f1"] is not None:
                m1, m2 = st.columns(2)
                m1.metric("Last Val F1", f"{progress['last_val_f1']:.4f}")
                if progress["last_val_loss"] is not None:
                    m2.metric("Last Val Loss", f"{progress['last_val_loss']:.4f}")

            if run.epoch_csv_path and run.epoch_csv_path.exists():
                live_df = _load_df(str(run.log_path), str(run.epoch_csv_path))
                if not live_df.empty:
                    fig_live = go.Figure()
                    if "val_f1" in live_df.columns:
                        fig_live.add_trace(go.Scatter(
                            x=live_df["epoch"], y=live_df["val_f1"],
                            name="Val F1", mode="lines+markers",
                            line=dict(color=COLORS[0], width=2), marker=dict(size=4),
                        ))
                    if "val_loss" in live_df.columns:
                        fig_live.add_trace(go.Scatter(
                            x=live_df["epoch"], y=live_df["val_loss"],
                            name="Val Loss", mode="lines+markers",
                            line=dict(color=COLORS[1], width=2), marker=dict(size=4),
                        ))
                    fig_live.update_layout(**_base_layout(280, "Metrics"), xaxis_title="Epoch")
                    _show(fig_live, "live_metrics")

            st.subheader("Log tail")
            st.code(_read_log_tail(run.log_path, n=40), language="text")

        _live_panel(live_run)


def _launcher(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    run = ctx.run
    refresh_interval = ctx.refresh_interval
    subtab_single, subtab_ddp_l = st.tabs(["Single GPU", "DDP (multi-GPU)"])

    configs_l = _get_configs()
    model_opts_l = [
        "vit_tiny_patch16_224", "vit_small_patch16_224", "vit_base_patch16_224",
        "resnet50", "efficientnet_b0", "deit_tiny_patch16_224",
    ]

    with subtab_single:
        st.subheader("Single-GPU training")
        with st.form("launcher_single_form"):
            la1, la2 = st.columns(2)
            with la1:
                l_model = st.selectbox("Model", model_opts_l)
                l_config = st.selectbox("YAML config", configs_l if configs_l else ["(none)"])
                l_epochs = st.number_input("Override epochs", min_value=0, value=0,
                                           help="0 = use the config value")
                l_batch = st.number_input("Override batch size", min_value=0, value=0,
                                          help="0 = use the config value")
            with la2:
                l_trace = st.selectbox("Trace mode", ["simple", "off", "deep"])
                l_layers = st.multiselect("Layers", ["plot", "hooks", "confusion", "batch-monitor"],
                                          default=["confusion"])
                l_fn = st.multiselect("Fn decorators", ["timing", "energy"])
                l_inspect = st.multiselect("Inspect features",
                                           ["model-summary", "grad-monitor", "anomalies", "batch-table"])
                from src.gpu_specs import detect_all as _detect
                from src.precision import available_precisions as _avail, label as _plabel
                _g = _detect()
                _precs_l = _avail(_g[0].compute_capability if _g else None, is_cuda=bool(_g))
                l_precision = st.selectbox(
                    "Precision (Tensor cores)", _precs_l, format_func=_plabel,
                    help="fp32 = CUDA cores; tf32/amp/bf16 = Tensor cores (faster, less VRAM).",
                )
            launched_single = st.form_submit_button("Launch")

        if launched_single:
            parts_l = [
                "uv run python scripts/train_single_gpu.py",
                f"--config configs/{l_config}",
                f"--model {l_model}",
                f"--trace {l_trace}",
            ]
            if l_epochs > 0:
                parts_l.append(f"--epochs {l_epochs}")
            if l_batch > 0:
                parts_l.append(f"--batch-size {l_batch}")
            if l_layers:
                parts_l.append(f"--layers {' '.join(l_layers)}")
            if l_fn:
                parts_l.append(f"--fn {' '.join(l_fn)}")
            if l_inspect:
                parts_l.append(f"--inspect {' '.join(l_inspect)}")
            if l_precision and l_precision != "fp32":
                parts_l.append(f"--precision {l_precision}")
            cmd_l = " ".join(parts_l)
            st.code(cmd_l, language="bash")
            out_ph_l = st.empty()
            rc_l = _launch_process(cmd_l, out_ph_l)
            if rc_l == 0:
                st.success("Training complete.")
                _get_runs.clear()
            else:
                st.error(f"Process exited with code {rc_l}.")

    with subtab_ddp_l:
        st.subheader("DDP training")
        with st.form("launcher_ddp_form"):
            dd1, dd2 = st.columns(2)
            with dd1:
                d_nproc = st.number_input("GPUs (--nproc_per_node)", min_value=1, max_value=8, value=2)
                d_model = st.selectbox("Model", model_opts_l, key="ddp_model")
                d_config = st.selectbox("YAML config", configs_l if configs_l else ["(none)"],
                                         key="ddp_config")
                d_epochs = st.number_input("Override epochs", min_value=0, value=0, key="ddp_ep")
            with dd2:
                d_trace = st.selectbox("Trace mode", ["simple", "off", "deep"], key="ddp_trace")
                d_layers = st.multiselect("Layers", ["plot", "confusion", "batch-monitor"],
                                          default=["confusion"], key="ddp_layers")
                d_fn = st.multiselect("Fn decorators", ["timing", "energy"], key="ddp_fn")
            launched_ddp = st.form_submit_button("Launch")

        if launched_ddp:
            parts_d = [
                f"torchrun --nproc_per_node={d_nproc} scripts/train_ddp.py",
                f"--config configs/{d_config}",
                f"--model {d_model}",
                f"--trace {d_trace}",
            ]
            if d_epochs > 0:
                parts_d.append(f"--epochs {d_epochs}")
            if d_layers:
                parts_d.append(f"--layers {' '.join(d_layers)}")
            if d_fn:
                parts_d.append(f"--fn {' '.join(d_fn)}")
            cmd_d = " ".join(parts_d)
            st.code(cmd_d, language="bash")
            out_ph_d = st.empty()
            rc_d = _launch_process(cmd_d, out_ph_d)
            if rc_d == 0:
                st.success("DDP training complete.")
                _get_runs.clear()
            else:
                st.error(f"Process exited with code {rc_d}.")

