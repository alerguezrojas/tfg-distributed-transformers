"""Tab render module — see src/web/app.py for the orchestrator.

System = local hardware Monitor + Import runs. The old Live monitor and
Launcher were removed: trainings now run on Kaggle/Verode, not from this
machine, so launching/streaming from the web was never usable. Import runs
replaces them with something useful for every machine — drop the zip a
remote run produced and it shows up in the dashboard.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.web.run_import import import_run_archive, import_run_folder, summarize_import
from src.web.system_monitor import get_snapshot

from src.web.ui.context import DashboardContext
from src.web.ui.helpers import ROOT, _get_runs, _get_feasibility_csvs


def render(ctx: DashboardContext) -> None:
    st.markdown("## System")
    st.caption("Local hardware monitor and importer for runs trained elsewhere "
               "(Kaggle, the Verode cluster…).")
    sub = st.tabs(["Monitor", "Import runs"])
    with sub[0]:
        _monitor(ctx)
    with sub[1]:
        _import_runs(ctx)


def _monitor(ctx: DashboardContext) -> None:
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


def _import_runs(ctx: DashboardContext) -> None:
    st.markdown("### Import runs trained elsewhere")
    st.caption(
        "Trainings run on Kaggle or the cluster and produce a `logs/` folder. "
        "Download it as a zip (or copy it to this machine) and import it here — "
        "the artifacts are copied into the repo's `logs/` tree and appear in the "
        "dashboard immediately. Works for any environment (kaggle, verode, local)."
    )

    logs_root = ROOT / "logs"

    # ── Upload a zip ──────────────────────────────────────────────────────────
    st.markdown("#### From a zip file")
    uploaded = st.file_uploader(
        "Drop a zip of the run's `logs/` folder (or its contents)",
        type=["zip"], accept_multiple_files=False,
    )
    if uploaded is not None and st.button("Import zip", type="primary"):
        try:
            rel = import_run_archive(uploaded.getvalue(), logs_root)
        except Exception as e:
            st.error(f"Could not read the zip: {e}")
            rel = []
        _report_import(rel)

    st.markdown("---")

    # ── Import from a folder path ─────────────────────────────────────────────
    st.markdown("#### From a folder on this machine")
    st.caption("Useful when you already copied the `logs/` folder somewhere "
               "(e.g. via `scp` from the cluster).")
    folder_str = st.text_input("Folder path", placeholder="/home/alejandro/Downloads/kaggle_logs")
    if folder_str and st.button("Import folder"):
        folder = Path(folder_str).expanduser()
        if not folder.is_dir():
            st.error(f"Not a folder: {folder}")
        else:
            rel = import_run_folder(folder, logs_root)
            _report_import(rel)

    # ── What's already in the repo ────────────────────────────────────────────
    st.markdown("---")
    runs = ctx.runs
    envs = sorted({r.env for r in runs})
    c1, c2, c3 = st.columns(3)
    c1.metric("Runs indexed", len(runs))
    c2.metric("Feasibility reports", len(_get_feasibility_csvs()))
    c3.metric("Environments", ", ".join(envs) if envs else "—")


def _report_import(rel_paths: list[str]) -> None:
    """Reports the outcome of an import and refreshes the run caches."""
    if not rel_paths:
        st.warning("No recognizable artifacts found "
                   "(expected train_*.log / *_metrics_*.csv / confusion_matrix_*.csv / "
                   "feasibility_*).")
        return
    s = summarize_import(rel_paths)
    # New files on disk → invalidate the discovery caches so they show up now.
    _get_runs.clear()
    _get_feasibility_csvs.clear()
    st.success(
        f"Imported {s['total']} file(s): {s['runs']} run log(s), "
        f"{s['metric_csvs']} metric CSV(s), {s['feasibility']} feasibility report(s)."
    )
    with st.expander("Imported files"):
        for p in rel_paths:
            st.write(f"`logs/{p}`")
    st.info("Select the new run in the sidebar to explore it.")
