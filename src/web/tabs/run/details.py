"""Run results — details view (run metadata + YAML config + anomalies + log)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.web.ui.context import DashboardContext
from src.web.ui.helpers import (ROOT, _detect_anomalies, _dur_str, _get_configs,
                                 _load_df, _run_config, _run_metric, _safe_max,
                                 _safe_val_at_best)


def _info(ctx: DashboardContext) -> None:
    selected_run = ctx.selected_run
    run = ctx.run
    if selected_run is None:
        st.info("Select a run in the sidebar.")
        return

    df_info = _load_df(str(run.log_path), str(run.epoch_csv_path) if run.epoch_csv_path else None)
    n_ep_i = len(df_info)
    metric = _run_metric(run, df_info)
    best_f1_i = (_safe_max(df_info[metric.column]) if metric.column in df_info.columns
                 else float("nan"))
    best_ep_i_v = _safe_val_at_best(df_info, metric.column, "epoch")
    meta = run.meta

    col_m, col_f = st.columns(2)

    with col_m:
        st.subheader("Run metadata")
        _cfg_run = _run_config(str(run.log_path))
        rows_i = {
            "Dataset": run.dataset,
            "Task": ((meta.task if meta else None) or "—") + (" (ordered classes)" if meta and meta.ordinal else ""),
            "Model": run.model or "—",
            "Environment · Mode": f"{run.env} · {run.mode}",
            "Trace mode": run.trace_mode,
            "Epochs": n_ep_i,
            f"Best Val {metric.name}": f"{best_f1_i:.4f}" if not pd.isna(best_f1_i) else "—",
            "Best epoch": str(int(best_ep_i_v)) if best_ep_i_v is not None else "—",  # one dtype for Arrow
        }
        # What distinguishes the run from its neighbours (manifest / events / log line).
        if meta:
            rows_i["Loss"] = meta.loss_label
            if meta.seed is not None:
                rows_i["Seed"] = str(meta.seed)
            if meta.providers_label:
                rows_i["Providers"] = meta.providers_label
            if meta.providers_test:
                rows_i["Test providers"] = "+".join(meta.providers_test)
            if meta.color_jitter:
                rows_i["Color jitter"] = f"{meta.color_jitter:g}"
            if meta.img_size:
                rows_i["Input size"] = f"{meta.img_size}px"
            if meta.n_train and meta.n_val:
                rows_i["Train/val samples"] = f"{meta.n_train:,} / {meta.n_val:,}"
            if meta.command:
                rows_i["Command"] = f"`{meta.command}`"
        # Run config (only new / backfilled runs record it)
        if _cfg_run.get("batch"):
            rows_i["Batch size"] = _cfg_run["batch"]
        if _cfg_run.get("reparto"):
            rows_i["Data split"] = _cfg_run["reparto"]
        if _cfg_run.get("lr"):
            rows_i["Learning rate"] = _cfg_run["lr"]
        if _cfg_run.get("train"):
            rows_i["Train/val images"] = f"{_cfg_run.get('train', '?')} / {_cfg_run.get('val', '?')}"
        if "epoch_time" in df_info.columns and df_info["epoch_time"].notna().any():
            rows_i["Total time"] = _dur_str(df_info["epoch_time"].sum())
            rows_i["Average/epoch"] = f"{df_info['epoch_time'].mean()/60:.1f} min"
        for k, v in rows_i.items():
            st.markdown(f"**{k}:** {v}")
        if meta:
            st.caption(f"Run metadata from: {meta.source}"
                       + ("" if meta.source != "path" else " (the folder layout only)"))

    with col_f:
        st.subheader("Associated files")
        for label_f, path_f in [
            ("Batch CSV", run.batch_csv_path),
            ("Per-class CSV", run.perclass_csv_path),
            ("Epoch CSV", run.epoch_csv_path),
            ("Confusion matrix CSV", run.confusion_matrix_csv_path),
        ]:
            st.markdown(f"- **{label_f}:** `{path_f.name if path_f else '—'}`")

    st.markdown("---")

    st.subheader("YAML config")
    import yaml
    configs_i: list[Path] = []
    for cfg in _get_configs():
        try:
            cfg_path = ROOT / "configs" / cfg
            cfg_data = yaml.safe_load(cfg_path.read_text())
            env_cfg = cfg_data.get("output", {}).get("env", "")
            if env_cfg == run.env or (run.env == "local" and "cluster" not in cfg):
                configs_i.append(cfg_path)
        except Exception:
            pass
    if configs_i:
        cfg_sel = st.selectbox("Config", [p.name for p in configs_i])
        cfg_path_sel = next(p for p in configs_i if p.name == cfg_sel)
        st.code(cfg_path_sel.read_text(), language="yaml")
    else:
        st.caption("Could not determine the config for this run.")

    st.subheader("Anomaly detection")
    anomalies = _detect_anomalies(run.log_path)
    if anomalies:
        st.warning(f"{len(anomalies)} line(s) with detected anomalies.")
        with st.expander("View anomalies"):
            for line in anomalies:
                st.text(line)
    else:
        st.success("No anomalies detected in the log.")

    st.subheader("Log")
    search_term = st.text_input("Filter log lines", "")
    try:
        all_lines = run.log_path.read_text(errors="replace").splitlines()
        if search_term:
            disp_lines = [ln for ln in all_lines if search_term.lower() in ln.lower()]
            st.caption(f"{len(disp_lines)} / {len(all_lines)} lines")
        else:
            disp_lines = all_lines
            st.caption(f"{len(all_lines)} lines total")
        st.code("\n".join(disp_lines[-400:]), language="text")
    except Exception as exc:
        st.error(str(exc))
