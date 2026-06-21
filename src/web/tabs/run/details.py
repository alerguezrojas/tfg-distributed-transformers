"""Run results — details view (run metadata + YAML config + anomalies + log)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.web.ui.context import DashboardContext
from src.web.ui.helpers import (ROOT, _detect_anomalies, _dur_str, _get_configs,
                                 _load_df, _run_config, _safe_max, _safe_val_at_best)


def _info(ctx: DashboardContext) -> None:
    selected_run = ctx.selected_run
    run = ctx.run
    if selected_run is None:
        st.info("Select a run in the sidebar.")
        return

    df_info = _load_df(str(run.log_path), str(run.epoch_csv_path) if run.epoch_csv_path else None)
    n_ep_i = len(df_info)
    best_f1_i = _safe_max(df_info["val_f1"]) if "val_f1" in df_info.columns else float("nan")
    best_ep_i_v = _safe_val_at_best(df_info, "val_f1", "epoch")

    col_m, col_f = st.columns(2)

    with col_m:
        st.subheader("Run metadata")
        _cfg_run = _run_config(str(run.log_path))
        rows_i = {
            "Model": run.model or "—",
            "Environment · Mode": f"{run.env} · {run.mode}",
            "Trace mode": run.trace_mode,
            "Epochs": n_ep_i,
            "Best Val F1": f"{best_f1_i:.4f}" if not pd.isna(best_f1_i) else "—",
            "Best epoch": int(best_ep_i_v) if best_ep_i_v is not None else "—",
        }
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
        if not _cfg_run:
            st.caption("ℹ️ Batch size is only recorded in new runs (from this "
                       "version onwards). Earlier ones did not store it in the log.")

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
