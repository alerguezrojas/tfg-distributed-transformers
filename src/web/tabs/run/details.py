"""Run results — details view."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.feasibility_parser import (parse_feasibility_csv)
from src.web.ui.charts import (COLORS, _base_layout, _dl_csv, _show)
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import (ROOT, _detect_anomalies, _dur_str, _get_configs, _get_feasibility_csvs, _load_df, _run_config, _safe_max, _safe_val_at_best, _throughput_col)


def _time(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    run = ctx.run
    refresh_interval = ctx.refresh_interval
    if selected_run is None:
        st.info("Select a run in the sidebar.")
    else:
        df_time = _load_df(str(run.log_path), str(run.epoch_csv_path) if run.epoch_csv_path else None)

        if "epoch_time" not in df_time.columns or df_time["epoch_time"].isna().all():
            st.info("No per-epoch time data. Use `--trace simple` to generate it.")
        else:
            et = df_time[["epoch", "epoch_time"]].dropna()
            total_s_t = et["epoch_time"].sum()
            avg_s_t = et["epoch_time"].mean()

            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Total", _dur_str(total_s_t))
            t2.metric("Average/epoch", f"{avg_s_t/60:.1f} min")
            t3.metric("Min/epoch", f"{et['epoch_time'].min()/60:.1f} min")
            t4.metric("Max/epoch", f"{et['epoch_time'].max()/60:.1f} min")

            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=et["epoch"], y=et["epoch_time"] / 60,
                name="Real (min)", mode="lines+markers",
                line=dict(color=COLORS[0], width=2), marker=dict(size=4),
            ))

            has_train_t = ("epoch_time_train_s" in df_time.columns
                           and df_time["epoch_time_train_s"].notna().any())
            has_eval_t = ("epoch_time_eval_s" in df_time.columns
                          and df_time["epoch_time_eval_s"].notna().any())
            if has_train_t:
                fig_time.add_trace(go.Scatter(x=et["epoch"], y=df_time["epoch_time_train_s"] / 60,
                                              name="Train (min)", mode="lines",
                                              line=dict(color=COLORS[2], width=2, dash="dot")))
            if has_eval_t:
                fig_time.add_trace(go.Scatter(x=et["epoch"], y=df_time["epoch_time_eval_s"] / 60,
                                              name="Eval (min)", mode="lines",
                                              line=dict(color=COLORS[1], width=2, dash="dash")))

            if len(et) >= 2:
                x_arr = et["epoch"].values.astype(float)
                y_arr = et["epoch_time"].values / 60
                coeffs = np.polyfit(x_arr, y_arr, 1)
                fig_time.add_trace(go.Scatter(x=et["epoch"], y=np.polyval(coeffs, x_arr),
                                              name="Trend", mode="lines",
                                              line=dict(color="#94a3b8", width=1, dash="dash")))

            # Warmup region — only from a config that matches THIS run's env AND model.
            # (Picking any config would shade the wrong epochs, e.g. a 5-epoch warmup
            # over a 3-epoch demo run.) If nothing matches, don't shade.
            import yaml
            warmup_ep = None
            for cfg_name in _get_configs():
                try:
                    cfg = yaml.safe_load((ROOT / "configs" / cfg_name).read_text())
                    env_cfg = cfg.get("output", {}).get("env", "")
                    model_cfg = cfg.get("model", {}).get("name", "")
                    env_ok = env_cfg == run.env or (run.env == "local" and "cluster" not in cfg_name)
                    if env_ok and run.model and model_cfg == run.model:
                        warmup_ep = cfg.get("training", {}).get("warmup_epochs")
                        break
                except Exception:
                    pass
            if warmup_ep:
                # cap at the run's epoch count (a run can stop before warmup ends)
                x1 = min(warmup_ep + 0.5, len(et) + 0.5)
                fig_time.add_vrect(x0=0.5, x1=x1, fillcolor="#A8823E", opacity=0.08,
                                   annotation_text=f"Warmup ({warmup_ep} ep)",
                                   annotation_position="top left")

            feasibility_csvs_t = _get_feasibility_csvs()
            # Pick the feasibility whose MODEL matches the displayed run; otherwise
            # the estimate line would be for a different model (a false comparison).
            feas_match_t = None
            for _fc in feasibility_csvs_t:
                try:
                    _m_t, _ = parse_feasibility_csv(_fc)
                    if run.model and _m_t.get("model_name") == run.model:
                        feas_match_t = _fc
                        break
                except Exception:
                    pass
            # No fallback to a different model's report: only draw the estimate when
            # a feasibility report matches this run's model (else it's a false compare).
            if feas_match_t:
                try:
                    _, bdf_t = parse_feasibility_csv(feas_match_t)
                    viable_t = bdf_t[bdf_t["oom"] == "no"].copy()
                    tp_col_t = _throughput_col(viable_t)
                    per_ep_col = next((c for c in ["est_total_min_per_epoch", "est_min_per_epoch_30ep"]
                                       if c in viable_t.columns), None)
                    _idx_t = (viable_t[tp_col_t].idxmax()
                               if (tp_col_t and per_ep_col and not viable_t.empty) else None)
                    if _idx_t is not None and not pd.isna(_idx_t):
                        est_min = float(viable_t.loc[_idx_t, per_ep_col])
                        fig_time.add_hline(y=est_min, line_dash="dash", line_color=COLORS[1],
                                           annotation_text=f"Feasibility estimate: {est_min:.0f} min/epoch",
                                           annotation_position="top right")
                except Exception:
                    pass

            fig_time.update_layout(**_base_layout(380, "Time per epoch"),
                                   xaxis_title="Epoch", yaxis_title="Minutes")
            _show(fig_time, "time_per_epoch")
            _dl_csv(et.assign(epoch_time_min=et["epoch_time"] / 60),
                    "time_per_epoch.csv", "Download time data")

            # Estimated vs Real — use the SAME-model feasibility report as the chart
            # line (feas_match_t), never an arbitrary one, so the numbers are honest.
            if feas_match_t:
                try:
                    _, bdf_c = parse_feasibility_csv(feas_match_t)
                    viable_c = bdf_c[bdf_c["oom"] == "no"].copy()
                    tp_col_c = _throughput_col(viable_c)
                    per_ep_c = next((c for c in ["est_total_min_per_epoch", "est_min_per_epoch_30ep"]
                                     if c in viable_c.columns), None)
                    _idx_c = (viable_c[tp_col_c].idxmax()
                               if (tp_col_c and per_ep_c and not viable_c.empty) else None)
                    if _idx_c is not None and not pd.isna(_idx_c):
                        est_val = float(viable_c.loc[_idx_c, per_ep_c])
                        real_val = avg_s_t / 60
                        err_pct = (real_val - est_val) / est_val * 100 if est_val else 0
                        st.markdown("**Estimated vs Real**")
                        ce1, ce2, ce3 = st.columns(3)
                        ce1.metric("Estimated (min/epoch)", f"{est_val:.1f}")
                        ce2.metric("Real average (min/epoch)", f"{real_val:.1f}")
                        ce3.metric("Relative error", f"{err_pct:+.1f}%")
                except Exception:
                    pass
            elif feasibility_csvs_t:
                st.caption("No feasibility report for this run's model, so the "
                           "predicted-vs-real timing comparison is hidden (comparing "
                           "against a different model would be misleading).")



def _info(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    run = ctx.run
    refresh_interval = ctx.refresh_interval
    if selected_run is None:
        st.info("Select a run in the sidebar.")
    else:
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
                total_si = df_info["epoch_time"].sum()
                rows_i["Total time"] = _dur_str(total_si)
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

