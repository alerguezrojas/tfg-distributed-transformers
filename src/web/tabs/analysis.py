"""Analysis tab — hyperparameters vs results across all runs.

Inspired by TensorBoard's HParams dashboard and MLflow's run comparison: a parallel
-coordinates plot (every run as a line through strategy / precision / model / epochs
→ best F1) and a scatter of the run population, so patterns across the whole study
read at a glance — not one run at a time.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.run_registry import RunInfo
from src.web.ui.charts import COLORS, _show, _dl_csv
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import _load_df, _safe_max, _run_config, _dur_str

_MODE_ORDER = ["single", "ddp", "ddp_hetero", "model_parallel"]


def _hp_frame(runs: list[RunInfo]) -> pd.DataFrame:
    rows = []
    for r in runs:
        df = _load_df(str(r.log_path), str(r.epoch_csv_path) if r.epoch_csv_path else None)
        best = _safe_max(df["val_f1"]) if ("val_f1" in df.columns and not df.empty) else np.nan
        cfg = _run_config(str(r.log_path))
        m = re.search(r"\d+", cfg.get("batch", ""))
        batch = int(m.group()) if m else np.nan
        try:
            lr = float(cfg.get("lr", "nan"))
        except ValueError:
            lr = np.nan
        secs = df["epoch_time"].dropna().sum() if "epoch_time" in df.columns else np.nan
        rows.append({
            "label": r.label, "model": r.model.replace("_patch16_224", "") or "—",
            "strategy": r.mode, "precision": r.precision or "fp32", "env": r.env,
            "epochs": len(df) if not df.empty else 0, "batch": batch, "lr": lr,
            "best_f1": best, "duration_s": secs,
        })
    return pd.DataFrame(rows)


def _parcoords(hp: pd.DataFrame) -> go.Figure:
    def cat(label, col, order=None):
        cats = order or sorted(hp[col].dropna().unique().tolist())
        cats = [c for c in cats if c in set(hp[col])]
        code = {c: i for i, c in enumerate(cats)}
        return dict(label=label, values=[code.get(v, 0) for v in hp[col]],
                    tickvals=list(range(len(cats))), ticktext=cats,
                    range=[0, max(len(cats) - 1, 1)])

    def num(label, col):
        v = pd.to_numeric(hp[col], errors="coerce")
        lo, hi = np.nanmin(v), np.nanmax(v)
        return dict(label=label, values=v.fillna(lo), range=[lo, hi if hi > lo else lo + 1])

    dims = [cat("Strategy", "strategy", _MODE_ORDER), cat("Precision", "precision"),
            cat("Model", "model"), num("Epochs", "epochs"), num("Best Val F1", "best_f1")]
    f1 = pd.to_numeric(hp["best_f1"], errors="coerce").fillna(0)
    fig = go.Figure(go.Parcoords(
        line=dict(color=f1, colorscale="Blues", showscale=True,
                  cmin=float(f1.min()), cmax=float(f1.max()),
                  colorbar=dict(title="Best F1", thickness=12)),
        dimensions=dims, labelfont=dict(size=12), tickfont=dict(size=10)))
    fig.update_layout(height=420, margin=dict(l=70, r=40, t=40, b=20))
    return fig


def _scatter(hp: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, model in enumerate(sorted(hp["model"].unique())):
        sub = hp[hp["model"] == model]
        fig.add_trace(go.Scatter(
            x=sub["epochs"], y=sub["best_f1"], mode="markers", name=model,
            marker=dict(size=11, color=COLORS[i % len(COLORS)], line=dict(width=1, color="white"),
                        opacity=0.85),
            text=sub["label"], customdata=sub[["strategy", "precision"]],
            hovertemplate="<b>%{text}</b><br>epochs %{x} · F1 %{y:.3f}"
                          "<br>%{customdata[0]} · %{customdata[1]}<extra></extra>"))
    fig.update_layout(height=380, xaxis_title="Epochs", yaxis_title="Best Val F1",
                      title=dict(text="Run population — best F1 vs training length"))
    fig.update_yaxes(range=[0, 1])
    return fig


def render(ctx: DashboardContext) -> None:
    st.markdown("## Analysis")
    st.caption("Hyperparameters vs results across every run — parallel coordinates and a "
               "scatter of the run population (TensorBoard HParams / MLflow style).")
    runs = ctx.runs
    if not runs:
        st.info("No runs available.")
        return

    hp = _hp_frame(runs)

    # ── Filters (subset the population, like TensorBoard) ─────────────────────────
    c1, c2 = st.columns(2)
    envs = c1.multiselect("Environment", sorted(hp["env"].unique()),
                          default=sorted(hp["env"].unique()))
    models = c2.multiselect("Model", sorted(hp["model"].unique()),
                            default=sorted(hp["model"].unique()))
    view = hp[hp["env"].isin(envs) & hp["model"].isin(models)]
    if view.empty:
        st.warning("No runs match the filters.")
        return

    st.markdown("#### Parallel coordinates")
    st.caption("Each line is a run threading strategy → precision → model → epochs → best F1. "
               "Drag along an axis to brush a range; the line colour is the best F1.")
    _show(_parcoords(view), "analysis_parcoords")

    st.markdown("---")
    left, right = st.columns([1.1, 1])
    with left:
        _show(_scatter(view), "analysis_scatter")
    with right:
        st.markdown("#### Hyperparameter table")
        tbl = (view[["label", "model", "strategy", "precision", "epochs", "batch", "lr", "best_f1"]]
               .sort_values("best_f1", ascending=False).reset_index(drop=True))
        st.dataframe(
            tbl.style.format({"best_f1": "{:.4f}", "lr": "{:.5f}", "batch": "{:.0f}"}),
            use_container_width=True, height=380)
        _dl_csv(tbl, "hparams.csv", "Download HParams CSV")
