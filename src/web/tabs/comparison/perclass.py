"""Compare — perclass."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.run_registry import RunInfo
from src.web.ui import theme
from src.web.ui.charts import (_show)
from src.web.ui.helpers import (_load_perclass)


def _perclass_heatmap_section(sel: list[tuple[str, RunInfo]]) -> None:
    """Heatmap of per-class F1 across ALL selected runs — one row per class, one
    column per run, colour = F1. Scales to N runs (the dumbbell is for 2) and
    makes the rare classes that stay near zero jump out across every model."""
    with_pc = [(lbl, r) for lbl, r in sel
               if r.perclass_csv_path and Path(r.perclass_csv_path).exists()]
    if len(with_pc) < 2:
        return
    st.caption("F1 of every class (rows) for each selected run (columns), at each "
               "run's last epoch. Red = the model barely detects that class, green = "
               "strong. The rare classes that stay red across runs are the F1-macro ceiling.")

    def _last_f1(r: RunInfo) -> pd.Series:
        df = _load_perclass(str(r.perclass_csv_path))
        df = df[df["epoch"] == df["epoch"].max()]
        return df.set_index("class_name")["f1"]

    def _short(lbl: str) -> str:
        return re.sub(r"^\d{2}/\d{2}/\d{4}\s+", "", lbl)

    series = {_short(lbl): _last_f1(r) for lbl, r in with_pc}
    mat = pd.DataFrame(series)
    if mat.empty:
        return
    # Worst classes at the bottom (sorted by mean F1) so the ceiling clusters there.
    mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]

    fig = go.Figure(go.Heatmap(
        z=mat.values, x=list(mat.columns), y=list(mat.index),
        colorscale=[[0.0, theme.BAD], [0.5, theme.WARN], [1.0, theme.GOOD]],
        zmin=0.0, zmax=1.0,
        text=[[f"{v:.2f}" for v in row] for row in mat.values],
        texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title="F1", thickness=12),
        hovertemplate="%{y}<br>%{x}<br>F1=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Per-class F1 across runs"),
        height=max(420, 24 * len(mat) + 140),
        margin=dict(l=240, b=120, t=44),
        xaxis=dict(tickangle=30, side="top"),
    )
    _show(fig, "compare_perclass_heatmap")
    zeros = (mat <= 0.01).all(axis=1)
    if zeros.any():
        names = ", ".join(mat.index[zeros])
        st.caption(f"**Never detected by any selected run** ({int(zeros.sum())}): {names}. "
                   "These are the classes a loss like focal aims to rescue.")

