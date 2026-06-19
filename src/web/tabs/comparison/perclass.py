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
    st.markdown("### Per-class F1 heatmap")
    st.caption("F1 of every class (rows) for each selected run (columns), at each "
               "run's last epoch. Red = the model barely detects that class, green = "
               "strong. The rare classes that stay red across runs are the F1-macro ceiling.")

    def _last_f1(r: RunInfo) -> pd.Series:
        df = _load_perclass(str(r.perclass_csv_path))
        df = df[df["epoch"] == df["epoch"].max()]
        return df.set_index("class_name")["f1"]

    def _short(lbl: str) -> str:
        import re
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


def _perclass_compare_section(sel: list[tuple[str, RunInfo]]) -> None:
    """Dumbbell of per-class F1 for two runs — shows exactly which classes one run
    rescues or loses vs another (e.g. focal vs BCE on the rare classes)."""
    with_pc = [(lbl, r) for lbl, r in sel
               if r.perclass_csv_path and Path(r.perclass_csv_path).exists()]
    if len(with_pc) < 2:
        return  # needs two runs with per-class data
    st.markdown("### Per-class comparison")
    st.caption("F1 per class for two runs, sorted by the change. The connector is "
               "green where B beats A and red where it loses — the quickest way to "
               "see which classes a loss like focal rescues or sacrifices.")
    labels = [lbl for lbl, _ in with_pc]
    by_lbl = {lbl: r for lbl, r in with_pc}
    c1, c2 = st.columns(2)
    a_lbl = c1.selectbox("Run A", labels, index=0, key="pc_cmp_a")
    b_lbl = c2.selectbox("Run B", labels, index=min(1, len(labels) - 1), key="pc_cmp_b")
    if a_lbl == b_lbl:
        st.caption("Pick two different runs.")
        return

    def _last_f1(r: RunInfo) -> pd.Series:
        df = _load_perclass(str(r.perclass_csv_path))
        df = df[df["epoch"] == df["epoch"].max()]
        return df.set_index("class_name")["f1"]

    fa, fb = _last_f1(by_lbl[a_lbl]), _last_f1(by_lbl[b_lbl])
    classes = [c for c in fa.index if c in fb.index]
    data = pd.DataFrame({"class": classes,
                         "A": [fa[c] for c in classes],
                         "B": [fb[c] for c in classes]})
    data["delta"] = data["B"] - data["A"]
    data = data.sort_values("delta", ascending=True)   # biggest improvement on top

    def _short(lbl: str) -> str:
        import re
        return re.sub(r"^\d{2}/\d{2}/\d{4}\s+", "", lbl)

    fig = go.Figure()
    # Connectors, grouped by direction so the legend stays clean.
    for sign, color, name in ((1, theme.GOOD, "B better"), (-1, theme.BAD, "B worse")):
        xs: list = []
        ys: list = []
        for _, r in data.iterrows():
            if (r["delta"] > 0.01) == (sign > 0) and abs(r["delta"]) > 0.01:
                xs += [r["A"], r["B"], None]
                ys += [r["class"], r["class"], None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                     line=dict(color=color, width=2.5),
                                     name=name, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=data["A"], y=data["class"], mode="markers",
                             name=f"A · {_short(a_lbl)}",
                             marker=dict(color="#94A3B8", size=10)))
    fig.add_trace(go.Scatter(x=data["B"], y=data["class"], mode="markers",
                             name=f"B · {_short(b_lbl)}",
                             marker=dict(color=theme.ACCENT, size=10)))
    n_better = int((data["delta"] > 0.01).sum())
    n_worse = int((data["delta"] < -0.01).sum())
    fig.update_layout(
        title=dict(text="Per-class F1 — A vs B"),
        xaxis_title="F1", xaxis=dict(range=[0, 1.02]),
        height=max(380, 26 * len(classes) + 120),
        margin=dict(l=210, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0),
    )
    _show(fig, "compare_perclass_dumbbell")
    st.caption(f"**B improves {n_better} class(es)** and loses {n_worse} vs A "
               f"(macro F1: A={data['A'].mean():.3f}, B={data['B'].mean():.3f}).")


# ── Speedup vs baseline ─────────────────────────────────────────────────────────

