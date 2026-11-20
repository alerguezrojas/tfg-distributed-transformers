"""Dynamics sub-view: data cartography (Swayamdipta et al., 2020) and flips between epochs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis.dynamics import cartography, flips, prob_of_true, sample_correct
from src.web.tabs.inspector._common import InspectorCtx, card_grid, need_monitor, sample_cards
from src.web.tabs.inspector.predictions import epoch_table
from src.web.ui import theme
from src.web.ui.charts import _show

_REGION_COLORS = {"easy": theme.GOOD, "ambiguous": theme.WARN, "hard": theme.BAD}
_MAX_REGION_IMAGES = 24


def render(c: InspectorCtx) -> None:
    if not need_monitor(c):
        return
    eps, probs = c.stack("probs")
    if len(eps) < 2:
        st.info("Cartography needs at least two inspected epochs (it measures how each sample "
                "behaves *during* training). Keep `monitor.every` low and train a few epochs.")
        return
    targets = c.monitor.targets
    d = c.epoch_data()
    thr = d.threshold if d.threshold is not None else 0.5
    pt = np.stack([prob_of_true(p, targets, c.task) for p in probs])
    ok = np.stack([sample_correct(p, targets, c.task, thr) for p in probs])
    carto = cartography(pt, ok)
    table, _ = epoch_table(c)
    table = table.assign(confidence_mean=carto["confidence"], variability=carto["variability"],
                         correctness=carto["correctness"], region=carto["region"])

    st.caption(f"Epochs 1–{eps[-1]} ({len(eps)} inspected). **Confidence** = mean probability of the "
               "true label across epochs; **variability** = its standard deviation. "
               "*Hard*: low confidence, stable (often mislabelled or ambiguous); *easy*: high "
               "confidence, stable; *ambiguous*: the most variable third.")
    colour = st.radio("Colour by", ["Region", "Correctness"], horizontal=True, key="insp_dyn_color")
    fig = go.Figure()
    if colour == "Region":
        for region, col in _REGION_COLORS.items():
            _scatter(fig, table[table["region"] == region], region, col)
    else:
        fig.add_trace(_scatter_trace(table, "correctness"))
    fig.update_layout(height=440, xaxis_title="Variability", yaxis_title="Confidence",
                      title={"text": "Data cartography"}, yaxis_range=[-0.02, 1.02])
    _show(fig, "insp_cartography")

    counts = table["region"].value_counts()
    k = st.columns(3)
    for col, region in zip(k, _REGION_COLORS):
        col.metric(region.capitalize(), int(counts.get(region, 0)))

    region = st.radio("Show samples", ["hard", "ambiguous", "easy"], horizontal=True,
                      key="insp_dyn_region")
    part = table[table["region"] == region].sort_values("confidence_mean")
    if region == "hard":
        st.caption("Candidates for a wrong label: consistently low probability for the given "
                   "label. Worth checking by hand.")
    if part.empty:
        st.caption("No sample in this region.")
    else:
        st.caption(f"{len(part)} sample(s), lowest confidence first"
                   + (f" (showing {_MAX_REGION_IMAGES})" if len(part) > _MAX_REGION_IMAGES else ""))
        card_grid(sample_cards(c, part.head(_MAX_REGION_IMAGES)))

    st.markdown("---")
    rows = []
    for a, b, pa, pb in zip(eps[:-1], eps[1:], probs[:-1], probs[1:]):
        f = flips(pa, pb, targets, c.task, thr)
        rows.append({"epoch": b, "right → wrong": len(f["to_wrong"]),
                     "wrong → right": len(f["to_right"])})
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["epoch"], y=df["right → wrong"], name="right → wrong",
                         marker_color=theme.BAD))
    fig.add_trace(go.Bar(x=df["epoch"], y=df["wrong → right"], name="wrong → right",
                         marker_color=theme.GOOD))
    fig.update_layout(height=300, barmode="group", title={"text": "Flips vs the previous epoch"},
                      xaxis_title="epoch", yaxis_title="samples")
    fig.update_xaxes(dtick=1)
    _show(fig, "insp_flips")


def _hover(df: pd.DataFrame) -> list[str]:
    return [f"#{r.pos}<br>true: {r.true}<br>pred: {r.pred}<br>confidence {r.confidence_mean:.2f}"
            f"<br>variability {r.variability:.2f}<br>correct in {100 * r.correctness:.0f}% of epochs"
            for r in df.itertuples()]


def _scatter(fig: go.Figure, df: pd.DataFrame, name: str, color: str) -> None:
    fig.add_trace(go.Scatter(x=df["variability"], y=df["confidence_mean"], mode="markers",
                             name=name, marker={"color": color, "size": 8, "opacity": 0.8},
                             text=_hover(df), hoverinfo="text"))


def _scatter_trace(df: pd.DataFrame, colour_col: str) -> go.Scatter:
    return go.Scatter(x=df["variability"], y=df["confidence_mean"], mode="markers",
                      marker={"color": df[colour_col], "size": 8, "opacity": 0.85,
                              "colorscale": "RdYlGn", "cmin": 0, "cmax": 1,
                              "colorbar": {"title": "Correct<br>epochs"}},
                      text=_hover(df), hoverinfo="text", showlegend=False)
