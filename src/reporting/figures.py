"""Plotly figures of the report. Each function returns a ``go.Figure``; ``to_div`` turns it into
an HTML fragment (no JavaScript bundle: the report embeds plotly.js once)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go

# Okabe-Ito (colour-blind safe); black is reserved for reference lines.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442",
           "#7A7A7A"]
INK, GRID = "#1f2933", "#e3e7ea"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
MAX_POINTS = 1500
CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["sendDataToCloud", "lasso2d", "select2d"]}


def _layout(fig: go.Figure, title: str | None = None, x: str | None = None,
            y: str | None = None, height: int = 340) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 14}} if title else None,
        font={"family": FONT, "size": 12, "color": INK}, height=height,
        margin={"l": 56, "r": 16, "t": 40 if title else 16, "b": 44},
        paper_bgcolor="white", plot_bgcolor="white", colorway=PALETTE,
        legend={"orientation": "h", "y": -0.22, "font": {"size": 11}},
        hoverlabel={"font": {"family": FONT}})
    fig.update_xaxes(title=x, gridcolor=GRID, linecolor="#9aa5b1", zeroline=False)
    fig.update_yaxes(title=y, gridcolor=GRID, linecolor="#9aa5b1", zeroline=False)
    return fig


def to_div(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
                       default_width="100%", config=CONFIG)


def thin(n: int, limit: int = MAX_POINTS) -> slice | np.ndarray:
    """Indices of at most ``limit`` evenly spread points (ends included)."""
    if n <= limit:
        return slice(None)
    return np.unique(np.linspace(0, n - 1, limit).round().astype(int))


def _col(rows: list[dict[str, Any]], key: str) -> list[float]:
    out = []
    for r in rows:
        v = r.get(key)
        ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        out.append(float(v) if ok else float("nan"))
    return out


def has(rows: list[dict[str, Any]], key: str) -> bool:
    return any(not math.isnan(v) for v in _col(rows, key))


# ── Training ───────────────────────────────────────────────────────────────────

def loss_fig(rows: list[dict[str, Any]]) -> go.Figure:
    ep = [r["epoch"] for r in rows]
    fig = go.Figure()
    for key, name in (("train_loss", "train"), ("val_loss", "validation")):
        if has(rows, key):
            fig.add_scatter(x=ep, y=_col(rows, key), mode="lines+markers", name=name)
    return _layout(fig, "Loss per epoch", "epoch", "loss")


def metrics_fig(rows: list[dict[str, Any]]) -> go.Figure:
    ep = [r["epoch"] for r in rows]
    fig = go.Figure()
    for key, name, dash in (("val_auc", "val AUC", "solid"), ("val_f1", "val F1", "solid"),
                            ("val_accuracy", "val accuracy", "solid"),
                            ("val_f1_at_optimal_threshold", "val F1 @ optimal threshold", "dot"),
                            ("train_f1", "train F1", "dash")):
        if has(rows, key):
            fig.add_scatter(x=ep, y=_col(rows, key), mode="lines+markers", name=name,
                            line={"dash": dash})
    return _layout(fig, "Metrics per epoch", "epoch", "value")


def throughput_fig(rows: list[dict[str, Any]], batches: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    xs, ys = [], []
    for ev in batches:
        m, n = ev.get("metrics") or {}, ev.get("n_batches")
        v, e, b = m.get("samples_per_s"), ev.get("epoch"), ev.get("batch")
        if all(isinstance(t, (int, float)) for t in (v, e, b, n)) and n:
            xs.append((e - 1) + b / n)
            ys.append(float(v))
    if xs:
        idx = np.arange(len(xs))[thin(len(xs))]
        fig.add_scatter(x=[xs[i] for i in idx], y=[ys[i] for i in idx], mode="lines",
                        name="per batch", line={"width": 1, "color": "#9aa5b1"})
    if has(rows, "train_samples_per_s"):
        # epoch mean is plotted at the end of its epoch
        fig.add_scatter(x=[r["epoch"] for r in rows], y=_col(rows, "train_samples_per_s"),
                        mode="lines+markers", name="epoch mean", line={"color": PALETTE[0]})
    return _layout(fig, "Training throughput", "epoch", "samples / s")


def wait_fig(rows: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[r["epoch"] for r in rows],
                    y=[100 * v for v in _col(rows, "train_data_wait_frac")],
                    mode="lines+markers", name="DataLoader wait", line={"color": PALETTE[1]})
    fig.add_hline(y=30, line={"dash": "dot", "color": INK, "width": 1},
                  annotation_text="30 %: I/O-bound", annotation_position="top left")
    return _layout(fig, "Time waiting for the DataLoader", "epoch", "% of training time")


def gpu_figs(gpu_df) -> list[go.Figure]:
    """Utilisation, memory and power over time, one line per GPU."""
    figs = []
    for col, title, unit in (("util_pct", "GPU utilisation", "%"),
                             ("mem_pct", "GPU memory in use", "% of total"),
                             ("power_w", "GPU power", "W")):
        fig = go.Figure()
        for gpu, df in gpu_df.groupby("gpu"):
            df = df.dropna(subset=[col]).sort_values("t_s")
            if df.empty:
                continue
            idx = np.arange(len(df))[thin(len(df))]
            label = f"GPU {gpu}"
            fig.add_scatter(x=(df["t_s"].to_numpy()[idx] / 60.0), y=df[col].to_numpy()[idx],
                            mode="lines", name=label, line={"width": 1.5})
        if fig.data:
            figs.append(_layout(fig, title, "minutes since first sample", unit, height=280))
    return figs


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _curve_fig(curves: dict[str, Any], classes: list[str], kind: str) -> go.Figure:
    fig = go.Figure()
    for c, name in enumerate(classes):
        cur = curves["classes"][c][kind]
        x, y = (cur["fpr"], cur["tpr"]) if kind == "roc" else (cur["recall"], cur["precision"])
        if len(x) == 0:
            continue
        score = cur["auc"] if kind == "roc" else cur["ap"]
        fig.add_scatter(x=x, y=y, mode="lines", line={"width": 1.4, "shape": "hv" if kind == "pr"
                                                      else "linear"},
                        name=f"{name} ({score:.3f})")
    micro = curves["micro"][kind]
    mx, my = (micro["fpr"], micro["tpr"]) if kind == "roc" else (micro["recall"],
                                                                 micro["precision"])
    if len(mx) and len(curves["classes"]) > 1:
        score = micro["auc"] if kind == "roc" else micro["ap"]
        fig.add_scatter(x=mx, y=my, mode="lines", name=f"micro ({score:.3f})",
                        line={"color": INK, "width": 2, "dash": "dash"})
    if kind == "roc":
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False, hoverinfo="skip",
                        line={"color": "#9aa5b1", "dash": "dot", "width": 1})
        _layout(fig, "ROC (one-vs-rest)", "false positive rate", "true positive rate", 380)
    else:
        _layout(fig, "Precision-recall (one-vs-rest)", "recall", "precision", 380)
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return fig


def roc_fig(curves, classes):
    return _curve_fig(curves, classes, "roc")


def pr_fig(curves, classes):
    return _curve_fig(curves, classes, "pr")


def reliability_fig(cal: dict[str, Any]) -> go.Figure:
    bins = [b for b in cal["bins"] if b["count"] > 0]
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", hoverinfo="skip",
                    line={"color": "#9aa5b1", "dash": "dot", "width": 1})
    fig.add_scatter(x=[b["confidence"] for b in bins], y=[b["accuracy"] for b in bins],
                    mode="lines+markers", name="observed", customdata=[b["count"] for b in bins],
                    hovertemplate="confidence %{x:.3f}<br>accuracy %{y:.3f}<br>n=%{customdata}"
                                  "<extra></extra>", line={"color": PALETTE[0]})
    _layout(fig, "Reliability diagram", "mean confidence", "observed accuracy", 380)
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return fig


def confidence_hist_fig(cal: dict[str, Any]) -> go.Figure:
    fig = go.Figure(go.Bar(x=[(b["lower"] + b["upper"]) / 2 for b in cal["bins"]],
                           y=[b["count"] for b in cal["bins"]], marker_color=PALETTE[5]))
    return _layout(fig, "Confidence histogram", "confidence", "predictions", 380)


def confusion_fig(view: dict[str, Any]) -> go.Figure:
    m = np.asarray(view["matrix"])
    text = [[str(int(v)) for v in row] for row in m]
    fig = go.Figure(go.Heatmap(z=m, x=view["cols"], y=view["rows"], text=text,
                               texttemplate="%{text}" if m.size <= 400 else None,
                               colorscale=[[0, "#ffffff"], [1, "#0072B2"]], showscale=False,
                               hovertemplate="true %{y}<br>predicted %{x}<br>n=%{z}"
                                             "<extra></extra>"))
    side = max(340, min(760, 80 + 46 * len(view["rows"])))
    _layout(fig, None, view["col_title"], view["row_title"], side)
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(type="category", side="bottom")
    return fig


# ── Dynamics ───────────────────────────────────────────────────────────────────

REGION_COLOURS = {"easy": "#009E73", "ambiguous": "#E69F00", "hard": "#D55E00"}


def cartography_fig(carto: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    region = np.asarray(carto["region"])
    for name, colour in REGION_COLOURS.items():
        sel = region == name
        fig.add_scatter(x=carto["variability"][sel], y=carto["confidence"][sel], mode="markers",
                        name=f"{name} ({int(sel.sum())})",
                        marker={"size": 6, "color": colour, "opacity": 0.75})
    _layout(fig, "Data cartography (Swayamdipta et al., 2020)", "variability (std of p(true))",
            "confidence (mean p(true))", 380)
    return fig


def embedding_fig(xy: np.ndarray, labels: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    lab = np.asarray(labels)
    for i, name in enumerate(sorted(set(labels))):
        sel = lab == name
        fig.add_scatter(x=xy[sel, 0], y=xy[sel, 1], mode="markers", name=name,
                        marker={"size": 6, "opacity": 0.8, "color": PALETTE[i % len(PALETTE)]})
    return _layout(fig, title, "PC 1", "PC 2", 400)
