"""Run results — confusions view."""
from __future__ import annotations


import plotly.graph_objects as go
import streamlit as st

from src.web.confusion_matrix_parser import (get_matrix_for_epoch,
                                             parse_confusion_matrix_csv, top_confusions,
                                             top_confusions_by_lift)
from src.web.dataset_stats import CLASS_NAMES
from src.web.ui import theme
from src.web.ui.charts import (COLORS, _CLASS_GROUPS, _base_layout, _dl_csv, _show)
from src.web.ui.context import DashboardContext


def _group_of_name(class_name: str):
    """Ecosystem group of a class name (via its index in CLASS_NAMES), or None."""
    idx = CLASS_NAMES.index(class_name) if class_name in CLASS_NAMES else -1
    for gname, (idxs, _color) in _CLASS_GROUPS.items():
        if idx in idxs:
            return gname
    return None


def _same_ecosystem(a: str, b: str) -> bool:
    """True if both classes belong to the same land-cover group → a confusion
    between them is expected co-occurrence rather than a likely real error."""
    ga = _group_of_name(a)
    return ga is not None and ga == _group_of_name(b)


def _confusions_tab(ctx: DashboardContext) -> None:
    if ctx.selected_run is None:
        st.info("Select a run in the sidebar.")
        return
    _confusions_view(ctx.run)


def _confusions_view(run) -> None:
    """Which classes the model conflates / predicts together.

    Multi-label task → there is no classic confusion matrix (no single predicted
    class). We read the stored co-activation matrix and show only the off-diagonal:
    when a class is truly present, which OTHER labels the model also predicts.
    Per-class quality (including recall, the diagonal) lives in the Per-class tab,
    so this view stays focused on confusions alone.
    """
    if not (run.confusion_matrix_csv_path and run.confusion_matrix_csv_path.exists()):
        st.info("No confusion data. Use `--layers confusion` to generate it.")
        return

    cm_df = parse_confusion_matrix_csv(run.confusion_matrix_csv_path)
    epochs_cm = sorted(cm_df["epoch"].unique().tolist())
    ep = st.selectbox("Epoch", epochs_cm, index=len(epochs_cm) - 1,
                      format_func=lambda e: f"Epoch {e}", key="cm_epoch_sel")

    st.caption(
        "Multi-label task: an image carries several of the 19 classes, so there is no "
        "single 'predicted class' to confuse. This reads label **co-activation** — when "
        "a class is truly present, which other labels the model also predicts. Some pairs "
        "are genuine confusion, others natural co-occurrence (e.g. forest types share a "
        "scene). Per-class recall lives in the **Per-class** tab."
    )

    # ── Top label confusions (the off-diagonal of the co-activation matrix) ───────
    st.markdown("#### Top label confusions")
    cc1, cc2 = st.columns([2, 2])
    n_pairs = cc1.slider("How many pairs to show", 5, 30, 12, key="cm_topn")
    rank_by = cc2.radio("Rank by", ["Real confusion (lift)", "Co-prediction (raw)"],
                        horizontal=True, key="cm_rank")
    use_lift = rank_by.startswith("Real")
    st.caption("**Lift** divides each frequency by how often that label is predicted "
               "overall, so it surfaces real confusions and drops the noise of a "
               "common class (e.g. *Arable land*) being predicted everywhere. "
               "*Raw* shows the plain frequency.")
    top = (top_confusions_by_lift(cm_df, ep, k=n_pairs) if use_lift
           else top_confusions(cm_df, ep, k=n_pairs))
    if top.empty:
        st.info("No strong off-diagonal confusions at this epoch.")
    else:
        same_col, cross_col = COLORS[0], theme.WARN
        # Colour each pair by whether the two classes share an ecosystem: same group
        # = expected co-occurrence; different group = more likely a real confusion.
        bar_colors = [same_col if _same_ecosystem(r.true_class, r.pred_class) else cross_col
                      for r in top.itertuples()]
        pair_labels = [f"{r.true_class}  →  {r.pred_class}" for r in top.itertuples()]
        if use_lift:
            xvals, texts = list(top["lift"]), [f"{v:.1f}×" for v in top["lift"]]
            cdata = list(top["value"])
            hover = "%{y}<br>lift %{x:.2f}× · P=%{customdata:.2f}<extra></extra>"
        else:
            xvals, texts = list(top["value"]), [f"{v:.2f}" for v in top["value"]]
            cdata = None
            hover = "%{y}<br>P=%{x:.2f}<extra></extra>"
        fig_top = go.Figure(go.Bar(
            y=pair_labels, x=xvals, orientation="h", marker_color=bar_colors,
            text=texts, textposition="outside", customdata=cdata, hovertemplate=hover,
        ))
        fig_top.update_layout(
            **_base_layout(120 + 28 * len(top),
                           "When the left class is present, the model also predicts the right one",
                           margin=dict(l=300, r=40, t=48, b=40)),
            showlegend=False,
        )
        if use_lift:
            fig_top.update_xaxes(
                title="Lift (× vs base rate — &gt;1 = real confusion, not just a common class)")
            fig_top.add_vline(x=1.0, line=dict(color="gray", width=1, dash="dot"))
        else:
            fig_top.update_xaxes(range=[0, 1],
                                 title="P(also predicts right label | left is present)")
        fig_top.update_yaxes(automargin=True, autorange="reversed")
        _show(fig_top, f"top_confusions_ep{ep}")
        st.markdown(
            f"<span style='font-size:0.8rem'>"
            f"<span style='color:{same_col}'>■</span> same ecosystem (expected co-occurrence)"
            f" &nbsp;·&nbsp; "
            f"<span style='color:{cross_col}'>■</span> different ecosystem (more likely a real confusion)"
            f"</span>", unsafe_allow_html=True)
        _dl_csv(top, f"top_confusions_ep{ep}.csv", "Download confusions table")

    # ── Full matrix (advanced) ────────────────────────────────────────────────────
    with st.expander("Full 19×19 co-activation matrix (advanced)"):
        st.caption("Cell (row i, column j) = P(model predicts j | class i is truly "
                   "present). The diagonal is recall; bright off-diagonal cells are "
                   "the confusions above. Coloured borders group classes by ecosystem.")
        pivot = get_matrix_for_epoch(cm_df, ep)
        class_order = list(pivot.index)
        z_norm = pivot.reindex(index=class_order, columns=class_order).values
        n_classes = len(class_order)
        text = [[f"{v:.2f}" if v >= 0.05 else "" for v in row] for row in z_norm]

        shapes = []
        for _gname, (idxs, color) in _CLASS_GROUPS.items():
            positions = [i for i in range(n_classes) if i in idxs]
            if not positions:
                continue
            lo, hi = min(positions), max(positions)
            shapes.append(dict(
                type="rect", x0=lo - 0.5, x1=hi + 0.5, y0=lo - 0.5, y1=hi + 0.5,
                line=dict(color=color, width=2.5), fillcolor="rgba(0,0,0,0)", layer="above",
            ))

        fig_cm = go.Figure(go.Heatmap(
            z=z_norm.tolist(), x=class_order, y=class_order,
            colorscale="Blues", zmin=0, zmax=1,
            text=text, texttemplate="%{text}", textfont={"size": 8},
            hovertemplate="True: %{y}<br>Also predicts: %{x}<br>P = %{z:.3f}<extra></extra>",
            colorbar=dict(title="P(pred j | true i)"),
        ))
        fig_cm.update_layout(
            title=dict(text=f"Co-activation matrix — Epoch {ep}", font=dict(size=13)),
            xaxis=dict(title="Also predicted", tickangle=45, tickfont=dict(size=9),
                       tickmode="array", tickvals=list(range(n_classes)), ticktext=class_order),
            yaxis=dict(title="True class", tickfont=dict(size=9), autorange="reversed",
                       tickmode="array", tickvals=list(range(n_classes)), ticktext=class_order),
            height=660, margin=dict(l=180, r=20, t=50, b=180), shapes=shapes,
        )
        _show(fig_cm, f"confusion_matrix_ep{ep}")
