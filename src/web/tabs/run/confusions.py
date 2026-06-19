"""Run results — confusions view."""
from __future__ import annotations


import plotly.graph_objects as go
import streamlit as st

from src.web.confusion_matrix_parser import (confusion_profile, get_matrix_for_epoch, parse_confusion_matrix_csv, recall_by_class, top_confusions)
from src.web.ui.charts import (COLORS, _CLASS_GROUPS, _base_layout, _dl_csv, _show)
from src.web.ui.context import DashboardContext


def _confusions_tab(ctx: DashboardContext) -> None:
    if ctx.selected_run is None:
        st.info("Select a run in the sidebar.")
        return
    _confusions_view(ctx.run)


def _confusions_view(run) -> None:
    """Multi-label confusion diagnostics.

    This is a multi-label task, so a classic N×N confusion matrix does not apply.
    Instead we show the three things that ARE interpretable from the stored
    co-activation matrix: recall per class (did the model catch the class), the
    strongest label confusions, and a per-class 'what else fires' profile. The
    full 19×19 matrix stays available as an advanced view.
    """
    if not (run.confusion_matrix_csv_path and run.confusion_matrix_csv_path.exists()):
        st.info("No confusion data. Use `--layers confusion` to generate it.")
        return

    cm_df = parse_confusion_matrix_csv(run.confusion_matrix_csv_path)
    epochs_cm = sorted(cm_df["epoch"].unique().tolist())
    ep = st.selectbox("Epoch", epochs_cm, index=len(epochs_cm) - 1,
                      format_func=lambda e: f"Epoch {e}", key="cm_epoch_sel")

    st.caption(
        "Multi-label task: each image can carry several of the 19 classes, so there "
        "is no single 'predicted class' to confuse. These views read the model's "
        "label co-activation: **recall** (whether each class is detected) and which "
        "other labels are predicted when a class is present (confusion / co-occurrence)."
    )

    # ── 1) Recall per class (the diagonal) ────────────────────────────────────
    st.markdown("#### Recall by class")
    rec = recall_by_class(cm_df, ep)
    if not rec.empty:
        bar_colors = [
            COLORS[3] if v < 0.3 else (COLORS[1] if v < 0.6 else COLORS[2])
            for v in rec.values
        ]
        fig_rec = go.Figure(go.Bar(
            y=list(rec.index), x=list(rec.values), orientation="h",
            marker_color=bar_colors, text=[f"{v:.2f}" for v in rec.values],
            textposition="outside",
        ))
        fig_rec.update_layout(
            **_base_layout(120 + 26 * len(rec), "Recall by class (red < 0.30, amber < 0.60, green ≥ 0.60)",
                           margin=dict(l=200, r=40, t=48, b=40)),
            showlegend=False,
        )
        fig_rec.update_xaxes(range=[0, 1], title="Recall")
        fig_rec.update_yaxes(automargin=True)
        _show(fig_rec, f"recall_by_class_ep{ep}")
        failed = rec[rec < 0.3]
        if not failed.empty:
            st.warning(
                f"**{len(failed)} class(es) the model rarely catches** (recall < 0.30): "
                + ", ".join(f"{c} ({v:.2f})" for c, v in failed.items())
                + ". Typically rare classes, which lower the macro-F1."
            )

    st.markdown("---")

    # ── 2) Strongest confusions (off-diagonal) ────────────────────────────────
    st.markdown("#### Label confusions")
    st.caption("When the class on the left is truly present, the model also predicts "
               "the class on the right this often. Some pairs are real confusion, "
               "others are natural co-occurrence (e.g. forest types share scenes).")
    top = top_confusions(cm_df, ep, k=12)
    if top.empty:
        st.info("No strong off-diagonal confusions at this epoch.")
    else:
        pair_labels = [f"{r.true_class}  →  {r.pred_class}" for r in top.itertuples()]
        fig_top = go.Figure(go.Bar(
            y=pair_labels, x=list(top["value"]), orientation="h",
            marker_color=COLORS[0], text=[f"{v:.2f}" for v in top["value"]],
            textposition="outside",
        ))
        fig_top.update_layout(
            **_base_layout(120 + 28 * len(top), "Top confusions (true → also predicted)",
                           margin=dict(l=300, r=40, t=48, b=40)),
            showlegend=False,
        )
        fig_top.update_xaxes(range=[0, 1], title="P(also predicts the right label | left is present)")
        fig_top.update_yaxes(automargin=True, autorange="reversed")
        _show(fig_top, f"top_confusions_ep{ep}")
        _dl_csv(top, f"top_confusions_ep{ep}.csv", "Download confusions table")

    st.markdown("---")

    # ── 3) Per-class confusion profile ────────────────────────────────────────
    st.markdown("#### Per-class profile")
    all_classes = sorted(cm_df["true_class"].unique().tolist())
    sel_cls = st.selectbox("When this class is truly present…", all_classes, key="cm_profile_cls")
    prof = confusion_profile(cm_df, ep, sel_cls).head(10)
    diag = recall_by_class(cm_df, ep).get(sel_cls, float("nan"))
    st.caption(f"Recall of **{sel_cls}**: {diag:.2f} — detected in {diag*100:.0f}% of "
               f"cases. Below: the other labels also predicted when it is present.")
    if not prof.empty and prof.max() > 0:
        fig_prof = go.Figure(go.Bar(
            y=list(prof.index), x=list(prof.values), orientation="h",
            marker_color=COLORS[5], text=[f"{v:.2f}" for v in prof.values],
            textposition="outside",
        ))
        fig_prof.update_layout(
            **_base_layout(120 + 26 * len(prof), f"Also predicted when '{sel_cls}' is present",
                           margin=dict(l=200, r=40, t=48, b=40)),
            showlegend=False,
        )
        fig_prof.update_xaxes(range=[0, 1], title="Frequency")
        fig_prof.update_yaxes(automargin=True, autorange="reversed")
        _show(fig_prof, f"profile_{ep}")
    else:
        st.info("The model rarely turns on other labels for this class.")

    # ── 4) Full matrix (advanced) ─────────────────────────────────────────────
    with st.expander("Full 19×19 co-activation matrix (advanced)"):
        st.caption("Cell (row i, column j) = P(model predicts j | class i is truly "
                   "present). The diagonal is recall; bright off-diagonal cells are "
                   "the confusions above. Colored borders group classes by ecosystem.")
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
            height=660, margin=dict(l=180, r=20, t=50, b=180),
            paper_bgcolor="white", shapes=shapes,
        )
        _show(fig_cm, f"confusion_matrix_ep{ep}")



