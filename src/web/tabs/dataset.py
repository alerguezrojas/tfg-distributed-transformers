"""Dataset section — the BigEarthNet-S2 data story + the run-import utility.

Promoted out of the Overview (which had grown too dense) into its own section,
and it absorbs run import: bringing in artifacts is a data operation, not a
primary navigation destination, so it no longer deserves a top-level nav slot.
"""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.web.dataset_stats import SPLIT_SIZES, class_distribution_approximate
from src.web.tabs import data_models
from src.web.ui import theme
from src.web.ui.charts import _show
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import _run_config, _load_class_distribution, _class_gallery

_META_CANDIDATES = [
    "/media/alejandro/SSD/datasets/bigearthnet/metadata.parquet",
    "/home/bejeque/alu0101317038/datasets/bigearthnet/metadata.parquet",
]
_ROOT_CANDIDATES = [
    "/media/alejandro/SSD/datasets/bigearthnet/BigEarthNet-S2",
    "/home/bejeque/alu0101317038/datasets/bigearthnet/BigEarthNet-S2",
]


def render(ctx: DashboardContext) -> None:
    st.markdown("## Dataset")
    st.caption("BigEarthNet-S2 — the data behind every run.")
    _dataset_story(ctx.runs)
    st.markdown("---")
    data_models.render(ctx)          # run-import utility (keeps its own header)


def _dataset_story(runs) -> None:
    meta = next((p for p in _META_CANDIDATES if Path(p).exists()), None)
    root = next((p for p in _ROOT_CANDIDATES if Path(p).exists()), None)

    # Subset actually used in the most recent runs (from their config line).
    subset = ""
    for r in runs[:5]:
        cfg = _run_config(str(r.log_path))
        if cfg.get("train") and cfg.get("val"):
            try:
                tr, va = int(cfg["train"]), int(cfg["val"])
            except ValueError:
                continue
            full = tr >= SPLIT_SIZES["train"] * 0.9
            subset = (f"Latest runs use the **full** set ({tr:,}/{va:,})" if full
                      else f"Latest runs use a **subset**: {tr:,} train / {va:,} val "
                           f"({tr / SPLIT_SIZES['train'] * 100:.1f}% of train)")
            break

    left, right = st.columns([1, 1.3])
    with left:
        with st.container(border=True):
            s1, s2, s3 = st.columns(3)
            s1.metric("Train", f"{SPLIT_SIZES['train']:,}")
            s2.metric("Val", f"{SPLIT_SIZES['val']:,}")
            s3.metric("Test", f"{SPLIT_SIZES['test']:,}")
            st.caption(f"{sum(SPLIT_SIZES.values()):,} patches · 19 CORINE classes · "
                       "multi-label · RGB proxy (B04/B03/B02)")
            if subset:
                st.caption(subset)
    with right:
        with st.container(border=True):
            _imbalance_treemap(meta)

    # ── Gallery: one example patch per class, with multi-label info ─────────────
    avg_labels, gallery = _class_gallery(str(meta), str(root)) if (meta and root) else (0.0, [])
    if gallery:
        st.markdown("#### The 19 classes")
        st.caption(
            f"One example patch per class. Each patch is **multi-label** "
            f"(~{avg_labels:.1f} classes per patch on average): the caption shows the "
            f"class shown, how many other labels the patch also has (**+k**), and that "
            f"class's train count and share. Hover an image for its full label list."
        )
        ncols = 10
        for i in range(0, len(gallery), ncols):
            cols = st.columns(ncols, gap="small")
            for col, (cls, cnt, pct, img, labels) in zip(cols, gallery[i:i + ncols]):
                col.image(img, use_container_width=True)
                others = max(len(labels) - 1, 0)
                extra = f" <span style='color:#64748b'>+{others}</span>" if others else ""
                title = " · ".join(labels).replace("'", "")
                col.markdown(
                    f"<div title='{title}' style='font-size:0.66rem;line-height:1.1;"
                    f"height:3.0rem;overflow:hidden'><b>{cls}</b>{extra}<br>"
                    f"{cnt:,} · {pct:.0f}%</div>",
                    unsafe_allow_html=True)
    elif not (meta and root):
        st.caption("Dataset not mounted on this machine — splits and class counts "
                   "shown from metadata.")


def _imbalance_treemap(meta: str | None) -> None:
    """Treemap of train-set class frequency: tile area = #patches, so the class
    imbalance (a few huge classes, many tiny ones) is visible at a glance — the
    imbalance that caps macro-F1 and motivates focal loss / pos_weight."""
    dist = _load_class_distribution(str(meta)) if meta else None
    if dist is None:
        dist = class_distribution_approximate()
    dist = dist.sort_values("train_count", ascending=False)
    st.caption("Class frequency in the train split — area = number of patches")
    fig = go.Figure(go.Treemap(
        labels=dist["class"], parents=[""] * len(dist), values=dist["train_count"],
        marker=dict(colors=dist["train_count"], colorscale="Blues", showscale=False,
                    line=dict(width=1, color="white")),
        texttemplate="%{label}<br>%{value:,}", textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>%{value:,} patches<br>%{percentRoot} of train<extra></extra>",
        tiling=dict(pad=2),
    ))
    fig.update_layout(height=300, margin=dict(t=4, l=0, r=0, b=0))
    _show(fig, "dataset_imbalance_treemap")
