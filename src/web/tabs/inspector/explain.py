"""Explain sub-view: Grad-CAM panel and its evolution across epochs."""
from __future__ import annotations

import numpy as np
import streamlit as st

from src.web.inspector_data import cam_overlay, outcome_card, thin_epochs
from src.web.tabs.inspector._common import InspectorCtx, card_grid, need_monitor, thumb_of
from src.web.tabs.inspector.predictions import epoch_table

_STRIP = 8          # epochs shown in the evolution strip
_SIZE = 160


def _cls(c: InspectorCtx, d, k: int) -> str:
    if d.cam_class is None or int(d.cam_class[k]) >= c.n_classes:
        return "?"
    return c.classes[int(d.cam_class[k])]


def render(c: InspectorCtx) -> None:
    if not need_monitor(c):
        return
    d = c.epoch_data()
    panel = c.monitor.panel
    if d.cam is None or panel is None or len(panel) == 0:
        st.info("No Grad-CAM maps for this run. Enable them with `monitor.gradcam: 16` (number "
                "of panel samples) together with `--monitor`. Models without a compatible last "
                "layer (and model parallelism) are skipped.")
        return
    panel = np.asarray(panel)
    table, _ = epoch_table(c)
    alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.4, 0.05, key="insp_explain_alpha")

    st.markdown(f"##### Panel at epoch {c.epoch}")
    cards = []
    for k, pos in enumerate(panel):
        r = table.loc[int(pos)]
        img = cam_overlay(thumb_of(c, int(pos)), d.cam[k], alpha, _SIZE)
        cards.append((img, f"#{pos} · explains {_cls(c, d, k)} · true: {r['true']} · "
                           f"{'right' if r['correct'] else 'wrong'}"))
    card_grid(cards, cols=4)
    st.caption("Grad-CAM of the predicted class. ViT / DeiT: last block's patch tokens; CNN: last "
               "4-D activation map. Warm colours = regions that raised the score.")

    st.markdown("---")
    st.markdown("##### Evolution of one sample")
    pos = st.selectbox("Panel sample", panel.tolist(), key="insp_explain_pick",
                       format_func=lambda p: f"#{p} · true: {table.at[int(p), 'true']}")
    k = int(np.nonzero(panel == pos)[0][0])
    strip = thin_epochs([e for e in c.epochs if e <= c.epoch], _STRIP)
    cols = st.columns(len(strip))
    for col, e in zip(cols, strip):
        de = c.epoch_data(e)
        if de is None or de.cam is None or k >= len(de.cam):
            col.caption(f"epoch {e}: no map")
            continue
        col.image(cam_overlay(thumb_of(c, int(pos)), de.cam[k], alpha, _SIZE),
                  caption=f"epoch {e} · {_cls(c, de, k)}", width="stretch")
    if len(strip) < len([e for e in c.epochs if e <= c.epoch]):
        st.caption(f"Showing {len(strip)} evenly spaced epochs.")
    st.image(outcome_card(thumb_of(c, int(pos)), bool(table.at[int(pos), "correct"]), 96), width=96,
             caption="original")
