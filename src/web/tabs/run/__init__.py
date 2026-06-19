"""Run results page — one module per sub-view (curves, per-class, confusions,
batch, details). ``render`` dispatches to them."""
from __future__ import annotations

import streamlit as st

from src.web.ui.context import DashboardContext
from src.web.tabs.run.curves import _curves, _test_callout
from src.web.tabs.run.perclass import _per_class
from src.web.tabs.run.confusions import _confusions_tab
from src.web.tabs.run.batch import _batch
from src.web.tabs.run.details import _time, _info


def render(ctx: DashboardContext) -> None:
    st.markdown("## Run results")
    st.caption("Metrics and metadata of the run selected in the sidebar.")
    _test_callout(ctx)
    sub = st.tabs(["Curves", "Per-class", "Confusions", "Batch", "Details"])
    with sub[0]:
        _curves(ctx)
    with sub[1]:
        _per_class(ctx)
    with sub[2]:
        _confusions_tab(ctx)
    with sub[3]:
        _batch(ctx)
    with sub[4]:
        _time(ctx)
        st.markdown("---")
        _info(ctx)
