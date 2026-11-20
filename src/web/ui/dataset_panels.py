"""Streamlit panels about a run's dataset for every dataset but BigEarthNet.

Built only from what ``logs/`` holds (see ``src/web/dataset_view.py``): the class mix of
the validation set and example thumbnails from the monitor's panel. Shared by the
Overview and the Dataset page.
"""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.web.dataset_view import Example, inspect_dir_of, panel_examples, val_class_counts
from src.web.ui.charts import COLORS, _base_layout, _show


@st.cache_data(ttl=120, show_spinner=False)
def _cached_counts(dir_str: str):
    return val_class_counts(Path(dir_str))


@st.cache_data(ttl=120, show_spinner=False)
def _cached_examples(dir_str: str, limit: int) -> list[Example]:
    return panel_examples(Path(dir_str), limit)


def class_mix(run, height: int = 145, key: str = "class_mix") -> None:
    """Bar chart of the validation samples per class, from the run's inspection files."""
    d = inspect_dir_of(run) if run is not None else None
    found = _cached_counts(str(d)) if d else None
    if not found:
        st.caption("No class counts in the logs of this run (they come from its "
                   "`inspect_*` files).")
        return
    classes, counts = found
    fig = go.Figure(go.Bar(
        x=classes, y=counts, marker={"color": COLORS[0]},
        text=[f"{c:,}" for c in counts], textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:,} validation samples<extra></extra>"))
    fig.update_layout(**_base_layout(height), showlegend=False, yaxis_title=None,
                      margin={"l": 10, "r": 10, "t": 20, "b": 30})
    fig.update_xaxes(automargin=True, tickfont={"size": 10})
    fig.update_yaxes(visible=False)
    _show(fig, key)


def example_strip(run, n: int = 5, empty: str | None = None) -> None:
    """Up to ``n`` thumbnails of the monitor's fixed panel, captioned with their class."""
    d = inspect_dir_of(run) if run is not None else None
    examples = _cached_examples(str(d), n) if d else []
    if not examples:
        st.caption(empty or "No example images in the logs of this run (they come from its "
                            "`inspect_*` files).")
        return
    for col, ex in zip(st.columns(len(examples), gap="small"), examples):
        col.image(ex.thumb, width="stretch")
        col.markdown(f"<div style='font-size:0.66rem;line-height:1.1'>{ex.class_name}</div>",
                     unsafe_allow_html=True)
