"""Subgroups sub-view: performance by group (sex, age band, centre...) with intervals and the gap.

Only offered when the run has ``inspect_<ts>/groups.npz`` (one array per variable, aligned with
the monitor set; format in docs/inspeccion.md). The metrics are computed on the monitored
subset, which is what the groups are aligned with.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.analysis.subgroups import SUBGROUP_METRICS, subgroup_metrics
from src.web.inspector_stats import (
    DEFAULT_SEED, SUBGROUP_LABELS, bin_groups, ci_note, gap_frame, subgroup_frame,
)
from src.web.tabs.inspector._common import InspectorCtx, _mtime, fmt, need_monitor
from src.web.ui.charts import _show

GROUPS_HINT = ("Subgroup analysis needs per-sample metadata (sex, age, centre...). The monitor "
               "writes `inspect_<ts>/groups.npz` when the dataset exposes it; the format is in "
               "docs/inspeccion.md.")


@st.cache_data(show_spinner="Computing subgroup metrics…", max_entries=16)
def _subgroups(dir_str: str, epoch: int, mtime: float, gmtime: float, variable: str, bins: int,
               task: str, thr: float, min_size: int, n_boot: int, _probs, _targets,
               _groups) -> dict:
    return subgroup_metrics(_probs, _targets, task, _groups, thr, min_size, n_boot,
                            seed=DEFAULT_SEED)


def render(c: InspectorCtx) -> None:
    if not c.groups:
        st.info("This run has no group metadata. " + GROUPS_HINT)
        return
    if not need_monitor(c):
        return
    d = c.epoch_data()
    a, b = st.columns([2, 1])
    variable = a.selectbox("Variable", list(c.groups), key="insp_grp_var")
    bins = int(b.number_input("Bands for numeric variables", 2, 8, 4, key="insp_grp_bins",
                              help="A numeric variable with many distinct values (age) is "
                                   "cut in quantile bands; categorical ones are used as they are."))
    labels = bin_groups(c.groups[variable], bins)
    k = st.columns(3)
    thr = 0.5
    if c.task != "multiclass":
        default = float(d.threshold) if d.threshold is not None else 0.5
        thr = k[0].slider("Threshold", 0.05, 0.95, min(max(round(default, 2), 0.05), 0.95), 0.05,
                          key="insp_grp_thr")
    min_size = int(k[1].number_input("Minimum group size", 1, 1000, 20, key="insp_grp_min",
                                     help="Smaller groups are flagged and left out of the gap "
                                          "and the intervals."))
    n_boot = k[2].selectbox("Interval resamples", (0, 200, 1000), index=1, key="insp_grp_n",
                            format_func=lambda n: "no intervals" if n == 0 else f"{n:,}")
    mt = _mtime(c.inspect_dir / f"epoch_{c.epoch:03d}.npz")
    gmt = _mtime(c.inspect_dir / "groups.npz")
    res = _subgroups(str(c.inspect_dir), c.epoch, mt, gmt, variable, bins, c.task, thr, min_size,
                     n_boot, d.probs, c.monitor.targets, labels)

    st.caption(f"Computed on the monitored subset ({res['overall']['n']:,} samples, the ones "
               "the groups are aligned with), so small groups are noisy. "
               + (ci_note(n_boot) if n_boot else "No intervals: point estimates only."))
    st.dataframe(subgroup_frame(res), hide_index=True, width="stretch")

    gap = gap_frame(res)
    st.markdown("##### Gap between the best and the worst group")
    if gap["Gap (max - min)"].notna().any():
        worst = gap.loc[gap["Gap (max - min)"].idxmax()]
        st.caption(f"Largest gap: {worst['Metric']} {fmt(worst['Gap (max - min)'])} "
                   f"({worst['Best group']} vs {worst['Worst group']}). It is a difference of "
                   "point estimates: compare it with the intervals before calling it real.")
    else:
        st.caption("Needs at least two groups of the minimum size.")
    st.dataframe(gap.round(3), hide_index=True, width="stretch")

    metric = st.selectbox("Metric to plot", SUBGROUP_METRICS, key="insp_grp_metric",
                          format_func=SUBGROUP_LABELS.get)
    ok = [g for g in res["groups"] if not g["small"]]
    if ok:
        err = None
        if n_boot:
            est = [g["metrics"][metric] for g in ok]
            err = {"type": "data", "symmetric": False,
                   "array": [max(g["ci"][metric][1] - e, 0) for g, e in zip(ok, est)],
                   "arrayminus": [max(e - g["ci"][metric][0], 0) for g, e in zip(ok, est)]}
        fig = go.Figure(go.Bar(x=[str(g["group"]) for g in ok],
                               y=[g["metrics"][metric] for g in ok], error_y=err))
        fig.add_hline(y=res["overall"]["metrics"][metric], line_dash="dot", opacity=0.6,
                      annotation_text="all samples")
        fig.update_layout(height=320, title={"text": f"{SUBGROUP_LABELS[metric]} by {variable}"},
                          yaxis_title=SUBGROUP_LABELS[metric], xaxis_title=variable)
        _show(fig, "insp_grp_bar")
