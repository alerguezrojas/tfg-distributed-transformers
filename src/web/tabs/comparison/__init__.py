"""Compare page — one multiselect drives summary, speedup, per-class, radar,
energy and overlays. render orchestrates; the sections live in sibling modules.

Runs are compared within ONE dataset (its main metric is the common yardstick: QWK for
PANDA, AUC for MedMNIST, F1 for BigEarthNet), and runs that differ only in the seed can
be aggregated into a mean ± std (``src/web/seed_agg.py``)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.web.run_meta import F1, resolve_metric
from src.web.seed_agg import build_series, group_runs
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import _load_df
from src.web.tabs.comparison.summary import _summary_table
from src.web.tabs.comparison.perclass import _perclass_heatmap_section
from src.web.tabs.comparison.speedup import _speedup_section
from src.web.tabs.comparison.charts import _radar_section, _energy_section, _overlay_charts

MAX_RUNS = 12


def render(ctx: DashboardContext) -> None:
    st.markdown("## Compare")
    st.caption("Pick runs of one dataset and compare them in one place: summary, "
               "speedup against a baseline, radar, energy and per-epoch overlays. Runs that "
               "differ only in the seed can be averaged.")

    runs = ctx.runs
    if not runs:
        st.info("No runs available.")
        return

    # One dataset at a time: each has its own main metric, so mixing them would compare
    # a QWK with an F1.
    datasets = sorted({r.dataset for r in runs})
    default_ds = ctx.run.dataset if ctx.run is not None and ctx.run.dataset in datasets \
        else runs[0].dataset
    dataset = st.selectbox("Dataset", datasets, index=datasets.index(default_ds),
                           key="compare_dataset") if len(datasets) > 1 else datasets[0]
    pool = [r for r in runs if r.dataset == dataset]
    all_run_labels = {r.label: r for r in pool}
    all_labels_list = list(all_run_labels.keys())

    # Default: the seed siblings of the active run when it has some (to see the mean over
    # seeds at once), else the latest SESSION (runs sharing env and day with the most recent
    # run) — e.g. the whole 5-strategy Kaggle session in one click.
    _active = ctx.run
    _siblings = [r.label for g in group_runs(pool)
                 if _active is not None and any(x.timestamp == _active.timestamp for x in g)
                 for r in g]
    _latest = pool[0]
    _session = [r.label for r in pool
                if r.env == _latest.env and r.timestamp[:8] == _latest.timestamp[:8]][:MAX_RUNS]
    _default = (_siblings if len(_siblings) >= 2 else
                _session if len(_session) >= 2 else all_labels_list[:min(2, len(all_labels_list))])

    selected_compare = st.multiselect(
        f"Select runs to compare (max {MAX_RUNS})", all_labels_list,
        default=_default,
        max_selections=MAX_RUNS,
        key=f"compare_runs_{dataset}",
    )
    if len(selected_compare) < 2:
        st.info("Select at least 2 runs.")
        return

    compare_runs_list = [(lbl, all_run_labels[lbl]) for lbl in selected_compare]
    compare_dfs: list[tuple[str, pd.DataFrame]] = []
    for lbl, r in compare_runs_list:
        cdf = _load_df(str(r.log_path), str(r.epoch_csv_path) if r.epoch_csv_path else None)
        # The log reports train energy in Joules — derive Wh so energy
        # charts use one unit (eval already comes as energy_eval_wh).
        if "energy_train_j" in cdf.columns:
            cdf = cdf.assign(energy_train_wh=cdf["energy_train_j"] / 3600.0)
        compare_dfs.append((lbl, cdf))
    df_by_label = dict(compare_dfs)

    # The dataset's main metric, unless some selected run did not log it (then F1 for all,
    # so the numbers stay comparable).
    metric = pool[0].main_metric
    if any(resolve_metric(metric, df_by_label[lbl]) is F1 and metric is not F1
           for lbl in selected_compare):
        metric = F1

    sel_runs = [r for _, r in compare_runs_list]
    can_aggregate = any(len(g) > 1 for g in group_runs(sel_runs))
    aggregate = st.toggle(
        "Aggregate runs that differ only in the seed (mean ± std)", value=can_aggregate,
        disabled=not can_aggregate, key="compare_aggregate",
        help="Groups the selected runs with the same dataset, model, loss, providers, "
             "colour, resolution, batch and lr. Needs at least two seeds of one setting.")
    series = build_series(sel_runs, df_by_label, aggregate=aggregate and can_aggregate)

    # Primary, always visible: the summary + the views the user relies on (radar,
    # speedup, energy, overlays). Secondary/niche views go in expanders below.
    _summary_table(series, metric)
    st.markdown("---")
    _radar_section(series, metric)
    st.markdown("---")
    _speedup_section(compare_runs_list, df_by_label)
    st.markdown("---")
    _energy_section(compare_dfs)
    _overlay_charts(series, metric)

    st.markdown("---")
    with st.expander("Per-class F1 by run"):
        _perclass_heatmap_section(compare_runs_list)
