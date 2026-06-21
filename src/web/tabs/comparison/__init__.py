"""Compare page — one multiselect drives summary, speedup, per-class, radar,
energy and overlays. render orchestrates; the sections live in sibling modules."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.web.ui.context import DashboardContext
from src.web.ui.helpers import _load_df, _get_runs
from src.web.tabs.comparison.summary import _summary_table, _config_diff_section
from src.web.tabs.comparison.perclass import _perclass_heatmap_section
from src.web.tabs.comparison.speedup import _speedup_section
from src.web.tabs.comparison.charts import _radar_section, _energy_section, _overlay_charts


def render(ctx: DashboardContext) -> None:
    st.markdown("## Compare")
    st.caption("Pick any set of runs and compare them in one place: summary, "
               "speedup against a baseline, radar, energy and per-epoch overlays.")

    runs = ctx.runs
    if not runs:
        st.info("No runs available.")
        return

    all_run_labels = {r.label: r for r in runs}
    all_labels_list = list(all_run_labels.keys())

    # Default: the latest SESSION (runs sharing env and day with the most
    # recent run) — e.g. the whole 5-strategy Kaggle session in one click.
    _latest = runs[0]
    _session = [r.label for r in runs
                if r.env == _latest.env and r.timestamp[:8] == _latest.timestamp[:8]][:8]
    _default = _session if len(_session) >= 2 else all_labels_list[:min(2, len(all_labels_list))]

    selected_compare = st.multiselect(
        "Select runs to compare (max 8)", all_labels_list,
        default=_default,
        max_selections=8,
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

    # Primary, always visible: the summary + the views the user relies on (radar,
    # speedup, energy, overlays). Secondary/niche views go in expanders below.
    _summary_table(compare_runs_list, df_by_label)
    st.markdown("---")
    _radar_section(compare_dfs)
    st.markdown("---")
    _speedup_section(compare_runs_list, df_by_label)
    st.markdown("---")
    _energy_section(compare_dfs)
    _overlay_charts(compare_dfs)

    st.markdown("---")
    with st.expander("Configuration — hyperparameters side by side"):
        _config_diff_section(compare_runs_list)
    with st.expander("Per-class F1 by run"):
        _perclass_heatmap_section(compare_runs_list)


# ── Summary ─────────────────────────────────────────────────────────────────────

