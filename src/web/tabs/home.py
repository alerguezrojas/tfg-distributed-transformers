"""Tab render module — see src/web/app.py for the orchestrator."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.dataset_stats import (
    CLASS_NAMES, SPLIT_SIZES, class_distribution_approximate,
)

_BEST = "Best val"
_CURVE = "Val curve"

from src.web.ui.charts import (
    COLORS, _show, _dl_csv, _base_layout, _CLASS_GROUPS,
)
from src.web.data_paths import NOT_AVAILABLE
from src.web.run_registry import _DEFAULT_DATASET
from src.web.run_select import gpu_count, best_test_by_dataset
from src.web.ui import dataset_panels
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import (
    _bigearthnet_paths, _load_df, _load_test_aggregate, _get_benchmark_csvs,
    _load_class_distribution, _class_gallery,
    _safe_max, _safe_val_at_best, _dur_str,
    _run_metric, _scope_dataset, _scope_run, _loss_label, _fmt_span,
)


def render(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    st.markdown("## Overview")

    # The dataset the dataset-specific panels are about: the sidebar filter, else the
    # active run's. Metrics are never mixed across datasets: each run is read with ITS
    # main metric (QWK for PANDA, AUC for MedMNIST, F1 for BigEarthNet).
    scope = _scope_dataset(selected_run)
    scope_run = _scope_run(runs, selected_run, scope)

    # ── One pass over the runs (stats + per-run main-metric curve + epoch time) ──
    best_by_ds: dict[str, tuple] = {}     # dataset → (metric, value, label)
    scope_epoch_s: list[float] = []       # mean epoch time of each run of the scope dataset
    total_gpu_h = 0.0
    total_energy_wh = 0.0
    benchmark_csvs_home = _get_benchmark_csvs()
    curve_by_label: dict[str, list[float]] = {}
    gpu_secs_by_env: dict[str, float] = {}     # total GPU seconds per environment

    gpus_unknown = 0                      # runs whose log does not say how many GPUs they used
    for r in runs:
        try:
            df_r = _load_df(str(r.log_path),
                            str(r.epoch_csv_path) if r.epoch_csv_path else None)
            metric = _run_metric(r, df_r)
            if not df_r.empty and metric.column in df_r.columns:
                curve_by_label[r.label] = df_r[metric.column].dropna().round(4).tolist()
                run_best = _safe_max(df_r[metric.column])
                if not pd.isna(run_best) and (
                        r.dataset not in best_by_ds or run_best > best_by_ds[r.dataset][1]):
                    best_by_ds[r.dataset] = (metric, run_best, r.label)
            if not df_r.empty and "epoch_time" in df_r.columns and df_r["epoch_time"].notna().any():
                n_gpus = gpu_count(r)
                gpus_unknown += n_gpus is None
                # GPU time = wall time x GPUs; a run with an unknown count is counted once.
                secs = float(df_r["epoch_time"].dropna().sum()) * (n_gpus or 1)
                total_gpu_h += secs / 3600
                gpu_secs_by_env[r.env] = gpu_secs_by_env.get(r.env, 0.0) + secs
                if r.dataset == scope:
                    scope_epoch_s.append(float(df_r["epoch_time"].dropna().mean()))
            for _c in ("energy_train_wh", "energy_eval_wh"):
                if _c in df_r.columns and df_r[_c].notna().any():
                    total_energy_wh += float(df_r[_c].dropna().sum())
        except Exception:
            pass

    n_models = len({r.model for r in runs if r.model})
    n_envs = len({r.env for r in runs})
    n_scope = sum(1 for r in runs if r.dataset == scope)
    best_scope = best_by_ds.get(scope)

    # ── Compact KPI strip (one dense row) ───────────────────────────────────────
    _kpi_strip([
        ("Runs" if n_scope == len(runs) else "Runs (shown / all)",
         f"{n_scope} / {len(runs)}" if scope and n_scope != len(runs) else str(len(runs))),
        (f"Best val {best_scope[0].name}" if best_scope else "Best val metric",
         f"{best_scope[1]:.3f}" if best_scope else "—"),
        ("Fastest epoch", _fmt_span(min(scope_epoch_s)) if scope_epoch_s else "—"),
        ("GPU time", f"{total_gpu_h:.0f} h"),
        ("Energy", f"{total_energy_wh:.0f} Wh" if total_energy_wh else "—"),
        ("Models", str(n_models)),
        ("Environments", str(n_envs)),
        ("Benchmark", str(len(benchmark_csvs_home))),
    ])
    if gpus_unknown:
        st.caption(f"GPU time = wall time x number of GPUs; {gpus_unknown} run(s) do not record "
                   "their GPU count and are counted with one.")
    if scope:
        st.caption(f"Runs, best result and fastest epoch refer to **{scope}** (the sidebar's "
                   "dataset filter, else the active run's dataset); the rest to all runs.")

    # ── Row 1: GPU time + two panels about the dataset of the scope. BigEarthNet has
    #          its own (split pie, class-imbalance treemap); any other dataset gets the
    #          generic ones built from the run's logs ──────────────────────────────
    is_ben = scope == _DEFAULT_DATASET
    paths = _bigearthnet_paths()
    meta, root = paths.metadata, paths.root

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.caption("GPU time by environment — hours")
            _time_by_env_bars(gpu_secs_by_env)
    with c2:
        with st.container(border=True):
            if is_ben:
                st.caption("Dataset split — train / val / test")
                _split_pie()
            else:
                st.caption(f"Class mix — {scope} validation set")
                dataset_panels.class_mix(scope_run, key="ov_class_mix")
    with c3:
        with st.container(border=True):
            if is_ben:
                st.caption("Class imbalance — tile area = patches")
                _class_treemap(_dataset_dist(meta))
            else:
                st.caption("Best result per dataset")
                _best_by_dataset(best_by_ds, runs)

    # ── Row 2: active-run card (with mini metric/loss curves) + sample images ───────
    a_left, a_right = st.columns([1.4, 1])
    with a_left:
        with st.container(border=True):
            _df_active = pd.DataFrame()
            if selected_run is not None:
                try:
                    _df_active = _load_df(
                        str(selected_run.log_path),
                        str(selected_run.epoch_csv_path) if selected_run.epoch_csv_path else None)
                except Exception:
                    pass
            st.caption(f"Active run — {selected_run.label if selected_run else '—'}")
            if selected_run is not None and not _df_active.empty and \
                    _run_metric(selected_run, _df_active).column in _df_active.columns:
                _run_highlight(_df_active, selected_run)
            else:
                st.info("No metrics for this run.")
    with a_right:
        with st.container(border=True):
            if is_ben:
                st.caption("Sample Sentinel-2 patches — RGB proxy")
                _photo_strip(meta, root)
            else:
                st.caption(f"Examples — {scope}, from the monitor's fixed panel")
                dataset_panels.example_strip(scope_run)

    # ── Held-out test results, one row per dataset ──────────────────────────────
    st.markdown("#### Test results by dataset")
    _test_by_dataset(runs)

    # ── All runs — selectable table (click a row → active run) ──────────────────
    st.markdown("#### All runs")
    st.caption("Click a row to make that run active. Each run is scored with its own main "
               "metric (column *Metric*). Dataset & import: **Dataset** section.")
    _all_runs_table(runs, curve_by_label)


def _test_by_dataset(runs) -> None:
    """Best held-out test result of each dataset (from `scripts/eval.py`), with the main
    metric of that dataset: datasets are never ranked against each other."""
    rows = best_test_by_dataset(
        runs, parse=lambda p: _load_test_aggregate(str(p)))
    if not rows:
        st.caption("No held-out test results yet: evaluate a run with `paravit eval` and its "
                   "test CSV shows up here. Every other figure is validation.")
        return
    st.caption("Best run of each dataset on its test split (never seen in training), scored "
               "with the dataset's own main metric.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 height=min(38 + 35 * len(rows), 220))


def _best_by_dataset(best_by_ds: dict, runs) -> None:
    """Small table: for each dataset, its main metric and the best validation value."""
    rows = [{"Dataset": ds, "Runs": sum(1 for r in runs if r.dataset == ds),
             "Metric": m.name, "Best": round(v, 4)}
            for ds, (m, v, _lbl) in sorted(best_by_ds.items())]
    if not rows:
        st.info("No metrics yet.")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=145)


def _dataset_dist(meta):
    """Per-class train counts — from the parquet if mounted, else an approximation."""
    dist = _load_class_distribution(str(meta)) if meta else None
    if dist is None:
        dist = class_distribution_approximate()
    return dist


def _group_of(class_name: str):
    """(group name, colour) for a class name, via its index in CLASS_NAMES."""
    idx = CLASS_NAMES.index(class_name) if class_name in CLASS_NAMES else -1
    for gname, (idxs, color) in _CLASS_GROUPS.items():
        if idx in idxs:
            return gname, color
    return "Other", COLORS[0]


def _short_cls(name: str, n: int = 13) -> str:
    return name if len(name) <= n else name[:n - 1] + "…"


# Stable colour per environment for the GPU-time bar.
_ENV_COLOR = {"local": COLORS[0], "verode": COLORS[1], "kaggle": COLORS[2]}


def _env_color(env: str) -> str:
    return _ENV_COLOR.get(env, COLORS[3])


def _time_by_env_bars(gpu_secs_by_env: dict[str, float]) -> None:
    """Vertical bars of total GPU hours per environment — where the project's
    compute actually ran (local RTX 3060 Ti, Verode V100, Kaggle T4)."""
    if not gpu_secs_by_env:
        st.info("No timing data.")
        return
    items = sorted(gpu_secs_by_env.items(), key=lambda x: -x[1])
    envs = [e for e, _ in items]
    hrs = [s / 3600 for _, s in items]
    fig = go.Figure(go.Bar(
        x=[e.capitalize() for e in envs], y=hrs,
        marker={"color": [_env_color(e) for e in envs], "line": {"width": 0}},
        text=[f"{h:.0f} h" for h in hrs], textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:.1f} GPU hours<extra></extra>",
    ))
    fig.update_layout(**_base_layout(145), showlegend=False, yaxis_title="GPU hours",
                      margin={"l": 10, "r": 10, "t": 20, "b": 10})
    _show(fig, "ov_time_env_bars")


def _split_pie() -> None:
    """Pie of the dataset partition — train / val / test patch counts."""
    labels = ["Train", "Val", "Test"]
    vals = [SPLIT_SIZES["train"], SPLIT_SIZES["val"], SPLIT_SIZES["test"]]
    fig = go.Figure(go.Pie(
        labels=labels, values=vals, sort=False,
        marker={"colors": [COLORS[0], COLORS[1], COLORS[2]],
                    "line": {"width": 1.5, "color": "white"}},
        textinfo="label+percent", textfont={"size": 11},
        hovertemplate="<b>%{label}</b><br>%{value:,} patches · %{percent}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(145), showlegend=False,
                      margin={"l": 6, "r": 6, "t": 10, "b": 6})
    _show(fig, "ov_split_pie")


def _class_treemap(dist) -> None:
    """Treemap of the 19 classes (tile area = #train patches) — the imbalance that
    caps macro-F1, in a third, distinct chart type."""
    d = dist.sort_values("train_count", ascending=False)
    colors = [_group_of(c)[1] for c in d["class"]]
    fig = go.Figure(go.Treemap(
        labels=[_short_cls(c, 16) for c in d["class"]], parents=[""] * len(d),
        values=d["train_count"],
        marker={"colors": colors, "line": {"width": 1, "color": "white"}},
        texttemplate="%{label}", textfont={"size": 9}, customdata=list(d["class"]),
        hovertemplate="<b>%{customdata}</b><br>%{value:,} patches · %{percentRoot}<extra></extra>",
        tiling={"pad": 2},
    ))
    fig.update_layout(**_base_layout(145), margin={"t": 6, "l": 0, "r": 0, "b": 0})
    _show(fig, "ov_class_treemap")


def _photo_strip(meta, root) -> None:
    """Real Sentinel-2 patches (RGB proxy), one per class — a small carousel paged
    5 at a time with ◀/▶ arrows (5·5·5·4 = all 19). Skipped when not mounted."""
    if not (meta and root):
        st.info(f"{NOT_AVAILABLE} Sample patches need it.")
        return
    try:
        _avg, gallery = _class_gallery(str(meta), str(root))
    except Exception:
        gallery = []
    if not gallery:
        st.info("No sample patches available.")
        return

    per_page = 5
    n = len(gallery)
    pages = (n + per_page - 1) // per_page
    page = st.session_state.get("ov_photo_page", 0) % pages

    nav1, nav2, nav3 = st.columns([1, 5, 1])
    if nav1.button("◀", key="ov_photo_prev", help="Previous classes", width="stretch"):
        st.session_state["ov_photo_page"] = (page - 1) % pages
        st.rerun()
    lo, hi = page * per_page, min(page * per_page + per_page, n)
    nav2.markdown(
        f"<div style='text-align:center;font-size:0.72rem;color:#6B7280;"
        f"padding-top:0.4rem'>Classes {lo + 1}–{hi} of {n}</div>",
        unsafe_allow_html=True)
    if nav3.button("▶", key="ov_photo_next", help="Next classes", width="stretch"):
        st.session_state["ov_photo_page"] = (page + 1) % pages
        st.rerun()

    cols = st.columns(per_page, gap="small")   # fixed 5 columns → stable layout
    for col, (cls, cnt, pct, img, labels) in zip(cols, gallery[lo:hi]):
        col.image(img, width="stretch")
        title = " · ".join(labels).replace("'", "")
        col.markdown(
            f"<div title='{title}' style='font-size:0.6rem;line-height:1.05;"
            f"height:2.2rem;overflow:hidden'>{_short_cls(cls, 14)}</div>",
            unsafe_allow_html=True)


def _kpi_strip(items: list[tuple[str, str]]) -> None:
    """One dense row of stat cards (much more compact than st.metric cards)."""
    cells = "".join(
        f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>"
        for l, v in items
    )
    st.markdown(f"<div class='kpi-strip'>{cells}</div>", unsafe_allow_html=True)


def _run_highlight(df: pd.DataFrame, run) -> None:
    """Metric strip, the two mini curves (main metric and loss) and a one-line verdict
    (card body). Names and values follow the run's own main metric and loss."""
    metric = _run_metric(run, df)
    best = _safe_max(df[metric.column])
    best_ep = _safe_val_at_best(df, metric.column, "epoch")
    dur = (_dur_str(df["epoch_time"].dropna().sum())
           if "epoch_time" in df.columns and df["epoch_time"].notna().any() else "—")
    thr = (_safe_max(df["f1_at_threshold"])
           if "f1_at_threshold" in df.columns and df["f1_at_threshold"].notna().any() else None)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Epochs", len(df))
    m2.metric(f"Best val {metric.name}", f"{best:.4f}" if not pd.isna(best) else "—")
    m3.metric("Best epoch", int(best_ep) if best_ep is not None else "—")
    m4.metric("Duration", dur)
    if thr is not None and metric.key == "f1":
        st.caption(f"F1 at the optimal threshold: {thr:.4f}")

    cc1, cc2 = st.columns(2)
    with cc1:
        fig = go.Figure()
        train_col = f"train_{metric.key}"      # only F1 has a train-side value
        if train_col in df.columns:
            fig.add_trace(go.Scatter(x=df["epoch"], y=df[train_col], name="Train",
                                     line={"color": COLORS[0], "width": 2}))
        fig.add_trace(go.Scatter(x=df["epoch"], y=df[metric.column], name="Val",
                                 line={"color": COLORS[1], "width": 2}))
        fig.update_layout(**_base_layout(108, metric.long_name), xaxis_title="Epoch",
                          yaxis_title=metric.name)
        _show(fig, "hub_metric")
    with cc2:
        if "val_loss" in df.columns:
            fig = go.Figure()
            if "train_loss" in df.columns:
                fig.add_trace(go.Scatter(x=df["epoch"], y=df["train_loss"], name="Train",
                                         line={"color": COLORS[0], "width": 2}))
            fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"], name="Val",
                                     line={"color": COLORS[3], "width": 2}))
            fig.update_layout(**_base_layout(108, f"Loss ({_loss_label(run)})"),
                              xaxis_title="Epoch", yaxis_title="Loss")
            _show(fig, "hub_loss")

    # One-line verdict: overfitting gap at the best epoch (in F1, the only train-side metric).
    if "train_f1" in df.columns and "val_f1" in df.columns and not pd.isna(best):
        ep_f1 = _safe_val_at_best(df, "val_f1", "epoch")
        f1_best = _safe_max(df["val_f1"])
        _tr = df.loc[df["epoch"] == ep_f1, "train_f1"] if ep_f1 is not None else []
        gap = float(_tr.iloc[0]) - float(f1_best) if len(_tr) else None
        note = " — overfitting" if gap is not None and gap > 0.1 else ""
        gap_txt = f" · train–val F1 gap {gap:+.2f}{note}" if gap is not None else ""
        st.caption(f"Best val {metric.name} {best:.3f} at epoch "
                   f"{int(best_ep) if best_ep is not None else '—'}{gap_txt}")


def _all_runs_table(runs, curve_by_label: dict[str, list[float]]) -> None:
    """Runs table with a wandb-style sparkline of each run's own main metric."""
    rows = []
    for r in runs[:40]:
        try:
            df_r = _load_df(str(r.log_path),
                            str(r.epoch_csv_path) if r.epoch_csv_path else None)
            metric = _run_metric(r, df_r)
            if df_r.empty or metric.column not in df_r.columns:
                continue
            best = _safe_max(df_r[metric.column])
            if pd.isna(best):
                continue
            best_ep = _safe_val_at_best(df_r, metric.column, "epoch")
            dur_s = df_r["epoch_time"].dropna().sum() if "epoch_time" in df_r.columns else float("nan")
            energy_wh = (df_r["energy_eval_wh"].sum()
                         if "energy_eval_wh" in df_r.columns and df_r["energy_eval_wh"].notna().any()
                         else None)
            rows.append({
                "Run": r.label,
                "Dataset": r.dataset,
                "Metric": metric.name,
                _CURVE: curve_by_label.get(r.label, []),
                _BEST: round(best, 4),
                "Mode": r.mode,
                "Precision": r.precision or "fp32",
                "Env": r.env,
                "Epochs": len(df_r),
                "Best epoch": int(best_ep) if best_ep is not None else None,
                "Duration": _dur_str(dur_s) if not pd.isna(dur_s) else "—",
                "Eval Wh": round(energy_wh) if energy_wh else None,
            })
        except Exception:
            pass

    if not rows:
        st.info("No runs with parseable metrics found.")
        return

    ov_df = pd.DataFrame(rows)
    # Bounded height: ~8 rows are visible, the rest scroll *inside* the table so
    # the page itself stays within one screen (no page scroll on the Overview).
    _table_h = min(34 + 30 * len(ov_df), 160)
    event = st.dataframe(
        ov_df,
        width="stretch", hide_index=True, height=_table_h,
        on_select="rerun", selection_mode="single-row", key="runs_table",
        column_config={
            "Run": st.column_config.TextColumn("Run", width="large"),
            _CURVE: st.column_config.LineChartColumn(
                _CURVE, y_min=0.0, y_max=1.0, width="medium"),
            _BEST: st.column_config.ProgressColumn(
                _BEST, min_value=0.0, max_value=1.0, format="%.4f"),
        },
    )
    # Clicking a row makes that run active across the whole dashboard. We act
    # only when the selected ROW changes (tracked in _last_table_row); otherwise
    # a stale table selection would override a run picked from the sidebar.
    sel = event.selection.rows if event and event.selection else []
    if sel:
        chosen = ov_df.iloc[sel[0]]["Run"]
        if st.session_state.get("_last_table_row") != chosen:
            st.session_state["_last_table_row"] = chosen
            st.session_state["run_label"] = chosen
            st.rerun()
    _dl_csv(ov_df.drop(columns=[_CURVE]), "runs_summary.csv", "Download runs table")
