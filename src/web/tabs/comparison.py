"""Tab render module — see src/web/app.py for the orchestrator.

Compare is ONE unified section: a single run multiselect drives everything —
summary table, speedup vs a chosen baseline (the generalized version of the
old "Single vs Distributed" pair), radar, energy and per-epoch overlays.
"""
from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.feasibility_parser import parse_feasibility_csv, parse_ddp_scenarios
from src.web.run_registry import RunInfo

from src.web.ui import theme
from src.web.ui.charts import COLORS, _show, _dl_csv, _base_layout, _overlay_fig
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import (
    _load_df, _load_perclass, _run_config, _get_feasibility_csvs,
    _safe_max, _safe_val_at_best, _dur_str,
)
from pathlib import Path


def _predicted_2gpu_speedup(env: str, model: str) -> float | None:
    """The feasibility's predicted 2-GPU speedup for this env/model, if any."""
    for p in _get_feasibility_csvs():
        try:
            if p.parent.parent.name != env:
                continue
            meta, _ = parse_feasibility_csv(p)
            if meta.get("model_name") != model:
                continue
            scen = parse_ddp_scenarios(meta)
            if not scen.empty and {"n_gpus", "speedup"}.issubset(scen.columns):
                r2 = scen[scen["n_gpus"] == 2]
                if not r2.empty:
                    return float(r2.iloc[0]["speedup"])
        except Exception:
            pass
    return None


def _prec(r: RunInfo) -> str:
    """Runs without a precision marker predate the selector → fp32."""
    return r.precision or "fp32"


def _has(d: pd.DataFrame, c: str) -> bool:
    return c in d.columns and d[c].notna().any()


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

    _summary_table(compare_runs_list, df_by_label)
    st.markdown("---")
    _config_diff_section(compare_runs_list)
    st.markdown("---")
    _speedup_section(compare_runs_list, df_by_label)
    st.markdown("---")
    _perclass_compare_section(compare_runs_list)
    _radar_section(compare_dfs)
    st.markdown("---")
    _energy_section(compare_dfs)
    _overlay_charts(compare_dfs)


# ── Summary ─────────────────────────────────────────────────────────────────────

def _summary_table(sel: list[tuple[str, RunInfo]], df_by_label: dict[str, pd.DataFrame]) -> None:
    summary_rows = []
    for lbl, r in sel:
        cdf = df_by_label[lbl]
        best_f1_c = _safe_max(cdf["val_f1"]) if "val_f1" in cdf.columns and not cdf.empty else float("nan")
        best_ep_c_v = _safe_val_at_best(cdf, "val_f1", "epoch")
        _last = cdf["val_f1"].dropna() if "val_f1" in cdf.columns and not cdf.empty else pd.Series(dtype=float)
        total_s_c = cdf["epoch_time"].dropna().sum() if "epoch_time" in cdf.columns else float("nan")
        summary_rows.append({
            "Run": lbl,
            "Mode": r.mode,
            "Precision": _prec(r),
            "Best Val F1": f"{best_f1_c:.4f}" if not pd.isna(best_f1_c) else "—",
            "Best epoch": int(best_ep_c_v) if best_ep_c_v is not None else "—",
            "Final F1": f"{_last.iloc[-1]:.4f}" if not _last.empty else "—",
            "Epochs": len(cdf),
            "Duration": _dur_str(total_s_c) if not pd.isna(total_s_c) else "—",
            "Environment": r.env, "Trace": r.trace_mode,
        })
    sum_df = pd.DataFrame(summary_rows).set_index("Run")
    st.dataframe(sum_df, use_container_width=True)
    _dl_csv(sum_df.reset_index(), "runs_comparison.csv", "Download comparison")


# ── Configuration diff (W&B style) ────────────────────────────────────────────────

def _config_diff_section(sel: list[tuple[str, RunInfo]]) -> None:
    """Side-by-side hyperparameters with the differing rows highlighted — makes the
    apples-to-apples explicit (which params change between runs, which stay fixed)."""
    st.markdown("### Configuration")
    st.caption("Hyperparameters side by side; rows that differ between the selected "
               "runs are highlighted. Lets you confirm the comparison is apples-to-apples.")

    def _col(lbl: str, r: RunInfo) -> dict:
        cfg = _run_config(str(r.log_path))
        return {
            "Model": r.model.replace("_patch16_224", "") or "—",
            "Strategy": r.mode,
            "Precision": r.precision or "fp32",
            "Loss": cfg.get("loss", "—"),
            "Batch": cfg.get("batch", "—"),
            "Learning rate": cfg.get("lr", "—"),
            "Train / Val": f"{cfg.get('train', '?')}/{cfg.get('val', '?')}",
            "Environment": r.env,
            "Trace": r.trace_mode,
        }

    def _short(lbl: str) -> str:
        return re.sub(r"^\d{2}/\d{2}/\d{4}\s+", "", lbl)

    cols = {_short(lbl): _col(lbl, r) for lbl, r in sel}
    params = list(next(iter(cols.values())).keys())
    df = pd.DataFrame({c: [cols[c][p] for p in params] for c in cols}, index=params)
    differs = df.apply(lambda row: row.nunique() > 1, axis=1)

    only_diff = st.checkbox("Show only parameters that differ", value=True, key="cfg_only_diff")
    view = df[differs] if only_diff else df
    if view.empty:
        st.info("The selected runs share the same configuration — a clean apples-to-apples set.")
        return
    diff_idx = set(view.index[view.apply(lambda r: r.nunique() > 1, axis=1)])

    def _hl(row):
        on = row.name in diff_idx
        return [("background-color:#FBECEC" if on else "") for _ in row]

    st.dataframe(view.style.apply(_hl, axis=1), use_container_width=True)
    n = int(differs.sum())
    st.caption(f"{n} of {len(params)} parameters differ across the {len(sel)} selected runs.")


# ── Per-class comparison (dumbbell) ───────────────────────────────────────────────

def _perclass_compare_section(sel: list[tuple[str, RunInfo]]) -> None:
    """Dumbbell of per-class F1 for two runs — shows exactly which classes one run
    rescues or loses vs another (e.g. focal vs BCE on the rare classes)."""
    with_pc = [(lbl, r) for lbl, r in sel
               if r.perclass_csv_path and Path(r.perclass_csv_path).exists()]
    if len(with_pc) < 2:
        return  # needs two runs with per-class data
    st.markdown("### Per-class comparison")
    st.caption("F1 per class for two runs, sorted by the change. The connector is "
               "green where B beats A and red where it loses — the quickest way to "
               "see which classes a loss like focal rescues or sacrifices.")
    labels = [lbl for lbl, _ in with_pc]
    by_lbl = {lbl: r for lbl, r in with_pc}
    c1, c2 = st.columns(2)
    a_lbl = c1.selectbox("Run A", labels, index=0, key="pc_cmp_a")
    b_lbl = c2.selectbox("Run B", labels, index=min(1, len(labels) - 1), key="pc_cmp_b")
    if a_lbl == b_lbl:
        st.caption("Pick two different runs.")
        return

    def _last_f1(r: RunInfo) -> pd.Series:
        df = _load_perclass(str(r.perclass_csv_path))
        df = df[df["epoch"] == df["epoch"].max()]
        return df.set_index("class_name")["f1"]

    fa, fb = _last_f1(by_lbl[a_lbl]), _last_f1(by_lbl[b_lbl])
    classes = [c for c in fa.index if c in fb.index]
    data = pd.DataFrame({"class": classes,
                         "A": [fa[c] for c in classes],
                         "B": [fb[c] for c in classes]})
    data["delta"] = data["B"] - data["A"]
    data = data.sort_values("delta", ascending=True)   # biggest improvement on top

    def _short(lbl: str) -> str:
        import re
        return re.sub(r"^\d{2}/\d{2}/\d{4}\s+", "", lbl)

    fig = go.Figure()
    # Connectors, grouped by direction so the legend stays clean.
    for sign, color, name in ((1, theme.GOOD, "B better"), (-1, theme.BAD, "B worse")):
        xs: list = []
        ys: list = []
        for _, r in data.iterrows():
            if (r["delta"] > 0.01) == (sign > 0) and abs(r["delta"]) > 0.01:
                xs += [r["A"], r["B"], None]
                ys += [r["class"], r["class"], None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                     line=dict(color=color, width=2.5),
                                     name=name, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=data["A"], y=data["class"], mode="markers",
                             name=f"A · {_short(a_lbl)}",
                             marker=dict(color="#94A3B8", size=10)))
    fig.add_trace(go.Scatter(x=data["B"], y=data["class"], mode="markers",
                             name=f"B · {_short(b_lbl)}",
                             marker=dict(color=theme.ACCENT, size=10)))
    n_better = int((data["delta"] > 0.01).sum())
    n_worse = int((data["delta"] < -0.01).sum())
    fig.update_layout(
        title=dict(text="Per-class F1 — A vs B"),
        xaxis_title="F1", xaxis=dict(range=[0, 1.02]),
        height=max(380, 26 * len(classes) + 120),
        margin=dict(l=210, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0),
    )
    _show(fig, "compare_perclass_dumbbell")
    st.caption(f"**B improves {n_better} class(es)** and loses {n_worse} vs A "
               f"(macro F1: A={data['A'].mean():.3f}, B={data['B'].mean():.3f}).")


# ── Speedup vs baseline ─────────────────────────────────────────────────────────

def _speedup_section(sel: list[tuple[str, RunInfo]], df_by_label: dict[str, pd.DataFrame]) -> None:
    """Every selected run against one baseline (the generalized pair analysis)."""
    st.markdown("### Speedup analysis")
    st.caption(
        "All the runs selected at the top are compared against ONE of them — the "
        "baseline, which counts as 1.00×. To compare more runs, add them to the "
        "selector at the top of the page."
    )

    timed: list[tuple[str, RunInfo, float]] = []
    for lbl, r in sel:
        d = df_by_label[lbl]
        if _has(d, "epoch_time"):
            timed.append((lbl, r, float(d["epoch_time"].dropna().mean())))
    if len(timed) < 2:
        st.caption("Speedup needs at least 2 selected runs with epoch timing.")
        return
    if len(timed) < len(sel):
        st.caption(f"{len(sel) - len(timed)} selected run(s) have no epoch timing "
                   "and are excluded from the speedup table.")

    # Baseline default: a single-GPU fp32 simple-trace run (the natural 1.00x
    # reference); ties broken by recency.
    def _baseline_rank(t):
        _, r, _ = t
        return (r.mode == "single", _prec(r) == "fp32", r.trace_mode != "deep", r.sort_key)

    timed_labels = [lbl for lbl, _, _ in timed]
    default_lbl = max(timed, key=_baseline_rank)[0]
    # No fixed key: the widget is re-created when the selection changes, so the
    # smart default re-applies (a fixed key would freeze the first-render pick).
    base_lbl = st.selectbox(
        "Baseline run (= 1.00×) — every other selected run is measured against it",
        timed_labels, index=timed_labels.index(default_lbl),
    )
    _, base_r, base_t = next(t for t in timed if t[0] == base_lbl)

    rows, bar_lbls, bar_vals, bar_colors = [], [], [], []
    for lbl, r, t in timed:
        sp = base_t / t if t > 0 else float("nan")
        notes = []
        if r.model != base_r.model:
            notes.append("different model — not directly comparable")
        if _prec(r) != _prec(base_r):
            notes.append(f"precision {_prec(r)} vs {_prec(base_r)} (Tensor cores ~3-4×)")
        if r.mode == "ddp":
            # % of ideal only makes sense vs a same-precision baseline —
            # otherwise the Tensor-core effect pollutes the DDP efficiency.
            if _prec(r) == _prec(base_r):
                notes.append(f"{sp / 2 * 100:.0f}% of ideal 2×")
        elif r.mode == "ddp_hetero":
            notes.append("synchronous + imbalanced hardware")
        elif r.mode == "model_parallel":
            notes.append("naive pipeline — ≈1× expected")
        if r.trace_mode == "deep" and base_r.trace_mode != "deep":
            notes.append("deep trace ~20% overhead")
        rows.append({
            "Run": lbl, "Mode": r.mode, "Precision": _prec(r),
            "Avg epoch (min)": round(t / 60, 2),
            "Speedup": "baseline" if lbl == base_lbl else f"{sp:.2f}×",
            "Notes": "; ".join(notes) if lbl != base_lbl and notes else "—",
        })
        bar_lbls.append(lbl)
        bar_vals.append(sp)
        bar_colors.append("#94a3b8" if lbl == base_lbl else COLORS[0])

    st.dataframe(pd.DataFrame(rows).set_index("Run"), use_container_width=True)

    # Ranked bar: sort by speedup (fastest on top) and colour by faster / slower
    # than the baseline, so "scales up" vs "penalises" reads instantly.
    order = sorted(range(len(bar_vals)), key=lambda i: bar_vals[i])
    s_lbls = [bar_lbls[i] for i in order]
    s_vals = [bar_vals[i] for i in order]
    s_cols = ["#94A3B8" if abs(v - 1.0) < 0.05 else (theme.GOOD if v > 1.0 else theme.BAD)
              for v in s_vals]
    fig_sp = go.Figure(go.Bar(
        y=s_lbls, x=s_vals, orientation="h", marker_color=s_cols,
        text=[f"{v:.2f}×" for v in s_vals], textposition="outside",
        cliponaxis=False,
    ))
    fig_sp.update_layout(
        **_base_layout(150 + 40 * len(s_lbls), "Speedup vs baseline",
                       margin=dict(l=10, r=64, t=48, b=40)),
        xaxis_title="× speedup  (green = faster than baseline, red = slower)",
        showlegend=False,
    )
    fig_sp.update_yaxes(automargin=True)
    fig_sp.add_vline(x=1.0, line_dash="dash", line_color="#475569",
                     annotation_text="baseline 1.0×", annotation_position="top")
    _show(fig_sp, "compare_speedup")

    # One pedagogical banner per special mode present in the selection.
    others = [(lbl, r, base_t / t) for lbl, r, t in timed if lbl != base_lbl]
    if any(r.mode == "model_parallel" for _, r, _ in others):
        st.info(
            "**Model parallelism is not expected to accelerate training (≈1×).** "
            "The naive pipeline runs the stages sequentially — one GPU is idle while the "
            "other computes — so the theoretical ceiling is ≈1×. Its purpose is to **train "
            "models that do not fit on a single GPU**: vit_large exceeds the memory of one "
            "T4 but trains when split 12/24 across both."
        )
    if any(r.mode == "ddp_hetero" and sp < 1 for _, r, sp in others):
        st.warning(
            "**Heterogeneous DDP is slower than the GPU alone** — the expected outcome of "
            "**synchronous** DDP on imbalanced hardware (V100 + CPU): on every batch the "
            "GPU stalls on the CPU (~50× slower), so the system runs at the pace of the "
            "slowest worker. An example of when distribution is not beneficial."
        )

    # Feasibility validation: predicted vs measured for the first homogeneous
    # DDP run comparable with a single-GPU baseline (data-parallel prediction).
    if base_r.mode == "single":
        for lbl, r, sp in others:
            if r.mode == "ddp" and r.model == base_r.model and r.env == base_r.env \
                    and _prec(r) == _prec(base_r):
                pred_sp = _predicted_2gpu_speedup(r.env, r.model)
                if pred_sp:
                    err = (pred_sp - sp) / sp * 100
                    pp1, pp2, pp3 = st.columns(3)
                    pp1.metric("Predicted speedup (feasibility)", f"{pred_sp:.2f}×")
                    pp2.metric(f"Measured ({lbl})", f"{sp:.2f}×")
                    pp3.metric("Prediction error", f"{err:+.0f}%")
                    ok = abs(err) <= 15
                    (st.success if ok else st.info)(
                        f"The feasibility predicts the speedup from a **1-GPU** benchmark; "
                        f"here it is validated against the real multi-GPU run "
                        f"({'accurate' if ok else 'off'}: predicted {pred_sp:.2f}× vs "
                        f"measured {sp:.2f}×)."
                    )
                break

    # Theoretical (data-parallel) scaling — distributed runs as points at 2
    # workers vs the perfect-scaling line from the baseline.
    dist_timed = [(lbl, r, t) for lbl, r, t in timed
                  if r.mode in ("ddp", "ddp_hetero", "model_parallel")]
    if base_r.mode == "single" and dist_timed:
        st.markdown("### Theoretical vs real scaling")
        world_sizes = [1, 2, 4, 8]
        fig_scale = go.Figure()
        fig_scale.add_trace(go.Scatter(
            x=world_sizes, y=[base_t / ws / 60 for ws in world_sizes],
            name="Theoretical (100% efficiency)",
            line=dict(color=COLORS[4], width=2, dash="dash"), mode="lines+markers",
        ))
        for i, (lbl, r, t) in enumerate(dist_timed):
            fig_scale.add_trace(go.Scatter(
                x=[2], y=[t / 60], name=lbl, mode="markers",
                marker=dict(color=COLORS[i % len(COLORS)], size=14, symbol="star"),
            ))
        _n = len(dist_timed)
        fig_scale.update_layout(
            **_base_layout(320 + 20 * _n, "Epoch time vs number of workers",
                           margin=dict(l=50, r=16, t=48, b=70 + 20 * _n)),
            xaxis_title="Number of workers (processes)", yaxis_title="Minutes per epoch",
        )
        fig_scale.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.22,
                                            xanchor="left", x=0, font=dict(size=11)))
        fig_scale.update_xaxes(tickvals=world_sizes)
        _show(fig_scale, "ddp_scaling")
        st.caption(
            "The theoretical line assumes adding workers IDENTICAL to the baseline "
            "(perfect linear scaling, only valid for DATA parallelism). Real points fall "
            "below it due to communication overhead, the I/O bottleneck, imbalanced "
            "hardware (V100+CPU) or — for model parallelism — stage serialization."
        )


# ── Radar ───────────────────────────────────────────────────────────────────────

def _radar_section(compare_dfs: list[tuple[str, pd.DataFrame]]) -> None:
    st.markdown("#### Metric radar at the best epoch")
    radar_metrics = ["val_f1", "train_f1", "val_acc", "val_prec", "val_rec"]
    radar_fig = go.Figure()
    for i, (lbl, cdf) in enumerate(compare_dfs):
        vals = [
            float(v) if (v := _safe_val_at_best(cdf, "val_f1", m)) is not None else 0.0
            for m in radar_metrics
        ]
        vals_closed = vals + [vals[0]]
        radar_fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=radar_metrics + [radar_metrics[0]],
            fill="toself", name=lbl,
            line=dict(color=COLORS[i % len(COLORS)]), opacity=0.6,
        ))
    # Full-label legend below the radar: one row per run stays readable
    # even with 7-8 runs selected.
    _n_radar = len(compare_dfs)
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=380 + 20 * _n_radar,
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0),
        margin=dict(l=60, r=60, t=40, b=40 + 20 * _n_radar), paper_bgcolor="white",
        title=dict(text="Metrics at the best Val F1 epoch", font=dict(size=13)),
    )
    _show(radar_fig, "radar_comparison")


# ── Energy ──────────────────────────────────────────────────────────────────────

def _energy_section(compare_dfs: list[tuple[str, pd.DataFrame]]) -> None:
    energy_rows = []
    for lbl, cdf in compare_dfs:
        t_wh = cdf["energy_train_wh"].dropna().sum() if _has(cdf, "energy_train_wh") else 0.0
        e_wh = cdf["energy_eval_wh"].dropna().sum() if _has(cdf, "energy_eval_wh") else 0.0
        if t_wh or e_wh:
            energy_rows.append((lbl, t_wh, e_wh))
    if not energy_rows:
        return

    st.markdown("---")
    st.markdown("#### Energy consumption")
    st.caption(
        "Total energy over the whole run (Wh), as measured by pynvml on the "
        "logging GPU. Runs without energy measurement (no `--fn energy`, e.g. "
        "model-parallel) are not shown."
    )
    fig_energy = go.Figure()
    _lbls = [l for l, _, _ in energy_rows]
    fig_energy.add_trace(go.Bar(
        y=_lbls, x=[t for _, t, _ in energy_rows], name="Train",
        orientation="h", marker_color=COLORS[0],
    ))
    fig_energy.add_trace(go.Bar(
        y=_lbls, x=[e for _, _, e in energy_rows], name="Eval",
        orientation="h", marker_color=COLORS[1],
    ))
    fig_energy.update_layout(
        **_base_layout(160 + 44 * len(energy_rows), "Total energy per run (Wh)",
                       margin=dict(l=10, r=16, t=48, b=40)),
        barmode="stack", xaxis_title="Wh",
    )
    fig_energy.update_yaxes(autorange="reversed", automargin=True)
    # Outside the plot: the inside-top-left default would cover the first bar.
    fig_energy.update_layout(legend=dict(
        orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
        bgcolor="rgba(0,0,0,0)",
    ))
    _show(fig_energy, "compare_energy_total")

    _n_eff = [(l, (t + e)) for l, t, e in energy_rows]
    _best = min(_n_eff, key=lambda x: x[1])
    _worst = max(_n_eff, key=lambda x: x[1])
    if _worst[1] > 0 and _best != _worst:
        st.caption(
            f"Most efficient: **{_best[0]}** ({_best[1]:.1f} Wh) — "
            f"{_worst[1]/_best[1]:.1f}× less energy than **{_worst[0]}** "
            f"({_worst[1]:.1f} Wh)."
        )
    st.markdown("---")


# ── Per-epoch overlays ──────────────────────────────────────────────────────────

def _overlay_charts(compare_dfs: list[tuple[str, pd.DataFrame]]) -> None:
    has_energy = any(_has(d, "energy_train_wh") or _has(d, "energy_eval_wh")
                     for _, d in compare_dfs)
    _energy_opts = (["energy_train_wh", "energy_eval_wh", "power_train_w"]
                    if has_energy else [])
    metrics_to_compare = st.multiselect(
        "Metrics to overlay",
        ["val_f1", "val_loss", "train_f1", "train_loss", "val_prec", "val_rec",
         "epoch_time"] + _energy_opts,
        default=["val_f1", "val_loss"],
    )
    cols = st.columns(2)
    for idx, col_name in enumerate(metrics_to_compare):
        fig = _overlay_fig(compare_dfs, col=col_name,
                           title=col_name.replace("_", " "), y_label=col_name)
        with cols[idx % 2]:
            _show(fig, f"compare_{col_name}")
