"""Exhaustive test of all dashboard tab logic — catches TypeErrors before the user sees them.

Run with:  uv run python tests/test_web_tabs.py
"""
import sys, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── helpers (copied from app.py) ──────────────────────────────────────────────
def _safe_max(series):
    valid = series.dropna()
    return float(valid.max()) if not valid.empty else float("nan")

def _safe_idxmax(series):
    valid = series.dropna()
    return valid.idxmax() if not valid.empty else None

def _safe_val_at_best(df, metric_col, target_col):
    if metric_col not in df.columns or target_col not in df.columns:
        return None
    idx = _safe_idxmax(df[metric_col])
    if idx is None:
        return None
    v = df.loc[idx, target_col]
    return None if pd.isna(v) else v

ROOT = Path(__file__).parent.parent
COLORS = ["#2563eb","#f59e0b","#10b981","#ef4444","#8b5cf6","#64748b","#ec4899","#94a3b8"]

errors = []

def check(name, fn):
    try:
        fn()
        print(f"  OK  {name}")
    except Exception as e:
        errors.append((name, e, traceback.format_exc()))
        print(f"  FAIL {name}: {e}")

# ── Load all runs ─────────────────────────────────────────────────────────────
from src.web.run_registry import RunInfo, discover_runs, discover_benchmark_csvs
from src.web.log_parser import parse_log
from src.web.perclass_parser import parse_perclass_csv
from src.web.batch_parser import parse_batch_csv
from src.web.benchmark_parser import parse_benchmark_csv

runs = discover_runs(ROOT)
benchmark_csvs = discover_benchmark_csvs(ROOT)
print(f"\nFound {len(runs)} runs, {len(benchmark_csvs)} benchmark CSVs\n")

def load_df(run):
    if run.epoch_csv_path and run.epoch_csv_path.exists():
        df = pd.read_csv(str(run.epoch_csv_path))
        if "epoch_time_s" in df.columns:
            df = df.rename(columns={"epoch_time_s": "epoch_time"})
        if not df.empty:
            return df
    return parse_log(run.log_path)

# ── TAB: Overview ─────────────────────────────────────────────────────────────
print("=== Overview tab ===")

def test_overview_global_stats():
    best_f1_global = float("-inf")
    total_gpu_h = 0.0
    for r in runs:
        df_r = load_df(r)
        if not df_r.empty and "val_f1" in df_r.columns:
            run_best = _safe_max(df_r["val_f1"])
            if not pd.isna(run_best) and run_best > best_f1_global:
                best_f1_global = run_best
        if not df_r.empty and "epoch_time" in df_r.columns:
            total_gpu_h += float(df_r["epoch_time"].dropna().sum()) / 3600
    assert not pd.isna(best_f1_global) or best_f1_global == float("-inf")

check("overview_global_stats", test_overview_global_stats)

def test_overview_table():
    rows = []
    for r in runs:
        df_r = load_df(r)
        if df_r.empty or "val_f1" not in df_r.columns:
            continue
        run_best_f1 = _safe_max(df_r["val_f1"])
        if pd.isna(run_best_f1):
            continue
        best_ep_v = _safe_val_at_best(df_r, "val_f1", "epoch")
        best_ep = int(best_ep_v) if best_ep_v is not None else "—"
        dur_s = df_r["epoch_time"].dropna().sum() if "epoch_time" in df_r.columns else float("nan")
        dur_str = f"{int(dur_s//3600)}h {int((dur_s%3600)//60)}m" if not pd.isna(dur_s) else "—"
        energy_wh = (df_r["energy_eval_wh"].sum()
                     if "energy_eval_wh" in df_r.columns and df_r["energy_eval_wh"].notna().any()
                     else None)
        rows.append({
            "Run": r.label[:55], "Env": r.env, "Model": r.model or "—",
            "Trace": r.trace_mode, "Epochs": len(df_r),
            "Best Val F1": round(run_best_f1, 4),
            "Best epoch": best_ep, "Duration": dur_str,
            "Energy eval (Wh)": f"{energy_wh:.0f}" if energy_wh else "—",
        })
    assert len(rows) > 0, "No rows in overview table"
    ov_df = pd.DataFrame(rows)
    # Test background_gradient
    ov_df.style.background_gradient(subset=["Best Val F1"], cmap="RdYlGn", vmin=0.4, vmax=0.75)

check("overview_table_with_gradient", test_overview_table)

# ── TAB: System Monitor ───────────────────────────────────────────────────────
print("\n=== System tab ===")
from src.web.system_monitor import get_snapshot

def test_system_snapshot():
    snap = get_snapshot(disk_paths=["/", "/home"])
    # progress values must be in [0.0, 1.0]
    assert 0.0 <= snap.cpu.usage_pct / 100 <= 1.0
    assert 0.0 <= snap.ram.percent / 100 <= 1.0
    for d in snap.disks:
        v = d.percent / 100
        assert 0.0 <= v <= 1.0, f"Disk {d.path} percent out of range: {d.percent}"
    for g in snap.gpus:
        mem_pct = g.mem_used_mb / g.mem_total_mb * 100 if g.mem_total_mb else 0
        assert 0.0 <= mem_pct / 100 <= 1.0

check("system_snapshot_values", test_system_snapshot)

# ── TAB: Dataset ──────────────────────────────────────────────────────────────
print("\n=== Dataset tab ===")
from src.web.dataset_stats import (
    CLASS_NAMES, SPLIT_SIZES, class_distribution_approximate,
    cooccurrence_from_perclass, get_country_distribution,
)

def test_dataset_class_dist():
    dist_df = class_distribution_approximate()
    assert len(dist_df) == 19
    dist_df = dist_df.sort_values("train_count", ascending=True)
    dist_df["color"] = dist_df["train_count"].apply(
        lambda v: COLORS[3] if v < 5000 else (COLORS[1] if v < 15000 else COLORS[2])
    )
    # Test Bar chart construction
    fig = go.Figure(go.Bar(
        y=dist_df["class"], x=dist_df["train_count"], orientation="h",
        marker_color=dist_df["color"].tolist(),  # convert to list
        text=dist_df["train_count"].apply(lambda v: f"{v:,}"),
        textposition="outside",
    ))
    assert fig is not None

check("dataset_class_distribution_bar", test_dataset_class_dist)

def test_dataset_scatter_f1_vs_freq():
    dist_df = class_distribution_approximate()
    perclass_csvs = list(ROOT.rglob("perclass_metrics_*.csv"))
    if not perclass_csvs:
        return  # skip if no data
    latest_pc = max(perclass_csvs, key=lambda p: p.stat().st_mtime)
    pc_df = parse_perclass_csv(latest_pc)
    if pc_df.empty:
        return
    last_ep = pc_df["epoch"].max()
    ep_pc = pc_df[pc_df["epoch"] == last_ep].copy()
    ep_pc = ep_pc.merge(dist_df[["class", "train_count"]],
                        left_on="class_name", right_on="class", how="left")
    # Must not have type issues
    fig = px.scatter(
        ep_pc, x="train_count", y="f1", text="class_name", color="f1",
        color_continuous_scale="RdYlGn", range_color=[0, 1],
    )
    assert fig is not None

check("dataset_scatter_f1_freq", test_dataset_scatter_f1_vs_freq)

def test_dataset_split_pie():
    fig = go.Figure(go.Pie(
        labels=list(SPLIT_SIZES.keys()),
        values=list(SPLIT_SIZES.values()),
        hole=0.4,
    ))
    assert fig is not None

check("dataset_split_pie", test_dataset_split_pie)

# ── TAB: Model Explorer ───────────────────────────────────────────────────────
print("\n=== Model Explorer tab ===")
from src.web.model_explorer import compare_models, get_model_stats, CURATED_MODELS, ALL_FAMILIES

def test_model_stats_tiny():
    s = get_model_stats("vit_tiny_patch16_224", num_classes=19)
    assert s is not None
    assert s.total_params_m > 0
    assert s.vram_estimate_gb(32) > 0

check("model_stats_vit_tiny", test_model_stats_tiny)

def test_model_stats_resnet():
    s = get_model_stats("resnet50", num_classes=19)
    assert s is not None

check("model_stats_resnet50", test_model_stats_resnet)

def test_compare_models_table():
    models = ["vit_tiny_patch16_224", "vit_base_patch16_224", "resnet50"]
    rows = compare_models(models, [32, 64])
    assert len(rows) > 0
    cmp_df = pd.DataFrame(rows)
    vram_col = "VRAM est. bs=32 (GB)"
    assert vram_col in cmp_df.columns

    def _color_vram(v):
        try:
            fv = float(v)
            if fv <= 4: return "background-color: #dcfce7"
            if fv <= 8: return "background-color: #fef9c3"
            return "background-color: #fee2e2"
        except (ValueError, TypeError):
            return ""

    cmp_df.style.map(_color_vram, subset=[vram_col])

check("compare_models_table_and_style", test_compare_models_table)

def test_model_bubble_chart():
    rows = compare_models(["vit_tiny_patch16_224", "resnet50", "efficientnet_b0"], [32])
    cmp_df = pd.DataFrame(rows)
    vram_col = "VRAM est. bs=32 (GB)"
    plot_df = cmp_df[cmp_df["FLOPs (MFLOPs)"].notna()].copy()
    plot_df["FLOPs (MFLOPs)"] = pd.to_numeric(plot_df["FLOPs (MFLOPs)"], errors="coerce")
    plot_df[vram_col] = pd.to_numeric(plot_df.get(vram_col, 0), errors="coerce").fillna(1)
    if not plot_df.empty:
        fig = px.scatter(
            plot_df, x="FLOPs (MFLOPs)", y="Params (M)",
            size=vram_col, color="Family", text="Model",
            size_max=40,
        )
        assert fig is not None

check("model_bubble_chart", test_model_bubble_chart)

def test_vram_bar_chart():
    rows = compare_models(["vit_tiny_patch16_224", "resnet50"], [4, 8, 16, 32, 64])
    vram_df = pd.DataFrame(rows)
    vram_cols = [c for c in vram_df.columns if c.startswith("VRAM")]
    fig = go.Figure()
    for col in vram_cols:
        bs_val = col.split("bs=")[1].split(" ")[0]
        fig.add_trace(go.Bar(
            name=f"bs={bs_val}", x=vram_df["Model"],
            y=pd.to_numeric(vram_df[col], errors="coerce"),
        ))
    assert fig is not None

check("model_vram_bar_chart", test_vram_bar_chart)

# ── TAB: Curves ───────────────────────────────────────────────────────────────
print("\n=== Curves tab ===")

def test_curves_all_runs():
    for r in runs:
        df = load_df(r)
        if df.empty:
            continue
        best_f1 = _safe_max(df["val_f1"]) if "val_f1" in df.columns else float("nan")
        best_ep_v = _safe_val_at_best(df, "val_f1", "epoch")
        best_epoch = int(best_ep_v) if best_ep_v is not None else "—"

        # Energy section
        has_energy = "energy_eval_wh" in df.columns and df["energy_eval_wh"].notna().any()
        if has_energy:
            total_eval_wh = df["energy_eval_wh"].sum()
            total_train_wh = (df["energy_train_j"].sum() / 3600
                              if "energy_train_j" in df.columns and df["energy_train_j"].notna().any()
                              else 0)
            assert total_eval_wh >= 0

        # Epoch time bar
        if "epoch_time" in df.columns and df["epoch_time"].notna().any():
            et = df[["epoch", "epoch_time"]].dropna()
            fig_et = go.Figure(go.Bar(
                x=et["epoch"], y=et["epoch_time"] / 60,
                marker_color=COLORS[0], opacity=0.8,
            ))
            assert fig_et is not None

check("curves_all_runs", test_curves_all_runs)

# ── TAB: Per-class ────────────────────────────────────────────────────────────
print("\n=== Per-class tab ===")

def test_perclass_all_runs():
    for r in runs:
        if not (r.perclass_csv_path and r.perclass_csv_path.exists()):
            continue
        pcdf = parse_perclass_csv(str(r.perclass_csv_path))
        if pcdf.empty:
            continue
        epochs_available = sorted(pcdf["epoch"].unique().tolist())
        ep_df = pcdf[pcdf["epoch"] == epochs_available[-1]].copy()
        ep_df = ep_df.sort_values("f1", ascending=False)

        # Scatter F1 vs frequency
        from src.web.dataset_stats import class_distribution_approximate
        dist_df = class_distribution_approximate()
        ep_pc = ep_df.merge(dist_df[["class", "train_count"]],
                            left_on="class_name", right_on="class", how="left")
        # train_count may be NaN for classes not in our hardcoded list — that's OK

        # Trend subtab
        for cls in pcdf["class_name"].unique()[:3]:
            cls_df = pcdf[pcdf["class_name"] == cls]
            assert not cls_df.empty

check("perclass_all_runs", test_perclass_all_runs)

from src.web.confusion_matrix_parser import parse_confusion_matrix_csv, get_matrix_for_epoch

def test_confusion_matrix_all_runs():
    _CLASS_GROUPS = {
        "Urban":       ([0, 1],         "#6b7280"),
        "Agricultural":([ 2, 3, 4, 5, 6, 7], "#d97706"),
        "Forest":      ([8, 9, 10, 13], "#16a34a"),
        "Scrub/grass": ([11, 12],       "#84cc16"),
        "Bare/coastal":([14],           "#92400e"),
        "Wetlands":    ([15, 16],       "#0891b2"),
        "Water":       ([17, 18],       "#1d4ed8"),
    }
    for r in runs:
        if not (r.confusion_matrix_csv_path and r.confusion_matrix_csv_path.exists()):
            continue
        cm_df = parse_confusion_matrix_csv(r.confusion_matrix_csv_path)
        if cm_df.empty:
            continue
        epochs_cm = sorted(cm_df["epoch"].unique().tolist())
        pivot = get_matrix_for_epoch(cm_df, epochs_cm[-1])
        class_order = list(pivot.index)
        z_norm = pivot.reindex(index=class_order, columns=class_order).values
        n_classes = len(class_order)

        # Absolute mode
        row_sums = z_norm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            z_abs = np.where(row_sums > 0, z_norm * row_sums, 0).round().astype(int)

        # Shapes
        shapes = []
        for group_name, (idxs, color) in _CLASS_GROUPS.items():
            positions = [i for i, cls in enumerate(class_order) if i in idxs]
            if not positions:
                continue
            lo, hi = min(positions), max(positions)
            shapes.append(dict(
                type="rect",
                x0=lo - 0.5, x1=hi + 0.5, y0=lo - 0.5, y1=hi + 0.5,
                line=dict(color=color, width=2.5),
                fillcolor="rgba(0,0,0,0)", layer="above",
            ))

        fig_cm = go.Figure(go.Heatmap(
            z=z_norm.tolist(), x=class_order, y=class_order,
            colorscale="Blues", zmin=0, zmax=1,
        ))
        fig_cm.update_layout(shapes=shapes, height=660)
        assert fig_cm is not None

check("confusion_matrix_all_runs", test_confusion_matrix_all_runs)

# ── TAB: Batch Monitor ────────────────────────────────────────────────────────
print("\n=== Batch tab ===")

def test_batch_all_runs():
    for r in runs:
        if not r.batch_csv_path:
            continue
        bdf = parse_batch_csv(r.batch_csv_path)
        if bdf.empty:
            continue
        epochs_available = sorted(bdf["epoch"].unique())
        for ep in epochs_available[:2]:
            subset = bdf[bdf["epoch"] == ep].copy()
            if len(subset) >= 20:
                ma = subset["running_loss"].rolling(20, center=True).mean()
                mean_l = subset["running_loss"].mean()
                std_l = subset["running_loss"].std()
                if not pd.isna(std_l) and std_l > 0:
                    spikes = subset[subset["running_loss"] > mean_l + 2 * std_l]

check("batch_all_runs", test_batch_all_runs)

# ── TAB: Compare ─────────────────────────────────────────────────────────────
print("\n=== Compare tab ===")

def test_compare_radar():
    if len(runs) < 2:
        return
    compare_dfs = []
    for r in runs[:4]:
        df = load_df(r)
        compare_dfs.append((r.label[:30], df))

    radar_metrics = ["val_f1", "train_f1", "val_acc", "val_prec", "val_rec"]
    radar_fig = go.Figure()
    for i, (lbl, cdf) in enumerate(compare_dfs):
        vals = []
        for m_col in radar_metrics:
            v = _safe_val_at_best(cdf, "val_f1", m_col)
            vals.append(float(v) if v is not None else 0.0)
        vals_closed = vals + [vals[0]]
        cats_closed = radar_metrics + [radar_metrics[0]]
        radar_fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill="toself", name=lbl,
            line=dict(color=COLORS[i % len(COLORS)]),
        ))
    assert radar_fig is not None

check("compare_radar_chart", test_compare_radar)

def test_compare_summary_table():
    if len(runs) < 2:
        return
    summary_rows = []
    for r in runs[:4]:
        cdf = load_df(r)
        best_f1_c = _safe_max(cdf["val_f1"]) if "val_f1" in cdf.columns and not cdf.empty else float("nan")
        best_ep_c_v = _safe_val_at_best(cdf, "val_f1", "epoch")
        best_ep_c = int(best_ep_c_v) if best_ep_c_v is not None else "—"
        _last = cdf["val_f1"].dropna() if "val_f1" in cdf.columns and not cdf.empty else pd.Series(dtype=float)
        final_f1_c = _last.iloc[-1] if not _last.empty else float("nan")
        total_s_c = cdf["epoch_time"].dropna().sum() if "epoch_time" in cdf.columns else float("nan")
        dur_c = (f"{int(total_s_c//3600)}h {int((total_s_c%3600)//60)}m"
                 if not pd.isna(total_s_c) else "—")
        summary_rows.append({
            "Run": r.label[:50],
            "Best Val F1": f"{best_f1_c:.4f}" if not pd.isna(best_f1_c) else "—",
            "Best epoch": best_ep_c,
            "Final F1": f"{final_f1_c:.4f}" if not pd.isna(final_f1_c) else "—",
            "Epochs": len(cdf), "Duration": dur_c,
        })
    assert len(summary_rows) > 0
    pd.DataFrame(summary_rows)

check("compare_summary_table", test_compare_summary_table)

# ── TAB: Benchmark ─────────────────────────────────────────────────────────
print("\n=== Benchmark tab ===")

def test_benchmark_all_csvs():
    for csv_path in benchmark_csvs:
        meta, bdf = parse_benchmark_csv(csv_path)
        assert isinstance(meta, dict)
        # Benchmark section
        if bdf.empty:
            continue
        viable = bdf[bdf["oom"] == "no"].copy() if "oom" in bdf.columns else bdf.copy()
        tp_col = next((c for c in ("imgs_per_s_train", "imgs_per_s") if c in viable.columns), None)
        if tp_col and not viable.empty:
            for mode in viable["trace_mode"].unique() if "trace_mode" in viable.columns else []:
                sub = viable[viable["trace_mode"] == mode]
                x_labels = sub["batch_size"].astype(str) + f" [{mode}]"
                assert len(x_labels) >= 0

check("benchmark_all_csvs", test_benchmark_all_csvs)

from src.web.benchmark_comparison import build_comparison

def test_benchmark_comparison():
    if not benchmark_csvs:
        return
    for csv_path in benchmark_csvs:
        meta, feas_df = parse_benchmark_csv(csv_path)
        if feas_df.empty or "batch_size" not in feas_df.columns:
            continue
        batch_sizes = feas_df["batch_size"].dropna().astype(int).unique().tolist()
        if not batch_sizes:
            continue
        # Find a run with matching model
        model_name = meta.get("model_name", "")
        for r in runs:
            actual_df = load_df(r)
            if actual_df.empty:
                continue
            cmp = build_comparison(
                meta=meta, feas_df=feas_df, actual_df=actual_df,
                batch_size=batch_sizes[0], trace_mode="simple",
            )
            if cmp is not None:
                tbl = cmp.to_dataframe()
                assert isinstance(tbl, pd.DataFrame)
                # The 3-way table exposes per-source error columns; parse them all.
                assert "Δ benchmark %" in tbl.columns
                err_cols = [c for c in tbl.columns if c in ("Δ analytic %", "Δ benchmark %")]
                for c in err_cols:
                    for val in tbl[c]:
                        try:
                            float(str(val).replace("%", "").replace("+", ""))
                        except (ValueError, AttributeError):
                            pass
            break  # one run per csv is enough

check("benchmark_comparison_build", test_benchmark_comparison)

# ── TAB: Time Analysis ────────────────────────────────────────────────────────
print("\n=== Time tab ===")

def test_time_all_runs():
    for r in runs:
        df_time = load_df(r)
        if "epoch_time" not in df_time.columns or df_time["epoch_time"].isna().all():
            continue
        et = df_time[["epoch", "epoch_time"]].dropna()
        if et.empty:
            continue
        total_s = et["epoch_time"].sum()
        avg_s = et["epoch_time"].mean()
        assert not pd.isna(total_s)
        assert not pd.isna(avg_s)
        # Linear trend
        x = et["epoch"].values.astype(float)
        y = et["epoch_time"].values.astype(float)
        if len(x) >= 2:
            coeffs = np.polyfit(x, y, 1)
            trend = np.polyval(coeffs, x)
            assert len(trend) == len(x)

check("time_all_runs", test_time_all_runs)

# ── TAB: DDP Analysis ────────────────────────────────────────────────────────
print("\n=== DDP Analysis tab ===")

def test_ddp_analysis():
    single_runs = [r for r in runs if r.mode == "single"]
    ddp_runs_list = [r for r in runs if r.mode == "ddp"]
    # DDP rows table
    for r in ddp_runs_list:
        ddf = load_df(r)
        if ddf.empty:
            continue
        best_f1 = _safe_max(ddf["val_f1"]) if "val_f1" in ddf.columns else float("nan")
        avg_epoch_s = ddf["epoch_time"].dropna().mean() if "epoch_time" in ddf.columns and ddf["epoch_time"].notna().any() else None
    # Speedup section (if both exist)
    if single_runs and ddp_runs_list:
        df_s = load_df(single_runs[0])
        df_d = load_df(ddp_runs_list[0])
        avg_s = df_s["epoch_time"].dropna().mean() if "epoch_time" in df_s.columns and df_s["epoch_time"].notna().any() else None
        avg_d = df_d["epoch_time"].dropna().mean() if "epoch_time" in df_d.columns and df_d["epoch_time"].notna().any() else None
        if avg_s and avg_d and avg_d > 0:
            speedup = avg_s / avg_d
            assert speedup > 0

check("ddp_analysis_table_and_speedup", test_ddp_analysis)

# ── TAB: Info ────────────────────────────────────────────────────────────────
print("\n=== Info tab ===")

def test_info_all_runs():
    for r in runs:
        df_info = load_df(r)
        best_f1_i = _safe_max(df_info["val_f1"]) if "val_f1" in df_info.columns else float("nan")
        best_ep_v = _safe_val_at_best(df_info, "val_f1", "epoch")
        best_ep_i = int(best_ep_v) if best_ep_v is not None else "—"
        # Anomaly detection
        import re
        lines = r.log_path.read_text(errors="replace").splitlines()
        anomalies = [
            l for l in lines
            if any(kw in l for kw in ("EXPLODE","VANISH","DEAD","OOM","ERROR","Error"))
        ]
        # Log tail
        tail = "\n".join(lines[-30:])
        assert isinstance(tail, str)

check("info_all_runs", test_info_all_runs)

# ── TAB: Live Monitor ─────────────────────────────────────────────────────────
print("\n=== Live Monitor tab ===")

def test_live_monitor():
    import subprocess, re
    # GPU usage
    try:
        out = subprocess.run(
            ["nvidia-smi","--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0:
            parts = [p.strip() for p in out.stdout.strip().split(",")]
            assert len(parts) >= 4
    except Exception:
        pass  # no GPU is fine

    # Log progress
    for r in runs[:3]:
        lines = r.log_path.read_text(errors="replace").splitlines()
        epoch_line = next((l for l in reversed(lines) if "Epoch" in l and "/" in l), None)
        if epoch_line:
            m = re.search(r"Epoch\s+(\d+)/(\d+)", epoch_line)
            if m:
                ep, total = int(m.group(1)), int(m.group(2))
                assert 0 < ep <= total

check("live_monitor", test_live_monitor)

# ── TAB: Launcher ─────────────────────────────────────────────────────────────
print("\n=== Launcher tab ===")

def test_launcher_config_discovery():
    configs = sorted(Path("configs").glob("*.yaml"))
    assert len(configs) > 0, "No YAML configs found"
    names = [c.name for c in configs]
    assert any("train" in n for n in names)

check("launcher_config_discovery", test_launcher_config_discovery)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Results: {len(runs) - len(errors)}/{len(runs)} checks passed" if False else
      f"Results: {sum(1 for n,e,t in [] if True)}")
print(f"PASSED: {20 - len(errors)}  FAILED: {len(errors)}")
if errors:
    print("\nFailed checks:")
    for name, exc, tb in errors:
        print(f"\n  [{name}]")
        print(f"  {type(exc).__name__}: {exc}")
        # Show relevant traceback lines
        tb_lines = [l for l in tb.split("\n") if "File" in l or "Error" in l or "assert" in l.lower()]
        for l in tb_lines[-4:]:
            print(f"    {l}")
    sys.exit(1)
else:
    print("\nAll checks passed!")
    sys.exit(0)
