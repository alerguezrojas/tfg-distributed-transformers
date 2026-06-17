"""Tab render module — see src/web/app.py for the orchestrator."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.web.confusion_matrix_parser import get_matrix_for_epoch, parse_confusion_matrix_csv
from src.web.dataset_stats import (
    CLASS_NAMES, SPLIT_SIZES,
    class_distribution_approximate, class_distribution_from_parquet,
    get_country_distribution, find_example_patches, load_rgb_image,
)
from src.web.feasibility_comparison import build_comparison
from src.web.feasibility_parser import parse_feasibility_csv, parse_ddp_scenarios
from src.web.model_explorer import ALL_FAMILIES, CURATED_MODELS, compare_models
from src.web.perclass_parser import parse_perclass_csv
from src.web.run_import import import_run_archive, import_run_folder, summarize_import
from src.web.run_registry import RunInfo

from src.web.ui.charts import (
    COLORS, _show, _dl_csv, _base_layout, _metric_fig, _overlay_fig,
    _CLASS_GROUPS, _CLASS_GROUP_COLOR,
)
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import (
    ROOT, _load_df, _load_batch, _load_perclass, _get_runs, _get_feasibility_csvs,
    _feas_label, _run_config, _load_class_distribution, _load_example_images,
    _safe_max, _safe_idxmax, _safe_val_at_best, _throughput_col, _dur_str,
    _get_configs, _detect_anomalies, _read_log_tail, _parse_log_progress,
    _gpu_usage, _launch_process, _color_f1_cell,
)


def render(ctx: DashboardContext) -> None:
    st.markdown("## Data & runs")
    st.caption("BigEarthNet-S2 dataset explorer, timm model comparison and "
               "importer for runs trained elsewhere.")
    sub = st.tabs(["Dataset", "Models", "Import runs"])
    with sub[0]:
        _dataset(ctx)
    with sub[1]:
        _models(ctx)
    with sub[2]:
        _import_runs(ctx)


def _import_runs(ctx: DashboardContext) -> None:
    """Import the artifacts of a run trained on Kaggle / the cluster (zip or
    folder) into logs/ so the dashboard discovers it. Moved here from System."""
    st.markdown("### Import runs trained elsewhere")
    st.caption(
        "Trainings run on Kaggle or the cluster and produce a `logs/` folder. "
        "Download it as a zip (or copy it to this machine) and import it here — "
        "the artifacts are copied into the repo's `logs/` tree and appear in the "
        "dashboard immediately."
    )
    logs_root = ROOT / "logs"

    st.markdown("#### From a zip file")
    uploaded = st.file_uploader(
        "Drop a zip of the run's `logs/` folder (or its contents)",
        type=["zip"], accept_multiple_files=False,
    )
    if uploaded is not None and st.button("Import zip", type="primary"):
        try:
            rel = import_run_archive(uploaded.getvalue(), logs_root)
        except Exception as e:
            st.error(f"Could not read the zip: {e}")
            rel = []
        _report_import(rel)

    st.markdown("#### From a folder on this machine")
    st.caption("Useful when you already copied the `logs/` folder somewhere "
               "(for example via `scp` from the cluster).")
    folder_str = st.text_input("Folder path", placeholder="/home/alejandro/Downloads/kaggle_logs")
    if folder_str and st.button("Import folder"):
        folder = Path(folder_str).expanduser()
        if not folder.is_dir():
            st.error(f"Not a folder: {folder}")
        else:
            _report_import(import_run_folder(folder, logs_root))

    runs = ctx.runs
    envs = sorted({r.env for r in runs})
    c1, c2, c3 = st.columns(3)
    c1.metric("Runs indexed", len(runs))
    c2.metric("Feasibility reports", len(_get_feasibility_csvs()))
    c3.metric("Environments", ", ".join(envs) if envs else "—")


def _report_import(rel_paths: list[str]) -> None:
    if not rel_paths:
        st.warning("No recognizable artifacts found "
                   "(expected train_*.log / *_metrics_*.csv / confusion_matrix_*.csv / "
                   "feasibility_*).")
        return
    s = summarize_import(rel_paths)
    _get_runs.clear()
    _get_feasibility_csvs.clear()
    st.success(
        f"Imported {s['total']} file(s): {s['runs']} run log(s), "
        f"{s['metric_csvs']} metric CSV(s), {s['feasibility']} feasibility report(s)."
    )
    with st.expander("Imported files"):
        for p in rel_paths:
            st.write(f"`logs/{p}`")
    st.info("Select the new run in the sidebar to explore it.")


def _dataset(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    run = ctx.run
    refresh_interval = ctx.refresh_interval
    st.markdown("### Data explorer — BigEarthNet-S2 v2.0")

    meta_path: Path | None = None
    for candidate in [
        "/media/alejandro/SSD/datasets/bigearthnet/metadata.parquet",
        "/home/bejeque/alu0101317038/datasets/bigearthnet/metadata.parquet",
    ]:
        if Path(candidate).exists():
            meta_path = Path(candidate)
            break

    st.markdown("### Dataset splits")
    split_col, pie_col = st.columns([1, 1])
    with split_col:
        ds1, ds2 = st.columns(2)
        ds1.metric("Train", f"{SPLIT_SIZES['train']:,}")
        ds2.metric("Validation", f"{SPLIT_SIZES['val']:,}")
        ds3, ds4 = st.columns(2)
        ds3.metric("Test", f"{SPLIT_SIZES['test']:,}")
        ds4.metric("Total patches", f"{sum(SPLIT_SIZES.values()):,}")
    with pie_col:
        fig_splits = go.Figure(go.Pie(
            labels=["Train", "Validation", "Test"],
            values=list(SPLIT_SIZES.values()),
            hole=0.45,
            marker_colors=[COLORS[0], COLORS[2], COLORS[1]],
            textinfo="label+percent",
        ))
        fig_splits.update_layout(
            **_base_layout(260, "Split distribution", margin=dict(l=10, r=10, t=40, b=10)),
            showlegend=False,
        )
        _show(fig_splits, "splits")

    # ── Class distribution ──────────────────────────────────────────────────────
    st.markdown("### Class distribution (train split)")
    dist_df = None
    if meta_path:
        dist_df = _load_class_distribution(str(meta_path))
        if dist_df is not None:
            st.caption(f"Source: {meta_path.name} (real multi-label count)")
        else:
            st.caption("Could not read the parquet — using approximate statistics.")
    if dist_df is None:
        dist_df = class_distribution_approximate()
        if not meta_path:
            st.caption("metadata.parquet not found — using approximate statistics.")

    dist_df = dist_df.sort_values("train_count", ascending=True).reset_index(drop=True)
    dist_df["color"] = dist_df["train_count"].apply(
        lambda v: COLORS[3] if v < 10000 else (COLORS[1] if v < 40000 else COLORS[2])
    )
    fig_dist = go.Figure(go.Bar(
        y=dist_df["class"], x=dist_df["train_count"],
        orientation="h",
        marker_color=dist_df["color"].tolist(),
        text=dist_df["train_count"].apply(lambda v: f"{v:,}").tolist(),
        textposition="outside",
        cliponaxis=False,
    ))
    fig_dist.update_layout(
        **_base_layout(620, "Samples per class (train)", margin=dict(l=300, r=90, t=40, b=40)),
        xaxis_title="Number of samples (multi-label)", yaxis_title="",
    )
    fig_dist.update_yaxes(tickfont=dict(size=11), automargin=True)
    max_x = dist_df["train_count"].max()
    fig_dist.update_xaxes(range=[0, max_x * 1.15])
    _show(fig_dist, "class_distribution")
    st.caption(
        "Red = rare class (<10K), Orange = moderate (<40K), Green = frequent. "
        "Being multi-label, the sum of labels exceeds the number of patches. "
        "Rare classes cap the macro-F1 ceiling."
    )

    st.markdown("### Class imbalance")
    max_c = dist_df["train_count"].max()
    min_c = dist_df["train_count"].min()
    ratio = max_c / min_c if min_c > 0 else float("inf")
    ci1, ci2, ci3 = st.columns(3)
    ci1.metric("Most frequent class", dist_df.iloc[-1]["class"][:28],
               f"{int(max_c):,}")
    ci2.metric("Rarest class", dist_df.iloc[0]["class"][:28],
               f"{int(min_c):,}")
    ci3.metric("Imbalance ratio", f"{ratio:.1f}×")
    _dl_csv(dist_df[["class", "train_count"]], "class_distribution.csv",
            "Download distribution")

    # ── Example images per class ────────────────────────────────────────────────
    st.markdown("### Example images per class")
    if not meta_path:
        st.info("Requires dataset access (metadata.parquet not found).")
    else:
        # Path to the dataset root directory (next to the parquet)
        ds_root = meta_path.parent / "BigEarthNet-S2"
        if not ds_root.exists():
            st.info(f"Dataset directory not found at {ds_root}.")
        else:
            sel_class = st.selectbox(
                "Class", CLASS_NAMES,
                index=CLASS_NAMES.index("Marine waters") if "Marine waters" in CLASS_NAMES else 0,
            )
            with st.spinner("Loading RGB images from the dataset…"):
                examples = _load_example_images(str(meta_path), str(ds_root), sel_class, n=4)
            if examples:
                st.caption(
                    "RGB proxy (Sentinel-2 bands B04/B03/B02) with percentile stretch "
                    "for visibility. Each patch is 120×120 px (~1.2 km²)."
                )
                img_cols = st.columns(len(examples))
                for col, (pid, img) in zip(img_cols, examples):
                    col.image(img, caption=pid.split("_")[-2] + "_" + pid.split("_")[-1],
                              use_container_width=True)
            else:
                st.warning(
                    f"Could not load images for '{sel_class}'. "
                    "Is the dataset complete and accessible?"
                )

    # ── Country ──────────────────────────────────────────────────────────────────
    if meta_path:
        country_counts = get_country_distribution(meta_path)
        if country_counts is not None and not country_counts.empty:
            st.markdown("### Distribution by country (train)")
            top_n = country_counts.head(15).sort_values(ascending=True)
            fig_c = go.Figure(go.Bar(
                x=top_n.values, y=top_n.index, orientation="h",
                marker_color=COLORS[0], opacity=0.85,
                text=[f"{v:,}" for v in top_n.values], textposition="outside",
                cliponaxis=False,
            ))
            fig_c.update_layout(
                **_base_layout(420, "Top 15 countries by number of patches",
                               margin=dict(l=120, r=80, t=40, b=40)),
                xaxis_title="Patches", yaxis_title="",
            )
            fig_c.update_xaxes(range=[0, top_n.values.max() * 1.15])
            _show(fig_c, "countries")

    # ── Difficulty vs frequency ──────────────────────────────────────────────────
    st.markdown("### Per-class difficulty vs frequency")
    st.caption("Crosses each class's frequency with its validation F1 (most recent per-class CSV).")
    perclass_csvs_all = sorted(ROOT.rglob("perclass_metrics_*.csv"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
    if perclass_csvs_all:
        pc_df = parse_perclass_csv(perclass_csvs_all[0])
        if not pc_df.empty:
            last_ep = pc_df["epoch"].max()
            ep_pc = pc_df[pc_df["epoch"] == last_ep].copy()
            ep_pc = ep_pc.merge(dist_df[["class", "train_count"]],
                                left_on="class_name", right_on="class", how="left")
            fig_sc = px.scatter(
                ep_pc, x="train_count", y="f1",
                text="class_name", color="f1",
                color_continuous_scale="RdYlGn", range_color=[0, 1],
                labels={"train_count": "Training samples", "f1": "Val F1"},
                title=f"F1 vs class frequency (epoch {last_ep})",
            )
            fig_sc.update_traces(textposition="top center", textfont_size=9)
            fig_sc.update_layout(
                **_base_layout(480, margin=dict(l=60, r=40, t=40, b=50)),
                showlegend=False,
            )
            fig_sc.update_yaxes(range=[-0.05, 1.05])
            _show(fig_sc, "f1_vs_frequency")
            st.caption(
                "Classes with few samples tend to have low F1. "
                "The points at the bottom-left are the hardest to improve."
            )
    else:
        st.info("No per-class CSV found. Run a training with `--layers confusion`.")



def _models(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    run = ctx.run
    refresh_interval = ctx.refresh_interval
    st.markdown("### Model explorer")
    st.caption("Explore timm models and compare parameters and VRAM requirements.")

    col_fam, col_bs = st.columns([3, 1])
    with col_fam:
        selected_families = st.multiselect(
            "Model families", ALL_FAMILIES, default=["ViT", "ResNet", "EfficientNet"],
        )
    with col_bs:
        cmp_batch = st.selectbox("Batch size for VRAM estimate", [4, 8, 16, 32, 64, 128], index=3)

    candidate_models = []
    for fam in selected_families:
        candidate_models.extend(CURATED_MODELS.get(fam, []))

    extra_model = st.text_input("Add a custom timm model", placeholder="e.g. convnext_large")
    if extra_model.strip():
        candidate_models.append(extra_model.strip())

    if not candidate_models:
        st.info("Select at least one family.")
    else:
        with st.spinner("Loading model statistics…"):
            rows = compare_models(candidate_models, [cmp_batch], num_classes=19)

        if not rows:
            st.warning("Could not load any model.")
        else:
            cmp_df = pd.DataFrame(rows)
            vram_col = f"VRAM est. bs={cmp_batch} (GB)"

            def _color_vram(v):
                try:
                    fv = float(v)
                    if fv <= 4:
                        return "background-color: #dcfce7"
                    if fv <= 8:
                        return "background-color: #fef9c3"
                    return "background-color: #fee2e2"
                except (ValueError, TypeError):
                    return ""

            styled_cmp = cmp_df.style
            if vram_col in cmp_df.columns:
                styled_cmp = styled_cmp.map(_color_vram, subset=[vram_col])
            st.dataframe(styled_cmp, use_container_width=True, hide_index=True)
            st.caption("Green = fits in 4 GB | Orange = 4–8 GB | Red = >8 GB (RTX 3060 Ti limit)")
            _dl_csv(cmp_df, "model_comparison.csv", "Download model comparison")

            st.markdown("### Parameters vs FLOPs")
            plot_df = cmp_df[cmp_df["FLOPs (MFLOPs)"] != "—"].copy()
            if not plot_df.empty:
                plot_df["FLOPs (MFLOPs)"] = pd.to_numeric(plot_df["FLOPs (MFLOPs)"], errors="coerce")
                plot_df[vram_col] = pd.to_numeric(plot_df.get(vram_col, 0), errors="coerce").fillna(1)
                fig_bubble = px.scatter(
                    plot_df, x="FLOPs (MFLOPs)", y="Params (M)",
                    size=vram_col, color="Family", text="Model", hover_name="Model",
                    size_max=40,
                    labels={"FLOPs (MFLOPs)": "FLOPs per image (MFLOPs)", "Params (M)": "Parameters (M)"},
                )
                fig_bubble.update_traces(textposition="top center", textfont_size=8)
                fig_bubble.update_layout(**_base_layout(420, "Model complexity"), showlegend=True)
                _show(fig_bubble, "model_complexity")
                st.caption("Bubble size = estimated VRAM at the selected batch size.")

            st.markdown("### Required VRAM by batch size")
            vram_models = candidate_models[:8]
            with st.spinner("Computing VRAM estimates…"):
                vram_rows = compare_models(vram_models, [4, 8, 16, 32, 64, 128])

            if vram_rows:
                vram_df = pd.DataFrame(vram_rows)
                vram_cols_list = [c for c in vram_df.columns if c.startswith("VRAM")]
                fig_vram_m = go.Figure()
                for col_v in vram_cols_list:
                    bs_val = col_v.split("bs=")[1].split(" ")[0]
                    fig_vram_m.add_trace(go.Bar(
                        name=f"bs={bs_val}", x=vram_df["Model"],
                        y=pd.to_numeric(vram_df[col_v], errors="coerce"),
                    ))
                fig_vram_m.add_hline(y=8, line_dash="dash", line_color="red",
                                     annotation_text="RTX 3060 Ti (8 GB)")
                fig_vram_m.add_hline(y=32, line_dash="dash", line_color="orange",
                                     annotation_text="V100 (32 GB)")
                fig_vram_m.update_layout(
                    **_base_layout(380, "Estimated VRAM (GB) by batch size"),
                    barmode="group", xaxis_tickangle=30, yaxis_title="GB",
                )
                _show(fig_vram_m, "vram_by_batch")

            st.markdown("### Quick launch")
            selected_for_launch = st.selectbox(
                "Select a model to prefill the Launcher", [r["Model"] for r in rows]
            )
            if selected_for_launch:
                st.info(f"Go to the **Launcher** tab and use the model `{selected_for_launch}`.")
                st.session_state["preselected_model"] = selected_for_launch

