"""Feasibility — ddp."""
from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.feasibility_parser import (parse_ddp_scenarios)
from src.web.ui.charts import (COLORS, _base_layout, _dl_csv, _show)


def render_ddp_analysis(meta, feasibility_csvs) -> None:
    if not feasibility_csvs:
        st.info("Run the feasibility analysis first.")
    else:
        st.markdown("## Distributed scaling (predicted)")
        st.caption(
            "Compares configurations from 1 to 8 GPUs showing batch size, recommended workers, "
            "expected speedup, scaling efficiency and the identified bottleneck."
        )
        ddp_df = parse_ddp_scenarios(meta)

        if ddp_df.empty:
            st.info(
                "No DDP data in this report. "
                "Regenerate the analysis with the current version of check_feasibility.py."
            )
        else:
            # ── Scenario table ─────────────────────────────────────────────
            st.markdown("### Scenario table")

            def _color_bottleneck(val: str) -> str:
                if val == "io":
                    return "background-color: #fee2e2; color: #991b1b"
                if val == "sync":
                    return "background-color: #fef3c7; color: #92400e"
                return "background-color: #d1fae5; color: #065f46"

            if "bottleneck" in ddp_df.columns:
                styled_ddp = ddp_df.style.map(_color_bottleneck, subset=["bottleneck"])
                if "speedup" in ddp_df.columns:
                    styled_ddp = styled_ddp.background_gradient(
                        subset=["speedup"], cmap="RdYlGn", vmin=1.0, vmax=float(ddp_df["n_gpus"].max() or 8)
                    )
            else:
                styled_ddp = ddp_df.style
            st.dataframe(styled_ddp, use_container_width=True, hide_index=True)
            _dl_csv(ddp_df, "ddp_scenarios.csv", "Download DDP scenarios")

            # ── Load distribution rectangles ───────────────────────────────
            st.markdown("### Load distribution per GPU")
            st.caption(
                "Each bar shows the share of batch time: "
                "compute (green), data I/O (orange), gradient synchronization (red)."
            )

            # Calcular proporciones por GPU
            if {"speedup", "sync_overhead_pct", "n_gpus"}.issubset(ddp_df.columns):
                viable_ddp = ddp_df[pd.to_numeric(ddp_df["n_gpus"], errors="coerce") > 0].copy()
                for col in ["sync_overhead_pct", "speedup", "n_gpus"]:
                    viable_ddp[col] = pd.to_numeric(viable_ddp[col], errors="coerce")

                # Estimate I/O overhead from the ratio if available
                io_ratio = float(meta.get("dataset", {}).get("io_bottleneck_ratio", 0) or 0)
                io_pct_est = min(io_ratio * 30, 50)  # Estimate: if ratio=1, I/O ≈ 30% of the time

                fig_rect = go.Figure()
                labels = [f"{int(row['n_gpus'])} GPU(s)" for _, row in viable_ddp.iterrows()]
                sync_pcts = viable_ddp["sync_overhead_pct"].fillna(0).tolist()
                compute_pcts = [max(0, 100 - s - io_pct_est) for s in sync_pcts]
                io_pcts = [io_pct_est] * len(labels)

                fig_rect.add_trace(go.Bar(
                    name="GPU compute", x=labels, y=compute_pcts,
                    marker_color=COLORS[2], opacity=0.85,
                ))
                fig_rect.add_trace(go.Bar(
                    name="Data I/O", x=labels, y=io_pcts,
                    marker_color=COLORS[1], opacity=0.85,
                ))
                fig_rect.add_trace(go.Bar(
                    name="Gradient sync", x=labels, y=sync_pcts,
                    marker_color=COLORS[3], opacity=0.85,
                ))
                fig_rect.update_layout(
                    **_base_layout(360, "Batch time breakdown (%) — estimate"),
                    barmode="stack",
                    xaxis_title="DDP configuration",
                    yaxis_title="Percentage of batch time",
                )
                fig_rect.update_yaxes(range=[0, 100])
                _show(fig_rect, "ddp_load_distribution")

                # ── Speedup: feasibility estimate vs selectable scaling laws ──
                st.markdown("### Speedup: estimate vs scaling laws")
                if "speedup" in viable_ddp.columns:
                    from src.estimation_models import SPEEDUP_MODELS, speedup_curve
                    n_gpus_vals = viable_ddp["n_gpus"].tolist()
                    speedup_vals = viable_ddp["speedup"].tolist()

                    st.caption(
                        "The green line is the feasibility's own **compute/IO-aware** estimate "
                        "(includes NFS I/O and gradient-sync overhead). Overlay one or more "
                        "analytic scaling laws to compare, and tune the serial fraction *s* "
                        "used by Amdahl/Gustafson."
                    )
                    msc, mslider = st.columns([3, 2])
                    with msc:
                        chosen = st.multiselect(
                            "Scaling laws to overlay",
                            list(SPEEDUP_MODELS.keys()),
                            default=["linear", "amdahl"],
                            format_func=lambda k: SPEEDUP_MODELS[k].name,
                            key="ddp_speedup_models",
                        )
                    with mslider:
                        serial_frac = st.slider(
                            "Serial fraction s (Amdahl / Gustafson)",
                            0.0, 0.5, 0.05, 0.01, key="ddp_serial_frac",
                        )

                    fig_su = go.Figure()
                    fig_su.add_trace(go.Scatter(
                        x=n_gpus_vals, y=speedup_vals,
                        name="Feasibility (compute/IO-aware)",
                        mode="lines+markers",
                        line=dict(color=COLORS[2], width=3),
                        marker=dict(size=10),
                    ))
                    _dash = {"linear": "dash", "amdahl": "dot", "gustafson": "dashdot"}
                    for i, key in enumerate(chosen):
                        fig_su.add_trace(go.Scatter(
                            x=n_gpus_vals,
                            y=speedup_curve(key, n_gpus_vals, serial_frac),
                            name=SPEEDUP_MODELS[key].name,
                            mode="lines+markers",
                            line=dict(color=COLORS[(i + 3) % len(COLORS)], width=2,
                                      dash=_dash.get(key, "dash")),
                        ))
                    fig_su.update_layout(
                        **_base_layout(340, "Speedup vs number of GPUs"),
                        xaxis_title="Number of GPUs",
                        yaxis_title="Speedup",
                    )
                    fig_su.update_xaxes(tickvals=n_gpus_vals)
                    _show(fig_su, "ddp_speedup")
                    if chosen:
                        st.caption("  ·  ".join(
                            f"**{SPEEDUP_MODELS[k].name}:** `{SPEEDUP_MODELS[k].formula}`"
                            for k in chosen
                        ))

                # ── Estimated total time per configuration ─────────────────
                if "time_total_h" in viable_ddp.columns:
                    st.markdown("### Estimated total time per configuration")
                    viable_ddp["time_total_h_num"] = pd.to_numeric(viable_ddp["time_total_h"], errors="coerce")
                    fig_tt = go.Figure(go.Bar(
                        x=labels,
                        y=viable_ddp["time_total_h_num"].tolist(),
                        marker_color=[COLORS[i % len(COLORS)] for i in range(len(labels))],
                        opacity=0.85,
                        text=[f"{v:.1f}h" for v in viable_ddp["time_total_h_num"].tolist()],
                        textposition="outside",
                    ))
                    fig_tt.update_layout(
                        **_base_layout(280, "Total training time (h)"),
                        xaxis_title="DDP configuration",
                        yaxis_title="Hours",
                    )
                    _show(fig_tt, "ddp_total_time")


