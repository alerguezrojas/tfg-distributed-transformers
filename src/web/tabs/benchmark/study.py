"""Benchmark — study."""
from __future__ import annotations


import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.web.ui.charts import (COLORS, _base_layout, _show)


def render_study(meta, benchmark_csvs) -> None:
    if not benchmark_csvs:
        st.info("Run the benchmark analysis first.")
    else:
        study = meta.get("study")
        if not study:
            st.info(
                "This report does not include an empirical study. To generate it, run the "
                "analysis with `--convergence-study` (real mini-training with LR range test "
                "and gradient noise scale)."
            )
        else:
            st.markdown("## Empirical convergence study")
            st.caption(
                "Real measurements on this machine via a short mini-training, "
                "not extrapolation from historical data."
            )

            # ── LR range test ──────────────────────────────────────────────
            lr_data = study.get("lr", {})
            lr_lrs = study.get("lr_curve_lrs", [])
            lr_losses = study.get("lr_curve_losses", [])
            if lr_data and lr_lrs and lr_losses:
                st.markdown("### LR range test")
                sug = float(lr_data.get("suggested_lr", 0) or 0)
                minl = float(lr_data.get("min_loss_lr", 0) or 0)
                lr1, lr2, lr3 = st.columns(3)
                lr1.metric("Suggested LR", f"{sug:.2e}")
                lr2.metric("Min-loss LR", f"{minl:.2e}")
                div = lr_data.get("diverged_lr", "")
                lr3.metric("Divergence LR", f"{float(div):.2e}" if div else "—")

                fig_lr = go.Figure()
                fig_lr.add_trace(go.Scatter(
                    x=lr_lrs, y=lr_losses, mode="lines+markers",
                    line=dict(color=COLORS[0], width=2), name="Loss",
                ))
                if sug > 0:
                    fig_lr.add_vline(x=sug, line_dash="dash", line_color=COLORS[2],
                                     annotation_text=f"Suggested {sug:.1e}",
                                     annotation_position="top")
                fig_lr.update_layout(
                    **_base_layout(340, "Loss vs Learning Rate (sweep)"),
                    xaxis_title="Learning rate (log)", yaxis_title="Loss",
                )
                fig_lr.update_xaxes(type="log")
                _show(fig_lr, "lr_range_test")
                st.caption(
                    "The suggested LR is where the loss drops fastest (the steepest "
                    "negative-slope region), typically ~1 order below the minimum."
                )

            # ── Curva de convergencia medida ───────────────────────────────
            conv = study.get("conv", {})
            conv_steps = study.get("conv_steps", [])
            conv_losses = study.get("conv_losses", [])
            conv_f1s = study.get("conv_f1s", [])
            if conv and conv_steps:
                st.markdown("### Measured convergence curve")
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("Fit R²", f"{float(conv.get('r_squared', 0) or 0):.3f}")
                cc2.metric("Estimated Val F1", f"{float(conv.get('best_f1', 0) or 0):.3f}")
                cc3.metric("Plateau (epoch)", conv.get("epochs_to_plateau", "—"))
                cc4.metric("Real throughput", f"{float(conv.get('measured_imgs_per_s', 0) or 0):.0f} img/s")

                # Measured loss curve + extrapolated power-law fit
                fig_conv = go.Figure()
                fig_conv.add_trace(go.Scatter(
                    x=conv_steps, y=conv_losses, mode="markers",
                    marker=dict(color=COLORS[0], size=5), name="Measured loss",
                ))
                # Fitted curve a·t^-b+c
                a = float(conv.get("fit_a", 0) or 0)
                b = float(conv.get("fit_b", 0) or 0)
                c = float(conv.get("fit_c", 0) or 0)
                if a > 0 and conv_steps:
                    t_fit = np.linspace(min(conv_steps), max(conv_steps) * 3, 80)
                    y_fit = a * np.power(t_fit, -b) + c
                    fig_conv.add_trace(go.Scatter(
                        x=t_fit, y=y_fit, mode="lines",
                        line=dict(color=COLORS[1], width=2, dash="dash"),
                        name=f"Fit a·t^-b+c (R²={float(conv.get('r_squared',0) or 0):.2f})",
                    ))
                fig_conv.update_layout(
                    **_base_layout(360, "Measured loss + power-law fit"),
                    xaxis_title="Step", yaxis_title="Loss BCE",
                )
                _show(fig_conv, "convergence_loss")

                # Measured F1 per step
                if conv_f1s:
                    fig_cf1 = go.Figure(go.Scatter(
                        x=conv_steps, y=conv_f1s, mode="lines+markers",
                        line=dict(color=COLORS[2], width=2), marker=dict(size=4),
                        name="Train F1 (batch)",
                    ))
                    fig_cf1.update_layout(
                        **_base_layout(280, "F1 per step (mini-training)"),
                        xaxis_title="Step", yaxis_title="F1 (batch)",
                    )
                    fig_cf1.update_yaxes(range=[0, 1])
                    _show(fig_cf1, "convergence_f1")

                st.caption(
                    f"Loss extrapolated to 1 epoch: {float(conv.get('loss_1ep', 0) or 0):.4f} | "
                    f"final: {float(conv.get('loss_final', 0) or 0):.4f}. "
                    "The power-law fit (loss = a·t⁻ᵇ + c) models the initial drop; "
                    "it is extrapolated to the target number of epochs to estimate F1."
                )

            # ── Gradient noise scale ───────────────────────────────────────
            grad = study.get("grad", {})
            if grad:
                st.markdown("### Gradient noise scale")
                gg1, gg2, gg3 = st.columns(3)
                gg1.metric("Gradient norm",
                           f"{float(grad.get('grad_norm_mean', 0) or 0):.3f} "
                           f"± {float(grad.get('grad_norm_std', 0) or 0):.3f}")
                gg2.metric("Suggested batch size", grad.get("suggested_batch_size", "—"))
                gg3.metric("Coeff. of variation", f"{float(grad.get('cv', 0) or 0):.3f}")
                st.caption(
                    "The gradient noise scale (McCandlish 2018) estimates the critical batch size: "
                    "above it, increasing the batch yields diminishing returns. "
                    "A high CV indicates noisy gradients (suggests a larger batch)."
                )


