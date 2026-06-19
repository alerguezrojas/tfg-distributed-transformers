"""Compare — summary."""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from src.web.run_registry import RunInfo
from src.web.ui.charts import (_dl_csv)
from src.web.ui.helpers import (_dur_str, _run_config, _safe_max, _safe_val_at_best)
from src.web.tabs.comparison._common import (_prec)


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

