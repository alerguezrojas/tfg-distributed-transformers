"""Compare — summary."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.web.run_registry import RunInfo
from src.web.ui.charts import (_dl_csv)
from src.web.ui.helpers import (_dur_str, _run_config, _safe_max, _safe_val_at_best)
from src.web.tabs.comparison._common import (_prec)


def _summary_table(sel: list[tuple[str, RunInfo]], df_by_label: dict[str, pd.DataFrame]) -> None:
    """One table per run: key hyperparameters (model, loss, batch, lr, data split)
    AND results (F1, epochs, duration) side by side — the apples-to-apples at a
    glance, replacing the separate config table that showed the same fields."""
    summary_rows = []
    for lbl, r in sel:
        cdf = df_by_label[lbl]
        cfg = _run_config(str(r.log_path))
        best_f1_c = _safe_max(cdf["val_f1"]) if "val_f1" in cdf.columns and not cdf.empty else float("nan")
        best_ep_c_v = _safe_val_at_best(cdf, "val_f1", "epoch")
        _last = cdf["val_f1"].dropna() if "val_f1" in cdf.columns and not cdf.empty else pd.Series(dtype=float)
        total_s_c = cdf["epoch_time"].dropna().sum() if "epoch_time" in cdf.columns else float("nan")
        summary_rows.append({
            "Run": lbl,
            "Model": r.model.replace("_patch16_224", "") if r.model else "—",
            "Mode": r.mode,
            "Precision": _prec(r),
            "Loss": cfg.get("loss", "—"),
            "Batch": cfg.get("batch", "—"),
            "LR": cfg.get("lr", "—"),
            "Train/Val": f"{cfg.get('train', '?')}/{cfg.get('val', '?')}",
            "Epochs": len(cdf),
            "Best Val F1": f"{best_f1_c:.4f}" if not pd.isna(best_f1_c) else "—",
            "Best epoch": int(best_ep_c_v) if best_ep_c_v is not None else "—",
            "Final F1": f"{_last.iloc[-1]:.4f}" if not _last.empty else "—",
            "Duration": _dur_str(total_s_c) if not pd.isna(total_s_c) else "—",
            "Environment": r.env,
        })
    sum_df = pd.DataFrame(summary_rows).set_index("Run")
    st.caption("Per-run hyperparameters and results in one table.")
    st.dataframe(sum_df, use_container_width=True)
    _dl_csv(sum_df.reset_index(), "runs_comparison.csv", "Download comparison")

