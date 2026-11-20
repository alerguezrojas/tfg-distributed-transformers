"""Compare — summary."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.web.run_meta import AUC, MetricInfo
from src.web.seed_agg import Series, fmt_mean_std, run_numbers
from src.web.tabs.comparison._common import _prec
from src.web.ui.charts import _dl_csv
from src.web.ui.helpers import _dur_str

_DASH = "—"


def _cell(values: list[float], digits: int = 4) -> str:
    """'0.7216' for one run, '0.7216 ± 0.0112' for several (strings: one dtype for Arrow)."""
    return fmt_mean_std(values, digits)


def _epochs_cell(values: list[float]) -> str:
    """An epoch count: a plain integer for one run, 'mean ± std' for several."""
    return str(int(values[0])) if len(values) == 1 else _cell(values, 1)


def _hms(seconds: list[float]) -> str:
    vals = [s for s in seconds if s == s]
    if not vals:
        return _DASH
    mean = sum(vals) / len(vals)
    return _dur_str(mean) + ("" if len(vals) == 1 else " (mean)")


def _same(values: list) -> str:
    """The value shared by every run of the series; the distinct ones joined otherwise."""
    shown = sorted({str(v) for v in values if v not in (None, "")})
    return "/".join(shown) if shown else _DASH


def summary_frame(series: list[Series], metric: MetricInfo) -> pd.DataFrame:
    """One row per series: results first (the metric must stay in view), then what the
    run was (hyperparameters). Several runs → mean ± std of every numeric column."""
    show_auc = metric.key != AUC.key and any(
        "val_auc" in d.columns and d["val_auc"].notna().any() for s in series for d in s.dfs)
    rows = []
    for s in series:
        nums = [run_numbers(d, metric.column) for d in s.dfs]
        metas = [r.meta for r in s.runs if r.meta]
        auc_best = [run_numbers(d, AUC.column).get("best") for d in s.dfs]
        row = {
            "Run": s.label,
            f"Best Val {metric.name}": _cell([n["best"] for n in nums if "best" in n]),
            "Best epoch": _epochs_cell([n["best_epoch"] for n in nums if "best_epoch" in n]),
            f"Final {metric.name}": _cell([n["final"] for n in nums if "final" in n]),
            **({"Best Val AUC": _cell([a for a in auc_best if a is not None])}
               if show_auc else {}),
            "Seeds": ", ".join(str(x) for x in s.seeds) if s.seeds else _DASH,
            "Epochs": _epochs_cell([n["epochs"] for n in nums if "epochs" in n]),
            "Duration": _hms([n["duration_s"] for n in nums if "duration_s" in n]),
            "Model": (s.runs[0].model or _DASH).replace("_patch16_224", ""),
            "Mode": s.runs[0].mode,
            "Precision": _prec(s.runs[0]),
            "Loss": _same([m.effective_loss for m in metas]),
            "Providers": _same([m.providers_label for m in metas]),
            "Color jitter": _same([m.color_jitter for m in metas if m.color_jitter]),
            "Input px": _same([m.img_size for m in metas]),
            "Batch": _same([m.batch_size for m in metas]),
            "LR": _same([m.lr for m in metas]),
            "Train/Val": _same([f"{m.n_train}/{m.n_val}" for m in metas if m.n_train]),
            "Environment": s.runs[0].env,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_table(series: list[Series], metric: MetricInfo) -> None:
    """One table per series: results first and hyperparameters after, side by side — the
    apples-to-apples at a glance. Series of several seeds show mean ± std."""
    sum_df = summary_frame(series, metric)
    n_agg = sum(1 for s in series if s.n > 1)
    st.caption("Per-run results and hyperparameters in one table. "
               + (f"{n_agg} row(s) aggregate runs that differ only in the seed: "
                  "values are mean ± sample std." if n_agg else ""))
    st.dataframe(
        sum_df, width="stretch", hide_index=True,
        column_config={
            "Run": st.column_config.TextColumn("Run", width="large", pinned=True),
            f"Best Val {metric.name}": st.column_config.TextColumn(
                f"Best Val {metric.name}", width="medium"),
        })
    _dl_csv(sum_df, "runs_comparison.csv", "Download comparison")
