"""Parses the held-out test-set CSVs written by scripts/eval.py.

The file has per-class rows followed by a trailing aggregate comment line::

    class_idx,class_name,f1,precision,recall
    0,Urban fabric,0.4324,0.2857,0.8889
    ...
    # aggregate,loss=0.265,f1_t05=0.277,f1_opt=0.386,threshold=0.25,...

``parse_eval_csv`` returns ``(per_class_df, aggregate_dict)`` so the dashboard
can show the held-out number next to the validation curves. Pure + testable.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_eval_csv(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Return (per-class DataFrame, aggregate dict) from an eval.py CSV.

    The aggregate dict has float values for keys like ``loss``, ``f1_t05``,
    ``f1_opt``, ``threshold``, ``accuracy``, ``precision``, ``recall``.
    Missing/garbled files yield an empty frame and an empty dict.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return pd.DataFrame(), {}

    # Per-class rows: every non-comment, non-blank CSV line. comment='#' skips
    # the trailing aggregate line; pandas tolerates the blank line before it.
    try:
        per_class = pd.read_csv(path, comment="#", skip_blank_lines=True)
    except Exception:
        per_class = pd.DataFrame()
    if "f1" not in per_class.columns:
        per_class = pd.DataFrame()

    aggregate: dict = {}
    for line in text.splitlines():
        s = line.strip().lstrip("#").strip()
        if not s.lower().startswith("aggregate"):
            continue
        for tok in s.split(",")[1:]:
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            try:
                aggregate[k.strip()] = float(v)
            except ValueError:
                aggregate[k.strip()] = v.strip()
        break

    return per_class, aggregate
