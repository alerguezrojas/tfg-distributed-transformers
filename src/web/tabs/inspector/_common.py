"""Shared pieces of the Inspector sub-views: the run context, cached loaders, hints and
small drawing helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis.io import EpochData, MonitorSet, load_epoch, load_monitor_set
from src.web.events_reader import RunState
from src.web.inspector_data import outcome_card
from src.web.ui.charts import _show

MONITOR_HINT = ("Enable the prediction monitor to record this: `--monitor` in the training "
                "script (or `monitor.enabled: true` in the config).")
LAYERS_HINT = ("Enable the per-layer statistics with `monitor.layers: true` (together with "
               "`--monitor`); they are logged every `log_batch_every` batches.")
SYSTEM_HINT = ("The trainer emits `system` events every `monitor.system_interval_s` seconds "
               "(set it above 0 and use a `--trace` other than `off`).")
BATCH_HINT = ("Per-batch events need the batch monitor: `--layers batch-monitor` "
              "(`log_batch_every` sets how often).")

GALLERY_COLS = 6


# ── Cached loaders (key: path + mtime, so a rewritten file is re-read) ───────────────

def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_monitor(dir_str: str, mtime: float) -> MonitorSet | None:
    try:
        return load_monitor_set(dir_str)
    except Exception:           # missing file, half-written .npz (BadZipFile), bad keys…
        return None


@st.cache_data(show_spinner=False, max_entries=48)
def _cached_epoch(dir_str: str, epoch: int, mtime: float) -> EpochData | None:
    try:
        return load_epoch(dir_str, epoch)
    except Exception:
        return None


def load_monitor(inspect_dir: Path | None) -> MonitorSet | None:
    if inspect_dir is None:
        return None
    return _cached_monitor(str(inspect_dir), _mtime(inspect_dir / "monitor_set.npz"))


def load_epoch_cached(inspect_dir: Path, epoch: int) -> EpochData | None:
    return _cached_epoch(str(inspect_dir), epoch,
                         _mtime(inspect_dir / f"epoch_{epoch:03d}.npz"))


# ── Context ─────────────────────────────────────────────────────────────────────────

@dataclass
class InspectorCtx:
    """What every sub-view needs about the selected run and epoch."""

    state: RunState | None                  # None: inspection directory without events
    inspect_dir: Path | None
    monitor: MonitorSet | None
    epochs: list[int]                       # epochs with an inspection file, ascending
    epoch: int | None                       # the epoch the per-epoch views show
    task: str = "multilabel"
    classes: list[str] = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def has_monitor(self) -> bool:
        return self.monitor is not None and bool(self.epochs) and self.epoch is not None

    def epoch_data(self, epoch: int | None = None) -> EpochData | None:
        epoch = self.epoch if epoch is None else epoch
        if self.inspect_dir is None or epoch is None:
            return None
        return load_epoch_cached(self.inspect_dir, epoch)

    def previous_epoch(self) -> int | None:
        earlier = [e for e in self.epochs if self.epoch is not None and e < self.epoch]
        return earlier[-1] if earlier else None

    def stack(self, key: str = "probs") -> tuple[list[int], np.ndarray]:
        """(epochs, array) of ``key`` over every loadable epoch up to the selected one."""
        eps, arrs = [], []
        for e in self.epochs:
            if self.epoch is not None and e > self.epoch:
                break
            d = self.epoch_data(e)
            if d is not None and getattr(d, key) is not None:
                eps.append(e)
                arrs.append(np.asarray(getattr(d, key), dtype=np.float32))
        return eps, (np.stack(arrs) if arrs else np.empty((0, 0, 0)))


def need_monitor(c: InspectorCtx) -> bool:
    """Shows the standard 'no data' message and returns False when there is no inspection."""
    if c.has_monitor and c.epoch_data() is not None:
        return True
    st.info("No inspection data for this run yet. " + MONITOR_HINT)
    return False


def thumb_of(c: InspectorCtx, pos: int) -> np.ndarray:
    thumbs = c.monitor.thumbs if c.monitor is not None else None
    if thumbs is None:
        return np.full((64, 64, 3), 128, dtype=np.uint8)
    return np.asarray(thumbs[pos])


# ── Drawing helpers ─────────────────────────────────────────────────────────────────

def card_grid(cards: Sequence[tuple[np.ndarray, str]], cols: int = GALLERY_COLS) -> None:
    """Images with captions in a grid of ``cols`` columns."""
    for start in range(0, len(cards), cols):
        row = st.columns(cols)
        for col, (img, caption) in zip(row, cards[start:start + cols]):
            col.image(img, caption=caption, width="stretch")


def sample_cards(c: InspectorCtx, table: pd.DataFrame, extra: str | None = None
                 ) -> list[tuple[np.ndarray, str]]:
    """Gallery cards (green/red frame) for the rows of a prediction table."""
    cards = []
    for r in table.itertuples():
        cap = f"#{r.pos} · true: {r.true} · pred: {r.pred} · conf {r.confidence:.2f}"
        if extra:
            cap += f" · {extra}: {getattr(r, extra):.3f}"
        cards.append((outcome_card(thumb_of(c, r.pos), bool(r.correct)), cap))
    return cards


def lines(df: pd.DataFrame, x: str, y: str, color: str | None, title: str, ytitle: str,
          key: str, height: int = 260, hline: float | None = None,
          xtitle: str | None = None) -> None:
    """Line chart, one trace per value of ``color``."""
    fig = go.Figure()
    groups = [(None, df)] if color is None else list(df.groupby(color, sort=True))
    for name, part in groups:
        fig.add_trace(go.Scatter(x=part[x], y=part[y], mode="lines", name=str(name or ytitle),
                                 line={"width": 1.6}))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dot", opacity=0.6)
    fig.update_layout(height=height, title={"text": title}, xaxis_title=xtitle,
                      yaxis_title=ytitle, hovermode="x unified", showlegend=color is not None)
    _show(fig, key)


def fmt(v: Any, pattern: str = "{:.3f}") -> str:
    return pattern.format(v) if isinstance(v, (int, float)) and v == v else "—"
