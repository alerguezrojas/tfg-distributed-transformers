"""Live — follow a training run while it happens.

Reads the ``events_*.jsonl`` stream that rank 0 appends during training (see
``src/web/events_reader.py``). Read-only, like the rest of the web: it never
trains and never touches the GPU. Only the fragment below re-runs on the timer,
so the sidebar and the rest of the page stay still.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.events_reader import (
    COMPLETED, EARLY_STOPPED, FAILED, FINAL_STATES, RUNNING, STALLED, WAITING,
    EventsFile, EventsFollower, RunState, discover_event_files,
)
from src.web.ui import helpers
from src.web.ui.charts import COLORS, _base_layout, _metric_fig, _show
from src.web.ui.context import DashboardContext

_INTERVALS_S = [2, 5, 10, 30]
_DEFAULT_INTERVAL_S = 5
_DEFAULT_STALLED_MIN = 10
_FOLLOWERS_KEY = "_live_followers"
_NO_VALUE = "—"
_METRIC_NAMES = {"f1": "F1", "f1_optimal": "F1 (optimal thr.)", "auc": "AUC"}

# Streamlit colored-text names per derived state.
_STATUS_STYLE = {
    RUNNING: ("blue", "Running"), WAITING: ("gray", "Waiting for events"),
    COMPLETED: ("green", "Completed"), EARLY_STOPPED: ("orange", "Early stopped"),
    FAILED: ("red", "Failed"), STALLED: ("red", "Stalled"),
}


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return _NO_VALUE
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {sec:02d}s" if m else f"{sec}s"


def _fmt(value, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else _NO_VALUE


def _follower(path) -> EventsFollower:
    followers = st.session_state.setdefault(_FOLLOWERS_KEY, {})
    key = str(path)
    if key not in followers:
        followers[key] = EventsFollower(path)
    return followers[key]


def _epoch_fraction(state: RunState) -> float:
    """Progress inside the epoch being trained (0..1), from the latest batch event."""
    if not state.batches or state.current_epoch != state.epochs_done + 1:
        return 0.0
    last = state.batches[-1]
    b, n = last.get("batch"), last.get("n_batches")
    if last.get("epoch") == state.current_epoch and isinstance(b, int) and n:
        return min(b / n, 1.0)
    return 0.0


def _delta(rows: list[dict], key: str) -> float | None:
    """Change of ``key`` between the last two finished epochs."""
    vals = [r.get(key) for r in rows[-2:]]
    if len(vals) == 2 and all(isinstance(v, (int, float)) for v in vals):
        return vals[1] - vals[0]
    return None


def _header(state: RunState, status: str, eta: float | None) -> None:
    color, text = _STATUS_STYLE[status]
    start = state.run_start or {}
    parts = [start.get("dataset"), start.get("model"), start.get("mode")]
    ws = start.get("world_size")
    if isinstance(ws, int) and ws > 1:
        parts.append(f"{ws} processes")
    st.markdown(f":{color}[**{text}**] &nbsp; " + " · ".join(str(p) for p in parts if p))

    planned = state.planned_epochs
    if planned:
        frac = ((state.epochs_done + _epoch_fraction(state)) / planned
                if status not in FINAL_STATES else state.progress or 0.0)
        cur = state.current_epoch or state.epochs_done
        label = f"Epoch {min(cur, planned)}/{planned} · {state.epochs_done} finished"
        if eta is not None:
            label += f" · ETA ≤ {_fmt_duration(eta)}"
        st.progress(min(max(frac, 0.0), 1.0), text=label)
    if status == STALLED:
        st.warning("No new events and no run_end: the run may have been interrupted.")
    if state.run_end and state.run_end.get("error"):
        st.error(str(state.run_end["error"]))


def _kpis(state: RunState, rows: list[dict]) -> None:
    val = state.last_val
    cols = st.columns(6)
    cols[0].metric("Epochs done", f"{state.epochs_done}/{state.planned_epochs or '?'}")
    metric = state.select_metric
    cols[1].metric(f"Best {_METRIC_NAMES.get(metric, metric)}" if metric else "Best metric",
                   _fmt(state.best))
    for col, (label, key, inverse) in zip(cols[2:5], (
            ("Val loss", "val_loss", True), ("Val F1", "val_f1", False),
            ("Val AUC", "val_auc", False))):
        d = _delta(rows, key)
        col.metric(label, _fmt(val.get(key[4:])),
                   delta=None if d is None else f"{d:+.4f}",
                   delta_color="inverse" if inverse else "normal")
    cols[5].metric("Mean epoch time", _fmt_duration(state.mean_epoch_time_s))


def _curves(rows: list[dict]) -> None:
    if not rows:
        st.info("No epoch has finished yet.")
        return
    df = pd.DataFrame(rows)
    left, right = st.columns(2)
    with left:
        _show(_metric_fig(df, "train_loss", "val_loss", "Loss per epoch", "Loss", height=280),
              "live_loss")
    with right:
        if "val_f1" in df.columns or "val_auc" in df.columns:
            _show(_metric_fig(df, "val_f1", "val_auc", "Validation F1 / AUC", "Score",
                              name_train="Val F1", name_val="Val AUC", height=280),
                  "live_scores")


def _batch_chart(state: RunState) -> None:
    df = pd.DataFrame(state.batch_rows())
    if df.empty or "progress" not in df.columns:
        return
    fig = go.Figure()
    for i, (col, name) in enumerate((("running_loss", "Running mean loss"),
                                     ("batch_loss", "Batch loss"))):
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(x=df["progress"], y=df[col], name=name, mode="lines",
                                     line={"color": COLORS[i]}))
    if not fig.data:
        return
    fig.update_layout(**_base_layout(260, "Loss per batch"), xaxis_title="Epochs elapsed",
                      yaxis_title="Loss", hovermode="x unified")
    _show(fig, "live_batch_loss")
    st.caption(f"Last {len(df)} logged batches (the trainer logs every N-th batch).")


def _details(state: RunState) -> None:
    start = state.run_start
    if not start:
        return
    with st.expander("Configuration and environment"):
        env = start.get("env") or {}
        if env:
            st.markdown("**Environment**")
            st.dataframe(pd.DataFrame([{"key": k, "value": ", ".join(map(str, v))
                                        if isinstance(v, list) else str(v)}
                                       for k, v in env.items()]),
                         hide_index=True, width="stretch")
        st.markdown("**Config**")
        st.json(start.get("config") or {}, expanded=False)
        classes = start.get("classes") or []
        st.caption(f"Task: {start.get('task', _NO_VALUE)} · {len(classes)} classes · "
                   f"selection metric: {state.select_metric or _NO_VALUE}")
    if state.n_ignored:
        st.caption(f"{state.n_ignored} line(s) ignored (unknown event types or unreadable).")


def _view(path, stalled_s: float, armed: bool) -> None:
    """The auto-refreshing body: poll the file, fold the new events, draw."""
    follower = _follower(path)
    follower.poll()
    state = follower.state
    status = state.status(stalled_after_s=stalled_s)
    # The page armed the timer while the run was going; once it ends, one full
    # rerun switches the timer off (a finished run needs no polling).
    if armed and status in FINAL_STATES:
        st.rerun(scope="app")

    _header(state, status, state.eta_s(stalled_after_s=stalled_s))
    rows = state.epoch_rows()
    _kpis(state, rows)
    _curves(rows)
    _batch_chart(state)
    _details(state)


def render(ctx: DashboardContext) -> None:
    st.markdown("## Live")
    st.caption("Follows a training run in progress from its event stream "
               "(`events_*.jsonl`, written by the trainer). Read-only.")

    files: list[EventsFile] = discover_event_files(helpers.ROOT)
    if not files:
        st.info("No live events found under `logs/`. Start a training from the terminal: "
                "the trainer writes `events_<timestamp>.jsonl` next to the run's log.")
        return

    c_run, c_live, c_int, c_stall, c_now = st.columns([4, 1, 1.4, 1.4, 1.2],
                                                       vertical_alignment="bottom")
    by_path = {str(f.path): f for f in files}
    chosen = c_run.selectbox(
        "Run", list(by_path), key="live_run",
        format_func=lambda p: ("● " if by_path[p].active else "") + by_path[p].label,
        help="Active runs first (● = being written now).")
    live = c_live.toggle("Live", value=True, key="live_on",
                         help="Off pauses the automatic refresh.")
    interval = c_int.selectbox("Refresh every", _INTERVALS_S,
                               index=_INTERVALS_S.index(_DEFAULT_INTERVAL_S),
                               format_func=lambda s: f"{s} s", key="live_interval")
    stalled_min = c_stall.number_input("Stalled after (min)", min_value=1,
                                       value=_DEFAULT_STALLED_MIN, key="live_stalled_min")
    c_now.button("Refresh now")      # any interaction reruns the page and re-polls

    stalled_s = float(stalled_min) * 60
    # Poll once here so we know whether the run is still going before arming the timer.
    follower = _follower(by_path[chosen].path)
    follower.poll()
    armed = live and follower.state.status(stalled_after_s=stalled_s) not in FINAL_STATES

    st.fragment(run_every=f"{interval}s" if armed else None)(_view)(
        by_path[chosen].path, stalled_s, armed)
