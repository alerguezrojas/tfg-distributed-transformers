"""Inspector page — analyse a run while it trains and afterwards.

Reads two things the trainer writes and never touches the GPU: the event stream
(``events_*.jsonl``: system, throughput, layer statistics) and the inspection directory
(``inspect_<ts>/``: predictions, embeddings, Grad-CAM per epoch; contract in
docs/inspeccion.md). One module per sub-view; the heavy numerics live in ``src/analysis`` and
the pure helpers in ``src/web/inspector_data.py``.

Only the sub-view that is selected is computed (a segmented control, not ``st.tabs``, which
would run all ten on every refresh), and inspection files are cached by (path, mtime).
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.analysis.io import list_epochs
from src.web.events_reader import FINAL_STATES, EventsFollower
from src.web.inspector_data import (
    InspectorRun, discover_inspector_runs, resolve_inspect_dir,
)
from src.web.tabs.inspector import (
    alerts, calibration, clinical, curves, dynamics, embeddings, errors, explain, layers,
    performance, predictions, subgroups, system,
)
from src.web.tabs.inspector._common import InspectorCtx, load_groups_cached, load_monitor
from src.web.tabs.live import _FOLLOWERS_KEY, _STATUS_STYLE, _fmt_duration
from src.web.ui import helpers
from src.web.ui.alert_banner import render_alert_banner
from src.web.ui.context import DashboardContext

_INTERVALS_S = [2, 5, 10, 30]
_DEFAULT_INTERVAL_S = 5
_DEFAULT_STALLED_MIN = 3

VIEWS = {
    "System": system.render, "Performance": performance.render,
    "Predictions": predictions.render, "Errors": errors.render, "Curves": curves.render,
    "Clinical": clinical.render, "Calibration": calibration.render,
    "Subgroups": subgroups.render, "Dynamics": dynamics.render,
    "Embeddings": embeddings.render, "Explain": explain.render, "Layers": layers.render,
    "Alerts": alerts.render,
}
_EPOCH_VIEWS = {"Predictions", "Errors", "Curves", "Clinical", "Calibration", "Subgroups",
                "Dynamics", "Embeddings", "Explain"}


def view_names(c: InspectorCtx) -> list[str]:
    """The sub-views on offer: Subgroups only appears when the run has ``groups.npz``."""
    return [v for v in VIEWS if v != "Subgroups" or c.groups]


def _follower(path: Path) -> EventsFollower:
    followers = st.session_state.setdefault(_FOLLOWERS_KEY, {})
    key = str(path)
    if key not in followers:
        followers[key] = EventsFollower(path)
    return followers[key]


def _status_line(run: InspectorRun, state, status: str | None, n_epochs: int) -> None:
    parts = []
    if status is not None:
        color, text = _STATUS_STYLE[status]
        parts.append(f":{color}[**{text}**]")
    start = (state.run_start if state else None) or {}
    parts += [str(p) for p in (start.get("dataset"), start.get("model"), start.get("mode")) if p]
    if state is not None and state.planned_epochs:
        parts.append(f"epoch {state.epochs_done}/{state.planned_epochs}")
        eta = state.eta_s()
        if eta is not None:
            parts.append(f"ETA ≤ {_fmt_duration(eta)}")
    parts.append(f"{n_epochs} inspected epoch(s)" if n_epochs else "no inspection data")
    st.markdown(" · ".join(parts))


def _context(run: InspectorRun, state) -> InspectorCtx:
    inspect_dir = run.inspect_dir
    if inspect_dir is None and state is not None:
        inspect_dir = resolve_inspect_dir(run.events_path, state.monitor_start, helpers.ROOT)
    monitor = load_monitor(inspect_dir)
    epochs = list_epochs(inspect_dir) if inspect_dir is not None else []
    start = (state.run_start if state else None) or {}
    task = monitor.task if monitor else start.get("task") or "multilabel"
    classes = list(monitor.classes) if monitor else list(start.get("classes") or [])
    return InspectorCtx(state=state, inspect_dir=inspect_dir, monitor=monitor, epochs=epochs,
                        epoch=epochs[-1] if epochs else None, task=task, classes=classes,
                        groups=load_groups_cached(inspect_dir, monitor))


def _view(run: InspectorRun, stalled_s: float, armed: bool) -> None:
    """The auto-refreshing body: poll the events, rebuild the context, draw the sub-view."""
    state = status = None
    if run.events_path is not None:
        follower = _follower(run.events_path)
        follower.poll()
        state = follower.state
        status = state.status(stalled_after_s=stalled_s)
        if armed and status in FINAL_STATES:
            st.rerun(scope="app")           # the run ended: one full rerun disarms the timer
    c = _context(run, state)
    _status_line(run, state, status, len(c.epochs))
    render_alert_banner(state, status)

    # Epoch selection (by default the latest), only for the per-epoch views.
    names = view_names(c)
    default = "Predictions" if c.epochs else "System"
    view = st.segmented_control("View", names, default=default, key="insp_view",
                                label_visibility="collapsed") or default
    if view not in names:               # e.g. Subgroups of a previous run
        view = default
    if view in _EPOCH_VIEWS and len(c.epochs) > 1:
        a, b = st.columns([1, 4], vertical_alignment="center")
        follow = a.toggle("Latest epoch", value=True, key=f"insp_follow_{run.timestamp}")
        if follow:
            b.caption(f"Showing epoch {c.epochs[-1]} (follows new epochs as they arrive).")
        else:
            c.epoch = b.select_slider("Epoch", c.epochs, value=c.epochs[-1],
                                      key=f"insp_epoch_{run.timestamp}")
    elif view in _EPOCH_VIEWS and c.epochs:
        st.caption(f"Epoch {c.epochs[0]} (the only one inspected so far).")
    VIEWS[view](c)


def render(ctx: DashboardContext) -> None:
    st.markdown("## Inspector")
    st.caption("Analyse a run live and afterwards: system, throughput, predictions, errors, "
               "curves with confidence intervals, clinical operating point, calibration, "
               "subgroups, training dynamics, embeddings, Grad-CAM, per-layer statistics and "
               "alerts. Read-only: it only reads the run's event stream and `inspect_*` "
               "directory.")
    runs = discover_inspector_runs(helpers.ROOT)
    if not runs:
        st.info("No runs with an event stream or inspection data under `logs/`. Train with the "
                "prediction monitor: `--monitor` (see docs/inspeccion.md).")
        return

    by_key = {f"{r.timestamp}|{r.events_path or r.inspect_dir}": r for r in runs}
    keys = list(by_key)
    current = getattr(getattr(ctx, "run", None), "timestamp", None)
    index = next((i for i, k in enumerate(keys) if by_key[k].timestamp == current), 0)
    c_run, c_live, c_int, c_stall, c_now = st.columns([4, 1.3, 1.4, 1.4, 1.2],
                                                       vertical_alignment="bottom")
    chosen = c_run.selectbox(
        "Run", keys, index=index, key="insp_run",
        format_func=lambda k: ("● " if by_key[k].active else "") + by_key[k].label
        + ("" if by_key[k].inspect_dir else "  (no inspection data)"),
        help="Active runs first (● = being written now).")
    run = by_key[chosen]
    live = c_live.toggle("Live", value=True, key="insp_live", help="Off pauses the refresh.")
    interval = c_int.selectbox("Refresh every", _INTERVALS_S,
                               index=_INTERVALS_S.index(_DEFAULT_INTERVAL_S),
                               format_func=lambda s: f"{s} s", key="insp_interval")
    stalled_min = c_stall.number_input("Stalled after (min)", min_value=1,
                                       value=_DEFAULT_STALLED_MIN, key="insp_stalled_min")
    c_now.button("Refresh now")

    stalled_s = float(stalled_min) * 60
    armed = False
    if run.events_path is not None:
        follower = _follower(run.events_path)
        follower.poll()
        armed = live and follower.state.status(stalled_after_s=stalled_s) not in FINAL_STATES
    st.fragment(run_every=f"{interval}s" if armed else None)(_view)(run, stalled_s, armed)
