"""Header banner with the active alerts of a run (shared by Live and the Inspector).

Critical alerts in red, warnings in amber, each with its epoch. The alerts are the ``alert``
events of the trainer (docs/eventos.md); a run without them shows nothing.
"""
from __future__ import annotations

import streamlit as st

from src.web.events_reader import FINAL_STATES, RunState
from src.web.inspector_data import CRITICAL, active_alerts, alert_line


def render_alert_banner(state: RunState | None, status: str | None) -> None:
    """One red/amber box per active alert; nothing if there are none.

    While the run goes, *active* means fired in the current or the previous epoch; once it has
    ended, every code that fired is listed (the banner then summarises the run).
    """
    if state is None or not state.alerts:
        return
    current = None if status in FINAL_STATES else state.current_epoch
    for ev in active_alerts(state.alerts, current_epoch=current):
        line = alert_line(ev)
        if ev["level"] == CRITICAL:
            st.error(f"**CRITICAL** · {line}")
        else:
            st.warning(f"**Warning** · {line}")
