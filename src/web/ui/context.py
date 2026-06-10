"""Shared context object passed to every tab's render() function."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DashboardContext:
    """State produced by the sidebar and consumed by the tab modules."""

    runs: list[Any]
    selected_run: Any | None
    run: Any | None
    refresh_interval: int
