"""Discovers and indexes training runs from logs/, plots/, and checkpoints/."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")


@dataclass
class RunInfo:
    timestamp: str
    log_path: Path
    trace_mode: str
    plot_path: Path | None = None
    perclass_paths: list[Path] = field(default_factory=list)
    batch_csv_path: Path | None = None

    @property
    def label(self) -> str:
        ts = self.timestamp
        date = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
        time = f"{ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
        return f"{date} {time}  [{self.trace_mode}]"


def discover_runs(root: Path = Path(".")) -> list[RunInfo]:
    """Scan logs/ for training runs. Returns list sorted by timestamp descending."""
    logs_dir = root / "logs"
    plots_dir = root / "plots"

    if not logs_dir.exists():
        return []

    runs: dict[str, RunInfo] = {}

    for log_path in logs_dir.glob("train_*.log"):
        name = log_path.stem
        if "local" in name or name == "train":
            continue
        m = _TIMESTAMP_RE.search(name)
        if not m:
            continue
        ts = m.group(1)
        is_deep = "deep" in name
        runs[ts] = RunInfo(
            timestamp=ts,
            log_path=log_path,
            trace_mode="deep" if is_deep else "simple",
        )

    if plots_dir.exists():
        for plot_path in plots_dir.glob("training_*.png"):
            m = _TIMESTAMP_RE.search(plot_path.stem)
            if m and m.group(1) in runs:
                runs[m.group(1)].plot_path = plot_path

        for plot_path in sorted(plots_dir.glob("perclass_*.png")):
            m = _TIMESTAMP_RE.search(plot_path.stem)
            if m and m.group(1) in runs:
                runs[m.group(1)].perclass_paths.append(plot_path)

    for csv_path in logs_dir.glob("batch_metrics_*.csv"):
        m = _TIMESTAMP_RE.search(csv_path.stem)
        if m and m.group(1) in runs:
            runs[m.group(1)].batch_csv_path = csv_path

    return sorted(runs.values(), key=lambda r: r.timestamp, reverse=True)
