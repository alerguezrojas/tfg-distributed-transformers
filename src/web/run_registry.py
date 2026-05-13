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
    env: str = "local"          # "local" or "verode"
    plot_path: Path | None = None
    perclass_paths: list[Path] = field(default_factory=list)
    batch_csv_path: Path | None = None

    @property
    def label(self) -> str:
        ts = self.timestamp
        # Detect format: DDMMYYYY has a 4-digit year at positions 4-8 (>= 2000)
        # YYYYMMDD (legacy) has a 4-digit year at positions 0-4.
        if int(ts[4:8]) >= 2000:  # DDMMYYYY
            date = f"{ts[:2]}/{ts[2:4]}/{ts[4:8]}"
        else:                      # legacy YYYYMMDD
            date = f"{ts[6:8]}/{ts[4:6]}/{ts[:4]}"
        time_str = f"{ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
        return f"{date} {time_str}  [{self.trace_mode}]  [{self.env}]"


def discover_runs(root: Path = Path(".")) -> list[RunInfo]:
    """Scan logs/local/ and logs/verode/ for training runs.

    Returns list sorted by timestamp descending.
    """
    runs: dict[str, RunInfo] = {}

    for env in ("local", "verode"):
        logs_dir = root / "logs" / env
        plots_dir = root / "plots" / env

        if not logs_dir.exists():
            continue

        for log_path in logs_dir.glob("train_*.log"):
            name = log_path.stem
            if name in ("train_legacy", "train_local"):
                continue
            m = _TIMESTAMP_RE.search(name)
            if not m:
                continue
            ts = m.group(1)
            is_deep = "deep" in name
            key = f"{env}_{ts}"
            runs[key] = RunInfo(
                timestamp=ts,
                log_path=log_path,
                trace_mode="deep" if is_deep else "simple",
                env=env,
            )

        if plots_dir.exists():
            for plot_path in plots_dir.glob("training_*.png"):
                m = _TIMESTAMP_RE.search(plot_path.stem)
                if m:
                    key = f"{env}_{m.group(1)}"
                    if key in runs:
                        runs[key].plot_path = plot_path

            for plot_path in sorted(plots_dir.glob("perclass_*.png")):
                m = _TIMESTAMP_RE.search(plot_path.stem)
                if m:
                    key = f"{env}_{m.group(1)}"
                    if key in runs:
                        runs[key].perclass_paths.append(plot_path)

        for csv_path in logs_dir.glob("batch_metrics_*.csv"):
            m = _TIMESTAMP_RE.search(csv_path.stem)
            if m:
                key = f"{env}_{m.group(1)}"
                if key in runs:
                    runs[key].batch_csv_path = csv_path

    return sorted(runs.values(), key=lambda r: r.timestamp, reverse=True)
