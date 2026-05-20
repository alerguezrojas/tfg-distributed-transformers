"""Discovers and indexes training runs from logs/, plots/, and checkpoints/.

Scans both legacy (logs/train_*.log) and env-structured directories
(logs/local/, logs/verode/) for training logs and their associated artifacts.
"""

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
    epoch_csv_path: Path | None = None
    perclass_csv_path: Path | None = None

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
    """Scan logs/ for training runs (legacy root + env subdirs).

    Returns list sorted by timestamp descending.
    """
    logs_root = root / "logs"
    plots_root = root / "plots"
    runs: dict[str, RunInfo] = {}

    if not logs_root.exists():
        return []

    # Collect (log_path, env) pairs from both legacy root and env subdirs
    log_sources: list[tuple[Path, str]] = []
    for lp in logs_root.glob("train_*.log"):
        log_sources.append((lp, "legacy"))
    for env_dir in logs_root.iterdir():
        if env_dir.is_dir():
            for lp in env_dir.glob("train_*.log"):
                log_sources.append((lp, env_dir.name))

    for log_path, env in log_sources:
        name = log_path.stem
        if name in ("train", "train_legacy", "train_local"):
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

    # Attach plots
    plot_dirs: list[tuple[Path, str]] = []
    if plots_root.exists():
        plot_dirs.append((plots_root, "legacy"))
        for env_dir in plots_root.iterdir():
            if env_dir.is_dir():
                plot_dirs.append((env_dir, env_dir.name))

    for plot_dir, p_env in plot_dirs:
        for plot_path in plot_dir.glob("training_*.png"):
            m = _TIMESTAMP_RE.search(plot_path.stem)
            if m:
                key = f"{p_env}_{m.group(1)}"
                if key in runs:
                    runs[key].plot_path = plot_path
        for plot_path in sorted(plot_dir.glob("perclass_*.png")):
            m = _TIMESTAMP_RE.search(plot_path.stem)
            if m:
                key = f"{p_env}_{m.group(1)}"
                if key in runs:
                    runs[key].perclass_paths.append(plot_path)

    # Attach CSV artifacts
    csv_dirs: list[tuple[Path, str]] = [(logs_root, "legacy")]
    for env_dir in logs_root.iterdir():
        if env_dir.is_dir():
            csv_dirs.append((env_dir, env_dir.name))

    for csv_dir, c_env in csv_dirs:
        for csv_path in csv_dir.glob("batch_metrics_*.csv"):
            m = _TIMESTAMP_RE.search(csv_path.stem)
            if m:
                key = f"{c_env}_{m.group(1)}"
                if key in runs and runs[key].batch_csv_path is None:
                    runs[key].batch_csv_path = csv_path
        for csv_path in csv_dir.glob("epoch_metrics_*.csv"):
            m = _TIMESTAMP_RE.search(csv_path.stem)
            if m:
                key = f"{c_env}_{m.group(1)}"
                if key in runs and runs[key].epoch_csv_path is None:
                    runs[key].epoch_csv_path = csv_path
        for csv_path in csv_dir.glob("perclass_metrics_*.csv"):
            m = _TIMESTAMP_RE.search(csv_path.stem)
            if m:
                key = f"{c_env}_{m.group(1)}"
                if key in runs and runs[key].perclass_csv_path is None:
                    runs[key].perclass_csv_path = csv_path

    return sorted(runs.values(), key=lambda r: r.timestamp, reverse=True)


def discover_feasibility_csvs(root: Path = Path(".")) -> list[Path]:
    """Return all feasibility CSVs sorted by modification time (newest first)."""
    logs_root = root / "logs"
    paths: list[Path] = []
    if not logs_root.exists():
        return paths
    # Legacy root
    paths.extend(logs_root.glob("feasibility_*.csv"))
    # Env subdirs
    for env_dir in logs_root.iterdir():
        if env_dir.is_dir():
            paths.extend(env_dir.glob("feasibility_*.csv"))
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
