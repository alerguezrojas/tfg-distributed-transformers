"""Discovers and indexes training runs from logs/.

Scans recursively to handle both the old flat layout
(logs/{env}/train_*.log) and the new deep one
(logs/{env}/{mode}/{model}/train_*.log).

Only indexes CSVs — PNGs are no longer generated during training.
The web dashboard renders every chart interactively from the CSVs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")
_PRECISION_RE = re.compile(r"precision=(fp32|tf32|amp|bf16)")


def _read_precision(log_path: Path) -> str:
    """Reads the precision from the run's 'Configuración:' header line.

    Returns "" when the log has no precision marker (pre-precision runs).
    Only the first ~4 KB are read — the config line is always at the top.
    """
    try:
        with log_path.open(encoding="utf-8", errors="ignore") as f:
            head = f.read(4096)
    except OSError:
        return ""
    m = _PRECISION_RE.search(head)
    return m.group(1) if m else ""


@dataclass
class RunInfo:
    timestamp: str
    log_path: Path
    trace_mode: str
    env: str = "local"
    mode: str = "single"
    model: str = ""
    precision: str = ""
    confusion_matrix_csv_path: Path | None = None
    batch_csv_path: Path | None = None
    epoch_csv_path: Path | None = None
    perclass_csv_path: Path | None = None
    # Held-out test-set results (from scripts/eval.py). Keyed by folder, not
    # timestamp — eval CSVs are named by loss/config (test_bce.csv …), so all
    # test_*.csv in the run's folder are offered against runs of that model.
    test_csv_paths: list[Path] = field(default_factory=list)

    @property
    def sort_key(self) -> str:
        """Chronological sort key (YYYYMMDD_HHMMSS).

        Timestamps are written as DDMMYYYY_HHMMSS (current format) or
        YYYYMMDD_HHMMSS (legacy). Sorting the raw string is NOT chronological
        for the DDMMYYYY format ('27052026' > '02062026' even though May < June).
        This property normalizes both to YYYYMMDD_HHMMSS for a correct order.
        """
        ts = self.timestamp
        time_part = ts[9:15] if len(ts) >= 15 else "000000"
        if len(ts) >= 8 and int(ts[4:8]) >= 2000:  # DDMMYYYY
            yyyymmdd = f"{ts[4:8]}{ts[2:4]}{ts[:2]}"
        else:                                        # legacy YYYYMMDD
            yyyymmdd = ts[:8]
        return f"{yyyymmdd}_{time_part}"

    @property
    def label(self) -> str:
        """Compact display label.

        Kept short so the distinguishing tags survive the selectbox width:
        no seconds, the default trace (simple) and precision (fp32) are
        implicit, and the ubiquitous "_patch16_224" model suffix is dropped.
        E.g. "10/06/2026 21:18 [kaggle] vit_base [ddp] [amp]".
        """
        ts = self.timestamp
        if int(ts[4:8]) >= 2000:  # DDMMYYYY
            date = f"{ts[:2]}/{ts[2:4]}/{ts[4:8]}"
        else:                      # legacy YYYYMMDD
            date = f"{ts[6:8]}/{ts[4:6]}/{ts[:4]}"
        time_str = f"{ts[9:11]}:{ts[11:13]}"
        parts = [f"{date} {time_str}", f"[{self.env}]"]
        if self.model:
            parts.append(self.model.replace("_patch16_224", ""))
        if self.mode != "single":
            parts.append(f"[{self.mode}]")
        # Tag Tensor-core runs (amp/tf32/bf16); fp32 is the unlabeled default.
        if self.precision and self.precision != "fp32":
            parts.append(f"[{self.precision}]")
        if self.trace_mode == "deep":
            parts.append("[deep]")
        return " ".join(parts)


def _env_mode_model_from_path(log_path: Path, logs_root: Path) -> tuple[str, str, str]:
    try:
        parts = log_path.relative_to(logs_root).parts
    except ValueError:
        return "unknown", "single", ""
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        return parts[0], "single", ""
    return "legacy", "single", ""


def discover_runs(root: Path = Path(".")) -> list[RunInfo]:
    """Scans logs/ recursively and returns runs sorted by timestamp desc."""
    logs_root = root / "logs"
    runs: dict[str, RunInfo] = {}

    if not logs_root.exists():
        return []

    for log_path in logs_root.rglob("train_*.log"):
        name = log_path.stem
        if name in ("train", "train_legacy", "train_local"):
            continue
        m = _TIMESTAMP_RE.search(name)
        if not m:
            continue
        ts = m.group(1)
        is_deep = "deep" in name
        env, mode, model = _env_mode_model_from_path(log_path, logs_root)
        runs[ts] = RunInfo(
            timestamp=ts,
            log_path=log_path,
            trace_mode="deep" if is_deep else "simple",
            env=env, mode=mode, model=model,
            precision=_read_precision(log_path),
        )

    if not runs:
        return []

    for csv_path in logs_root.rglob("batch_metrics_*.csv"):
        m = _TIMESTAMP_RE.search(csv_path.stem)
        if m and m.group(1) in runs and runs[m.group(1)].batch_csv_path is None:
            runs[m.group(1)].batch_csv_path = csv_path

    for csv_path in logs_root.rglob("epoch_metrics_*.csv"):
        m = _TIMESTAMP_RE.search(csv_path.stem)
        if m and m.group(1) in runs and runs[m.group(1)].epoch_csv_path is None:
            runs[m.group(1)].epoch_csv_path = csv_path

    for csv_path in logs_root.rglob("perclass_metrics_*.csv"):
        m = _TIMESTAMP_RE.search(csv_path.stem)
        if m and m.group(1) in runs and runs[m.group(1)].perclass_csv_path is None:
            runs[m.group(1)].perclass_csv_path = csv_path

    for csv_path in logs_root.rglob("confusion_matrix_*.csv"):
        m = _TIMESTAMP_RE.search(csv_path.stem)
        if m and m.group(1) in runs and runs[m.group(1)].confusion_matrix_csv_path is None:
            runs[m.group(1)].confusion_matrix_csv_path = csv_path

    # Held-out test CSVs: associate by folder (they carry no run timestamp).
    # Every test_*.csv / eval_*.csv in a run's directory is offered to runs there.
    _test_by_dir: dict[Path, list[Path]] = {}
    for pat in ("test_*.csv", "eval_*.csv"):
        for csv_path in logs_root.rglob(pat):
            _test_by_dir.setdefault(csv_path.parent, []).append(csv_path)
    for r in runs.values():
        found = _test_by_dir.get(r.log_path.parent, [])
        if found:
            r.test_csv_paths = sorted(found)

    return sorted(runs.values(), key=lambda r: r.sort_key, reverse=True)


def discover_feasibility_csvs(root: Path = Path(".")) -> list[Path]:
    """Returns all feasibility CSVs sorted by modification time."""
    logs_root = root / "logs"
    if not logs_root.exists():
        return []
    paths = list(logs_root.rglob("feasibility_*.csv"))
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
