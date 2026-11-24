"""Discovers and indexes training runs from logs/.

Scans recursively to handle both the old flat layout
(logs/{env}/train_*.log) and the new deep one
(logs/{env}/{mode}/{model}/train_*.log).

Only indexes CSVs (and the live-events JSONL) — PNGs are no longer generated
during training. The web dashboard renders every chart interactively from them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.web.run_meta import MetricInfo, RunMeta, load_run_meta, metric_info

_TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")
_TEST_CSV_RE = re.compile(r"^(?:test|eval)_(\d{8}_\d{6})\.csv$")
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



# Kept as a literal (not imported from src.data) so the web stays torch-free.
_DEFAULT_DATASET = "bigearthnet"

def timestamp_sort_key(ts: str) -> str:
    """Chronological sort key (YYYYMMDD_HHMMSS) of a run timestamp.

    Timestamps are written as DDMMYYYY_HHMMSS (current format) or
    YYYYMMDD_HHMMSS (legacy). Sorting the raw string is NOT chronological
    for the DDMMYYYY format ('27052026' > '02062026' even though May < June).
    This normalizes both to YYYYMMDD_HHMMSS for a correct order.
    """
    time_part = ts[9:15] if len(ts) >= 15 else "000000"
    if len(ts) >= 8 and int(ts[4:8]) >= 2000:  # DDMMYYYY
        yyyymmdd = f"{ts[4:8]}{ts[2:4]}{ts[:2]}"
    else:                                        # legacy YYYYMMDD
        yyyymmdd = ts[:8]
    return f"{yyyymmdd}_{time_part}"


@dataclass
class RunInfo:
    timestamp: str
    log_path: Path
    trace_mode: str
    env: str = "local"
    mode: str = "single"
    model: str = ""
    precision: str = ""
    dataset: str = _DEFAULT_DATASET
    confusion_matrix_csv_path: Path | None = None
    batch_csv_path: Path | None = None
    epoch_csv_path: Path | None = None
    perclass_csv_path: Path | None = None
    # Live-events stream (events_*.jsonl, written by rank 0 while training).
    events_path: Path | None = None
    # Run manifest (run_<ts>.json, docs/eventos.md): dataset, task, metric, config.
    manifest_path: Path | None = None
    # What the run is (src/web/run_meta.py), read once by discover_runs.
    meta: RunMeta | None = None
    # Show seconds in the label: set when two runs would otherwise share one.
    label_seconds: bool = False
    # Held-out test-set results (from scripts/eval.py): the test_<ts>.csv / eval_<ts>.csv
    # whose stamp is this run's.
    test_csv_paths: list[Path] = field(default_factory=list)
    # test_*/eval_*.csv of the folder that no run claims by stamp (not attributed to any run)
    unassigned_test_csv_paths: list[Path] = field(default_factory=list)

    @property
    def sort_key(self) -> str:
        """Chronological sort key (YYYYMMDD_HHMMSS); see :func:`timestamp_sort_key`."""
        return timestamp_sort_key(self.timestamp)

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
        time_str = f"{ts[9:11]}:{ts[11:13]}" + (f":{ts[13:15]}" if self.label_seconds else "")
        parts = [f"{date} {time_str}", f"[{self.env}]"]
        if self.dataset != _DEFAULT_DATASET:
            parts.append(f"[{self.dataset}]")
        if self.model:
            parts.append(self.model.replace("_patch16_224", ""))
        if self.mode != "single":
            parts.append(f"[{self.mode}]")
        # Tag Tensor-core runs (amp/tf32/bf16); fp32 is the unlabeled default.
        if self.precision and self.precision != "fp32":
            parts.append(f"[{self.precision}]")
        if self.trace_mode == "deep":
            parts.append("[deep]")
        return " · ".join([" ".join(parts), *self._tags()])

    def _tags(self, with_seed: bool = True) -> list[str]:
        return self.meta.tags(with_seed=with_seed) if self.meta else []

    @property
    def group_label(self) -> str:
        """Label of the runs that differ only in the seed: no date, no seed."""
        parts = [f"[{self.env}]"]
        if self.dataset != _DEFAULT_DATASET:
            parts.append(f"[{self.dataset}]")
        if self.model:
            parts.append(self.model.replace("_patch16_224", ""))
        if self.mode != "single":
            parts.append(f"[{self.mode}]")
        if self.precision and self.precision != "fp32":
            parts.append(f"[{self.precision}]")
        return " · ".join([" ".join(parts), *self._tags(with_seed=False)])

    @property
    def seed_group_key(self) -> tuple:
        """Equal for runs that differ at most in the seed (never shared by a run that
        does not record its seed)."""
        extra = self.meta.group_key(self.timestamp) if self.meta else ("single", self.timestamp)
        return (self.env, self.dataset, self.model, self.mode, self.trace_mode, *extra)

    @property
    def main_metric(self) -> MetricInfo:
        """The run's main metric (QWK for PANDA, AUC for MedMNIST, F1 otherwise)."""
        return self.meta.metric if self.meta else metric_info(None, self.dataset)


def _env_mode_model_from_path(log_path: Path, logs_root: Path) -> tuple[str, str, str, str]:
    """(env, mode, model, dataset) from the run's folder.

    BigEarthNet runs use logs/{env}/{mode}/{model}/ (the historical layout); runs of
    any other dataset add a level: logs/{env}/{dataset}/{mode}/{model}/.
    """
    try:
        parts = log_path.relative_to(logs_root).parts
    except ValueError:
        return "unknown", "single", "", _DEFAULT_DATASET
    if len(parts) >= 5:
        return parts[0], parts[2], parts[3], parts[1]
    if len(parts) == 4:
        return parts[0], parts[1], parts[2], _DEFAULT_DATASET
    if len(parts) == 2:
        return parts[0], "single", "", _DEFAULT_DATASET
    return "legacy", "single", "", _DEFAULT_DATASET


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
        env, mode, model, dataset = _env_mode_model_from_path(log_path, logs_root)
        runs[ts] = RunInfo(
            timestamp=ts,
            log_path=log_path,
            trace_mode="deep" if is_deep else "simple",
            env=env, mode=mode, model=model, dataset=dataset,
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

    for jsonl_path in logs_root.rglob("events_*.jsonl"):
        m = _TIMESTAMP_RE.search(jsonl_path.stem)
        if m and m.group(1) in runs and runs[m.group(1)].events_path is None:
            runs[m.group(1)].events_path = jsonl_path

    for json_path in logs_root.rglob("run_*.json"):
        m = _TIMESTAMP_RE.search(json_path.stem)
        if m and m.group(1) in runs and runs[m.group(1)].manifest_path is None:
            runs[m.group(1)].manifest_path = json_path

    for csv_path in logs_root.rglob("confusion_matrix_*.csv"):
        m = _TIMESTAMP_RE.search(csv_path.stem)
        if m and m.group(1) in runs and runs[m.group(1)].confusion_matrix_csv_path is None:
            runs[m.group(1)].confusion_matrix_csv_path = csv_path

    # Held-out test CSVs: test_<ts>.csv / eval_<ts>.csv belong to the run with that stamp.
    # Any other test_*/eval_*.csv (no stamp, or one of no run) is NOT attributed: the folder's
    # runs only list it in unassigned_test_csv_paths. Exception: a folder with a single run
    # gives its loose CSVs to that run.
    _unassigned_by_dir: dict[Path, list[Path]] = {}
    for pat in ("test_*.csv", "eval_*.csv"):
        for csv_path in logs_root.rglob(pat):
            m = _TEST_CSV_RE.match(csv_path.name)
            if m and m.group(1) in runs:
                runs[m.group(1)].test_csv_paths.append(csv_path)
            else:
                _unassigned_by_dir.setdefault(csv_path.parent, []).append(csv_path)
    runs_per_dir: dict[Path, int] = {}
    for r in runs.values():
        runs_per_dir[r.log_path.parent] = runs_per_dir.get(r.log_path.parent, 0) + 1
    for r in runs.values():
        loose = sorted(_unassigned_by_dir.get(r.log_path.parent, []))
        if runs_per_dir[r.log_path.parent] == 1:
            # the folder's only run: no ambiguity, an unstamped CSV (test_best_epoch007.csv) is its
            r.test_csv_paths.extend(loose)
            loose = []
        r.test_csv_paths.sort()
        r.unassigned_test_csv_paths = loose

    for r in runs.values():
        r.meta = load_run_meta(r)
    ordered = sorted(runs.values(), key=lambda r: r.sort_key, reverse=True)
    # The label is the run's key in the web: two runs of the same minute and settings
    # get their seconds so they stay distinguishable.
    seen: dict[str, list[RunInfo]] = {}
    for r in ordered:
        seen.setdefault(r.label, []).append(r)
    for twins in seen.values():
        if len(twins) > 1:
            for r in twins:
                r.label_seconds = True
    return ordered


def filter_runs(runs: list[RunInfo], env: str | None = None,
                dataset: str | None = None) -> list[RunInfo]:
    """Runs matching an environment and/or a dataset (None = no filter)."""
    return [r for r in runs
            if (env is None or r.env == env)
            and (dataset is None or r.dataset == dataset)]


def discover_benchmark_csvs(root: Path = Path(".")) -> list[Path]:
    """Returns all benchmark CSVs sorted by modification time."""
    logs_root = root / "logs"
    if not logs_root.exists():
        return []
    paths = list(logs_root.rglob("benchmark_*.csv"))
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
