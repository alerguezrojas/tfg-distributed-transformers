"""Logic of the web that does not need Streamlit: choosing a run, linking to it, comparing
datasets, counting GPUs.

* ``filter_runs_by`` / ``run_rows``: the sidebar's run picker (search text + filters), meant
  for hundreds of runs.
* ``query_target`` / ``query_params_for``: direct links (``?run=<id>&tab=<page>``).
* ``best_test_by_dataset``: best held-out test result per dataset, each with the main
  metric of its own task (QWK for PANDA, macro F1 for BigEarthNet, ...).
* ``gpu_count``: how many GPUs a run used (to turn wall time into GPU time).
* ``run_classes``: class names of a run, from its manifest (written from its
  ``DatasetSpec``), so the web never imports the dataset registry (it needs torch).

Pure: no Streamlit, no torch.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from src.data.bigearthnet_classes import CLASSES as BIGEARTHNET_CLASSES
from src.run_manifest import read_config_line, read_manifest
from src.web.eval_parser import parse_eval_csv

ALL = "All"

# Key of the aggregate line of an eval CSV (scripts/eval.py) holding each main metric.
_AGG_KEY = {"f1": "f1_t05", "f1_optimal": "f1_opt", "auc": "auc", "qwk": "qwk"}

_GPUS_RE = re.compile(r"(\d+)\s*GPUs")


# ── Run picker ────────────────────────────────────────────────────────────────

def run_id(run) -> str:
    """Stable id of a run (its timestamp, unique across ``logs/``): what the URL carries."""
    return run.timestamp


def _is_all(value: str | None) -> bool:
    return not value or value == ALL


def filter_runs_by(runs: Iterable, query: str = "", *, dataset: str | None = None,
                   model: str | None = None, mode: str | None = None,
                   env: str | None = None) -> list:
    """Runs matching every filter (``None`` / ``"All"`` = no filter) whose label, model,
    dataset, env or mode contain every word of ``query`` (case-insensitive)."""
    words = query.lower().split()
    out = []
    for r in runs:
        if not _is_all(dataset) and r.dataset != dataset:
            continue
        if not _is_all(model) and r.model != model:
            continue
        if not _is_all(mode) and r.mode != mode:
            continue
        if not _is_all(env) and r.env != env:
            continue
        hay = " ".join((r.label, r.model, r.dataset, r.env, r.mode, r.timestamp)).lower()
        if all(w in hay for w in words):
            out.append(r)
    return out


def run_rows(runs: Sequence) -> list[dict]:
    """One table row per run for the picker (same order as ``runs``)."""
    return [{"Run": r.label, "Dataset": r.dataset, "Model": r.model or "—",
             "Strategy": r.mode, "Env": r.env} for r in runs]


def filter_options(runs: Iterable, field: str) -> list[str]:
    """``["All", ...]`` with the distinct non-empty values of a RunInfo attribute."""
    return [ALL, *sorted({getattr(r, field) for r in runs if getattr(r, field)})]


# ── Direct links ──────────────────────────────────────────────────────────────

def query_target(params: Mapping, runs: Sequence,
                 nav_keys: Sequence[str]) -> tuple[str | None, str | None]:
    """``(run label, page key)`` a URL asks for; each is ``None`` when absent or unknown
    (a stale link degrades to the default instead of failing)."""
    rid, tab = params.get("run"), params.get("tab")
    label = next((r.label for r in runs if run_id(r) == rid), None) if rid else None
    return label, (tab if tab in nav_keys else None)


def query_params_for(run, nav: str | None) -> dict[str, str]:
    """Query string of the current view (the inverse of :func:`query_target`)."""
    out: dict[str, str] = {}
    if run is not None:
        out["run"] = run_id(run)
    if nav:
        out["tab"] = nav
    return out


# ── Held-out test results per dataset ─────────────────────────────────────────

def metric_from_aggregate(metric_key: str, aggregate: Mapping) -> float | None:
    """Value of the run's main metric in an eval CSV aggregate (``None`` if it lacks it)."""
    value = aggregate.get(_AGG_KEY.get(metric_key, "f1_t05"))
    return float(value) if isinstance(value, (int, float)) else None


def best_test_by_dataset(runs: Iterable,
                            parse: Callable[[Path], tuple] = parse_eval_csv) -> list[dict]:
    """Best held-out test result per dataset, scored with each dataset's main metric.

    Datasets are never mixed in one ranking: each row is the best (run, test CSV) of its own
    dataset by the metric of that dataset's runs. ``Evaluated`` counts the runs of the
    dataset with at least one test CSV."""
    best: dict[str, dict] = {}
    evaluated: dict[str, set] = {}
    for r in runs:
        metric = r.main_metric
        for path in r.test_csv_paths:
            _per_class, agg = parse(path)
            value = metric_from_aggregate(metric.key, agg)
            if value is None:
                continue
            evaluated.setdefault(r.dataset, set()).add(run_id(r))
            cur = best.get(r.dataset)
            if cur is None or value > cur["Test"]:
                best[r.dataset] = {"Dataset": r.dataset, "Metric": metric.name,
                                   "Test": round(value, 4), "Run": r.label,
                                   "Model": r.model or "—"}
    return [{**row, "Evaluated": len(evaluated[ds])} for ds, row in sorted(best.items())]


# ── GPUs of a run ─────────────────────────────────────────────────────────────

def gpu_count(run) -> int | None:
    """GPUs the run used, or ``None`` when its logs do not say.

    Order: the manifest's ``world_size``; the ``Configuración:`` line (``N GPUs`` of DDP, the
    ``devices=a+b`` of model parallel); ``single`` runs use one. A heterogeneous run uses
    ONE GPU (its other rank is a CPU worker), whatever its ``world_size``; the strategies that
    split the model run ONE process (``world_size`` 1) over the cuda devices they list."""
    if run.mode == "heterogeneous":
        return 1
    split = run.mode in ("model_parallel", "pipeline")
    manifest = read_manifest(getattr(run, "manifest_path", None)) or {}
    world = manifest.get("world_size")
    if not split and isinstance(world, int) and not isinstance(world, bool) and world > 0:
        return world
    cfg = read_config_line(run.log_path)
    m = _GPUS_RE.search(cfg.get("batch", ""))
    if m:
        return int(m.group(1))
    if cfg.get("devices"):
        gpus = {d for d in cfg["devices"].split("+") if d.startswith("cuda")}
        return len(gpus) or None
    return 1 if run.mode == "single" else None


# ── Classes of a run ──────────────────────────────────────────────────────────

def run_classes(run) -> tuple[str, ...]:
    """Class names of the run's dataset: the manifest's (from its ``DatasetSpec``), else
    BigEarthNet's fixed list, else ``()`` (the caller reads them from the run's CSVs)."""
    meta = getattr(run, "meta", None)
    if meta is not None and meta.classes:
        return tuple(meta.classes)
    return tuple(BIGEARTHNET_CLASSES) if run.dataset == "bigearthnet" else ()


def order_classes(run, present: Sequence[str]) -> list[str]:
    """``present`` in the dataset's own class order when that covers all of them; otherwise
    as given (names that do not match the dataset's list)."""
    known = set(present)
    ordered = [c for c in run_classes(run) if c in known]
    return ordered if len(ordered) == len(known) else list(present)
