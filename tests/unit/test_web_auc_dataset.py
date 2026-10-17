"""Web helpers for validation AUC and the dataset run filter (torch-free)."""
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.web.run_registry import RunInfo, discover_runs, filter_runs  # noqa: E402
from src.web.ui.helpers import _best_metric  # noqa: E402


# ── _best_metric ───────────────────────────────────────────────────────────────

def test_best_metric_returns_value_and_epoch():
    df = pd.DataFrame({"epoch": [1, 2, 3], "val_auc": [0.70, 0.85, 0.80]})
    best = _best_metric(df, "val_auc")
    assert best is not None
    assert math.isclose(best[0], 0.85)
    assert best[1] == 2


def test_best_metric_ignores_nan_rows():
    df = pd.DataFrame({"epoch": [1, 2, 3], "val_auc": [float("nan"), 0.6, float("nan")]})
    assert _best_metric(df, "val_auc") == (0.6, 2)


def test_best_metric_none_without_column_or_values():
    """Older epoch CSVs have no val_auc (or an empty one): callers keep the old layout."""
    assert _best_metric(pd.DataFrame({"epoch": [1], "val_f1": [0.5]}), "val_auc") is None
    assert _best_metric(pd.DataFrame({"epoch": [1, 2], "val_auc": [None, None]}),
                        "val_auc") is None
    assert _best_metric(pd.DataFrame(), "val_auc") is None


def test_auc_fig_marks_best_epoch():
    from src.web.tabs.run.curves import _auc_fig
    df = pd.DataFrame({"epoch": [1, 2, 3], "val_auc": [0.70, 0.85, 0.80]})
    fig = _auc_fig(df, (0.85, 2))
    names = [t.name for t in fig.data]
    assert names == ["Val AUC", "Best (epoch 2)"]   # no train trace: AUC is val-only
    assert list(fig.data[1].x) == [2] and math.isclose(fig.data[1].y[0], 0.85)


# ── filter_runs ────────────────────────────────────────────────────────────────

def _run(ts: str, env: str, dataset: str) -> RunInfo:
    return RunInfo(timestamp=ts, log_path=Path("x.log"), trace_mode="simple",
                   env=env, dataset=dataset)


def test_filter_runs_by_env_and_dataset():
    runs = [_run("17102026_100000", "local", "bigearthnet"),
            _run("17102026_110000", "local", "medmnist_chest"),
            _run("17102026_120000", "kaggle", "medmnist_chest")]
    assert filter_runs(runs) == runs
    assert [r.timestamp for r in filter_runs(runs, dataset="medmnist_chest")] == \
        ["17102026_110000", "17102026_120000"]
    assert [r.timestamp for r in filter_runs(runs, env="local", dataset="medmnist_chest")] == \
        ["17102026_110000"]
    assert filter_runs(runs, env="verode") == []


def test_discover_runs_reads_dataset_from_folder(tmp_path):
    """logs/{env}/{dataset}/{mode}/{model}/ → dataset; the 4-level layout is BigEarthNet."""
    ben = tmp_path / "logs/local/single/vit_tiny_patch16_224"
    med = tmp_path / "logs/local/medmnist_chest/single/vit_tiny_patch16_224"
    for d, ts in ((ben, "17102026_100000"), (med, "17102026_110000")):
        d.mkdir(parents=True)
        (d / f"train_{ts}.log").write_text("", encoding="utf-8")
    by_ts = {r.timestamp: r for r in discover_runs(tmp_path)}
    assert by_ts["17102026_100000"].dataset == "bigearthnet"
    assert by_ts["17102026_110000"].dataset == "medmnist_chest"
    assert by_ts["17102026_110000"].mode == "single"
    assert by_ts["17102026_110000"].model == "vit_tiny_patch16_224"
