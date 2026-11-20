"""Aggregation of runs that differ only in the seed (mean ± std, per-epoch bands) and the
summary table of the Compare page."""
import math
from pathlib import Path

import pandas as pd
import pytest

from src.web.run_meta import QWK, RunMeta
from src.web.run_registry import RunInfo
from src.web.seed_agg import (
    build_series, curve_band, fmt_mean_std, group_runs, mean_std, run_numbers)
from src.web.tabs.comparison.summary import summary_frame


def _df(qwk, epoch_time=10.0):
    return pd.DataFrame({"epoch": range(1, len(qwk) + 1), "val_qwk": qwk,
                         "epoch_time": epoch_time})


def _run(ts, seed, loss="ce", **kw):
    ts = f"19112026_1000{ts}"             # two digits of seconds identify the run
    return RunInfo(timestamp=ts, log_path=Path(f"/tmp/train_{ts}.log"), trace_mode="simple",
                   env="local", mode="single", model="vit_small_patch16_224", dataset="panda",
                   precision="amp", meta=RunMeta(dataset="panda", seed=seed, loss=loss,
                                                 task="multiclass", **kw))


def test_mean_std_is_the_sample_std_and_ignores_missing_values():
    mean, std, n = mean_std([0.70, 0.74, None, float("nan"), 0.78])
    assert n == 3 and mean == pytest.approx(0.74) and std == pytest.approx(0.04)
    assert mean_std([0.5]) == (0.5, 0.0, 1)
    assert math.isnan(mean_std([])[0]) and mean_std([])[2] == 0


def test_fmt_mean_std():
    assert fmt_mean_std([0.70, 0.74, 0.78]) == "0.7400 ± 0.0400"
    assert fmt_mean_std([0.7216]) == "0.7216"
    assert fmt_mean_std([]) == "—" and fmt_mean_std([0.7, 0.8], digits=1) == "0.8 ± 0.1"


def test_run_numbers_best_final_and_duration():
    n = run_numbers(_df([0.3, 0.8, 0.6]), "val_qwk")
    assert n == {"epochs": 3.0, "best": 0.8, "best_epoch": 2.0, "final": 0.6,
                 "duration_s": 30.0}
    assert run_numbers(pd.DataFrame(), "val_qwk") == {}
    assert "best" not in run_numbers(pd.DataFrame({"epoch": [1]}), "val_qwk")


def test_curve_band_mean_std_and_runs_that_stopped_early():
    band = curve_band([_df([0.2, 0.4, 0.6]), _df([0.4, 0.6])], "val_qwk")
    assert band["epoch"].tolist() == [1, 2, 3]
    assert band["mean"].tolist() == pytest.approx([0.3, 0.5, 0.6])
    assert band["n"].tolist() == [2, 2, 1]
    assert band["std"].iloc[0] == pytest.approx(0.1414213, rel=1e-4)
    assert band["std"].iloc[2] == 0.0                         # one run reached epoch 3
    assert band["lo"].iloc[0] == pytest.approx(0.3 - band["std"].iloc[0])
    assert curve_band([pd.DataFrame({"epoch": [1]})], "val_qwk").empty


def test_group_runs_by_everything_but_the_seed():
    a, b, c = _run("01", 42), _run("02", 43), _run("03", 43, loss="ce_emd")
    d = _run("04", 44, color_jitter=0.2)
    assert [[r.timestamp[-2:] for r in g] for g in group_runs([a, b, c, d])] == \
        [["01", "02"], ["03"], ["04"]]


def test_build_series_aggregates_or_keeps_every_run():
    runs = [_run("01", 42), _run("02", 43), _run("03", 43, loss="ce_emd")]
    dfs = {r.label: _df([0.5, 0.6]) for r in runs}
    agg = build_series(runs, dfs, aggregate=True)
    assert [s.n for s in agg] == [2, 1] and agg[0].seeds == [42, 43]
    assert agg[0].label.endswith("×2") and "seed" not in agg[0].label
    assert agg[1].label == runs[2].label                       # a lone run keeps its label
    assert [s.n for s in build_series(runs, dfs, aggregate=False)] == [1, 1, 1]


def test_summary_frame_puts_the_metric_first_and_shows_mean_std_for_seed_groups():
    runs = [_run("01", 42), _run("02", 43), _run("03", 43, loss="ce_emd")]
    dfs = {runs[0].label: _df([0.3, 0.7, 0.6]), runs[1].label: _df([0.3, 0.8, 0.7]),
           runs[2].label: _df([0.4, 0.5])}
    frame = summary_frame(build_series(runs, dfs), QWK)
    assert list(frame.columns)[:4] == ["Run", "Best Val QWK", "Best epoch", "Final QWK"]
    row = frame.iloc[0]
    assert row["Best Val QWK"] == "0.7500 ± 0.0707" and row["Seeds"] == "42, 43"
    assert row["Best epoch"] == "2.0 ± 0.0" and row["Loss"] == "ce"
    single = frame.iloc[1]
    assert single["Best Val QWK"] == "0.5000" and single["Best epoch"] == "2"
    assert single["Loss"] == "ce_emd"
    # every column is text: one dtype for Arrow
    assert all(frame[c].map(type).nunique() == 1 for c in frame.columns)
