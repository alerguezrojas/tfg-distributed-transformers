"""Renders the real Streamlit app (AppTest) on a synthetic logs/ tree.

Two runs from the same session: a BigEarthNet run whose epoch CSV predates
``val_auc`` and a MedMNIST run that logs it. Every sidebar section must render
without exceptions with either run active, the AUC views must appear only for
the run that has it, and the dataset filter must narrow the run list.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(ROOT / "src" / "web" / "app.py")
NAV_KEYS = ["overview", "live", "inspector", "run", "compare", "benchmark", "data"]
PAGE_TITLES = {"overview": "## Overview", "live": "## Live", "inspector": "## Inspector", "run": "## Run results", "compare": "## Compare",
               "benchmark": "## Estimate / Benchmark", "data": "## Dataset"}

_BEN_TS, _MED_TS = "17102026_100000", "17102026_110000"
_CFG = ("2026-10-17 10:00:00 [INFO ] Configuración: modelo=vit_tiny_patch16_224 | "
        "batch=32/GPU (global=32) | epochs=3 | lr=0.0001 | precision=fp32 | loss=bce | "
        "train=500 | val=100{extra}\n")


def _write_run(folder: Path, ts: str, with_auc: bool, dataset_tag: str) -> None:
    folder.mkdir(parents=True)
    (folder / f"train_{ts}.log").write_text(_CFG.format(extra=dataset_tag), encoding="utf-8")
    df = pd.DataFrame({
        "epoch": [1, 2, 3],
        "train_loss": [0.30, 0.22, 0.18], "val_loss": [0.28, 0.24, 0.25],
        "train_f1": [0.20, 0.35, 0.45], "val_f1": [0.18, 0.30, 0.29],
        "train_acc": [0.80, 0.85, 0.88], "val_acc": [0.79, 0.83, 0.84],
        "val_prec": [0.30, 0.40, 0.41], "val_rec": [0.15, 0.25, 0.24],
        "epoch_time_s": [12.0, 11.5, 11.8],
    })
    if with_auc:
        df["val_auc"] = [0.71, 0.78, 0.76]
    df.to_csv(folder / f"epoch_metrics_{ts}.csv", index=False)


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    import streamlit as st
    from src.web.ui import helpers

    root = tmp_path_factory.mktemp("webroot")
    _write_run(root / "logs/local/single/vit_tiny_patch16_224", _BEN_TS, False, "")
    _write_run(root / "logs/local/medmnist_chest/single/vit_tiny_patch16_224", _MED_TS, True,
               " | dataset=medmnist_chest")
    old = helpers.ROOT
    helpers.ROOT = root          # _get_runs / _get_benchmark_csvs read it at call time
    st.cache_data.clear()
    yield root
    helpers.ROOT = old
    st.cache_data.clear()


@pytest.fixture(scope="module")
def labels(fake_root):
    from src.web.run_registry import discover_runs
    by_ts = {r.timestamp: r.label for r in discover_runs(fake_root)}
    return {"bigearthnet": by_ts[_BEN_TS], "medmnist": by_ts[_MED_TS]}


def _run_app(nav: str, run_label: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["nav"] = nav
    at.session_state["run_label"] = run_label
    at.run()
    return at


def _metric_labels(at: AppTest) -> list[str]:
    return [m.label for m in at.metric]


@pytest.mark.parametrize("dataset", ["medmnist", "bigearthnet"])
@pytest.mark.parametrize("nav", NAV_KEYS)
def test_every_section_renders(labels, nav, dataset):
    at = _run_app(nav, labels[dataset])
    assert not at.exception, [e.value for e in at.exception]
    assert any(md.value == PAGE_TITLES[nav] for md in at.markdown), f"{nav} did not render"


def test_curves_show_auc_only_when_logged(labels):
    med = _run_app("run", labels["medmnist"])
    assert "Best Val AUC" in _metric_labels(med)
    assert any("best Val AUC" in i.value for i in med.info) or \
        any("best Val AUC" in w.value for w in med.warning)

    ben = _run_app("run", labels["bigearthnet"])
    assert "Best Val AUC" not in _metric_labels(ben)
    assert "Best Val F1" in _metric_labels(ben)


def test_compare_summary_and_overlay_offer_auc(labels):
    at = _run_app("compare", labels["medmnist"])
    assert not at.exception
    summary = at.dataframe[0].value       # default selection = the latest session (both runs)
    assert "Best Val AUC" in summary.columns
    assert summary["Best Val AUC"].map(type).nunique() == 1   # one dtype → Arrow-safe
    assert set(summary["Best Val AUC"]) == {"0.7800", "—"}
    overlay = next(m for m in at.multiselect if m.label == "Metrics to overlay")
    assert "val_auc" in overlay.options


def test_dataset_filter_narrows_run_list(labels):
    at = _run_app("overview", labels["bigearthnet"])
    ds = at.selectbox(key="run_dataset_filter")
    assert ds.options == ["All", "bigearthnet", "medmnist_chest"]
    ds.set_value("medmnist_chest").run()
    assert not at.exception
    picks = {b.label for b in at.button if b.key and b.key.startswith("runpick::")}
    # The active run stays listed (and selected) even when the filter hides it.
    assert picks == {labels["medmnist"], labels["bigearthnet"]}
    at.button(key=f"runpick::{labels['medmnist']}").click().run()
    assert at.session_state["run_label"] == labels["medmnist"]
    picks = {b.label for b in at.button if b.key and b.key.startswith("runpick::")}
    assert picks == {labels["medmnist"]}
