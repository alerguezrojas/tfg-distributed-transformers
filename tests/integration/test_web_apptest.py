"""Renders the real Streamlit app (AppTest) on a synthetic logs/ tree.

Three runs from the same session: a BigEarthNet run whose epoch CSV predates
``val_auc`` and two MedMNIST runs (seeds 42 and 43) that log it. Every sidebar section must render
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

_BEN_TS, _MED_TS, _MED2_TS = "17102026_100000", "17102026_110000", "17102026_120000"
_CFG = ("2026-10-17 10:00:00 [INFO ] Configuración: modelo=vit_tiny_patch16_224 | "
        "batch=32/GPU (global=32) | epochs=3 | lr=0.0001 | precision=fp32 | loss=bce | "
        "train=500 | val=100{extra}\n")


def _write_run(folder: Path, ts: str, with_auc: bool, dataset_tag: str,
               auc=(0.71, 0.78, 0.76)) -> None:
    folder.mkdir(parents=True, exist_ok=True)
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
        df["val_auc"] = list(auc)
    df.to_csv(folder / f"epoch_metrics_{ts}.csv", index=False)


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    import streamlit as st
    from src.web.ui import helpers

    root = tmp_path_factory.mktemp("webroot")
    _write_run(root / "logs/local/single/vit_tiny_patch16_224", _BEN_TS, False, "")
    _write_run(root / "logs/local/medmnist_chest/single/vit_tiny_patch16_224", _MED_TS, True,
               " | dataset=medmnist_chest | seed=42")
    _write_run(root / "logs/local/medmnist_chest/single/vit_tiny_patch16_224", _MED2_TS, True,
               " | dataset=medmnist_chest | seed=43", auc=(0.73, 0.82, 0.80))
    for folder, ts, agg in (("logs/local/single/vit_tiny_patch16_224", _BEN_TS, "f1_t05=0.31"),
                            ("logs/local/medmnist_chest/single/vit_tiny_patch16_224", _MED_TS,
                             "f1_t05=0.2,auc=0.91")):
        (root / folder / f"test_{ts}.csv").write_text(
            "class_idx,class_name,f1,precision,recall\n0,a,0.5,0.5,0.5\n\n"
            f"# aggregate,loss=0.2,{agg}\n", encoding="utf-8")
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
    return {"bigearthnet": by_ts[_BEN_TS], "medmnist": by_ts[_MED_TS],
            "medmnist2": by_ts[_MED2_TS]}


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


def test_curves_lead_with_the_runs_own_metric(labels):
    med = _run_app("run", labels["medmnist"])
    assert "Best Val AUC" in _metric_labels(med)       # MedMNIST is judged by AUC
    assert "Best Val F1" in _metric_labels(med)        # F1 stays as a secondary card
    assert any("Best Val AUC" in i.value for i in med.info) or \
        any("Best Val AUC" in w.value for w in med.warning)

    ben = _run_app("run", labels["bigearthnet"])
    assert "Best Val AUC" not in _metric_labels(ben)   # not logged by this run
    assert "Best Val F1" in _metric_labels(ben)


def test_compare_aggregates_seeds_of_one_dataset(labels):
    at = _run_app("compare", labels["medmnist"])
    assert not at.exception
    summary = at.dataframe[0].value       # default selection = both MedMNIST runs, one group
    assert len(summary) == 1               # the two seeds are aggregated into one row
    row = summary.iloc[0]
    assert row["Seeds"] == "42, 43" and "×2" in row["Run"]
    assert "±" in row["Best Val AUC"]      # mean ± std over the seeds
    assert row["Best Val AUC"] == "0.8000 ± 0.0283"   # mean ± std of 0.78 and 0.82
    overlay = next(m for m in at.multiselect if m.label == "Metrics to overlay")
    assert overlay.options[0] == "val_auc"            # the main metric leads
    at.toggle(key="compare_aggregate").set_value(False).run()
    assert not at.exception
    assert len(at.dataframe[0].value) == 2            # one row per run again


def test_overview_scopes_kpis_and_panels_to_the_dataset(labels):
    med = _run_app("overview", labels["medmnist"])
    assert not med.exception
    kpi = " ".join(m.value for m in med.markdown if "kpi-strip" in m.value)
    assert "Best val AUC" in kpi and "medmnist_chest" in " ".join(c.value for c in med.caption)
    assert "Sentinel-2" not in " ".join(c.value for c in med.caption)
    ben = _run_app("overview", labels["bigearthnet"])
    kpi_ben = " ".join(m.value for m in ben.markdown if "kpi-strip" in m.value)
    assert "Best val F1" in kpi_ben and "bigearthnet" in " ".join(c.value for c in ben.caption)
    # The fastest epoch lasts seconds: it is shown in seconds, never as "0.0 min".
    assert "11.8 s" in kpi_ben and "0.0 min" not in kpi_ben


def _picker(at: AppTest):
    """Options of the sidebar's run selector."""
    return at.selectbox(key="run_picker").options


def test_dataset_filter_narrows_run_list(labels):
    at = _run_app("overview", labels["bigearthnet"])
    ds = at.selectbox(key="run_dataset_filter")
    assert ds.options == ["All", "bigearthnet", "medmnist_chest"]
    assert len(_picker(at)) == 3
    ds.set_value("medmnist_chest").run()
    assert not at.exception
    # Only that dataset's runs are listed, plus the active run (filters never unselect it).
    assert set(_picker(at)) == {labels["medmnist"], labels["medmnist2"], labels["bigearthnet"]}
    at.selectbox(key="run_picker").set_value(labels["medmnist"]).run()
    assert at.session_state["run_label"] == labels["medmnist"]
    assert set(_picker(at)) == {labels["medmnist"], labels["medmnist2"]}


def test_run_chosen_outside_the_picker_is_not_reverted(labels):
    # Pick a run in the sidebar, then make another active as an Overview row click does:
    # the picker's stale value must not switch the active run back.
    at = _run_app("overview", labels["bigearthnet"])
    at.selectbox(key="run_picker").set_value(labels["medmnist"]).run()
    at.session_state["run_label"] = labels["medmnist2"]
    at.run()
    assert not at.exception
    assert at.session_state["run_label"] == labels["medmnist2"]
    assert at.selectbox(key="run_picker").value == labels["medmnist2"]


def test_search_filters_the_picker(labels):
    at = _run_app("overview", labels["medmnist"])
    at.text_input(key="run_search").set_value("seed 43").run()
    assert not at.exception
    assert set(_picker(at)) == {labels["medmnist2"], labels["medmnist"]}   # + active run
    at.text_input(key="run_search").set_value("no-such-run").run()
    assert _picker(at) == [labels["medmnist"]]


def test_title_is_the_project_name(labels):
    at = _run_app("overview", labels["bigearthnet"])
    assert any(m.value == "### ParaViT-Lab" for m in at.sidebar.markdown)


def test_direct_link_opens_run_and_tab(labels):
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["run"] = _MED2_TS
    at.query_params["tab"] = "compare"
    at.run()
    assert not at.exception
    assert at.session_state["run_label"] == labels["medmnist2"]
    assert any(md.value == PAGE_TITLES["compare"] for md in at.markdown)
    # The URL keeps pointing at what is on screen.
    assert at.query_params["run"] == _MED2_TS and at.query_params["tab"] == "compare"


def test_stale_link_falls_back_to_defaults(labels):
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["run"] = "00000000_000000"
    at.query_params["tab"] = "nonsense"
    at.run()
    assert not at.exception
    assert any(md.value == PAGE_TITLES["overview"] for md in at.markdown)


def test_overview_lists_test_results_by_dataset(labels):
    at = _run_app("overview", labels["bigearthnet"])
    assert not at.exception
    table = next(d.value for d in at.dataframe if "Evaluated" in d.value.columns)
    by_ds = {row["Dataset"]: row for _, row in table.iterrows()}
    assert by_ds["bigearthnet"]["Metric"] == "F1" and by_ds["bigearthnet"]["Test"] == 0.31
    assert by_ds["medmnist_chest"]["Metric"] == "AUC" and by_ds["medmnist_chest"]["Test"] == 0.91
