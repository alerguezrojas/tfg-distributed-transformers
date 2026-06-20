"""Web dashboard tests (English UI, modular layout).

Checks:
- That app.py and every tab/ui module parse without errors.
- That the key helpers (_safe_max, _dur_str, _safe_val_at_best) work.
- That RunInfo no longer has PNG attributes (plot_path, perclass_paths, etc.).
- That the English tab/section labels are present in the right modules.
- That _show / _dl_csv / _PLOTLY_CFG live in ui/charts.py and every chart
  goes through _show (a single direct st.plotly_chart call, inside _show).
- That app.py is a thin orchestrator (not the old 3000-line monolith).
"""
import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "src" / "web"
sys.path.insert(0, str(ROOT))


# ── internal helpers copied from the app to test in isolation ─────────────────

def _safe_max(series: pd.Series) -> float:
    valid = series.dropna()
    return float(valid.max()) if not valid.empty else float("nan")


def _safe_idxmax(series: pd.Series):
    valid = series.dropna()
    return valid.idxmax() if not valid.empty else None


def _safe_val_at_best(df: pd.DataFrame, metric_col: str, target_col: str):
    if metric_col not in df.columns or target_col not in df.columns:
        return None
    idx = _safe_idxmax(df[metric_col])
    if idx is None:
        return None
    v = df.loc[idx, target_col]
    return None if pd.isna(v) else v


def _dur_str(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h {m}m"


# ── helper tests ──────────────────────────────────────────────────────────────


def test_safe_max_normal():
    s = pd.Series([0.5, 0.7, 0.6])
    assert _safe_max(s) == pytest.approx(0.7)


def test_safe_max_all_nan():
    s = pd.Series([float("nan"), float("nan")])
    assert pd.isna(_safe_max(s))


def test_safe_max_with_nans():
    s = pd.Series([float("nan"), 0.8, float("nan")])
    assert _safe_max(s) == pytest.approx(0.8)


def test_safe_val_at_best_returns_correct_value():
    df = pd.DataFrame({"epoch": [1, 2, 3], "val_f1": [0.5, 0.8, 0.6], "val_loss": [0.3, 0.2, 0.25]})
    result = _safe_val_at_best(df, "val_f1", "val_loss")
    assert result == pytest.approx(0.2)


def test_safe_val_at_best_missing_column():
    df = pd.DataFrame({"epoch": [1, 2], "val_f1": [0.5, 0.6]})
    result = _safe_val_at_best(df, "val_f1", "nonexistent")
    assert result is None


def test_dur_str_exact():
    assert _dur_str(3600) == "1h 0m"
    assert _dur_str(3661) == "1h 1m"
    assert _dur_str(90) == "0h 1m"
    assert _dur_str(0) == "0h 0m"


# ── module layout ─────────────────────────────────────────────────────────────

_FEAS_PKG = [
    "tabs/feasibility/__init__.py", "tabs/feasibility/predict.py",
    "tabs/feasibility/validate.py", "tabs/feasibility/report.py",
    "tabs/feasibility/study.py", "tabs/feasibility/ddp.py", "tabs/feasibility/run_form.py",
]
_CMP_PKG = [
    "tabs/comparison/__init__.py", "tabs/comparison/_common.py",
    "tabs/comparison/summary.py", "tabs/comparison/perclass.py",
    "tabs/comparison/speedup.py", "tabs/comparison/charts.py",
]
_RUN_PKG = [
    "tabs/run/__init__.py", "tabs/run/curves.py", "tabs/run/perclass.py",
    "tabs/run/confusions.py", "tabs/run/batch.py", "tabs/run/details.py",
]
_MODULES = [
    "app.py",
    "ui/__init__.py", "ui/context.py", "ui/charts.py", "ui/helpers.py",
    "tabs/__init__.py", "tabs/home.py",
    "tabs/analysis.py", "tabs/dataset.py", "tabs/data_models.py",
    *_RUN_PKG,
    *_CMP_PKG,
    *_FEAS_PKG,
]


def _src(rel: str) -> str:
    return (WEB / rel).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_source() -> str:
    return _src("app.py")


@pytest.fixture(scope="module")
def charts_source() -> str:
    return _src("ui/charts.py")


@pytest.fixture(scope="module")
def tabs_source() -> str:
    mods = ["home.py", "analysis.py",
            "dataset.py", "data_models.py",
            "run/curves.py", "run/perclass.py", "run/confusions.py",
            "run/batch.py", "run/details.py", "run/__init__.py",
            "comparison/summary.py", "comparison/perclass.py",
            "comparison/speedup.py", "comparison/charts.py",
            "comparison/__init__.py",
            "feasibility/validate.py", "feasibility/report.py",
            "feasibility/study.py", "feasibility/ddp.py",
            "feasibility/run_form.py", "feasibility/predict.py",
            "feasibility/__init__.py"]
    return "\n".join(_src(f"tabs/{m}") for m in mods)


@pytest.fixture(scope="module")
def all_web_source() -> str:
    return "\n".join(_src(m) for m in _MODULES)


def test_all_modules_parse():
    """Every web module must parse without syntax errors."""
    for rel in _MODULES:
        ast.parse(_src(rel), filename=rel)


def test_app_is_thin_orchestrator(app_source):
    """app.py must be a thin orchestrator that dispatches to the page modules."""
    n_lines = len(app_source.splitlines())
    # Thin = page config + CSS + sidebar + dispatch (not the old 3000-line monolith).
    assert n_lines < 280, f"app.py has {n_lines} lines — expected a thin orchestrator"
    for mod in ("home", "comparison", "analysis", "feasibility", "dataset"):
        assert f"{mod}.render" in app_source, mod
    assert "run_tab.render" in app_source


def test_sidebar_nav_sections(app_source):
    """The single-level icon-menu navigation labels (English) live in app.py."""
    for t in ('"Overview"', '"Run results"', '"Compare"', '"Analysis"', '"Feasibility"',
              '"Dataset"'):
        assert t in app_source, f"missing nav item {t}"
    # Icon menu (streamlit-option-menu), single level, session-state driven.
    assert "option_menu" in app_source
    assert "_NAV_KEYS" in app_source and "st.session_state" in app_source


def test_sub_tab_names(tabs_source):
    """The sub-tab names (English) live in the tab modules.

    Compare has no sub-tabs anymore: it is ONE unified section (multiselect →
    summary + speedup vs baseline + radar + energy + overlays).
    """
    for t in ('"Curves"', '"Per-class"', '"Confusions"', '"Batch"', '"Details"',
              '"Predict"', '"Validate"'):
        assert t in tabs_source, f"missing sub-tab {t}"
    # The unified Compare keeps its key sections.
    for s in ("Speedup analysis", "Baseline run (= 1.00×)", "Energy consumption",
              "Metrics to overlay"):
        assert s in tabs_source, f"missing Compare section {s}"


def test_home_sections(tabs_source):
    """The Overview hub: KPIs, the active-run card, relevant charts (no
    redundant nav cards — navigation is the sidebar) and a selectable All-runs
    table with sparklines."""
    assert '"## Overview"' in tabs_source
    assert "All runs" in tabs_source
    assert "Active run" in tabs_source
    # Relevant charts: the quality/cost landscape + the per-model ceiling.
    assert "Quality vs training cost" in tabs_source
    assert "Best Val F1 reached per model" in tabs_source
    assert "LineChartColumn" in tabs_source
    assert 'selection_mode="single-row"' in tabs_source


def test_show_and_dl_csv_defined_in_charts(charts_source):
    """The _show() and _dl_csv() helpers live in ui/charts.py."""
    assert "def _show(" in charts_source
    assert "def _dl_csv(" in charts_source


def test_plotly_config_in_charts(charts_source):
    """The Plotly config with PNG download lives in ui/charts.py."""
    assert "_PLOTLY_CFG" in charts_source
    assert "toImageButtonOptions" in charts_source
    assert '"format": "png"' in charts_source


def test_no_pil_import(all_web_source):
    """No web module should import PIL (PNGs were removed from the flow)."""
    assert "from PIL import" not in all_web_source
    assert "import PIL" not in all_web_source


def test_download_buttons_present(tabs_source):
    """The tab modules must offer download buttons (_dl_csv)."""
    count = tabs_source.count("_dl_csv(")
    assert count >= 5, f"expected >=5 download buttons, found {count}"


def test_single_direct_plotly_chart_call(all_web_source):
    """All charts must go through _show(); the only direct st.plotly_chart
    call must be inside _show() in ui/charts.py."""
    show_calls = all_web_source.count("_show(")
    assert show_calls > 10, f"expected >10 _show() calls, found {show_calls}"

    direct = [
        (i + 1, line.strip())
        for i, line in enumerate(all_web_source.splitlines())
        if "st.plotly_chart(" in line
    ]
    assert len(direct) == 1, (
        f"expected exactly 1 direct st.plotly_chart call (inside _show), found {len(direct)}: {direct}"
    )
    assert "cfg" in direct[0][1], "the only st.plotly_chart call must pass config=cfg (inside _show)"


# ── RunInfo (no PNG attributes) ───────────────────────────────────────────────


def test_run_info_has_no_png_attributes():
    """RunInfo must not have any PNG path attributes."""
    from src.web.run_registry import RunInfo
    import inspect
    source = inspect.getsource(RunInfo)
    assert "plot_path" not in source
    assert "perclass_paths" not in source
    assert "confusion_matrix_paths" not in source


def test_run_info_csv_attributes_present():
    """RunInfo must have the expected CSV attributes."""
    from src.web.run_registry import RunInfo
    import dataclasses
    fields = {f.name for f in dataclasses.fields(RunInfo)}
    assert "epoch_csv_path" in fields
    assert "batch_csv_path" in fields
    assert "perclass_csv_path" in fields
    assert "confusion_matrix_csv_path" in fields


def test_run_info_label_format():
    """RunInfo.label: compact — date+time, env, short model; the defaults
    (simple trace, fp32, single mode) are implicit; deep/mode/precision are
    tagged so e.g. the Kaggle AMP runs are distinguishable in the selector."""
    from src.web.run_registry import RunInfo
    from pathlib import Path
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        tmp = Path(f.name)

    info = RunInfo(
        timestamp="27052026_210223",
        log_path=tmp,
        trace_mode="simple",
        env="local",
        mode="single",
        model="vit_base_patch16_224",
    )
    label = info.label
    assert label.startswith("27/05/2026 21:02")
    assert "[local]" in label
    assert "vit_base" in label and "_patch16_224" not in label  # short model
    assert "[simple]" not in label                              # default: implicit

    # Non-default runs get tags: mode, precision (Tensor cores) and deep trace.
    tagged = RunInfo(
        timestamp="10062026_211814", log_path=tmp, trace_mode="deep",
        env="kaggle", mode="ddp", model="vit_base_patch16_224", precision="amp",
    )
    assert "[ddp]" in tagged.label
    assert "[amp]" in tagged.label
    assert "[deep]" in tagged.label
    fp32 = RunInfo(
        timestamp="10062026_173904", log_path=tmp, trace_mode="simple",
        env="kaggle", mode="single", model="vit_base_patch16_224", precision="fp32",
    )
    assert "[fp32]" not in fp32.label                           # default: implicit
    tmp.unlink(missing_ok=True)


def test_read_precision_from_log(tmp_path):
    """discover_runs must read precision= from the log's config header line."""
    from src.web.run_registry import _read_precision

    log = tmp_path / "train_10062026_203609.log"
    log.write_text(
        "2026-06-10 20:36:11 [INFO ] Configuración: modelo=vit_base_patch16_224 "
        "| batch=96/GPU (global=96) | epochs=15 | lr=0.0001 | precision=amp "
        "| train=5000 | val=1500\n",
        encoding="utf-8",
    )
    assert _read_precision(log) == "amp"

    legacy = tmp_path / "train_27052026_210223.log"
    legacy.write_text(
        "2026-05-27 21:02:23 [INFO ] Configuración: modelo=x | batch=64\n",
        encoding="utf-8",
    )
    assert _read_precision(legacy) == ""
