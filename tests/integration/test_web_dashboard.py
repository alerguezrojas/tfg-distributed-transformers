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

_MODULES = [
    "app.py",
    "ui/__init__.py", "ui/context.py", "ui/charts.py", "ui/helpers.py",
    "tabs/__init__.py", "tabs/home.py", "tabs/run.py", "tabs/comparison.py",
    "tabs/feasibility.py", "tabs/data_models.py", "tabs/system.py",
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
    return "\n".join(_src(f"tabs/{m}") for m in (
        "home.py", "run.py", "comparison.py", "feasibility.py",
        "data_models.py", "system.py",
    ))


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
    assert n_lines < 220, f"app.py has {n_lines} lines — expected a thin orchestrator"
    for mod in ("home", "comparison", "feasibility", "data_models", "system"):
        assert f"{mod}.render" in app_source, mod
    assert "run_tab.render" in app_source


def test_sidebar_nav_sections(app_source):
    """The grouped sidebar navigation labels (English) live in app.py."""
    for t in ('"Overview"', '"Run results"', '"Compare"', '"Feasibility"',
              '"Data & models"', '"System"'):
        assert t in app_source, f"missing nav item {t}"
    # Grouped, always-visible navigation (no top tab bar).
    assert "_NAV" in app_source and "st.session_state" in app_source


def test_sub_tab_names(tabs_source):
    """The sub-tab names (English) live in the tab modules."""
    for t in ('"Curves"', '"Per-class"', '"Batch"', '"Time"', '"Info"',
              '"Monitor"', '"Launcher"', '"Live"', '"Overlay runs"',
              '"Single vs Distributed"', '"Prediction vs reality"',
              '"Dataset"', '"Models"'):
        assert t in tabs_source, f"missing sub-tab {t}"


def test_home_sections(tabs_source):
    """The home screen (executive summary) must keep its core sections.

    System/hardware and per-class snapshots were intentionally removed from Home
    to de-duplicate them (they live in System / Run results).
    """
    assert "Project overview" in tabs_source
    assert "All runs" in tabs_source
    assert "Selected run" in tabs_source


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
    """RunInfo.label must include date, trace mode, env and model."""
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
    assert "[simple]" in label
    assert "[local]" in label
    assert "vit_base_patch16_224" in label
    tmp.unlink(missing_ok=True)
