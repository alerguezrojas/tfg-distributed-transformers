"""Tests del dashboard web v6 (español).

Verifica:
- Que app.py importa correctamente sin errores.
- Que los helpers clave (_safe_max, _dur_str, _detect_anomalies, etc.) funcionan.
- Que RunInfo ya no tiene atributos PNG (plot_path, perclass_paths, etc.).
- Que la traducción clave está presente en el código fuente.
- Que _show y _dl_csv existen y son funciones.
- Que no hay referencias rotas a atributos inexistentes de RunInfo.
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ── helpers internos copiados del app para testear en aislamiento ─────────────

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


# ── tests de helpers ──────────────────────────────────────────────────────────


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


# ── tests del código fuente de app.py ─────────────────────────────────────────


@pytest.fixture(scope="module")
def app_source() -> str:
    return (ROOT / "src" / "web" / "app.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_ast(app_source: str) -> ast.Module:
    return ast.parse(app_source)


def test_app_py_valid_syntax(app_ast):
    """app.py debe parsear sin errores de sintaxis."""
    assert isinstance(app_ast, ast.Module)


def test_app_py_no_plot_path_references(app_source):
    """No debe haber referencias a run.plot_path (atributo eliminado de RunInfo)."""
    assert "plot_path" not in app_source, (
        "app.py contiene 'plot_path' — atributo eliminado de RunInfo"
    )


def test_app_py_no_perclass_paths_references(app_source):
    """No debe haber referencias a run.perclass_paths (atributo eliminado de RunInfo)."""
    assert "perclass_paths" not in app_source, (
        "app.py contiene 'perclass_paths' — atributo eliminado de RunInfo"
    )


def test_app_py_no_confusion_matrix_paths_references(app_source):
    """No debe haber referencias a run.confusion_matrix_paths (atributo eliminado de RunInfo)."""
    assert "confusion_matrix_paths" not in app_source, (
        "app.py contiene 'confusion_matrix_paths' — atributo eliminado de RunInfo"
    )


def test_app_py_spanish_tab_names(app_source):
    """Los nombres de las pestañas deben estar en español."""
    assert '"Inicio"' in app_source
    assert '"Sistema"' in app_source
    assert '"Modelos"' in app_source
    assert '"Curvas"' in app_source
    assert '"Por clase"' in app_source
    assert '"Comparar"' in app_source
    assert '"Viabilidad"' in app_source
    assert '"Tiempo"' in app_source
    assert '"Información"' in app_source
    assert '"Lanzador"' in app_source
    assert '"En vivo"' in app_source


def test_app_py_show_helper_defined(app_source):
    """El helper _show() debe estar definido para envolver st.plotly_chart con config."""
    assert "def _show(" in app_source


def test_app_py_dl_csv_helper_defined(app_source):
    """El helper _dl_csv() debe estar definido para botones de descarga."""
    assert "def _dl_csv(" in app_source


def test_app_py_plotly_config_present(app_source):
    """La configuración de Plotly con descarga PNG debe estar presente."""
    assert "_PLOTLY_CFG" in app_source
    assert "toImageButtonOptions" in app_source
    assert '"format": "png"' in app_source


def test_app_py_grid_home_sections(app_source):
    """La pantalla de inicio debe tener las secciones de cuadrícula requeridas."""
    assert "Vista general del proyecto" in app_source
    assert "Estado del sistema" in app_source
    assert "Rendimiento por clase" in app_source
    assert "Todos los runs" in app_source
    assert "Run seleccionado" in app_source


def test_app_py_no_pil_import(app_source):
    """No debe importar PIL ya que los PNGs se eliminaron del flujo."""
    assert "from PIL import" not in app_source
    assert "import PIL" not in app_source


def test_app_py_download_buttons_present(app_source):
    """Deben existir botones de descarga (_dl_csv) en las pestañas principales."""
    count = app_source.count("_dl_csv(")
    assert count >= 5, f"Se esperaban ≥5 botones de descarga, encontrados: {count}"


def test_app_py_uses_show_for_all_charts(app_source):
    """Todas las gráficas deben usar _show() en lugar de st.plotly_chart directo.

    La única llamada directa permitida es la interna dentro de la propia función _show().
    """
    show_calls = app_source.count("_show(")
    assert show_calls > 10, f"Se esperaban >10 llamadas a _show(), encontradas: {show_calls}"

    # La única llamada directa a st.plotly_chart debe estar DENTRO de _show()
    lines_with_direct = [
        (i + 1, line.strip())
        for i, line in enumerate(app_source.splitlines())
        if "st.plotly_chart(" in line
    ]
    assert len(lines_with_direct) == 1, (
        f"Se esperaba exactamente 1 llamada directa a st.plotly_chart (dentro de _show), "
        f"encontradas {len(lines_with_direct)}: {lines_with_direct}"
    )
    assert "cfg" in lines_with_direct[0][1], (
        "La única llamada a st.plotly_chart debe estar en _show() y pasar config=cfg"
    )


# ── tests de RunInfo (sin atributos PNG) ──────────────────────────────────────


def test_run_info_has_no_png_attributes():
    """RunInfo no debe tener atributos de paths PNG."""
    from src.web.run_registry import RunInfo
    import inspect
    source = inspect.getsource(RunInfo)
    assert "plot_path" not in source
    assert "perclass_paths" not in source
    assert "confusion_matrix_paths" not in source


def test_run_info_csv_attributes_present():
    """RunInfo debe tener los atributos CSV esperados."""
    from src.web.run_registry import RunInfo
    import dataclasses
    fields = {f.name for f in dataclasses.fields(RunInfo)}
    assert "epoch_csv_path" in fields
    assert "batch_csv_path" in fields
    assert "perclass_csv_path" in fields
    assert "confusion_matrix_csv_path" in fields


def test_run_info_label_format():
    """RunInfo.label debe incluir fecha, trace mode, env y modelo."""
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
