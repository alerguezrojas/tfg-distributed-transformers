"""Unit tests for the optional Spanish translation layer (src/web/ui/i18n.py)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.web.ui import i18n


def test_translate_known_and_unknown():
    assert i18n._tr("Home") == "Inicio"
    assert i18n._tr("Project overview") == "Vista general del proyecto"
    # Unknown strings pass through unchanged (dynamic text / identifiers).
    assert i18n._tr("Best Val F1: 0.68") == "Best Val F1: 0.68"
    assert i18n._tr(123) == 123


def test_install_es_then_restore_en():
    from streamlit.delta_generator import DeltaGenerator
    orig_markdown = DeltaGenerator.markdown
    i18n.install("es")
    assert DeltaGenerator.markdown is not orig_markdown  # wrapped
    i18n.install("en")
    assert DeltaGenerator.markdown is orig_markdown        # restored
    # Idempotent: a second restore keeps the original.
    i18n.install("en")
    assert DeltaGenerator.markdown is orig_markdown


def test_wrap_first_translates_module_level_call():
    """st.* call (no self): the text is args[0]."""
    seen = {}
    wrapped = i18n._wrap_first(lambda *a, **k: seen.update(args=a))
    wrapped("Project overview")
    assert seen["args"][0] == "Vista general del proyecto"
    wrapped("Unknown phrase")
    assert seen["args"][0] == "Unknown phrase"


def test_wrap_first_translates_bound_method_call():
    """Class-method call: args[0] is the DeltaGenerator self, text is args[1]."""
    from streamlit.delta_generator import DeltaGenerator
    seen = {}
    wrapped = i18n._wrap_first(lambda self, *a, **k: seen.update(text=a[0] if a else None))
    dummy = DeltaGenerator.__new__(DeltaGenerator)   # an instance without running __init__
    wrapped(dummy, "Home")
    assert seen["text"] == "Inicio"


def test_wrap_tabs_translates_list():
    seen = {}
    wrapped = i18n._wrap_tabs(lambda *a, **k: seen.update(labels=a[0]))
    wrapped(["Home", "Run", "Comparison"])
    assert seen["labels"] == ["Inicio", "Run", "Comparativa"]


def test_top_tabs_present():
    for t in ("Home", "Comparison", "Feasibility", "Data & models", "System"):
        assert t in i18n.TRANSLATIONS
