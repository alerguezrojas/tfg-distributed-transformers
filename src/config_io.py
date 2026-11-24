"""Loading a training config (YAML) given on the command line, with basic checks."""
from __future__ import annotations

from pathlib import Path

import yaml

YAML_SUFFIXES = (".yaml", ".yml")


def load_config(path: str | Path) -> dict:
    """The YAML at ``path`` as a dict. It must be an existing regular file with a YAML suffix
    and hold a mapping; anything else raises ``SystemExit`` with a clear message (these are
    command-line tools: a config the user chose, read as given, never substituted)."""
    p = Path(path).expanduser()
    if p.suffix.lower() not in YAML_SUFFIXES:
        raise SystemExit(f"ERROR: la config tiene que ser un .yaml: {path}")
    if not p.is_file():
        raise SystemExit(f"ERROR: no existe la config {path}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: {path} no es una config (se esperaba un diccionario YAML)")
    return data
