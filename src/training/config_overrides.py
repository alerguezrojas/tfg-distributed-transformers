"""Sobrescrituras de la config desde la línea de comandos, comunes a los scripts de entrenamiento."""

import logging
import os
import re

import yaml

from src.config_io import load_config
from src.machine import apply_profile, describe, select_machine

logger = logging.getLogger(__name__)


def add_data_root_override(parser) -> None:
    """Añade ``--data-root`` (lo comparten los scripts de entrenamiento y ``eval.py``)."""
    parser.add_argument("--data-root", type=str, help="Override data.root (only that key: absolute data.metadata, "
                        "data.splits or data.mosaics paths in the YAML are not moved)")


def add_machine_options(parser) -> None:
    """Añade ``--machine`` y ``--set`` (perfil de máquina y sobrescritura genérica de la config)."""
    parser.add_argument("--machine", type=str, help="Machine profile (configs/machines/<name>.yaml; "
                        "'none' = no profile). Default: $PARAVIT_MACHINE, else the profile whose "
                        "hosts match this hostname, else none")
    parser.add_argument("--set", dest="set_", action="append", metavar="KEY=VALUE", default=[],
                        help="Override any config key, e.g. --set training.lr=1e-4 --set "
                        "data.num_workers=2 (repeatable; the value is read as YAML; applied "
                        "after the machine profile)")


def add_common_overrides(parser) -> None:
    """Añade al parser las opciones que sobrescriben la config en todas las estrategias."""
    add_data_root_override(parser)
    add_machine_options(parser)
    parser.add_argument("--seed", type=int, help="Override training.seed (repetitions)")


def apply_common_overrides(cfg: dict, args) -> None:
    """Aplica ``--data-root``, ``--epochs`` y ``--seed`` sobre ``cfg`` (in situ).

    Si ``--epochs`` deja ``epochs <= warmup_epochs`` se baja ``warmup_epochs`` a
    ``epochs - 1`` (mínimo 0) en vez de dejar que el validador falle; un YAML que
    ya traiga esa combinación sigue fallando en el validador.
    """
    if getattr(args, "data_root", None):
        cfg["data"]["root"] = args.data_root
    if getattr(args, "seed", None) is not None:
        cfg["training"]["seed"] = args.seed
    if getattr(args, "epochs", None):
        train = cfg["training"]
        train["epochs"] = args.epochs
        warmup = train.get("warmup_epochs")
        if isinstance(warmup, int) and warmup >= args.epochs:
            train["warmup_epochs"] = max(0, args.epochs - 1)
            logger.warning(
                "--epochs %d <= warmup_epochs %d: se baja warmup_epochs a %d",
                args.epochs, warmup, train["warmup_epochs"])


# YAML 1.1 (PyYAML) reads "1e-4" as a string: a float needs a dot and a signed exponent. This
# loader also resolves it (unquoted, at any depth: lists, dicts) as a float; quoted stays a string.
class _SetLoader(yaml.SafeLoader):
    pass


_SetLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float", re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)[eE][+-]?\d+$"),
    list("-+0123456789."))


def parse_set_value(text: str):
    """The value of a ``--set`` as YAML would read it (numbers, booleans, lists, null…), except
    that scientific notation without a dot (``1e-4``) is a float, not a string."""
    return yaml.load(text, Loader=_SetLoader)  # noqa: S506 - SafeLoader subclass


def apply_set_overrides(cfg: dict, items: list[str] | None) -> list[str]:
    """Apply ``key.sub=value`` items in order on ``cfg`` (in place); returns a line per change.
    Intermediate sections are created; a key that did not exist is flagged (likely a typo)."""
    lines = []
    for item in items or []:
        key, sep, raw = item.partition("=")
        key = key.strip()
        parts = key.split(".")
        if not sep or not key or not all(parts):
            raise SystemExit(f"ERROR: --set espera clave.sub=valor, no '{item}'")
        try:
            value = parse_set_value(raw)
        except yaml.YAMLError as err:
            raise SystemExit(f"ERROR: --set {key}: valor YAML no válido ({err})") from None
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise SystemExit(f"ERROR: --set {key}: '{part}' no es una sección de la config")
        old = node.get(parts[-1], "<new key>")
        node[parts[-1]] = value
        lines.append(f"{key}: {old!r} -> {value!r}" + ("  (new key: check the spelling)"
                                                       if old == "<new key>" else ""))
    return lines


def resolve_machine(args, cfg: dict):
    """Apply the active machine profile to ``cfg``; returns ``(selection, changes)``."""
    selection = select_machine(getattr(args, "machine", None), os.environ)
    changes = apply_profile(cfg, selection.profile) if selection else []
    return selection, changes


def prepare_config(args, *, common: bool = True, path: str | None = None) -> dict:
    """The config of a run: ``args.config`` loaded, then the machine profile, then the explicit
    options (``--data-root``, ``--seed``, ``--epochs``) and finally ``--set``. The single place
    where every entry point (training scripts, eval, predict, benchmark) builds its config."""
    cfg = load_config(path or args.config)
    selection, changes = resolve_machine(args, cfg)
    if common:
        apply_common_overrides(cfg, args)
    set_lines = apply_set_overrides(cfg, getattr(args, "set_", None))
    if os.environ.get("RANK", "0") == "0":           # one line per run, not one per DDP rank
        print(describe(selection, changes))
        for line in set_lines:
            print(f"  --set {line}")
    return cfg
