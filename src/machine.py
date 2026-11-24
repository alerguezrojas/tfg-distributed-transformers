"""Machine profiles: where the data and the checkpoints live on each computer.

An experiment YAML says WHAT to train; a profile (``configs/machines/<name>.yaml``) says WHERE this
computer keeps the dataset, the checkpoints and how many loader workers it affords. With a profile
active, ``apply_profile`` rewrites those keys of the config; with none, the config is left exactly
as written. Pure apart from reading the profile files (through ``load_config``).

Selection order: ``--machine`` > ``PARAVIT_MACHINE`` > the profile whose ``hosts`` patterns match
the hostname > none. The name ``none`` (flag or variable) disables profiles, even the host match.
"""
from __future__ import annotations

import fnmatch
import re
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from src.config_io import load_config

MACHINES_DIR = Path(__file__).resolve().parent.parent / "configs" / "machines"
ENV_VAR = "PARAVIT_MACHINE"
NO_PROFILE = "none"
DEFAULT_DATASET = "bigearthnet"      # config without data.name (same default as the registry)

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PROFILE_KEYS = {"env", "data_roots", "checkpoint_dir", "num_workers", "hosts"}
_FAMILY_KEY = "medmnist"             # one data_roots entry serves every medmnist_<subset>


@dataclass(frozen=True)
class MachineProfile:
    name: str
    env: str | None = None
    data_roots: dict = field(default_factory=dict)   # data.name -> "path" | {data key: value}
    checkpoint_dir: str | None = None
    num_workers: int | None = None
    hosts: tuple[str, ...] = ()


@dataclass(frozen=True)
class Selection:
    profile: MachineProfile
    reason: str                                      # "--machine", "PARAVIT_MACHINE" or "host ..."


def available_profiles(machines_dir: Path | None = None) -> list[str]:
    d = Path(machines_dir) if machines_dir else MACHINES_DIR
    return sorted(p.stem for p in d.glob("*.yaml")) if d.is_dir() else []


def parse_profile(name: str, raw: Mapping) -> MachineProfile:
    """Validate the contents of a profile (a typo in a key fails instead of being ignored)."""
    unknown = set(raw) - _PROFILE_KEYS
    if unknown:
        raise SystemExit(f"ERROR: perfil '{name}': claves desconocidas {sorted(unknown)} "
                         f"(válidas: {sorted(_PROFILE_KEYS)})")
    roots = raw.get("data_roots") or {}
    if not isinstance(roots, dict):
        raise SystemExit(f"ERROR: perfil '{name}': data_roots debe ser un diccionario")
    for key, val in roots.items():
        if not isinstance(val, (str, dict)):
            raise SystemExit(f"ERROR: perfil '{name}': data_roots.{key} debe ser una ruta o un "
                             f"diccionario de claves de data")
    hosts = raw.get("hosts") or []
    if not isinstance(hosts, list):
        raise SystemExit(f"ERROR: perfil '{name}': hosts debe ser una lista de patrones")
    workers = raw.get("num_workers")
    if workers is not None and (not isinstance(workers, int) or isinstance(workers, bool)
                                or workers < 0):
        raise SystemExit(f"ERROR: perfil '{name}': num_workers debe ser un entero >= 0")
    return MachineProfile(
        name=name, env=raw.get("env"), data_roots=dict(roots),
        checkpoint_dir=raw.get("checkpoint_dir"), num_workers=workers,
        hosts=tuple(str(h) for h in hosts))


def load_profile(name: str, machines_dir: Path | None = None) -> MachineProfile:
    d = Path(machines_dir) if machines_dir else MACHINES_DIR
    if not _NAME_RE.match(name):                     # no path separators: a name, not a path
        raise SystemExit(f"ERROR: nombre de perfil no válido: {name!r}")
    if not (d / f"{name}.yaml").is_file():
        raise SystemExit(f"ERROR: no existe el perfil de máquina '{name}' en {d} "
                         f"(disponibles: {', '.join(available_profiles(d)) or 'ninguno'})")
    return parse_profile(name, load_config(d / f"{name}.yaml"))


def select_machine(name: str | None = None, environ: Mapping | None = None,
                   hostname: str | None = None,
                   machines_dir: Path | None = None) -> Selection | None:
    """The active profile (see the module docstring) or ``None``."""
    environ = environ if environ is not None else {}
    for value, reason in ((name, "--machine"), (environ.get(ENV_VAR), ENV_VAR)):
        if value:
            if value.lower() == NO_PROFILE:
                return None
            return Selection(load_profile(value, machines_dir), reason)
    host = socket.gethostname() if hostname is None else hostname
    if not host:
        return None
    for profile_name in available_profiles(machines_dir):
        try:
            profile = load_profile(profile_name, machines_dir)
        except SystemExit as err:          # a broken profile must not break every command
            print(f"WARNING: perfil '{profile_name}' ignorado en la búsqueda por host: {err}",
                  file=sys.stderr)
            continue
        for pattern in profile.hosts:
            if fnmatch.fnmatch(host.lower(), pattern.lower()):
                return Selection(profile, f"host '{host}' ~ '{pattern}'")
    return None


def _dataset_entry(profile: MachineProfile, dataset: str):
    if dataset in profile.data_roots:
        return profile.data_roots[dataset]
    if dataset.startswith(_FAMILY_KEY + "_"):
        return profile.data_roots.get(_FAMILY_KEY)
    return None


def apply_profile(cfg: dict, profile: MachineProfile) -> list[str]:
    """Rewrite in place the machine-dependent keys of ``cfg``; return a line per change.

    A config whose ``output.env`` already is the profile's ``env`` belongs to this machine and
    is left untouched. Otherwise the profile translates what is machine-specific (data paths,
    ``checkpoint.dir``, ``env``) and only FILLS ``data.num_workers`` when the config has none.

    ``data_roots`` entry: a string sets ``data.root``; a mapping sets each ``data.<key>``
    verbatim, except ``<key>_dir``, which moves ``data.<key>`` to that folder keeping its file
    name (so ``metadata_dir`` serves both ``metadata.parquet`` and ``metadata_demo.parquet``).
    """
    changes: list[str] = []
    if profile.env and (cfg.get("output") or {}).get("env") == profile.env:
        return changes          # already this machine's config: nothing to translate or change

    def put(section: str, key: str, value) -> None:
        sec = cfg.setdefault(section, {})
        old = sec.get(key)
        if old != value:
            sec[key] = value
            changes.append(f"{section}.{key}: {old!r} -> {value!r}")

    dataset = str((cfg.get("data") or {}).get("name", DEFAULT_DATASET)).lower()
    entry = _dataset_entry(profile, dataset)
    if isinstance(entry, str):
        entry = {"root": entry}
    for key, value in (entry or {}).items():
        if key.endswith("_dir") and len(key) > 4:
            current = (cfg.get("data") or {}).get(key[:-4])
            if current:
                put("data", key[:-4], str(Path(str(value)) / Path(str(current)).name))
        else:
            put("data", key, value)
    if profile.num_workers is not None and "num_workers" not in (cfg.get("data") or {}):
        put("data", "num_workers", profile.num_workers)    # fills a gap, never overrides
    if profile.checkpoint_dir:
        put("checkpoint", "dir", profile.checkpoint_dir)
    if profile.env:
        put("output", "env", profile.env)
    return changes


def describe(selection: Selection | None, changes: list[str]) -> str:
    if selection is None:
        return "Machine profile: none (config used as written)"
    head = f"Machine profile: {selection.profile.name} ({selection.reason})"
    return head + ("".join(f"\n  {c}" for c in changes) if changes else "; nothing to change")
