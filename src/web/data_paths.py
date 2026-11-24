"""Where the BigEarthNet data of THIS machine lives, read from the machine profiles.

The paths are not constants of the web: they come from ``configs/machines/<profile>.yaml``
(``src/machine.py``): the active profile first (``PARAVIT_MACHINE``, else the host), then
the other profiles, so a checkout mounting the data under another profile's path still
finds it. When nothing exists, ``NOT_AVAILABLE`` says what to do. Pure (no Streamlit).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.machine import MACHINES_DIR, MachineProfile, available_profiles, load_profile, select_machine

DATASET = "bigearthnet"
METADATA_FILE = "metadata.parquet"

NOT_AVAILABLE = ("BigEarthNet-S2 is not available on this machine. Add its path to a machine "
                 "profile (`data_roots.bigearthnet` in `configs/machines/`, see "
                 "`example.yaml`) and select it with `PARAVIT_MACHINE`.")


@dataclass(frozen=True)
class BigEarthNetPaths:
    metadata: str | None      # metadata.parquet, if it exists
    root: str | None          # BigEarthNet-S2 folder, if it exists


def _profiles(environ, machines_dir: Path) -> list[MachineProfile]:
    """The active profile (if any) followed by the others."""
    out: list[MachineProfile] = []
    try:
        sel = select_machine(None, environ, machines_dir=machines_dir)
    except SystemExit:               # a bad PARAVIT_MACHINE must not break the page
        sel = None
    if sel is not None:
        out.append(sel.profile)
    for name in available_profiles(machines_dir):
        if sel is not None and name == sel.profile.name:
            continue
        try:
            out.append(load_profile(name, machines_dir))
        except SystemExit:
            continue
    return out


def bigearthnet_paths(environ=None, machines_dir: Path | None = None) -> BigEarthNetPaths:
    """First existing metadata file and image root among the profiles' ``data_roots``."""
    environ = os.environ if environ is None else environ
    meta = root = None
    for profile in _profiles(environ, Path(machines_dir) if machines_dir else MACHINES_DIR):
        entry = profile.data_roots.get(DATASET)
        if isinstance(entry, str):
            entry = {"root": entry}
        if not isinstance(entry, dict):
            continue
        if root is None and entry.get("root") and Path(str(entry["root"])).is_dir():
            root = str(entry["root"])
        folder = entry.get("metadata_dir")
        if meta is None and folder and (Path(str(folder)) / METADATA_FILE).is_file():
            meta = str(Path(str(folder)) / METADATA_FILE)
    return BigEarthNetPaths(metadata=meta, root=root)
