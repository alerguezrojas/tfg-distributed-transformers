"""Dataset registry — look up a :class:`DatasetSpec` by name.

The config picks the dataset with ``data.name``; configs written before the
registry existed have no ``name`` and keep meaning BigEarthNet.

Built-in datasets register themselves when their module is imported; the
registry imports them lazily on first lookup so that importing it stays cheap.
"""
from __future__ import annotations

import importlib

from src.data.spec import DatasetSpec

DEFAULT_DATASET = "bigearthnet"

# Modules that define (and register) the built-in datasets.
_BUILTIN_MODULES = ("src.data.bigearthnet", "src.data.medmnist")

_REGISTRY: dict[str, DatasetSpec] = {}
_builtins_loaded = False


def register_dataset(spec: DatasetSpec) -> DatasetSpec:
    """Add ``spec`` to the registry (idempotent for the same object)."""
    current = _REGISTRY.get(spec.name)
    if current is not None and current is not spec:
        raise ValueError(f"dataset {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec
    return spec


def _load_builtins() -> None:
    global _builtins_loaded
    if not _builtins_loaded:
        for module in _BUILTIN_MODULES:
            importlib.import_module(module)
        _builtins_loaded = True


def available_datasets() -> list[str]:
    _load_builtins()
    return sorted(_REGISTRY)


def get_dataset_spec(name: str) -> DatasetSpec:
    _load_builtins()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; available: "
                         f"{available_datasets()}") from None


def dataset_name_from_config(cfg: dict) -> str:
    return str((cfg.get("data") or {}).get("name", DEFAULT_DATASET)).lower()


def dataset_spec_from_config(cfg: dict) -> DatasetSpec:
    """Spec selected by ``data.name`` (default: BigEarthNet)."""
    return get_dataset_spec(dataset_name_from_config(cfg))
