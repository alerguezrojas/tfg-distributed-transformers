"""Política de checkpoints: escritura atómica y localización de ficheros.

Cada run escribe en su propia carpeta (``.../<modelo>/<DDMMYYYY_HHMMSS>/``) y, por
defecto, solo conserva ``best.pt`` y ``last.pt``. ``keep: all`` mantiene el esquema
antiguo (un ``checkpoint_epoch_NNN.pt`` por época mejorada).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

KEEP_MODES = ("best_last", "all")
DEFAULT_KEEP = "best_last"
BEST_NAME = "best.pt"
LAST_NAME = "last.pt"
_EPOCH_GLOB = "checkpoint_epoch_*.pt"


def atomic_save(state: dict, path: Path) -> None:
    """``torch.save`` a un temporal y ``os.replace``: un disco lleno o un corte no dejan
    un fichero a medias en ``path`` (el anterior sigue intacto)."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


_WRAPPER_PREFIXES = ("module.", "base.")   # DDP/DataParallel, ModelParallelViT


def unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """El módulo bajo ``DistributedDataParallel``/``DataParallel`` (o el propio modelo)."""
    wrappers = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)
    return model.module if isinstance(model, wrappers) else model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """El clasificador de verdad bajo DDP y bajo ``ModelParallelViT`` (su atributo ``base``):
    sus pesos se guardan sin prefijo y cargan igual en una GPU, en DDP o en dos etapas."""
    model = unwrap_ddp(model)
    return getattr(model, "base", model)


def strip_wrapper_prefix(state_dict: dict) -> dict:
    """Quita ``module.`` y ``base.`` de las claves mientras **todas** lo lleven: los
    checkpoints DDP y de paralelismo de modelo anteriores al arreglo guardaban el
    ``state_dict`` del envoltorio y no cargaban en un modelo suelto."""
    while state_dict:
        prefix = next((p for p in _WRAPPER_PREFIXES if all(k.startswith(p) for k in state_dict)), None)
        if prefix is None:
            break
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _newest(files: list[Path]) -> Path:
    return max(files, key=lambda p: (p.stat().st_mtime, p.name))


def resolve_checkpoint(path: str | Path, prefer: str = "best") -> Path:
    """Devuelve el fichero de checkpoint al que apunta ``path``.

    - Un fichero (incluidos los ``checkpoint_epoch_*.pt`` antiguos) se devuelve tal cual.
    - Una carpeta de run: ``best.pt`` o ``last.pt`` según ``prefer`` (con el otro de
      reserva); si es una carpeta antigua, el ``checkpoint_epoch_*.pt`` más reciente.
    - Una carpeta que contiene varios runs: el run con el checkpoint más reciente.
    """
    p = Path(path)
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"No existe el checkpoint: {p}")
    names = (LAST_NAME, BEST_NAME) if prefer == "last" else (BEST_NAME, LAST_NAME)
    candidates = [f for f in p.rglob("*.pt")
                  if f.name in (BEST_NAME, LAST_NAME) or f.match(_EPOCH_GLOB)]
    if not candidates:
        raise FileNotFoundError(f"No hay checkpoints (best.pt, last.pt o checkpoint_epoch_*.pt) en {p}")
    newest_dir = _newest(candidates).parent   # run más reciente
    in_run = [f for f in candidates if f.parent == newest_dir]
    for name in names:
        for f in in_run:
            if f.name == name:
                return f
    return _newest(in_run)
