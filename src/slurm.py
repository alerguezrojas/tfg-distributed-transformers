"""Slurm helpers for ``paravit train --launcher slurm`` (pure: they take the environment as a dict).

Inside a Slurm job every node runs the SAME torchrun command; the nodes find each other through
the ``c10d`` rendezvous, whose endpoint is the first node of the allocation. Nothing here needs
Slurm to be installed, so it is unit-tested on a plain CPU machine.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from typing import Mapping

LAUNCHERS = ("local", "slurm")
_VAR_JOB = "SLURM_JOB_ID"
_VAR_NODES = "SLURM_NNODES"
_VAR_NODELIST = "SLURM_JOB_NODELIST"
_RANGE = re.compile(r"^(\d+)(?:-(\d+))?$")
_GROUP = re.compile(r"([^\[\]]*)\[([^\[\]]+)\]([^\[\]]*)")
_SCONTROL = "scontrol"
_BAD_LIST = "lista de nodos no válida: "


def _split_top_level(expr: str) -> list[str]:
    """Split on the commas that are outside ``[...]``."""
    parts, depth, current = [], 0, ""
    for ch in expr:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    parts.append(current)
    return [p.strip() for p in parts if p.strip()]


def _expand_ranges(spec: str) -> list[str]:
    """``16,21`` / ``01-03`` / ``1,3-4`` -> the numbers as strings (zero padding is kept)."""
    out: list[str] = []
    for item in spec.split(","):
        m = _RANGE.match(item.strip())
        if not m:
            raise ValueError(f"{_BAD_LIST}{item!r}")
        lo, hi = m.group(1), m.group(2) or m.group(1)
        if int(hi) < int(lo):
            raise ValueError(f"rango descendente en la lista de nodos: {item!r}")
        out += [str(n).zfill(len(lo)) for n in range(int(lo), int(hi) + 1)]
    return out


def expand_nodelist(expr: str) -> list[str]:
    """Hostnames of a compact Slurm list: ``verode[16,21]``, ``n[01-03],m5`` -> one name each."""
    hosts: list[str] = []
    for part in _split_top_level(expr):
        if "[" not in part and "]" not in part:
            hosts.append(part)
            continue
        m = _GROUP.fullmatch(part)
        if not m:
            raise ValueError(f"{_BAD_LIST}{part!r}")
        prefix, ranges, suffix = m.groups()
        hosts += [f"{prefix}{n}{suffix}" for n in _expand_ranges(ranges)]
    if not hosts:
        raise ValueError("lista de nodos vacía")
    return hosts


def _scontrol_hostnames(expr: str) -> list[str] | None:
    """``scontrol show hostnames`` if the tool exists and works; ``None`` otherwise."""
    exe = shutil.which(_SCONTROL)
    if not exe:
        return None
    try:
        res = subprocess.run([exe, "show", "hostnames", expr], capture_output=True,
                             text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.split() or None


def first_node(env: Mapping[str, str], use_scontrol: bool = True) -> str:
    """First node of ``SLURM_JOB_NODELIST`` (the rendezvous host)."""
    expr = env.get(_VAR_NODELIST, "")
    if not expr:
        raise ValueError(f"{_VAR_NODELIST} no está definida")
    names = (_scontrol_hostnames(expr) if use_scontrol else None) or expand_nodelist(expr)
    return names[0]


def in_slurm_job(env: Mapping[str, str]) -> bool:
    return bool(env.get(_VAR_JOB)) and bool(env.get(_VAR_NODELIST))


def slurm_torchrun(nproc: int, env: Mapping[str, str], port: int = 29500,
                   use_scontrol: bool = True) -> list[str]:
    """torchrun prefix for a Slurm job: identical on every node (c10d rendezvous, no node_rank)."""
    if not in_slurm_job(env):
        raise ValueError("--launcher slurm solo funciona dentro de un trabajo de Slurm "
                         f"({_VAR_JOB} y {_VAR_NODELIST} no están definidas): use sbatch o "
                         "srun, o quite --launcher")
    nnodes = int(env.get(_VAR_NODES) or 1)
    return [sys.executable, "-m", "torch.distributed.run", f"--nnodes={nnodes}",
            f"--nproc_per_node={nproc}", "--rdzv-backend=c10d",
            f"--rdzv-endpoint={first_node(env, use_scontrol)}:{port}",
            f"--rdzv-id={env[_VAR_JOB]}"]
