"""``paravit doctor``: is this machine ready to train?

Two separate parts. ``collect`` gathers facts from the machine (Python, torch, GPUs, CPU flags,
disk, paths) into an ``Environment`` and is the only part with dependencies or I/O; ``evaluate``
turns an ``Environment`` into OK / WARN / FAIL checks and is pure, so it is tested with simulated
machines. The GPU is only READ (``nvidia-smi``, ``torch.cuda`` counters): no memory is reserved.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.performance_model import gpu_spec

OK, WARN, FAIL = "OK", "WARN", "FAIL"
MIN_PYTHON = (3, 12)                 # requires-python of pyproject.toml
MIN_FREE_GB_WARN = 20.0              # a ViT-B run keeps best.pt + last.pt (~1 GB each) per run
MIN_FREE_GB_FAIL = 2.0
_AVX2 = "CPU AVX2"
_PATH_KEYS = ("root", "metadata")    # data keys whose existence is checked


@dataclass(frozen=True)
class GpuInfo:
    name: str
    memory_gb: float | None = None


@dataclass
class Environment:
    """What ``collect`` found. ``None`` = could not be determined."""
    python: tuple[int, int, int] = (0, 0, 0)
    platform: str = ""
    torch_version: str | None = None             # None = torch not importable / not tried
    torch_cuda: str | None = None                # CUDA version of the torch build (None = CPU build)
    cuda_available: bool = False                 # torch.cuda.is_available()
    driver: str | None = None
    gpus: list[GpuInfo] = field(default_factory=list)
    avx2: bool | None = None
    torch_error: str | None = None               # import failed with something other than ImportError
    cuda_hidden: str | None = None               # CUDA_VISIBLE_DEVICES value when it hides the GPUs
    machine: str | None = None                   # active profile and why ("local (host ...)")
    # Only with --config:
    config: str | None = None
    wants_gpu: bool = False
    paths: list[tuple[str, str, bool]] = field(default_factory=list)   # (label, path, exists)
    checkpoint_dir: str | None = None
    free_gb: float | None = None
    env_label: str | None = None                 # output.env of the config (after the profile)


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


# ── Pure evaluation ───────────────────────────────────────────────────────────

def evaluate(env: Environment, *, require_gpu: bool = False) -> list[Check]:
    checks: list[Check] = []
    add = lambda name, status, detail: checks.append(Check(name, status, detail))  # noqa: E731

    py = ".".join(map(str, env.python))
    add("Python", OK if env.python[:2] >= MIN_PYTHON else FAIL,
        py if env.python[:2] >= MIN_PYTHON else f"{py} (needs >= {'.'.join(map(str, MIN_PYTHON))})")

    if env.avx2 is None:
        add(_AVX2, WARN, "could not be checked on this platform")
    elif env.avx2:
        add(_AVX2, OK, "supported")
    else:
        add(_AVX2, FAIL, "not supported: numpy/torch crash with 'Illegal instruction' "
                              "(VERODE login node: run through srun on a compute node)")

    if env.torch_version is None:
        add("torch", FAIL if env.avx2 is not False else WARN,
            (f"import failed: {env.torch_error}" if env.torch_error else "not importable")
            if env.avx2 is not False else "not tried (no AVX2)")
    else:
        add("torch", OK, f"{env.torch_version} (CUDA build: {env.torch_cuda or 'none, CPU only'})")

    wants_gpu = require_gpu or env.wants_gpu
    smi_gpus = bool(env.gpus)
    if env.cuda_available:
        add("CUDA", OK, f"torch sees {len(env.gpus) or '?'} GPU(s); driver {env.driver or '?'}")
    elif smi_gpus and env.torch_version and env.torch_cuda is None:
        add("CUDA", FAIL if wants_gpu else WARN,
            "GPU present but this torch is a CPU build: training would run on CPU silently "
            "(reinstall torch with CUDA)")
    elif smi_gpus and env.cuda_hidden is not None:
        add("CUDA", FAIL if wants_gpu else WARN,
            f"GPU present but CUDA_VISIBLE_DEVICES={env.cuda_hidden!r} hides it from torch")
    elif smi_gpus:
        add("CUDA", FAIL if wants_gpu else WARN,
            f"GPU present (driver {env.driver or '?'}) but torch.cuda is unavailable "
            f"(driver/CUDA mismatch?)")
    else:
        why = "the config requires one" if wants_gpu else "training would run on CPU"
        add("CUDA", FAIL if wants_gpu else WARN, f"no GPU found; {why}")

    for i, gpu in enumerate(env.gpus):
        mem = f", {gpu.memory_gb:.1f} GB" if gpu.memory_gb else ""
        known = gpu_spec(gpu.name) is not None
        add(f"GPU {i}", OK if known else WARN,
            f"{gpu.name}{mem}" + ("" if known else
                                  " (not in the predictor's GPU table: `estimate` cannot cover it)"))

    add("Machine profile", OK if env.machine else WARN,
        env.machine or "none (configs used as written; see configs/machines/example.yaml)")

    if env.config:
        for label, path, exists in env.paths:
            add(f"Path {label}", OK if exists else FAIL, path if exists else f"{path} does not exist")
        if env.env_label:
            add("output.env", OK, f"{env.env_label} (logs/{env.env_label}/...)")
        if env.free_gb is None:
            add("Checkpoint disk", WARN, f"could not measure the free space for {env.checkpoint_dir}")
        else:
            status = (FAIL if env.free_gb < MIN_FREE_GB_FAIL
                      else WARN if env.free_gb < MIN_FREE_GB_WARN else OK)
            add("Checkpoint disk", status, f"{env.free_gb:.1f} GB free for {env.checkpoint_dir}")
    return checks


def exit_code(checks: list[Check]) -> int:
    return 1 if any(c.status == FAIL for c in checks) else 0


# ── Collection (impure) ──────────────────────────────────────────────────────

def _has_avx2() -> bool | None:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return None
    flags = [ln for ln in text.splitlines() if ln.startswith("flags")]
    return any(" avx2" in ln for ln in flags) if flags else None


def _nvidia_smi() -> tuple[list[GpuInfo], str | None]:
    """GPUs and driver as ``nvidia-smi`` reports them (read-only; no CUDA context is created)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return [], None
    gpus, driver = [], None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append(GpuInfo(parts[0], float(parts[1]) / 1024 if parts[1].replace(".", "").isdigit()
                                else None))
            driver = parts[2]
    return gpus, driver


def _free_gb(path: str) -> float | None:
    p = Path(path).expanduser()
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free / 1024 ** 3
    except OSError:
        return None


def config_wants_gpu(cfg: dict) -> bool:
    """A config asks for a GPU when its distributed backend is NCCL (it cannot run on CPU)."""
    return str((cfg.get("distributed") or {}).get("backend", "")).lower() == "nccl"


def collect(cfg: dict | None = None, config_path: str | None = None,
            machine: str | None = None) -> Environment:
    """Gather the machine's facts. ``cfg`` is the config AFTER the profile and ``--set``."""
    env = Environment(python=tuple(sys.version_info[:3]), platform=platform.platform(),
                      avx2=_has_avx2(), machine=machine)
    env.gpus, env.driver = _nvidia_smi()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() in ("", "-1", "NoDevFiles"):
        env.cuda_hidden = visible
    if env.avx2 is not False:                    # without AVX2 importing torch would crash
        try:
            import torch
            env.torch_version = torch.__version__
            env.torch_cuda = torch.version.cuda
            env.cuda_available = torch.cuda.is_available()
            if env.cuda_available and not env.gpus:      # no nvidia-smi: counters only
                env.gpus = [GpuInfo(torch.cuda.get_device_name(i))
                            for i in range(torch.cuda.device_count())]
        except ImportError:
            pass
        except OSError as err:                   # broken CUDA libraries abort the import
            env.torch_error = str(err)
    if cfg is not None:
        env.config = config_path or "config"
        env.wants_gpu = config_wants_gpu(cfg)
        data = cfg.get("data") or {}
        for key in _PATH_KEYS:
            if data.get(key):
                path = str(Path(str(data[key])).expanduser())
                env.paths.append((f"data.{key}", path, os.path.exists(path)))
        env.checkpoint_dir = str((cfg.get("checkpoint") or {}).get("dir", "checkpoints"))
        env.free_gb = _free_gb(env.checkpoint_dir)
        env.env_label = (cfg.get("output") or {}).get("env", "local")
    return env
