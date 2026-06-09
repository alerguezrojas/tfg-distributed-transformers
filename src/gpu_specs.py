"""GPU hardware specs derived from compute capability and SM count.

Neither PyTorch nor ``nvidia-smi`` expose the number of CUDA cores or Tensor
cores directly. They can be derived deterministically from the GPU's compute
capability (which fixes the architecture, and therefore the cores-per-SM) times
the number of streaming multiprocessors (SMs):

    cuda_cores   = cuda_cores_per_SM(arch)   * sm_count
    tensor_cores = tensor_cores_per_SM(arch) * sm_count

Examples (verified against vendor specs):
    V100      cc 7.0, 80 SM -> 5120 CUDA / 640 Tensor
    T4        cc 7.5, 40 SM -> 2560 CUDA / 320 Tensor
    RTX 3060 Ti cc 8.6, 38 SM -> 4864 CUDA / 152 Tensor
    A100      cc 8.0, 108 SM -> 6912 CUDA / 432 Tensor

The lookup tables and ``specs_for`` are pure functions (no GPU required), so
they are unit-testable. ``detect_all`` is the only part that touches torch/CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass

# FP32 ("CUDA") cores per SM, keyed by (major, minor) compute capability.
_CUDA_CORES_PER_SM: dict[tuple[int, int], int] = {
    (2, 0): 32, (2, 1): 48,                       # Fermi
    (3, 0): 192, (3, 2): 192, (3, 5): 192, (3, 7): 192,  # Kepler
    (5, 0): 128, (5, 2): 128, (5, 3): 128,        # Maxwell
    (6, 0): 64, (6, 1): 128, (6, 2): 128,         # Pascal
    (7, 0): 64, (7, 2): 64, (7, 5): 64,           # Volta / Turing
    (8, 0): 64, (8, 6): 128, (8, 7): 128, (8, 9): 128,   # Ampere / Ada
    (9, 0): 128,                                  # Hopper
    (10, 0): 128, (12, 0): 128,                   # Blackwell (best effort)
}

# Tensor cores per SM (0 = architecture has no tensor cores, pre-Volta).
_TENSOR_CORES_PER_SM: dict[tuple[int, int], int] = {
    (7, 0): 8, (7, 2): 8, (7, 5): 8,              # 1st/2nd gen (Volta/Turing)
    (8, 0): 4, (8, 6): 4, (8, 7): 4, (8, 9): 4,   # 3rd/4th gen (Ampere/Ada)
    (9, 0): 4, (10, 0): 4, (12, 0): 4,            # Hopper / Blackwell
}

# Architecture name by major compute-capability version.
_ARCH_BY_MAJOR: dict[int, str] = {
    2: "Fermi", 3: "Kepler", 5: "Maxwell", 6: "Pascal",
    7: "Volta/Turing", 8: "Ampere/Ada", 9: "Hopper",
    10: "Blackwell", 12: "Blackwell",
}

# Finer architecture names where the minor version matters.
_ARCH_BY_CC: dict[tuple[int, int], str] = {
    (7, 0): "Volta", (7, 2): "Volta", (7, 5): "Turing",
    (8, 0): "Ampere", (8, 6): "Ampere", (8, 7): "Ampere", (8, 9): "Ada Lovelace",
    (9, 0): "Hopper", (10, 0): "Blackwell", (12, 0): "Blackwell",
}


@dataclass
class GpuSpecs:
    """Derived hardware specs for a single GPU."""

    name: str
    compute_capability: str
    architecture: str
    sm_count: int
    cuda_cores: int           # 0 if the (major, minor) is unknown
    tensor_cores: int         # 0 if the architecture has no tensor cores
    total_vram_gb: float = 0.0
    index: int = 0


def _cores_per_sm(major: int, minor: int) -> int:
    """CUDA cores per SM, with a sensible fallback for unknown capabilities."""
    if (major, minor) in _CUDA_CORES_PER_SM:
        return _CUDA_CORES_PER_SM[(major, minor)]
    # Fallback by architecture family.
    if major >= 8:
        return 128
    if major == 7:
        return 64
    if major == 6:
        return 64 if minor == 0 else 128
    if major == 5:
        return 128
    if major == 3:
        return 192
    if major == 2:
        return 32
    return 0


def _tensor_cores_per_sm(major: int, minor: int) -> int:
    """Tensor cores per SM (0 for pre-Volta), with a fallback for unknowns."""
    if (major, minor) in _TENSOR_CORES_PER_SM:
        return _TENSOR_CORES_PER_SM[(major, minor)]
    if major >= 8:
        return 4
    if major == 7:
        return 8
    return 0


def architecture_name(major: int, minor: int) -> str:
    """Human-readable architecture name for a compute capability."""
    return _ARCH_BY_CC.get((major, minor)) or _ARCH_BY_MAJOR.get(major, "Unknown")


def specs_for(
    name: str,
    major: int,
    minor: int,
    sm_count: int,
    total_vram_gb: float = 0.0,
    index: int = 0,
) -> GpuSpecs:
    """Build :class:`GpuSpecs` from a compute capability and SM count."""
    cps = _cores_per_sm(major, minor)
    tps = _tensor_cores_per_sm(major, minor)
    return GpuSpecs(
        name=name,
        compute_capability=f"{major}.{minor}",
        architecture=architecture_name(major, minor),
        sm_count=int(sm_count),
        cuda_cores=cps * int(sm_count) if cps else 0,
        tensor_cores=tps * int(sm_count),
        total_vram_gb=total_vram_gb,
        index=index,
    )


def detect_all() -> list[GpuSpecs]:
    """Enumerate every visible CUDA GPU and return its derived specs.

    Returns an empty list if torch/CUDA is unavailable (e.g. CPU-only host).
    """
    try:
        import torch
    except Exception:
        return []
    if not torch.cuda.is_available():
        return []
    out: list[GpuSpecs] = []
    for i in range(torch.cuda.device_count()):
        try:
            p = torch.cuda.get_device_properties(i)
            out.append(specs_for(
                name=p.name,
                major=p.major,
                minor=p.minor,
                sm_count=p.multi_processor_count,
                total_vram_gb=p.total_memory / 1e9,
                index=i,
            ))
        except Exception:
            continue
    return out
