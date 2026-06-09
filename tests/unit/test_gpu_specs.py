"""Unit tests for src/gpu_specs.py — CUDA/Tensor core derivation.

These check the pure lookup logic against vendor-published specs; no GPU needed.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.gpu_specs import (
    GpuSpecs, architecture_name, detect_all, specs_for,
    _cores_per_sm, _tensor_cores_per_sm,
)


@pytest.mark.parametrize("name,major,minor,sm,cuda,tensor", [
    ("Tesla V100", 7, 0, 80, 5120, 640),       # Verode
    ("Tesla T4", 7, 5, 40, 2560, 320),         # Kaggle
    ("RTX 3060 Ti", 8, 6, 38, 4864, 152),      # local
    ("A100", 8, 0, 108, 6912, 432),
    ("RTX 4090", 8, 9, 128, 16384, 512),
    ("Tesla K40m", 3, 5, 15, 2880, 0),         # Kepler — no tensor cores
])
def test_known_gpus(name, major, minor, sm, cuda, tensor):
    s = specs_for(name, major, minor, sm)
    assert s.cuda_cores == cuda
    assert s.tensor_cores == tensor
    assert s.compute_capability == f"{major}.{minor}"


def test_pre_volta_has_no_tensor_cores():
    assert _tensor_cores_per_sm(6, 1) == 0   # Pascal
    assert _tensor_cores_per_sm(5, 2) == 0   # Maxwell
    assert _tensor_cores_per_sm(3, 5) == 0   # Kepler


def test_volta_turing_tensor_cores():
    assert _tensor_cores_per_sm(7, 0) == 8
    assert _tensor_cores_per_sm(7, 5) == 8
    assert _tensor_cores_per_sm(8, 0) == 4   # Ampere reduced TC count per SM


def test_architecture_names():
    assert architecture_name(7, 0) == "Volta"
    assert architecture_name(7, 5) == "Turing"
    assert architecture_name(8, 6) == "Ampere"
    assert architecture_name(8, 9) == "Ada Lovelace"
    assert architecture_name(9, 0) == "Hopper"


def test_unknown_capability_falls_back_gracefully():
    # A hypothetical future 8.x should still resolve to 128 cores/SM, 4 TC/SM.
    s = specs_for("Future", 8, 5, 100)
    assert s.cuda_cores == 12800
    assert s.tensor_cores == 400
    assert s.architecture == "Ampere/Ada"


def test_cores_per_sm_unknown_returns_zero_for_ancient():
    # Compute capability 1.x predates the lookup; cuda_cores must be 0, not crash.
    assert _cores_per_sm(1, 0) == 0
    s = specs_for("Ancient", 1, 0, 10)
    assert s.cuda_cores == 0


def test_specs_dataclass_fields():
    s = specs_for("Tesla V100", 7, 0, 80, total_vram_gb=32.0, index=1)
    assert isinstance(s, GpuSpecs)
    assert s.total_vram_gb == 32.0
    assert s.index == 1
    assert s.sm_count == 80


def test_detect_all_runs_without_crashing():
    # Must return a list whether or not a GPU is present.
    result = detect_all()
    assert isinstance(result, list)
    for s in result:
        assert isinstance(s, GpuSpecs)
        assert s.sm_count > 0


def test_feasibility_parser_reads_gpu_block(tmp_path):
    """The feasibility parser must round-trip the #gpu block into meta['gpu']."""
    from src.web.feasibility_parser import parse_feasibility_csv

    csv = tmp_path / "feasibility_test.csv"
    csv.write_text(
        "#meta,model_name,total_params_M,flops_mflops,hardware_name,total_vram_gb,free_vram_gb\n"
        "#meta,vit_tiny_patch16_224,5.53,34.3,Tesla V100,32.0,30.0\n"
        "#gpu,compute_capability,architecture,sm_count,cuda_cores,tensor_cores\n"
        "#gpu,7.0,Volta,80,5120,640\n"
        "batch_size,trace_mode,oom\n"
        "64,off,no\n"
    )
    meta, df = parse_feasibility_csv(csv)
    gpu = meta.get("gpu")
    assert gpu is not None
    assert gpu["architecture"] == "Volta"
    assert gpu["cuda_cores"] == 5120        # int, not str
    assert gpu["tensor_cores"] == 640
    assert gpu["sm_count"] == 80


def test_gpu_info_dataclass_has_spec_fields():
    """system_monitor.GpuInfo must carry the derived spec fields."""
    import dataclasses
    from src.web.system_monitor import GpuInfo
    fields = {f.name for f in dataclasses.fields(GpuInfo)}
    for f in ("cuda_cores", "tensor_cores", "sm_count", "architecture", "compute_capability"):
        assert f in fields
