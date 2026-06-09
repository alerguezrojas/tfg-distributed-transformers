"""System hardware monitor — CPU, RAM, disk, GPU, network."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

import psutil


@dataclass
class CpuInfo:
    usage_pct: float
    count_logical: int
    count_physical: int
    freq_mhz: float | None


@dataclass
class RamInfo:
    total_gb: float
    used_gb: float
    available_gb: float
    percent: float
    swap_total_gb: float
    swap_used_gb: float


@dataclass
class DiskInfo:
    path: str
    total_gb: float
    used_gb: float
    free_gb: float
    percent: float
    read_mb_s: float | None = None
    write_mb_s: float | None = None


@dataclass
class GpuInfo:
    index: int
    name: str
    mem_used_mb: int
    mem_total_mb: int
    util_pct: int
    temp_c: int
    power_w: float | None = None
    power_limit_w: float | None = None
    # Derived specs (from compute capability × SM count); None if torch/CUDA
    # is unavailable on the host running the dashboard.
    compute_capability: str | None = None
    architecture: str | None = None
    sm_count: int | None = None
    cuda_cores: int | None = None
    tensor_cores: int | None = None


@dataclass
class NetworkInfo:
    bytes_sent_mb: float
    bytes_recv_mb: float


@dataclass
class SystemSnapshot:
    cpu: CpuInfo
    ram: RamInfo
    disks: list[DiskInfo]
    gpus: list[GpuInfo]
    network: NetworkInfo


def get_snapshot(disk_paths: list[str] | None = None) -> SystemSnapshot:
    cpu = _cpu()
    ram = _ram()
    disks = _disks(disk_paths or ["/", "/home"])
    gpus = _gpus()
    net = _network()
    return SystemSnapshot(cpu=cpu, ram=ram, disks=disks, gpus=gpus, network=net)


def _cpu() -> CpuInfo:
    freq = psutil.cpu_freq()
    return CpuInfo(
        usage_pct=psutil.cpu_percent(interval=0.2),
        count_logical=psutil.cpu_count(logical=True),
        count_physical=psutil.cpu_count(logical=False) or 0,
        freq_mhz=freq.current if freq else None,
    )


def _ram() -> RamInfo:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return RamInfo(
        total_gb=vm.total / 1e9,
        used_gb=vm.used / 1e9,
        available_gb=vm.available / 1e9,
        percent=vm.percent,
        swap_total_gb=sw.total / 1e9,
        swap_used_gb=sw.used / 1e9,
    )


def _disks(paths: list[str]) -> list[DiskInfo]:
    result = []
    try:
        io = psutil.disk_io_counters()
    except Exception:
        io = None
    for path in paths:
        try:
            usage = psutil.disk_usage(path)
            result.append(DiskInfo(
                path=path,
                total_gb=usage.total / 1e9,
                used_gb=usage.used / 1e9,
                free_gb=usage.free / 1e9,
                percent=usage.percent,
            ))
        except Exception:
            pass
    return result


_SPECS_CACHE: dict | None = None


def _gpu_specs_by_index() -> dict:
    """Cached map {gpu_index: GpuSpecs} derived via torch (computed once)."""
    global _SPECS_CACHE
    if _SPECS_CACHE is None:
        try:
            from src.gpu_specs import detect_all
            _SPECS_CACHE = {s.index: s for s in detect_all()}
        except Exception:
            _SPECS_CACHE = {}
    return _SPECS_CACHE


def _gpus() -> list[GpuInfo]:
    specs = _gpu_specs_by_index()
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,"
                "utilization.gpu,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=4,
        )
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            idx = int(parts[0])
            sp = specs.get(idx)
            gpus.append(GpuInfo(
                index=idx,
                name=parts[1],
                mem_used_mb=int(parts[2]),
                mem_total_mb=int(parts[3]),
                util_pct=int(parts[4]),
                temp_c=int(parts[5]),
                power_w=float(parts[6]) if len(parts) > 6 and parts[6] not in ("[N/A]", "N/A") else None,
                power_limit_w=float(parts[7]) if len(parts) > 7 and parts[7] not in ("[N/A]", "N/A") else None,
                compute_capability=sp.compute_capability if sp else None,
                architecture=sp.architecture if sp else None,
                sm_count=sp.sm_count if sp else None,
                cuda_cores=sp.cuda_cores if sp else None,
                tensor_cores=sp.tensor_cores if sp else None,
            ))
        return gpus
    except Exception:
        return []


def _network() -> NetworkInfo:
    try:
        net = psutil.net_io_counters()
        return NetworkInfo(
            bytes_sent_mb=net.bytes_sent / 1e6,
            bytes_recv_mb=net.bytes_recv / 1e6,
        )
    except Exception:
        return NetworkInfo(bytes_sent_mb=0, bytes_recv_mb=0)
