"""Hardware / disk / dataset probes — profile the machine and storage."""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import torch

from src.feasibility.value_objects import HardwareInfo, CPUInfo, DiskInfo, DatasetProfile


class HardwareProbe:
    def probe_gpu(self, index: int = 0) -> HardwareInfo:
        if not torch.cuda.is_available():
            return HardwareInfo(device_name="CPU", total_vram_gb=0.0,
                                free_vram_gb=0.0, is_cuda=False)
        props = torch.cuda.get_device_properties(index)
        total_gb = props.total_memory / 1e9
        reserved_gb = torch.cuda.memory_reserved(index) / 1e9
        cc = f"{props.major}.{props.minor}"
        # Estimated memory bandwidth by CC (GB/s)
        bw_map = {"7.0": 900, "8.0": 2000, "8.6": 600, "9.0": 3350,
                  "6.1": 352, "6.0": 720, "5.2": 336}
        bw = float(bw_map.get(cc, 0))
        # CUDA / Tensor cores derived from compute capability × SM count.
        from src.gpu_specs import specs_for
        sp = specs_for(props.name, props.major, props.minor,
                       props.multi_processor_count, total_gb, index)
        return HardwareInfo(
            device_name=props.name,
            total_vram_gb=total_gb,
            free_vram_gb=total_gb - reserved_gb,
            is_cuda=True,
            compute_capability=cc,
            memory_bandwidth_gb_s=bw,
            architecture=sp.architecture,
            sm_count=sp.sm_count,
            cuda_cores=sp.cuda_cores,
            tensor_cores=sp.tensor_cores,
        )

    def probe_cpu(self) -> CPUInfo:
        try:
            import psutil, platform
            cpu = psutil.cpu_percent  # trigger import
            freq = psutil.cpu_freq()
            ram = psutil.virtual_memory()
            return CPUInfo(
                logical_cores=psutil.cpu_count(logical=True) or 1,
                physical_cores=psutil.cpu_count(logical=False) or 1,
                freq_mhz=freq.current if freq else 0.0,
                ram_total_gb=ram.total / 1e9,
                ram_free_gb=ram.available / 1e9,
                platform=platform.platform(),
            )
        except Exception:
            return CPUInfo(logical_cores=1, physical_cores=1, freq_mhz=0.0,
                           ram_total_gb=0.0, ram_free_gb=0.0, platform="unknown")


class DiskProbe:
    """Detecta el tipo de disco del dataset y mide velocidad de lectura."""

    def probe(self, dataset_path: str | None) -> DiskInfo:
        path = Path(dataset_path) if dataset_path else None
        is_nfs = self._detect_nfs(path)
        disk_type = "NFS" if is_nfs else self._detect_disk_type(path)
        read_mb_s, files_s = self._measure_io(path)
        return DiskInfo(
            dataset_path=str(path) if path else "",
            is_nfs=is_nfs,
            disk_type=disk_type,
            read_mb_per_s=read_mb_s,
            files_per_second=files_s,
        )

    @staticmethod
    def _detect_nfs(path: Path | None) -> bool:
        if path is None:
            return False
        try:
            import subprocess
            result = subprocess.run(
                ["df", "-T", str(path)], capture_output=True, text=True, timeout=3
            )
            return "nfs" in result.stdout.lower()
        except Exception:
            # Heurística: rutas NFS comunes en el proyecto
            return path is not None and (
                "bejeque" in str(path) or "/nfs" in str(path) or "/net" in str(path)
            )

    @staticmethod
    def _detect_disk_type(path: Path | None) -> str:
        if path is None:
            return "Unknown"
        try:
            import subprocess
            # Intentar detectar si es SSD o HDD via rotational flag
            mount = subprocess.run(
                ["df", str(path), "--output=source"],
                capture_output=True, text=True, timeout=3
            ).stdout.strip().splitlines()
            if len(mount) > 1:
                dev = mount[1].replace("/dev/", "").rstrip("0123456789")
                rot_path = f"/sys/block/{dev}/queue/rotational"
                if Path(rot_path).exists():
                    rotational = int(Path(rot_path).read_text().strip())
                    return "HDD" if rotational else "SSD"
        except Exception:
            pass
        return "Unknown"

    @staticmethod
    def _measure_io(path: Path | None) -> tuple[float, float]:
        """Mide velocidad de lectura con hasta N_SAMPLE parches TIFF.

        Usa islice sobre el generador rglob para NO materializar todo el árbol
        del dataset (BigEarthNet tiene ~1.6M ficheros; un list(rglob) sobre NFS
        tarda decenas de minutos). islice corta en cuanto reúne la muestra.
        """
        if path is None or not path.exists():
            return 0.0, 0.0
        import itertools
        N_SAMPLE = 50
        try:
            # Solo los primeros ~300 .tif (lazy, no recorre el árbol completo)
            tif_files = list(itertools.islice(path.rglob("*.tif"), 300))
            if not tif_files:
                return 0.0, 0.0
            sample = random.sample(tif_files, min(N_SAMPLE, len(tif_files)))
            total_bytes = 0
            t0 = time.perf_counter()
            for f in sample:
                data = f.read_bytes()
                total_bytes += len(data)
            elapsed = time.perf_counter() - t0 + 1e-9
            read_mb_s = total_bytes / 1e6 / elapsed
            files_s = len(sample) / elapsed
            return round(read_mb_s, 1), round(files_s, 1)
        except Exception:
            return 0.0, 0.0


# ═════════════════════════════════════════════════════════════════════════════
# DatasetProfiler
# ═════════════════════════════════════════════════════════════════════════════

class DatasetProfiler:
    """Analiza el dataset para medir I/O y determinar si es el cuello de botella."""

    def __init__(self, dataset_path: str | None, disk_info: DiskInfo | None = None):
        self._path = Path(dataset_path) if dataset_path else None
        self._disk_info = disk_info

    def profile(self, benchmark_sec_per_batch: float, batch_size: int) -> DatasetProfile:
        path_str = str(self._path) if self._path else ""
        n_found = 0
        n_est = 0
        read_mb_s = 0.0
        files_s = 0.0

        if self._path and self._path.exists():
            try:
                # Estimar el nº de patches sin recorrer todo el árbol (1.6M ficheros
                # en NFS colgaría). Contamos las escenas del primer nivel (scandir,
                # 1 nivel = rápido) y multiplicamos por los patches de una escena.
                scenes = [e for e in os.scandir(self._path) if e.is_dir()]
                n_scenes = len(scenes)
                patches_per_scene = 0
                if scenes:
                    sample_scene = scenes[0].path
                    patches_per_scene = sum(
                        1 for e in os.scandir(sample_scene) if e.is_dir()
                    )
                n_found = n_scenes * max(1, patches_per_scene)
                n_est = n_found
            except Exception:
                pass

        if self._disk_info:
            read_mb_s = self._disk_info.read_mb_per_s
            files_s = self._disk_info.files_per_second

        # Ratio I/O vs cómputo: cuánto tiempo de I/O por batch vs tiempo de cómputo
        # Cada imagen del batch necesita leer ~3 TIFF de ~1 MB cada uno
        if files_s > 0 and batch_size > 0:
            io_time_per_batch = batch_size * 3 / files_s   # segundos para leer un batch
            io_ratio = io_time_per_batch / (benchmark_sec_per_batch + 1e-9)
        else:
            io_ratio = 0.0

        return DatasetProfile(
            path=path_str,
            n_files_sampled=min(50, n_found),
            n_files_total_est=n_est,
            sample_read_mb_per_s=read_mb_s,
            files_per_second=files_s,
            io_bottleneck_ratio=round(io_ratio, 2),
        )
