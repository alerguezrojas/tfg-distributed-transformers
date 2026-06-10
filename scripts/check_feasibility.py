"""Analizador de viabilidad pre-entrenamiento — v3.

Mide complejidad teórica y throughput real para cada combinación
(batch_size, trace_mode) y produce estimaciones de tiempo + recursos.
No toca el dataset real — usa datos sintéticos para el benchmark de cómputo.

Nuevas capacidades v3:
  - Perfil completo del sistema: CPU (cores, RAM), GPU (VRAM, CC), disco
  - Detección de NFS y medición de velocidad de lectura del dataset
  - Predicción empírica de curva F1 (basada en datos históricos BigEarthNet-ViT)
  - Optimizador DDP: distribución óptima de carga + eficiencia real
  - Visualización CSV ampliada para el dashboard web

Uso:
    uv run python scripts/check_feasibility.py
    uv run python scripts/check_feasibility.py --batch-sizes 32 64
    uv run python scripts/check_feasibility.py --nfs-factor 1.3
    uv run python scripts/check_feasibility.py --dataset-path /path/to/BigEarthNet-S2
    uv run python scripts/check_feasibility.py --model vit_base_patch16_224 vit_tiny_patch16_224
"""

import argparse
import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vit import build_model

# ═════════════════════════════════════════════════════════════════════════════
# Value objects
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class ModelInfo:
    name: str
    total_params: int
    trainable_params: int
    flops_per_image_mflops: float
    weight_mb: float
    gradient_mb: float
    optimizer_mb: float
    activation_mb_per_image: float = 0.0

    @property
    def total_static_mb(self) -> float:
        return self.weight_mb + self.gradient_mb + self.optimizer_mb

    def total_mb(self, batch_size: int) -> float:
        return self.total_static_mb + self.activation_mb_per_image * batch_size


@dataclass
class HardwareInfo:
    device_name: str
    total_vram_gb: float
    free_vram_gb: float
    is_cuda: bool
    compute_capability: str = ""
    memory_bandwidth_gb_s: float = 0.0
    architecture: str = ""
    sm_count: int = 0
    cuda_cores: int = 0
    tensor_cores: int = 0


@dataclass
class CPUInfo:
    logical_cores: int
    physical_cores: int
    freq_mhz: float
    ram_total_gb: float
    ram_free_gb: float
    platform: str


@dataclass
class DiskInfo:
    dataset_path: str
    is_nfs: bool
    disk_type: str          # "SSD", "HDD", "NFS", "Unknown"
    read_mb_per_s: float    # medido o estimado
    files_per_second: float # parches TIFF leídos por segundo


@dataclass
class DatasetProfile:
    path: str
    n_files_sampled: int
    n_files_total_est: int
    sample_read_mb_per_s: float
    files_per_second: float
    io_bottleneck_ratio: float  # io_time / compute_time (>1 = limitado por I/O)


@dataclass
class DDPScenario:
    n_gpus: int
    batch_per_gpu: int
    global_batch: int
    num_workers_per_gpu: int
    estimated_speedup: float
    scaling_efficiency: float      # 0-1
    sync_overhead_pct: float       # % del tiempo de batch dedicado a sync
    bottleneck: str                # "compute", "io", "sync"
    time_train_per_epoch_s: float
    time_total_s: float


@dataclass
class PerformancePrediction:
    model_name: str
    predicted_best_f1: float
    predicted_best_epoch: int
    predicted_early_stop_epoch: int
    confidence: str                # "alta", "media", "baja"
    curve_epochs: list[int] = field(default_factory=list)
    curve_f1_train: list[float] = field(default_factory=list)
    curve_f1_val: list[float] = field(default_factory=list)
    notes: str = ""


@dataclass
class BenchmarkResult:
    batch_size: int
    trace_mode: str
    seconds_per_batch_train: float
    seconds_per_batch_eval: float
    images_per_second_train: float
    images_per_second_eval: float
    peak_vram_gb: float
    oom: bool = False
    avg_power_w: float = 0.0


@dataclass
class FeasibilityReport:
    model_info: ModelInfo
    hardware_info: HardwareInfo
    dataset_train: int
    dataset_val: int
    nfs_factor: float
    results: list[BenchmarkResult] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    epochs_list: list[int] = field(default_factory=list)
    trace_modes: list[str] = field(default_factory=list)
    # v3 additions
    cpu_info: Optional[CPUInfo] = None
    disk_info: Optional[DiskInfo] = None
    dataset_profile: Optional[DatasetProfile] = None
    ddp_scenarios: list[DDPScenario] = field(default_factory=list)
    performance_prediction: Optional[PerformancePrediction] = None
    # v4: estudio empírico real (mini-training + LR range + gradient noise)
    study_report: object = None  # StudyReport | None (evita import circular)
    # precision used for the benchmark + optional FP32-vs-Tensor-core comparison
    precision: str = "fp32"
    precision_comparison: dict | None = None  # {batch, fp32_imgs_s, tc_prec, tc_imgs_s, speedup}


# ═════════════════════════════════════════════════════════════════════════════
# ModelAnalyzer
# ═════════════════════════════════════════════════════════════════════════════

class ModelAnalyzer:
    def __init__(self, model: nn.Module, model_name: str, device: torch.device):
        self._model = model
        self._name = model_name
        self._device = device

    def analyze(self) -> ModelInfo:
        total = sum(p.numel() for p in self._model.parameters())
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        weight_mb = total * 4 / 1e6
        gradient_mb = trainable * 4 / 1e6
        optimizer_mb = trainable * 8 / 1e6
        flops, activation_mb = self._run_summary()
        return ModelInfo(
            name=self._name,
            total_params=total,
            trainable_params=trainable,
            flops_per_image_mflops=flops,
            weight_mb=weight_mb,
            gradient_mb=gradient_mb,
            optimizer_mb=optimizer_mb,
            activation_mb_per_image=activation_mb,
        )

    def _run_summary(self) -> tuple[float, float]:
        try:
            from torchinfo import summary
            stats = summary(self._model, input_size=(1, 3, 224, 224),
                            verbose=0, device=torch.device("cpu"))
            return stats.total_mult_adds / 1e6, getattr(stats, "total_output_bytes", 0) / 1e6
        except Exception:
            return 0.0, 0.0


# ═════════════════════════════════════════════════════════════════════════════
# HardwareProbe (GPU + CPU + disco)
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
# PerformancePredictor — predicción empírica de curva F1
# ═════════════════════════════════════════════════════════════════════════════

# Datos históricos reales de BigEarthNet-ViT (del CLAUDE.md)
_HISTORICAL_CURVES = {
    # (model_family, config_level): (best_f1, best_epoch, early_stop_epoch, tau, confidence)
    # tau = constante de tiempo del crecimiento exponencial
    ("vit_base", "v3"):    (0.68, 7,  17, 3.0, "alta"),
    ("vit_base", "v2"):    (0.67, 7,  17, 3.5, "alta"),
    ("vit_base", "v1"):    (0.66, 28, 30, 5.0, "media"),
    ("vit_small", "v3"):   (0.63, 8,  18, 3.0, "media"),
    ("vit_tiny", "v3"):    (0.53, 8,  18, 3.0, "media"),
    ("resnet50", "v3"):    (0.55, 10, 22, 4.0, "baja"),
    ("efficientnet", "v3"): (0.52, 9, 20, 3.5, "baja"),
}


class PerformancePredictor:
    """Predicción empírica de la curva F1 de validación basada en datos históricos."""

    def predict(
        self,
        model_name: str,
        n_epochs: int,
        has_llrd: bool = True,
        has_label_smoothing: bool = True,
    ) -> PerformancePrediction:
        family = self._model_family(model_name)
        config = "v3" if (has_llrd and has_label_smoothing) else ("v2" if has_llrd else "v1")

        key = (family, config)
        if key not in _HISTORICAL_CURVES:
            key = (family, "v3") if (family, "v3") in _HISTORICAL_CURVES else None

        if key:
            best_f1, best_epoch, early_stop_ep, tau, confidence = _HISTORICAL_CURVES[key]
        else:
            best_f1, best_epoch, early_stop_ep, tau, confidence = 0.50, 10, 20, 4.0, "baja"

        curve_epochs = list(range(1, min(n_epochs, 30) + 1))
        curve_val, curve_train = self._generate_curves(
            best_f1=best_f1,
            best_epoch=best_epoch,
            tau=tau,
            epochs=curve_epochs,
        )

        notes = self._build_notes(family, config, best_f1, best_epoch, early_stop_ep)

        return PerformancePrediction(
            model_name=model_name,
            predicted_best_f1=best_f1,
            predicted_best_epoch=best_epoch,
            predicted_early_stop_epoch=min(early_stop_ep, n_epochs),
            confidence=confidence,
            curve_epochs=curve_epochs,
            curve_f1_train=curve_train,
            curve_f1_val=curve_val,
            notes=notes,
        )

    @staticmethod
    def _model_family(name: str) -> str:
        n = name.lower()
        if "vit_base" in n:
            return "vit_base"
        if "vit_small" in n:
            return "vit_small"
        if "vit_tiny" in n:
            return "vit_tiny"
        if "resnet" in n:
            return "resnet50"
        if "efficientnet" in n:
            return "efficientnet"
        if "deit" in n:
            return "vit_base"  # DeiT se comporta similar a ViT
        return "vit_base"

    @staticmethod
    def _generate_curves(
        best_f1: float,
        best_epoch: int,
        tau: float,
        epochs: list[int],
    ) -> tuple[list[float], list[float]]:
        """Genera curvas de F1 val y train como series de números."""
        val_curve = []
        train_curve = []
        for ep in epochs:
            # Val: sube rápido hasta best_epoch, luego plateau con ligera degradación
            if ep <= best_epoch:
                # Crecimiento exponencial hacia best_f1
                frac = 1 - math.exp(-ep / tau)
                val_f1 = round(best_f1 * frac * 0.95 + 0.05 * best_f1, 3)
            else:
                # Degradación lenta por overfitting
                degradation = min(0.02, (ep - best_epoch) * 0.001)
                val_f1 = round(max(best_f1 - degradation, best_f1 * 0.97), 3)
            val_curve.append(val_f1)

            # Train: sube más alto que val (overfitting gradual)
            train_frac = 1 - math.exp(-ep / (tau * 0.6))
            train_ceiling = min(0.98, best_f1 + 0.25)  # tren puede llegar mucho más alto
            train_f1 = round(train_ceiling * train_frac, 3)
            train_curve.append(train_f1)

        return val_curve, train_curve

    @staticmethod
    def _build_notes(family: str, config: str, best_f1: float, best_epoch: int, early_stop: int) -> str:
        lines = []
        lines.append(f"Estimación empírica basada en {len(_HISTORICAL_CURVES)} runs históricos en BigEarthNet-S2.")
        if family == "vit_base":
            lines.append(f"ViT-Base pretrained ImageNet suele alcanzar Val F1 ≈ {best_f1:.2f} en epoch ≈ {best_epoch}.")
        elif family == "vit_tiny":
            lines.append(f"ViT-Tiny converge más rápido pero con menor techo ({best_f1:.2f}).")
        lines.append(f"Early stopping recomendado (patience=10) detiene en epoch ≈ {early_stop}.")
        lines.append("Variabilidad observada: ±0.008 F1 entre runs con mismo config.")
        if config != "v3":
            lines.append("Con label smoothing + mixup (v3 config) se reduce el gap train-val.")
        return " | ".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# DDPOptimizer
# ═════════════════════════════════════════════════════════════════════════════

# Ancho de banda de red estimado por tipo de interconexión (GB/s)
_NETWORK_BW = {
    "NVLink":    600.0,   # GPU-GPU en el mismo nodo con NVLink
    "PCIe":       16.0,   # GPU-GPU en el mismo nodo vía PCIe
    "InfiniBand": 25.0,   # Multi-nodo con IB (100 Gbps = ~12.5 GB/s)
    "Ethernet":    1.25,  # Multi-nodo con Ethernet 10 GbE
    "Gigabit":     0.125, # Ethernet Gigabit (Verode cluster)
}

# Eficiencia DDP observada en función del tipo de red
_DDP_EFF = {
    "NVLink":    0.92,
    "PCIe":      0.85,
    "InfiniBand": 0.80,
    "Ethernet":  0.70,
    "Gigabit":   0.60,
}


class DDPOptimizer:
    """Calcula la distribución óptima de recursos para entrenamiento distribuido."""

    def __init__(
        self,
        model_info: ModelInfo,
        hardware_info: HardwareInfo,
        cpu_info: CPUInfo | None,
        disk_info: DiskInfo | None,
        benchmark_results: list[BenchmarkResult],
        dataset_train: int,
        dataset_val: int,
        nfs_factor: float = 1.0,
    ):
        self._model = model_info
        self._gpu = hardware_info
        self._cpu = cpu_info
        self._disk = disk_info
        self._benchmarks = benchmark_results
        self._n_train = dataset_train
        self._n_val = dataset_val
        self._nfs_factor = nfs_factor

    def compute_scenarios(self, n_epochs: int) -> list[DDPScenario]:
        """Calcula escenarios DDP para 1, 2, 4 GPUs."""
        scenarios = []
        # Obtener benchmark base (batch más grande viable, trace=off)
        base_result = self._best_viable_result()
        if base_result is None:
            return scenarios

        # Detectar tipo de interconexión
        net_type = self._infer_network_type()
        bw_gb_s = _NETWORK_BW[net_type]
        eff_base = _DDP_EFF[net_type]

        for n_gpus in [1, 2, 4, 8]:
            scenario = self._build_scenario(
                n_gpus=n_gpus,
                base_result=base_result,
                n_epochs=n_epochs,
                bw_gb_s=bw_gb_s,
                eff_base=eff_base,
            )
            scenarios.append(scenario)

        return scenarios

    def _best_viable_result(self) -> BenchmarkResult | None:
        viable = [r for r in self._benchmarks if not r.oom and r.trace_mode == "off"]
        if not viable:
            viable = [r for r in self._benchmarks if not r.oom]
        if not viable:
            return None
        return max(viable, key=lambda r: r.images_per_second_train)

    def _infer_network_type(self) -> str:
        """Infiere la interconexión GPU-GPU para el escenario DDP multi-GPU.

        El caso realista de multi-GPU es `torchrun --nproc_per_node=N` en UN
        nodo: las GPUs se comunican por NVLink (datacenter) o PCIe (resto).
        Que el DISCO sea NFS NO implica que las GPUs hablen por Ethernet — son
        cosas independientes (Kaggle, Verode y local tienen NFS pero las GPUs
        están en el mismo nodo). Inferir "Gigabit" del NFS daba sync ~128×
        sobreestimado → predicciones ~6× pesimistas (0.29× cuando el real es
        1.90× con vit_base en 2×T4).
        """
        if not self._gpu.is_cuda:
            return "Ethernet"  # multi-nodo solo-CPU (gloo)
        name = self._gpu.device_name or ""
        # GPUs de datacenter suelen llevar NVLink entre sí en el mismo nodo.
        if any(g in name for g in ("V100", "A100", "H100", "A800", "A30", "A40")):
            return "NVLink"
        # T4, RTX y demás: PCIe en el mismo nodo.
        return "PCIe"

    def _build_scenario(
        self,
        n_gpus: int,
        base_result: BenchmarkResult,
        n_epochs: int,
        bw_gb_s: float,
        eff_base: float,
    ) -> DDPScenario:
        # Batch por GPU: el batch del benchmark ya es lo óptimo por GPU
        batch_per_gpu = base_result.batch_size
        global_batch = batch_per_gpu * n_gpus

        # Workers por GPU
        logical = self._cpu.logical_cores if self._cpu else 8
        workers_per_gpu = min(max(1, logical // max(1, n_gpus)), 8)

        # ── Modelo a nivel de EPOCH (no de batch) ──────────────────────────
        # Cómputo: el throughput sintético (sin I/O) escala con 1/n_gpus, porque
        # cada GPU procesa n_train/n_gpus imágenes.
        imgs_per_s_compute = base_result.images_per_second_train or 1.0
        compute_time_epoch = (self._n_train / imgs_per_s_compute) / n_gpus

        # I/O: leer las n_train imágenes (3 ficheros c/u) del disco. Es un total
        # ~CONSTANTE en n_gpus — el dataset es fijo y el disco se comparte entre
        # los lectores, así que el disco es el límite global, no por-GPU.
        if self._disk and self._disk.files_per_second > 0:
            io_imgs_per_s = self._disk.files_per_second / 3.0
            io_time_epoch = self._n_train / io_imgs_per_s
        else:
            io_time_epoch = 0.0

        # Sincronización de gradientes (All-Reduce = 2 × params × 4 bytes).
        sync_bytes = 2 * self._model.total_params * 4
        sync_per_batch = (sync_bytes / (bw_gb_s * 1e9)) if (bw_gb_s > 0 and n_gpus > 1) else 0.0
        n_batches_per_gpu = math.ceil(self._n_train / global_batch)
        sync_time_epoch = sync_per_batch * n_batches_per_gpu

        # Cómputo e I/O se solapan (los dataloader workers prefetchan mientras la
        # GPU computa) → max(); la sincronización es serie → se suma.
        step_time_epoch = max(compute_time_epoch, io_time_epoch)
        time_train_epoch = (step_time_epoch + sync_time_epoch) * self._nfs_factor
        time_eval_epoch = math.ceil(self._n_val / batch_per_gpu) * base_result.seconds_per_batch_eval

        # Speedup vs 1 GPU (mismo modelo, mismo disco).
        base_train_1gpu = max(self._n_train / imgs_per_s_compute, io_time_epoch)
        if n_gpus == 1 or time_train_epoch <= 0:
            efficiency = 1.0
            speedup = 1.0
            sync_pct = 0.0
        else:
            speedup = (base_train_1gpu * self._nfs_factor) / time_train_epoch
            efficiency = speedup / n_gpus
            sync_pct = sync_time_epoch / (step_time_epoch + sync_time_epoch) * 100

        # Cuello de botella
        if io_time_epoch > compute_time_epoch * 1.2:
            bottleneck = "io"
        elif sync_pct > 30:
            bottleneck = "sync"
        else:
            bottleneck = "compute"

        total_s = (time_train_epoch + time_eval_epoch) * n_epochs

        return DDPScenario(
            n_gpus=n_gpus,
            batch_per_gpu=batch_per_gpu,
            global_batch=global_batch,
            num_workers_per_gpu=workers_per_gpu,
            estimated_speedup=round(speedup, 2),
            scaling_efficiency=round(efficiency * 100, 1),
            sync_overhead_pct=round(sync_pct, 1),
            bottleneck=bottleneck,
            time_train_per_epoch_s=time_train_epoch,
            time_total_s=total_s,
        )

    def recommend_config(self) -> dict:
        """Devuelve la configuración recomendada basada en los escenarios.

        Criterio: el MAYOR nº de GPUs que aún escala con eficiencia ≥ 75%.
        Así, si es compute-bound se recomienda distribuir (varias GPUs); si es
        I/O-bound o sync-bound (eficiencia cae <75% al añadir GPUs), se queda en
        1 GPU porque distribuir no compensa. (El criterio anterior usaba
        speedup/n_gpus = eficiencia, que siempre daba 1 GPU como "óptimo".)
        """
        scenarios = self.compute_scenarios(30)
        if not scenarios:
            return {}
        EFF_MIN = 75.0  # % de eficiencia mínima para que merezca la pena escalar
        efficient = [s for s in scenarios
                     if s.n_gpus == 1 or s.scaling_efficiency >= EFF_MIN]
        best = max(efficient, key=lambda s: s.n_gpus)
        return {
            "n_gpus": best.n_gpus,
            "batch_per_gpu": best.batch_per_gpu,
            "global_batch": best.global_batch,
            "num_workers_per_gpu": best.num_workers_per_gpu,
            "estimated_speedup": best.estimated_speedup,
            "efficiency_pct": best.scaling_efficiency,
            "bottleneck": best.bottleneck,
        }


# ═════════════════════════════════════════════════════════════════════════════
# Benchmarker — throughput real de train y eval
# ═════════════════════════════════════════════════════════════════════════════

class Benchmarker:
    N_WARMUP = 3
    N_MEASURE = 8

    def __init__(self, model: nn.Module, device: torch.device, precision: str = "fp32"):
        from src import precision as precision_mod
        self._model = model.to(device)
        self._device = device
        self._criterion = nn.BCEWithLogitsLoss()
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-4)
        # Numeric precision = Tensor-core switch (fp32 -> CUDA cores).
        self._precision = precision if device.type == "cuda" else "fp32"
        precision_mod.apply_backend_flags(self._precision)
        self._amp_dtype = precision_mod.autocast_dtype(self._precision)
        self._use_amp = self._amp_dtype is not None and device.type == "cuda"
        _scaler_on = precision_mod.needs_scaler(self._precision) and device.type == "cuda"
        try:
            self._scaler = torch.amp.GradScaler("cuda", enabled=_scaler_on)
        except (AttributeError, TypeError):
            self._scaler = torch.cuda.amp.GradScaler(enabled=_scaler_on)

    def _autocast(self):
        import contextlib
        if self._use_amp:
            return torch.autocast(device_type="cuda", dtype=self._amp_dtype)
        return contextlib.nullcontext()

    def run(self, batch_size: int, trace_mode: str) -> BenchmarkResult:
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        batches = self._make_batches(batch_size)
        try:
            sec_train, sec_eval, peak_vram, avg_power = self._benchmark(batches, trace_mode)
            return BenchmarkResult(
                batch_size=batch_size,
                trace_mode=trace_mode,
                seconds_per_batch_train=sec_train,
                seconds_per_batch_eval=sec_eval,
                images_per_second_train=batch_size / sec_train if sec_train > 0 else 0.0,
                images_per_second_eval=batch_size / sec_eval if sec_eval > 0 else 0.0,
                peak_vram_gb=peak_vram,
                avg_power_w=avg_power,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return BenchmarkResult(
                batch_size=batch_size, trace_mode=trace_mode,
                seconds_per_batch_train=0.0, seconds_per_batch_eval=0.0,
                images_per_second_train=0.0, images_per_second_eval=0.0,
                peak_vram_gb=0.0, oom=True,
            )

    def _make_batches(self, batch_size: int):
        n = self.N_WARMUP + self.N_MEASURE
        return [
            (torch.randn(batch_size, 3, 224, 224),
             torch.randint(0, 2, (batch_size, 19)).float())
            for _ in range(n)
        ]

    def _benchmark(self, batches, trace_mode) -> tuple[float, float, float, float]:
        hooks = self._register_deep_hooks() if trace_mode == "deep" else []
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        self._model.train()
        for images, labels in batches[:self.N_WARMUP]:
            self._train_step(images, labels)

        power_samples = []
        pynvml_handle = self._get_pynvml_handle()

        if self._device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for images, labels in batches[self.N_WARMUP:]:
            self._train_step(images, labels)
            if pynvml_handle is not None:
                try:
                    import pynvml
                    power_samples.append(pynvml.nvmlDeviceGetPowerUsage(pynvml_handle) / 1000.0)
                except Exception:
                    pass
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        sec_train = (time.perf_counter() - t0) / self.N_MEASURE
        avg_power = sum(power_samples) / len(power_samples) if power_samples else 0.0

        self._model.eval()
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for images, labels in batches[self.N_WARMUP:]:
            self._eval_step(images, labels)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        sec_eval = (time.perf_counter() - t0) / self.N_MEASURE

        peak_vram = torch.cuda.max_memory_allocated() / 1e9 if self._device.type == "cuda" else 0.0

        for h in hooks:
            h.remove()
        return sec_train, sec_eval, peak_vram, avg_power

    @staticmethod
    def _get_pynvml_handle():
        try:
            import pynvml
            pynvml.nvmlInit()
            return pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return None

    def _train_step(self, images, labels):
        images, labels = images.to(self._device), labels.to(self._device)
        self._optimizer.zero_grad()
        with self._autocast():
            loss = self._criterion(self._model(images), labels)
        self._scaler.scale(loss).backward()
        self._scaler.step(self._optimizer)
        self._scaler.update()

    def _eval_step(self, images, labels):
        with torch.no_grad(), self._autocast():
            self._model(images.to(self._device))

    def _register_deep_hooks(self):
        hooks = []
        for _name, module in self._model.named_modules():
            if not list(module.children()):
                hooks.append(module.register_forward_hook(self._noop_hook()))
                hooks.append(module.register_full_backward_hook(self._noop_bw_hook()))
        for param in self._model.parameters():
            if param.requires_grad:
                hooks.append(param.register_hook(lambda g: None))
        return hooks

    @staticmethod
    def _noop_hook():
        def hook(_m, _i, output):
            if isinstance(output, torch.Tensor):
                output.detach().float().abs().mean().item()
        return hook

    @staticmethod
    def _noop_bw_hook():
        def hook(_m, _gi, grad_output):
            if grad_output[0] is not None:
                grad_output[0].detach().float().norm().item()
        return hook


# ═════════════════════════════════════════════════════════════════════════════
# TimeEstimator
# ═════════════════════════════════════════════════════════════════════════════

class TimeEstimator:
    def estimate(
        self,
        result: BenchmarkResult,
        dataset_train: int,
        dataset_val: int,
        epochs: int,
        nfs_factor: float = 1.0,
        model_info: ModelInfo | None = None,
    ) -> Optional[dict]:
        if result.oom or result.images_per_second_train == 0:
            return None

        train_batches = math.ceil(dataset_train / result.batch_size)
        eval_batches = math.ceil(dataset_val / result.batch_size)

        sec_train = train_batches * result.seconds_per_batch_train * nfs_factor
        sec_eval = eval_batches * result.seconds_per_batch_eval
        sec_epoch = sec_train + sec_eval
        sec_total = sec_epoch * epochs

        energy_train_wh = energy_eval_wh = 0.0
        if result.avg_power_w > 0:
            energy_train_wh = result.avg_power_w * sec_train / 3600
            energy_eval_wh = result.avg_power_w * 0.4 * sec_eval / 3600

        flops_train = flops_eval = 0.0
        if model_info and model_info.flops_per_image_mflops:
            flops_img = model_info.flops_per_image_mflops / 1000
            flops_train = flops_img * 3 * dataset_train
            flops_eval = flops_img * dataset_val

        ddp_eff = 0.85
        ddp_projections = {n: sec_total / (n * ddp_eff) for n in (2, 4, 8)}

        return {
            "train_per_epoch": sec_train,
            "eval_per_epoch": sec_eval,
            "total_per_epoch": sec_epoch,
            "total": sec_total,
            "energy_train_wh_per_epoch": energy_train_wh,
            "energy_eval_wh_per_epoch": energy_eval_wh,
            "energy_total_wh": (energy_train_wh + energy_eval_wh) * epochs,
            "flops_train_gflops_per_epoch": flops_train,
            "flops_eval_gflops_per_epoch": flops_eval,
            "optimizer_steps_per_epoch": train_batches,
            "ddp_total_2gpu_h": ddp_projections[2] / 3600,
            "ddp_total_4gpu_h": ddp_projections[4] / 3600,
            "ddp_total_8gpu_h": ddp_projections[8] / 3600,
        }

    @staticmethod
    def format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m:02d}m"


# ═════════════════════════════════════════════════════════════════════════════
# ReportFormatter
# ═════════════════════════════════════════════════════════════════════════════

class ReportFormatter:
    W = 72

    def __init__(self, output_path: Path | None = None):
        self._output_path = output_path
        self._lines: list[str] = []

    def _emit(self, line: str = ""):
        print(line)
        self._lines.append(line)

    def flush(self):
        if self._output_path:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            self._output_path.write_text("\n".join(self._lines) + "\n")
            print(f"\n  Informe guardado en: {self._output_path}")

    def print(self, report: FeasibilityReport):
        self._header(report)
        self._model_section(report.model_info)
        self._hardware_section(report)
        self._cpu_section(report.cpu_info)
        self._disk_section(report.disk_info, report.dataset_profile)
        self._memory_section(report)
        self._benchmark_section(report)
        self._estimates_section(report)
        self._ddp_section(report)
        self._prediction_section(report.performance_prediction)
        self._study_section(report)
        self._recommendations_section(report)
        self.flush()

    def _header(self, report: FeasibilityReport):
        self._emit("═" * self.W)
        self._emit("  ANÁLISIS DE VIABILIDAD — BigEarthNet ViT  (v3)")
        self._emit(f"  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        if report.nfs_factor != 1.0:
            self._emit(f"  Factor NFS aplicado: ×{report.nfs_factor:.2f}")
        self._emit("═" * self.W)

    def _model_section(self, m: ModelInfo):
        self._emit(f"\n{'─'*self.W}  MODELO")
        self._emit(f"  Nombre:               {m.name}")
        self._emit(f"  Parámetros totales:   {m.total_params:,} ({m.total_params/1e6:.1f}M)")
        if m.flops_per_image_mflops:
            self._emit(f"  FLOPs/imagen:         {m.flops_per_image_mflops:.1f} MFLOPs")
        self._emit(f"  Memoria estática:     {m.total_static_mb/1024:.2f} GB")

    def _hardware_section(self, report: FeasibilityReport):
        h = report.hardware_info
        self._emit(f"\n{'─'*self.W}  GPU")
        if h.is_cuda:
            self._emit(f"  {h.device_name}  |  VRAM: {h.total_vram_gb:.1f} GB total / {h.free_vram_gb:.1f} GB libre")
            if h.compute_capability:
                self._emit(f"  Compute Capability: {h.compute_capability}  ({h.architecture})")
            if h.cuda_cores:
                self._emit(f"  {h.sm_count} SMs  |  {h.cuda_cores:,} CUDA cores  |  "
                           f"{h.tensor_cores:,} Tensor cores")
            self._emit(f"  Precisión del benchmark: {report.precision} "
                       f"({'Tensor cores' if report.precision != 'fp32' else 'CUDA cores'})")
            pc = report.precision_comparison
            if pc:
                self._emit(f"  FP32 {pc['fp32_imgs_s']:.0f} img/s  vs  "
                           f"{pc['tc_precision'].upper()} {pc['tc_imgs_s']:.0f} img/s  "
                           f"→ {pc['speedup']}× con Tensor cores "
                           f"(VRAM {pc['fp32_vram_gb']} → {pc['tc_vram_gb']} GB)")
        else:
            self._emit("  GPU: no disponible (CUDA no activo)")

    def _cpu_section(self, cpu: CPUInfo | None):
        if cpu is None:
            return
        self._emit(f"\n{'─'*self.W}  CPU / SISTEMA")
        self._emit(f"  Núcleos: {cpu.logical_cores} lógicos / {cpu.physical_cores} físicos")
        if cpu.freq_mhz:
            self._emit(f"  Frecuencia: {cpu.freq_mhz:.0f} MHz")
        self._emit(f"  RAM: {cpu.ram_total_gb:.1f} GB total / {cpu.ram_free_gb:.1f} GB libre")

    def _disk_section(self, disk: DiskInfo | None, profile: DatasetProfile | None):
        if disk is None and profile is None:
            return
        self._emit(f"\n{'─'*self.W}  DISCO / DATASET I/O")
        if disk:
            self._emit(f"  Tipo: {disk.disk_type}  |  NFS: {'sí' if disk.is_nfs else 'no'}")
            if disk.read_mb_per_s > 0:
                self._emit(f"  Velocidad lectura: {disk.read_mb_per_s:.0f} MB/s  |  {disk.files_per_second:.0f} patches/s")
        if profile:
            self._emit(f"  Patches encontrados: ~{profile.n_files_total_est:,}")
            if profile.io_bottleneck_ratio > 0:
                if profile.io_bottleneck_ratio > 1.2:
                    self._emit(f"  ⚠ I/O-BOUND (ratio={profile.io_bottleneck_ratio:.2f}) — data loading más lento que cómputo")
                else:
                    self._emit(f"  ✓ Compute-bound (ratio={profile.io_bottleneck_ratio:.2f}) — GPU es el cuello de botella")

    def _memory_section(self, report: FeasibilityReport):
        m, h = report.model_info, report.hardware_info
        if not m.activation_mb_per_image:
            return
        self._emit(f"\n{'─'*self.W}  MEMORIA POR BATCH SIZE")
        self._emit(f"  {'Batch':>5}  {'Total est.':>11}  {'Estado':>10}")
        for bs in report.batch_sizes:
            total_gb = m.total_mb(bs) / 1024
            if h.is_cuda and total_gb > h.total_vram_gb:
                estado = "OOM ✗"
            elif h.is_cuda and total_gb > h.total_vram_gb * 0.85:
                estado = "⚠ Límite"
            else:
                estado = "✓ OK"
            self._emit(f"  {bs:>5}  {total_gb:>9.2f} GB  {estado:>10}")

    def _benchmark_section(self, report: FeasibilityReport):
        self._emit(f"\n{'─'*self.W}  BENCHMARK  ({Benchmarker.N_MEASURE} batches sintéticos)")
        self._emit(f"  {'Batch':>5}  {'Modo':<8}  {'imgs/s(train)':>13}  {'imgs/s(eval)':>12}  {'VRAM':>7}")
        for r in report.results:
            if r.oom:
                self._emit(f"  {r.batch_size:>5}  {r.trace_mode:<8}  {'OOM':>13}")
            else:
                self._emit(
                    f"  {r.batch_size:>5}  {r.trace_mode:<8}  "
                    f"{r.images_per_second_train:>11.1f}  "
                    f"{r.images_per_second_eval:>10.1f}  "
                    f"{r.peak_vram_gb:>5.2f} GB"
                )

    def _estimates_section(self, report: FeasibilityReport):
        estimator = TimeEstimator()
        nfs, mi = report.nfs_factor, report.model_info
        self._emit(f"\n{'─'*self.W}  ESTIMACIONES DE TIEMPO")
        for epochs in report.epochs_list:
            self._emit(f"\n  {epochs} epochs:")
            self._emit(f"  {'Batch':>5}  {'Modo':<8}  {'Train/ep':>8}  {'Eval/ep':>7}  {'Total/ep':>8}  {'TOTAL':>8}")
            for r in report.results:
                est = estimator.estimate(r, report.dataset_train, report.dataset_val, epochs, nfs, mi)
                if est is None:
                    self._emit(f"  {r.batch_size:>5}  {r.trace_mode:<8}  OOM")
                else:
                    self._emit(
                        f"  {r.batch_size:>5}  {r.trace_mode:<8}  "
                        f"{TimeEstimator.format_time(est['train_per_epoch']):>8}  "
                        f"{TimeEstimator.format_time(est['eval_per_epoch']):>7}  "
                        f"{TimeEstimator.format_time(est['total_per_epoch']):>8}  "
                        f"{TimeEstimator.format_time(est['total']):>8}"
                    )

    def _ddp_section(self, report: FeasibilityReport):
        if not report.ddp_scenarios:
            return
        self._emit(f"\n{'─'*self.W}  ANÁLISIS DDP — DISTRIBUCIÓN DE RECURSOS")
        self._emit(f"  {'GPUs':>4}  {'Batch/GPU':>9}  {'Global batch':>12}  {'Workers':>7}  {'Speedup':>7}  {'Efic.':>6}  {'Cuello':>8}")
        for s in report.ddp_scenarios:
            self._emit(
                f"  {s.n_gpus:>4}  {s.batch_per_gpu:>9}  {s.global_batch:>12}  "
                f"{s.num_workers_per_gpu:>7}  {s.estimated_speedup:>6.2f}×  "
                f"{s.scaling_efficiency:>5.1f}%  {s.bottleneck:>8}"
            )
        best = max(report.ddp_scenarios, key=lambda s: s.estimated_speedup / max(s.n_gpus, 1))
        self._emit(f"\n  Configuración recomendada: {best.n_gpus} GPU(s) con batch={best.batch_per_gpu}/GPU")
        if best.bottleneck == "io":
            self._emit("  ⚠ I/O es el cuello de botella — más GPUs no ayudarán sin disco más rápido")
        elif best.bottleneck == "sync":
            self._emit("  ⚠ Sincronización de gradientes es el cuello de botella — red lenta")

    def _prediction_section(self, pred: PerformancePrediction | None):
        if pred is None:
            return
        self._emit(f"\n{'─'*self.W}  PREDICCIÓN DE RENDIMIENTO (empírica)")
        self._emit(f"  Modelo: {pred.model_name}")
        self._emit(f"  Val F1 esperado:    ~{pred.predicted_best_f1:.3f}  (epoch ≈ {pred.predicted_best_epoch})")
        self._emit(f"  Early stop aprox.:  epoch ≈ {pred.predicted_early_stop_epoch}  (patience=10)")
        self._emit(f"  Confianza:          {pred.confidence}")
        self._emit(f"  Nota: {pred.notes[:100]}")

    def _study_section(self, report: FeasibilityReport):
        study = report.study_report
        if study is None:
            return
        self._emit(f"\n{'─'*self.W}  ESTUDIO EMPÍRICO DE CONVERGENCIA (medido)")

        if getattr(study, "lr_range", None):
            lr = study.lr_range
            self._emit("  LR range test:")
            self._emit(f"    LR sugerido (mayor descenso): {lr.suggested_lr:.2e}")
            self._emit(f"    LR del mínimo de loss:        {lr.min_loss_lr:.2e}")
            if lr.diverged_lr:
                self._emit(f"    LR de divergencia:            {lr.diverged_lr:.2e}")

        if getattr(study, "convergence", None):
            cv = study.convergence
            self._emit("  Convergencia (mini-training real):")
            self._emit(f"    Steps medidos:        {len(cv.steps)}  |  throughput {cv.measured_imgs_per_s:.1f} imgs/s")
            self._emit(f"    Ajuste loss=a·t^-b+c: a={cv.fit_a:.3f}, b={cv.fit_b:.3f}, c={cv.fit_c:.3f}  (R²={cv.r_squared:.3f})")
            self._emit(f"    Loss extrapolada 1 epoch:  {cv.extrapolated_loss_1ep:.4f}")
            self._emit(f"    Loss extrapolada final:    {cv.extrapolated_loss_final:.4f}")
            self._emit(f"    Val F1 estimado (medido):  ~{cv.extrapolated_best_f1:.3f}")
            self._emit(f"    Plateau estimado:          epoch ≈ {cv.epochs_to_plateau}")

        if getattr(study, "gradient_noise", None):
            gn = study.gradient_noise
            self._emit("  Gradient noise scale:")
            self._emit(f"    Norma gradiente: {gn.grad_norm_mean:.3f} ± {gn.grad_norm_std:.3f}  (CV={gn.cv:.3f})")
            self._emit(f"    Batch size sugerido: {gn.suggested_batch_size}  (noise scale ≈ {gn.noise_scale:.1f})")

    def _recommendations_section(self, report: FeasibilityReport):
        self._emit(f"\n{'─'*self.W}  RECOMENDACIONES")
        estimator = TimeEstimator()
        nfs = report.nfs_factor
        target_epochs = max(report.epochs_list)
        viable = [r for r in report.results if not r.oom and r.trace_mode == "off"]
        if not viable:
            self._emit("  ✗ Ningún batch size viable")
            return
        best = max(viable, key=lambda r: r.images_per_second_train)
        self._emit(f"  ✓ Batch size óptimo: {best.batch_size} ({best.images_per_second_train:.1f} imgs/s)")
        for mode in report.trace_modes:
            r = next((x for x in report.results if x.batch_size == best.batch_size and x.trace_mode == mode), None)
            if r and not r.oom:
                est = estimator.estimate(r, report.dataset_train, report.dataset_val, target_epochs, nfs, report.model_info)
                if est:
                    self._emit(
                        f"  → --trace {mode:<8} {target_epochs} epochs: "
                        f"~{TimeEstimator.format_time(est['total'])}"
                        f"  [DDP×2: ~{est['ddp_total_2gpu_h']:.1f}h]"
                    )
        if nfs == 1.0:
            self._emit("\n  💡 En Verode (NFS), usa --nfs-factor 1.3 para estimaciones más precisas")
        self._emit()

    def write_csv(self, report: FeasibilityReport, env: str = "local"):
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        out_dir = Path(f"logs/{env}/feasibility")
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"feasibility_{timestamp}.csv"

        estimator = TimeEstimator()
        target_epochs = max(report.epochs_list) if report.epochs_list else 0
        mi, hi = report.model_info, report.hardware_info

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Metadata del modelo
            writer.writerow(["#meta", "model_name", "total_params_M", "flops_mflops",
                              "hardware_name", "total_vram_gb", "free_vram_gb"])
            writer.writerow(["#meta", mi.name, round(mi.total_params/1e6, 2),
                              round(mi.flops_per_image_mflops, 1), hi.device_name,
                              round(hi.total_vram_gb, 2), round(hi.free_vram_gb, 2)])

            # GPU hardware specs (compute capability × SM count)
            writer.writerow(["#gpu", "compute_capability", "architecture",
                              "sm_count", "cuda_cores", "tensor_cores"])
            writer.writerow(["#gpu", hi.compute_capability, hi.architecture,
                              hi.sm_count, hi.cuda_cores, hi.tensor_cores])

            # Precision used + optional FP32-vs-Tensor-core comparison
            writer.writerow(["#precision", "mode"])
            writer.writerow(["#precision", report.precision])
            pc = report.precision_comparison
            if pc:
                writer.writerow(["#precision_cmp", "batch_size", "tc_precision",
                                  "fp32_imgs_s", "tc_imgs_s", "speedup",
                                  "fp32_vram_gb", "tc_vram_gb"])
                writer.writerow(["#precision_cmp", pc.get("batch_size"), pc.get("tc_precision"),
                                  pc.get("fp32_imgs_s"), pc.get("tc_imgs_s"), pc.get("speedup"),
                                  pc.get("fp32_vram_gb"), pc.get("tc_vram_gb")])

            # Tamaño REAL del dataset usado (n imágenes por split) — clave para
            # que la comparación estimación-vs-real no asuma el full set.
            writer.writerow(["#sizes", "n_train", "n_val", "nfs_factor"])
            writer.writerow(["#sizes", report.dataset_train, report.dataset_val,
                              report.nfs_factor])

            # Memoria del modelo
            writer.writerow(["#model_mem", "weight_mb", "gradient_mb", "optimizer_mb",
                              "activation_mb_per_image", "total_static_mb"])
            writer.writerow(["#model_mem", round(mi.weight_mb, 1), round(mi.gradient_mb, 1),
                              round(mi.optimizer_mb, 1), round(mi.activation_mb_per_image, 1),
                              round(mi.total_static_mb, 1)])

            # CPU info
            if report.cpu_info:
                cpu = report.cpu_info
                writer.writerow(["#cpu", "logical_cores", "physical_cores", "freq_mhz",
                                  "ram_total_gb", "ram_free_gb"])
                writer.writerow(["#cpu", cpu.logical_cores, cpu.physical_cores,
                                  round(cpu.freq_mhz, 0), round(cpu.ram_total_gb, 1),
                                  round(cpu.ram_free_gb, 1)])

            # Disk info
            if report.disk_info:
                disk = report.disk_info
                writer.writerow(["#disk", "type", "is_nfs", "read_mb_per_s", "files_per_second"])
                writer.writerow(["#disk", disk.disk_type, "yes" if disk.is_nfs else "no",
                                  round(disk.read_mb_per_s, 1), round(disk.files_per_second, 1)])

            # Dataset profile
            if report.dataset_profile:
                dp = report.dataset_profile
                writer.writerow(["#dataset", "n_files_est", "read_mb_per_s",
                                  "files_per_second", "io_bottleneck_ratio"])
                writer.writerow(["#dataset", dp.n_files_total_est,
                                  round(dp.sample_read_mb_per_s, 1),
                                  round(dp.files_per_second, 1),
                                  round(dp.io_bottleneck_ratio, 3)])

            # Predicción de rendimiento
            if report.performance_prediction:
                pred = report.performance_prediction
                writer.writerow(["#prediction", "predicted_best_f1", "predicted_best_epoch",
                                  "predicted_early_stop_epoch", "confidence"])
                writer.writerow(["#prediction", pred.predicted_best_f1,
                                  pred.predicted_best_epoch, pred.predicted_early_stop_epoch,
                                  pred.confidence])
                # Curva F1 (una fila por epoch)
                if pred.curve_epochs:
                    writer.writerow(["#curve_val_f1"] + pred.curve_f1_val)
                    writer.writerow(["#curve_train_f1"] + pred.curve_f1_train)
                    writer.writerow(["#curve_epochs"] + pred.curve_epochs)

            # Escenarios DDP
            if report.ddp_scenarios:
                writer.writerow(["#ddp", "n_gpus", "batch_per_gpu", "global_batch",
                                  "workers_per_gpu", "speedup", "efficiency_pct",
                                  "sync_overhead_pct", "bottleneck",
                                  "time_train_epoch_min", "time_total_h"])
                for s in report.ddp_scenarios:
                    writer.writerow([
                        "#ddp", s.n_gpus, s.batch_per_gpu, s.global_batch,
                        s.num_workers_per_gpu, round(s.estimated_speedup, 2),
                        round(s.scaling_efficiency, 1), round(s.sync_overhead_pct, 1),
                        s.bottleneck,
                        round(s.time_train_per_epoch_s / 60, 1),
                        round(s.time_total_s / 3600, 2),
                    ])

            # Estudio empírico de convergencia (v4)
            study = report.study_report
            if study is not None:
                lr = getattr(study, "lr_range", None)
                cv = getattr(study, "convergence", None)
                gn = getattr(study, "gradient_noise", None)
                if lr is not None:
                    writer.writerow(["#study_lr", "suggested_lr", "min_loss_lr", "diverged_lr"])
                    writer.writerow(["#study_lr", f"{lr.suggested_lr:.3e}",
                                      f"{lr.min_loss_lr:.3e}",
                                      f"{lr.diverged_lr:.3e}" if lr.diverged_lr else ""])
                    writer.writerow(["#study_lr_curve_lrs"] + [f"{x:.3e}" for x in lr.lrs])
                    writer.writerow(["#study_lr_curve_losses"] + [round(x, 5) for x in lr.losses])
                if cv is not None:
                    writer.writerow(["#study_conv", "fit_a", "fit_b", "fit_c", "r_squared",
                                      "loss_1ep", "loss_final", "best_f1", "epochs_to_plateau",
                                      "measured_imgs_per_s"])
                    writer.writerow(["#study_conv", round(cv.fit_a, 5), round(cv.fit_b, 5),
                                      round(cv.fit_c, 5), round(cv.r_squared, 4),
                                      round(cv.extrapolated_loss_1ep, 5),
                                      round(cv.extrapolated_loss_final, 5),
                                      round(cv.extrapolated_best_f1, 4),
                                      cv.epochs_to_plateau, round(cv.measured_imgs_per_s, 1)])
                    writer.writerow(["#study_conv_steps"] + cv.steps)
                    writer.writerow(["#study_conv_losses"] + [round(x, 5) for x in cv.losses])
                    writer.writerow(["#study_conv_f1s"] + [round(x, 5) for x in cv.f1s])
                if gn is not None:
                    writer.writerow(["#study_grad", "grad_norm_mean", "grad_norm_std",
                                      "noise_scale", "suggested_batch_size", "cv"])
                    writer.writerow(["#study_grad", round(gn.grad_norm_mean, 5),
                                      round(gn.grad_norm_std, 5), round(gn.noise_scale, 2),
                                      gn.suggested_batch_size, round(gn.cv, 4)])

            # Benchmarks (filas principales)
            writer.writerow([
                "batch_size", "trace_mode",
                "s_per_batch_train", "imgs_per_s_train",
                "s_per_batch_eval", "imgs_per_s_eval",
                "peak_vram_gb", "avg_power_w", "oom",
                "est_train_min_per_epoch", "est_eval_min_per_epoch",
                "est_total_min_per_epoch",
                f"est_total_h_{target_epochs}ep",
                "est_energy_train_wh_per_epoch", "est_energy_eval_wh_per_epoch",
                "est_energy_total_wh",
                "flops_train_gflops_per_epoch", "flops_eval_gflops_per_epoch",
                "optimizer_steps_per_epoch",
                f"est_ddp_2gpu_h_{target_epochs}ep",
                f"est_ddp_4gpu_h_{target_epochs}ep",
            ])
            for r in report.results:
                est = (estimator.estimate(r, report.dataset_train, report.dataset_val,
                                          target_epochs, report.nfs_factor, model_info=mi)
                       if not r.oom else None)
                writer.writerow([
                    r.batch_size, r.trace_mode,
                    round(r.seconds_per_batch_train, 4) if not r.oom else "",
                    round(r.images_per_second_train, 1) if not r.oom else "",
                    round(r.seconds_per_batch_eval, 4) if not r.oom else "",
                    round(r.images_per_second_eval, 1) if not r.oom else "",
                    round(r.peak_vram_gb, 2) if not r.oom else "",
                    round(r.avg_power_w, 1) if not r.oom and r.avg_power_w > 0 else "",
                    "yes" if r.oom else "no",
                    round(est["train_per_epoch"] / 60, 1) if est else "",
                    round(est["eval_per_epoch"] / 60, 1) if est else "",
                    round(est["total_per_epoch"] / 60, 1) if est else "",
                    round(est["total"] / 3600, 2) if est else "",
                    round(est["energy_train_wh_per_epoch"], 2) if est and est["energy_train_wh_per_epoch"] else "",
                    round(est["energy_eval_wh_per_epoch"], 2) if est and est["energy_eval_wh_per_epoch"] else "",
                    round(est["energy_total_wh"], 1) if est and est["energy_total_wh"] else "",
                    round(est["flops_train_gflops_per_epoch"], 1) if est and est["flops_train_gflops_per_epoch"] else "",
                    round(est["flops_eval_gflops_per_epoch"], 1) if est and est["flops_eval_gflops_per_epoch"] else "",
                    est["optimizer_steps_per_epoch"] if est else "",
                    round(est["ddp_total_2gpu_h"], 2) if est else "",
                    round(est["ddp_total_4gpu_h"], 2) if est else "",
                ])

        print(f"  → CSV guardado: {csv_path}")
        return csv_path


# ═════════════════════════════════════════════════════════════════════════════
# FeasibilityChecker — Facade
# ═════════════════════════════════════════════════════════════════════════════

class FeasibilityChecker:
    def __init__(
        self,
        model_name: str,
        batch_sizes: list[int],
        epochs_list: list[int],
        trace_modes: list[str],
        dataset_train: int,
        dataset_val: int,
        nfs_factor: float = 1.0,
        dataset_path: str | None = None,
        profile_disk: bool = True,
        predict_performance: bool = True,
        analyze_ddp: bool = True,
        config: dict | None = None,
        convergence_study: bool = False,
        study_steps: int = 60,
        device_index: int = 0,
        precision: str = "fp32",
        compare_precision: bool = False,
    ):
        self._model_name = model_name
        self._batch_sizes = batch_sizes
        self._epochs_list = epochs_list
        self._trace_modes = trace_modes
        self._dataset_train = dataset_train
        self._dataset_val = dataset_val
        self._nfs_factor = nfs_factor
        self._dataset_path = dataset_path
        self._profile_disk = profile_disk
        self._predict_performance = predict_performance
        self._analyze_ddp = analyze_ddp
        self._config = config or {}
        self._convergence_study = convergence_study
        self._study_steps = study_steps
        self._precision = precision
        self._compare_precision = compare_precision
        self._device_index = device_index if torch.cuda.is_available() else 0
        self._device = torch.device(
            f"cuda:{self._device_index}" if torch.cuda.is_available() else "cpu"
        )

    def run(self) -> FeasibilityReport:
        print(f"Cargando modelo {self._model_name}...")
        model = build_model(model_name=self._model_name, pretrained=False)

        hw_probe = HardwareProbe()
        model_info = ModelAnalyzer(model, self._model_name, self._device).analyze()
        hardware_info = hw_probe.probe_gpu(self._device_index)
        cpu_info = hw_probe.probe_cpu()

        print("Perfilando CPU y disco...")
        disk_probe = DiskProbe()
        disk_info = disk_probe.probe(self._dataset_path) if self._profile_disk else None

        dataset_profile: DatasetProfile | None = None
        if disk_info and self._profile_disk:
            # Usar el primer resultado viable del benchmark para el ratio I/O
            # (lo calculamos después del benchmark, lo actualizamos en _finalize)
            dataset_profile = DatasetProfiler(
                self._dataset_path, disk_info
            ).profile(0.5, self._batch_sizes[0])  # placeholder, actualizado después

        benchmarker = Benchmarker(model, self._device, precision=self._precision)
        report = FeasibilityReport(
            model_info=model_info,
            hardware_info=hardware_info,
            dataset_train=self._dataset_train,
            dataset_val=self._dataset_val,
            nfs_factor=self._nfs_factor,
            batch_sizes=self._batch_sizes,
            epochs_list=self._epochs_list,
            trace_modes=self._trace_modes,
            precision=self._precision,
            cpu_info=cpu_info,
            disk_info=disk_info,
        )

        total = len(self._batch_sizes) * len(self._trace_modes)
        done = 0
        for batch_size in self._batch_sizes:
            for mode in self._trace_modes:
                done += 1
                print(f"Benchmark {done}/{total}: batch={batch_size}, trace={mode}...")
                result = benchmarker.run(batch_size, mode)
                report.results.append(result)

        # Actualizar dataset profile con tiempo de cómputo real
        if disk_info and self._profile_disk:
            base = next((r for r in report.results if not r.oom), None)
            if base:
                report.dataset_profile = DatasetProfiler(
                    self._dataset_path, disk_info
                ).profile(base.seconds_per_batch_train, base.batch_size)

        # Predicción de rendimiento
        if self._predict_performance:
            training_cfg = self._config.get("training", {})
            has_llrd = "llrd_decay" in training_cfg
            has_ls = training_cfg.get("label_smoothing", 0.0) > 0
            target_epochs = max(self._epochs_list)
            report.performance_prediction = PerformancePredictor().predict(
                model_name=self._model_name,
                n_epochs=target_epochs,
                has_llrd=has_llrd,
                has_label_smoothing=has_ls,
            )

        # Análisis DDP
        if self._analyze_ddp:
            optimizer = DDPOptimizer(
                model_info=model_info,
                hardware_info=hardware_info,
                cpu_info=cpu_info,
                disk_info=disk_info,
                benchmark_results=report.results,
                dataset_train=self._dataset_train,
                dataset_val=self._dataset_val,
                nfs_factor=self._nfs_factor,
            )
            report.ddp_scenarios = optimizer.compute_scenarios(max(self._epochs_list))

        # Comparación de precisión FP32 vs Tensor cores (mismo batch, dos pasadas)
        if self._compare_precision and self._device.type == "cuda":
            report.precision_comparison = self._compare_precisions(model, hardware_info)

        # Estudio empírico de convergencia (mini-training real)
        if self._convergence_study:
            report.study_report = self._run_convergence_study(model)

        return report

    def _compare_precisions(self, model, hardware_info) -> dict | None:
        """Benchmark FP32 vs the best Tensor-core precision at one batch size,
        to quantify the speedup the Tensor cores give."""
        from src import precision as precision_mod
        avail = precision_mod.available_precisions(hardware_info.compute_capability, True)
        tc = next((p for p in ("amp", "bf16", "tf32") if p in avail), None)
        if tc is None:
            return None
        bs = self._batch_sizes[0]
        out = {"batch_size": bs, "tc_precision": tc}
        for key, prec in (("fp32", "fp32"), ("tc", tc)):
            bench = Benchmarker(model, self._device, precision=prec)
            try:
                res = bench.run(bs, "off")
            except Exception:
                return None
            out[f"{key}_imgs_s"] = round(res.images_per_second_train, 1)
            out[f"{key}_vram_gb"] = round(res.peak_vram_gb, 2)
        f, t = out.get("fp32_imgs_s", 0), out.get("tc_imgs_s", 0)
        out["speedup"] = round(t / f, 2) if f > 0 else 0.0
        print(f"  Precisión: FP32 {f:.0f} img/s  vs  {tc.upper()} {t:.0f} img/s  "
              f"→ {out['speedup']}× (Tensor cores)")
        return out

    def _model_family(self) -> str:
        n = self._model_name.lower()
        for fam in ("vit_base", "vit_small", "vit_tiny", "resnet", "efficientnet"):
            if fam in n:
                return "resnet50" if fam == "resnet" else fam
        return "vit_base"

    def _build_real_loader(self, batch_size: int):
        """Construye un DataLoader del dataset real para el estudio.

        Devuelve None si el dataset no está disponible (se hace fallback a sintético).
        """
        data_cfg = self._config.get("data", {})
        root = data_cfg.get("root")
        metadata = data_cfg.get("metadata")
        if not root or not metadata:
            return None
        if not (Path(root).exists() and Path(metadata).exists()):
            return None
        try:
            from src.data.dataset import BigEarthNetDataset, get_transforms
            from torch.utils.data import DataLoader
            ds = BigEarthNetDataset(root, metadata, split="train",
                                    transform=get_transforms("train"))
            return DataLoader(ds, batch_size=batch_size, shuffle=True,
                              num_workers=data_cfg.get("num_workers", 2),
                              pin_memory=(self._device.type == "cuda"))
        except Exception as exc:
            print(f"  [aviso] no se pudo construir loader real: {exc}")
            return None

    def _build_synthetic_loader(self, batch_size: int):
        """Loader sintético de respaldo si el dataset no está disponible."""
        from torch.utils.data import DataLoader, TensorDataset
        n = batch_size * (self._study_steps + 25)
        x = torch.randn(n, 3, 224, 224)
        y = torch.randint(0, 2, (n, 19)).float()
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    def _run_convergence_study(self, model):
        from src.training.convergence_study import ConvergenceStudy

        # Elegir el batch viable más grande para el estudio
        viable_bs = [r.batch_size for r in []]  # placeholder
        batch_size = min(self._batch_sizes)  # conservador (menos VRAM)
        lr = float(self._config.get("training", {}).get("lr", 1e-4))
        target_epochs = max(self._epochs_list)

        loader = self._build_real_loader(batch_size)
        source = "datos reales"
        if loader is None:
            loader = self._build_synthetic_loader(batch_size)
            source = "datos sintéticos (dataset no disponible)"

        print(f"Estudio de convergencia ({self._study_steps} steps, batch={batch_size}, {source})…")
        study = ConvergenceStudy(self._device, self._model_family())
        try:
            report = study.run_full_study(
                model, loader, lr=lr, batch_size=batch_size,
                n_train_images=self._dataset_train, n_epochs_target=target_epochs,
                n_steps=self._study_steps,
            )
            report.notes += f" Fuente: {source}."
            return report
        except Exception as exc:
            print(f"  [aviso] estudio de convergencia falló: {exc}")
            return None


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analizador de viabilidad pre-entrenamiento — BigEarthNet ViT v3"
    )
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", type=str, nargs="+", default=None,
                        help="Modelo(s) timm (separados por espacio)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, nargs="+", default=None)
    parser.add_argument("--trace-modes", nargs="+",
                        choices=["off", "simple", "deep"], default=["off", "simple"])
    parser.add_argument("--nfs-factor", type=float, default=1.0, metavar="FACTOR",
                        help="Multiplicador de overhead para almacenamiento NFS (p.ej. 1.3 para Verode)")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Ruta al dataset BigEarthNet-S2 para medir I/O real")
    parser.add_argument("--no-disk-profile", action="store_true",
                        help="Omitir medición de I/O del disco (más rápido)")
    parser.add_argument("--no-ddp-analysis", action="store_true",
                        help="Omitir análisis DDP")
    parser.add_argument("--no-prediction", action="store_true",
                        help="Omitir predicción de rendimiento F1")
    parser.add_argument("--convergence-study", action="store_true",
                        help="Ejecuta un mini-training REAL: LR range test + curva de "
                             "convergencia medida + gradient noise scale (más lento, ~3-8 min)")
    parser.add_argument("--study-steps", type=int, default=60,
                        help="Número de steps del mini-training de convergencia (default 60)")
    parser.add_argument("--device", type=int, default=0, metavar="INDEX",
                        help="Índice de GPU CUDA a usar (default 0). Útil en máquinas multi-GPU.")
    parser.add_argument("--precision", choices=["fp32", "tf32", "amp", "bf16"], default="fp32",
                        help="Precisión del benchmark = interruptor de Tensor cores "
                             "(fp32=CUDA cores; tf32/amp/bf16=Tensor cores).")
    parser.add_argument("--compare-precision", action="store_true",
                        help="Mide FP32 vs la mejor precisión Tensor-core y reporta el speedup.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    batch_sizes = args.batch_sizes or [cfg["training"]["batch_size"]]
    epochs_list = args.epochs or [cfg["training"]["epochs"]]
    model_names = args.model or [cfg["model"]["name"]]
    env = cfg.get("output", {}).get("env", "local")

    # Auto-detect dataset path from config if not provided
    dataset_path = args.dataset_path or cfg.get("data", {}).get("root")

    # Tamaño REAL del dataset según el metadata del config — NO asumir el full
    # set. Si el config apunta a un subset (p.ej. metadata_demo.parquet con 5000
    # imágenes), las estimaciones deben usar ese tamaño para que sean comparables
    # con el run real. Fallback al full BigEarthNet si no se puede leer.
    n_train, n_val = 237871, 122342
    meta_path = cfg.get("data", {}).get("metadata")
    if meta_path and Path(meta_path).exists():
        try:
            import pandas as pd
            counts = pd.read_parquet(meta_path, columns=["split"])["split"].value_counts()
            n_train = int(counts.get("train", n_train))
            n_val = int(counts.get("validation", n_val))
            print(f"Dataset (metadata): train={n_train:,}  val={n_val:,}")
        except Exception as exc:
            print(f"[aviso] no se pudo leer el tamaño del metadata ({exc}); "
                  f"usando full set {n_train:,}/{n_val:,}")

    for model_name in model_names:
        output_path = args.output
        if output_path is None:
            ts = datetime.now().strftime("%d%m%Y_%H%M%S")
            output_path = Path(f"logs/{env}/feasibility/feasibility_{ts}.log")

        checker = FeasibilityChecker(
            model_name=model_name,
            batch_sizes=batch_sizes,
            epochs_list=epochs_list,
            trace_modes=args.trace_modes,
            dataset_train=n_train,
            dataset_val=n_val,
            nfs_factor=args.nfs_factor,
            dataset_path=dataset_path,
            profile_disk=not args.no_disk_profile,
            predict_performance=not args.no_prediction,
            analyze_ddp=not args.no_ddp_analysis,
            config=cfg,
            convergence_study=args.convergence_study,
            study_steps=args.study_steps,
            device_index=args.device,
            precision=args.precision,
            compare_precision=args.compare_precision,
        )

        report = checker.run()
        formatter = ReportFormatter(output_path=output_path)
        formatter.print(report)
        formatter.write_csv(report, env=env)
        output_path = None  # reset para el siguiente modelo


if __name__ == "__main__":
    main()
