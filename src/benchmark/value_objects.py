"""Value objects (dataclasses) shared across the benchmark analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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
class BenchmarkReport:
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
