"""DDPOptimizer — distributed scaling scenarios (1/2/4/8 GPUs) and efficiency."""
from __future__ import annotations

import math

from src.benchmark.value_objects import (
    ModelInfo, HardwareInfo, CPUInfo, DiskInfo, DDPScenario, BenchmarkResult,
)


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
