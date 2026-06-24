"""BenchmarkChecker — Facade coordinating probes, benchmark, prediction and
DDP analysis into one BenchmarkReport."""
from __future__ import annotations

from pathlib import Path

import torch

from src.models.vit import build_model
from src.benchmark.value_objects import DatasetProfile, BenchmarkReport
from src.benchmark.model_analyzer import ModelAnalyzer
from src.benchmark.probes import HardwareProbe, DiskProbe, DatasetProfiler
from src.benchmark.predictor import PerformancePredictor
from src.benchmark.ddp_optimizer import DDPOptimizer
from src.benchmark.benchmarker import Benchmarker


class BenchmarkChecker:
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

    def run(self) -> BenchmarkReport:
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
        report = BenchmarkReport(
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
                dataset_size=self._dataset_train,
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
