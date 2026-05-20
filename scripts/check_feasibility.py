"""Pre-training feasibility checker for BigEarthNet ViT.

Measures theoretical complexity and real throughput for each
(batch_size, trace_mode) combination and produces time estimates
for the full training run without touching the real dataset.

Usage:
    uv run python scripts/check_feasibility.py
    uv run python scripts/check_feasibility.py --batch-sizes 16 32 64
    uv run python scripts/check_feasibility.py --batch-sizes 16 32 64 --epochs 10 30
    uv run python scripts/check_feasibility.py --batch-sizes 32 64 --trace-modes off deep
"""

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vit import build_model

# ──────────────────────────────────────────────────────────────
# Value objects
# ──────────────────────────────────────────────────────────────

@dataclass
class ModelInfo:
    name: str
    total_params: int
    trainable_params: int
    flops_per_image_mflops: float
    weight_mb: float
    gradient_mb: float
    optimizer_mb: float
    activation_mb_per_image: float = 0.0  # from torchinfo forward/backward pass size

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


@dataclass
class BenchmarkResult:
    batch_size: int
    trace_mode: str
    seconds_per_batch: float
    images_per_second: float
    peak_vram_gb: float
    oom: bool = False


@dataclass
class FeasibilityReport:
    model_info: ModelInfo
    hardware_info: HardwareInfo
    dataset_train: int
    dataset_val: int
    results: list[BenchmarkResult] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    epochs_list: list[int] = field(default_factory=list)
    trace_modes: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# ModelAnalyzer — SRP: only analyses model complexity
# ──────────────────────────────────────────────────────────────

class ModelAnalyzer:
    """Computes theoretical complexity metrics for a given model."""

    def __init__(self, model: nn.Module, model_name: str, device: torch.device):
        self._model = model
        self._name = model_name
        self._device = device

    def analyze(self) -> ModelInfo:
        total = sum(p.numel() for p in self._model.parameters())
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)

        weight_mb = total * 4 / 1e6          # float32
        gradient_mb = trainable * 4 / 1e6
        optimizer_mb = trainable * 8 / 1e6   # AdamW: m + v buffers

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
        """Returns (flops_mflops, activation_mb_per_image) via torchinfo on CPU."""
        try:
            from torchinfo import summary
            # Run on CPU to avoid fragmenting GPU memory before benchmarks
            stats = summary(
                self._model,
                input_size=(1, 3, 224, 224),
                verbose=0,
                device=torch.device("cpu"),
            )
            flops = stats.total_mult_adds / 1e6
            activation_mb = getattr(stats, "total_output_bytes", 0) / 1e6
            return flops, activation_mb
        except Exception:
            return 0.0, 0.0


# ──────────────────────────────────────────────────────────────
# HardwareProbe — SRP: only reads hardware capabilities
# ──────────────────────────────────────────────────────────────

class HardwareProbe:
    """Reads GPU memory and device information."""

    def probe(self) -> HardwareInfo:
        if not torch.cuda.is_available():
            return HardwareInfo(
                device_name="CPU",
                total_vram_gb=0.0,
                free_vram_gb=0.0,
                is_cuda=False,
            )

        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1e9
        reserved_gb = torch.cuda.memory_reserved(0) / 1e9
        free_gb = total_gb - reserved_gb

        return HardwareInfo(
            device_name=props.name,
            total_vram_gb=total_gb,
            free_vram_gb=free_gb,
            is_cuda=True,
        )


# ──────────────────────────────────────────────────────────────
# Benchmarker — SRP: measures real throughput per configuration
# ──────────────────────────────────────────────────────────────

class Benchmarker:
    """Runs timing and memory benchmarks using synthetic data.

    Uses random tensors instead of the real dataset so the benchmark
    is fast and independent of I/O speed.
    """

    N_WARMUP = 3
    N_MEASURE = 8

    def __init__(self, model: nn.Module, device: torch.device):
        self._model = model.to(device)
        self._device = device
        self._criterion = nn.BCEWithLogitsLoss()
        # Single optimizer instance reused across all runs
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-4)

    def run(self, batch_size: int, trace_mode: str) -> BenchmarkResult:
        # Release fragmented memory from previous run
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        batches = self._make_batches(batch_size)
        try:
            seconds, peak_vram = self._benchmark(batches, trace_mode)
            imgs_per_sec = batch_size / seconds if seconds > 0 else 0.0
            return BenchmarkResult(
                batch_size=batch_size,
                trace_mode=trace_mode,
                seconds_per_batch=seconds,
                images_per_second=imgs_per_sec,
                peak_vram_gb=peak_vram,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return BenchmarkResult(
                batch_size=batch_size,
                trace_mode=trace_mode,
                seconds_per_batch=0.0,
                images_per_second=0.0,
                peak_vram_gb=0.0,
                oom=True,
            )

    def _make_batches(self, batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        n = self.N_WARMUP + self.N_MEASURE
        return [
            (torch.randn(batch_size, 3, 224, 224), torch.randint(0, 2, (batch_size, 19)).float())
            for _ in range(n)
        ]

    def _benchmark(self, batches: list, trace_mode: str) -> tuple[float, float]:
        hooks: list = []
        if trace_mode == "deep":
            hooks = self._register_deep_hooks()

        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        self._model.train()
        for images, labels in batches[:self.N_WARMUP]:
            self._step(images, labels)

        if self._device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for images, labels in batches[self.N_WARMUP:]:
            self._step(images, labels)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        peak_vram = (
            torch.cuda.max_memory_allocated() / 1e9
            if self._device.type == "cuda"
            else 0.0
        )

        for h in hooks:
            h.remove()

        return elapsed / self.N_MEASURE, peak_vram

    def _step(self, images: torch.Tensor, labels: torch.Tensor):
        images = images.to(self._device)
        labels = labels.to(self._device)
        self._optimizer.zero_grad()
        loss = self._criterion(self._model(images), labels)
        loss.backward()
        self._optimizer.step()

    def _register_deep_hooks(self) -> list:
        """Register the same hooks as DeepTracingDecorator to measure overhead."""
        hooks = []
        for _name, module in self._model.named_modules():
            if not list(module.children()):
                hooks.append(module.register_forward_hook(self._make_noop_hook()))
                hooks.append(module.register_full_backward_hook(self._make_noop_backward_hook()))
        for param in self._model.parameters():
            if param.requires_grad:
                hooks.append(param.register_hook(lambda g: None))
        return hooks

    @staticmethod
    def _make_noop_hook():
        def hook(_m, _i, output):
            if isinstance(output, torch.Tensor):
                with torch.no_grad():
                    _ = output.detach().float().abs().mean().item()
        return hook

    @staticmethod
    def _make_noop_backward_hook():
        def hook(_m, _gi, grad_output):
            if grad_output[0] is not None:
                with torch.no_grad():
                    _ = grad_output[0].detach().float().norm().item()
        return hook


# ──────────────────────────────────────────────────────────────
# TimeEstimator — SRP: converts benchmark results into estimates
# ──────────────────────────────────────────────────────────────

class TimeEstimator:
    """Converts per-batch timings into full training time estimates."""

    def estimate(
        self,
        result: BenchmarkResult,
        dataset_size: int,
        epochs: int,
    ) -> Optional[tuple[float, float]]:
        """Returns (seconds_per_epoch, total_seconds) or None if OOM."""
        if result.oom or result.images_per_second == 0:
            return None
        batches_per_epoch = dataset_size / result.batch_size
        seconds_per_epoch = batches_per_epoch * result.seconds_per_batch
        return seconds_per_epoch, seconds_per_epoch * epochs

    @staticmethod
    def format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m:02d}m"


# ──────────────────────────────────────────────────────────────
# ReportFormatter — SRP: formats and prints the report
# ──────────────────────────────────────────────────────────────

class ReportFormatter:
    """Formats a FeasibilityReport as a human-readable console output."""

    W = 70

    def print(self, report: FeasibilityReport):
        self._header()
        self._model_section(report.model_info)
        self._hardware_section(report.hardware_info)
        self._memory_section(report)
        self._benchmark_section(report)
        self._estimates_section(report)
        self._recommendations_section(report)

    # ------------------------------------------------------------------

    def _header(self):
        print("═" * self.W)
        title = "ANÁLISIS DE VIABILIDAD — BigEarthNet ViT"
        print(f"  {title}")
        print("═" * self.W)

    def _model_section(self, m: ModelInfo):
        print(f"\n{'─'*self.W}")
        print("  MODELO")
        print(f"{'─'*self.W}")
        print(f"  Nombre:               {m.name}")
        print(f"  Parámetros totales:   {m.total_params:,} ({m.total_params/1e6:.1f}M)")
        print(f"  Parámetros train.:    {m.trainable_params:,} ({m.trainable_params/1e6:.1f}M)")
        if m.flops_per_image_mflops:
            print(f"  FLOPs/imagen:         {m.flops_per_image_mflops:.1f} MFLOPs")
        print(f"  Memoria pesos:        {m.weight_mb:.0f} MB")
        print(f"  Gradientes:           {m.gradient_mb:.0f} MB")
        print(f"  Estado AdamW:         {m.optimizer_mb:.0f} MB")
        print(f"  Total estático:       {m.total_static_mb/1024:.2f} GB  (sin activaciones)")

    def _hardware_section(self, h: HardwareInfo):
        print(f"\n{'─'*self.W}")
        print("  HARDWARE")
        print(f"{'─'*self.W}")
        if h.is_cuda:
            print(f"  GPU:                  {h.device_name}")
            print(f"  VRAM total:           {h.total_vram_gb:.2f} GB")
            print(f"  VRAM libre:           {h.free_vram_gb:.2f} GB")
        else:
            print("  Dispositivo:          CPU (CUDA no disponible)")

    def _memory_section(self, report: FeasibilityReport):
        m = report.model_info
        h = report.hardware_info
        if not m.activation_mb_per_image:
            return

        print(f"\n{'─'*self.W}")
        print("  MEMORIA POR BATCH SIZE")
        print(f"{'─'*self.W}")
        print(f"  Memoria estática (pesos + grad + AdamW): {m.total_static_mb/1024:.2f} GB")
        print(f"  Activaciones por imagen (forward+backward): {m.activation_mb_per_image:.1f} MB")
        print()
        print(f"  {'Batch':>5}  {'Estática':>10}  {'Activaciones':>13}  {'Total est.':>11}  {'Estado':>8}")
        print(f"  {'─'*5}  {'─'*10}  {'─'*13}  {'─'*11}  {'─'*8}")

        for bs in report.batch_sizes:
            total_gb = m.total_mb(bs) / 1024
            act_gb = m.activation_mb_per_image * bs / 1024
            static_gb = m.total_static_mb / 1024
            if h.is_cuda and total_gb > h.total_vram_gb:
                estado = "OOM ✗"
            elif h.is_cuda and total_gb > h.total_vram_gb * 0.85:
                estado = "⚠ Límite"
            else:
                estado = "✓ OK"
            print(
                f"  {bs:>5}  {static_gb:>8.2f} GB  {act_gb:>11.2f} GB  "
                f"{total_gb:>9.2f} GB  {estado:>8}"
            )

    def _benchmark_section(self, report: FeasibilityReport):
        print(f"\n{'─'*self.W}")
        print(f"  BENCHMARK  ({Benchmarker.N_MEASURE} batches sintéticos por configuración)")
        print(f"{'─'*self.W}")
        print(f"  {'Batch':>5}  {'Modo':<8}  {'s/batch':>8}  {'imgs/s':>7}  {'VRAM pico':>10}  {'Estado':>8}")
        print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*10}  {'─'*8}")

        baseline = {
            bs: next((r for r in report.results if r.batch_size == bs and r.trace_mode == "off"), None)
            for bs in report.batch_sizes
        }

        for r in report.results:
            if r.oom:
                estado = "OOM ✗"
                row = f"  {r.batch_size:>5}  {r.trace_mode:<8}  {'—':>8}  {'—':>7}  {'—':>10}  {estado:>8}"
            else:
                base = baseline.get(r.batch_size)
                overhead = ""
                if base and not base.oom and r.trace_mode != "off":
                    pct = (r.seconds_per_batch / base.seconds_per_batch - 1) * 100
                    overhead = f"+{max(0, round(pct)):.0f}%"
                estado = f"✓ OK {overhead}"
                row = (
                    f"  {r.batch_size:>5}  {r.trace_mode:<8}  "
                    f"{r.seconds_per_batch:>7.3f}s  {r.images_per_second:>6.1f}  "
                    f"{r.peak_vram_gb:>8.2f} GB  {estado}"
                )
            print(row)

    def _estimates_section(self, report: FeasibilityReport):
        estimator = TimeEstimator()
        print(f"\n{'─'*self.W}")
        print(f"  ESTIMACIONES  (train={report.dataset_train:,} imgs)")
        print(f"{'─'*self.W}")

        for epochs in report.epochs_list:
            print(f"\n  {epochs} epochs:")
            print(f"  {'Batch':>5}  {'Modo':<8}  {'Tiempo/epoch':>13}  {'Total':>10}")
            print(f"  {'─'*5}  {'─'*8}  {'─'*13}  {'─'*10}")
            for r in report.results:
                est = estimator.estimate(r, report.dataset_train, epochs)
                if est is None:
                    print(f"  {r.batch_size:>5}  {r.trace_mode:<8}  {'—':>13}  {'OOM':>10}")
                else:
                    spe, total = est
                    print(
                        f"  {r.batch_size:>5}  {r.trace_mode:<8}  "
                        f"{TimeEstimator.format_time(spe):>13}  "
                        f"{TimeEstimator.format_time(total):>10}"
                    )

    def _recommendations_section(self, report: FeasibilityReport):
        print(f"\n{'─'*self.W}")
        print("  RECOMENDACIONES")
        print(f"{'─'*self.W}")
        estimator = TimeEstimator()
        target_epochs = max(report.epochs_list)

        viable = [r for r in report.results if not r.oom and r.trace_mode == "off"]
        if not viable:
            print("  ✗ Ningún batch size viable en esta GPU")
            return

        best = max(viable, key=lambda r: r.images_per_second)
        print(f"  ✓ Batch size óptimo:  {best.batch_size} ({best.images_per_second:.1f} imgs/s)")

        for mode in report.trace_modes:
            r = next((x for x in report.results if x.batch_size == best.batch_size and x.trace_mode == mode), None)
            if r and not r.oom:
                est = estimator.estimate(r, report.dataset_train, target_epochs)
                if est:
                    print(f"  → --trace {mode:<8} para {target_epochs} epochs:  ~{TimeEstimator.format_time(est[1])}")

        off_result = next((r for r in report.results if r.batch_size == best.batch_size and r.trace_mode == "off"), None)
        deep_result = next((r for r in report.results if r.batch_size == best.batch_size and r.trace_mode == "deep"), None)
        if off_result and deep_result and not off_result.oom and not deep_result.oom:
            overhead = (deep_result.seconds_per_batch / off_result.seconds_per_batch - 1) * 100
            print(f"  ⚠ --trace deep añade un {overhead:.0f}% de overhead — úsalo solo para análisis puntual")

        print()

    def write_csv(self, report: FeasibilityReport, env: str = "local"):
        """Write benchmark results to a structured CSV in logs/{env}/."""
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        out_dir = Path(f"logs/{env}")
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"feasibility_{timestamp}.csv"

        estimator = TimeEstimator()
        target_epochs = max(report.epochs_list) if report.epochs_list else 0
        mi = report.model_info
        hi = report.hardware_info

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Metadata row (prefixed with '#' so parsers can identify it)
            writer.writerow([
                "#meta", "model_name", "total_params_M", "flops_mflops",
                "hardware_name", "total_vram_gb", "free_vram_gb",
            ])
            writer.writerow([
                "#meta",
                mi.name,
                round(mi.total_params / 1e6, 2),
                round(mi.flops_per_image_mflops, 1),
                hi.device_name,
                round(hi.total_vram_gb, 2),
                round(hi.free_vram_gb, 2),
            ])
            # Benchmark rows
            writer.writerow([
                "batch_size", "trace_mode", "s_per_batch", "imgs_per_s",
                "peak_vram_gb", "oom",
                f"est_min_per_epoch_{target_epochs}ep",
                f"est_total_h_{target_epochs}ep",
            ])
            for r in report.results:
                est = estimator.estimate(r, report.dataset_train, target_epochs) if not r.oom else None
                writer.writerow([
                    r.batch_size,
                    r.trace_mode,
                    round(r.seconds_per_batch, 4) if not r.oom else "",
                    round(r.images_per_second, 1) if not r.oom else "",
                    round(r.peak_vram_gb, 2) if not r.oom else "",
                    "yes" if r.oom else "no",
                    round(est[0] / 60, 1) if est else "",
                    round(est[1] / 3600, 2) if est else "",
                ])

        print(f"  → CSV guardado: {csv_path}")


# ──────────────────────────────────────────────────────────────
# FeasibilityChecker — Facade: orchestrates all components
# ──────────────────────────────────────────────────────────────

class FeasibilityChecker:
    """Facade that coordinates analysis, benchmarking and reporting."""

    def __init__(
        self,
        model_name: str,
        batch_sizes: list[int],
        epochs_list: list[int],
        trace_modes: list[str],
        dataset_train: int,
        dataset_val: int,
    ):
        self._model_name = model_name
        self._batch_sizes = batch_sizes
        self._epochs_list = epochs_list
        self._trace_modes = trace_modes
        self._dataset_train = dataset_train
        self._dataset_val = dataset_val
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> FeasibilityReport:
        print(f"Cargando modelo {self._model_name}...")
        model = build_model(model_name=self._model_name, pretrained=False)

        model_info = ModelAnalyzer(model, self._model_name, self._device).analyze()
        hardware_info = HardwareProbe().probe()
        benchmarker = Benchmarker(model, self._device)

        report = FeasibilityReport(
            model_info=model_info,
            hardware_info=hardware_info,
            dataset_train=self._dataset_train,
            dataset_val=self._dataset_val,
            batch_sizes=self._batch_sizes,
            epochs_list=self._epochs_list,
            trace_modes=self._trace_modes,
        )

        total = len(self._batch_sizes) * len(self._trace_modes)
        done = 0
        for batch_size in self._batch_sizes:
            for mode in self._trace_modes:
                done += 1
                print(f"Benchmarking {done}/{total}: batch_size={batch_size}, trace={mode}...")
                result = benchmarker.run(batch_size, mode)
                report.results.append(result)

        return report


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-training feasibility checker for BigEarthNet ViT"
    )
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name from config (any timm ID)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, nargs="+", default=None)
    parser.add_argument(
        "--trace-modes",
        nargs="+",
        choices=["off", "simple", "deep"],
        default=["off", "simple", "deep"],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    batch_sizes = args.batch_sizes or [cfg["training"]["batch_size"]]
    epochs_list = args.epochs or [cfg["training"]["epochs"]]
    model_name = args.model or cfg["model"]["name"]
    env = cfg.get("output", {}).get("env", "local")

    checker = FeasibilityChecker(
        model_name=model_name,
        batch_sizes=batch_sizes,
        epochs_list=epochs_list,
        trace_modes=args.trace_modes,
        dataset_train=237871,
        dataset_val=122342,
    )

    report = checker.run()
    formatter = ReportFormatter()
    formatter.print(report)
    formatter.write_csv(report, env=env)


if __name__ == "__main__":
    main()
