"""Pre-training feasibility checker for BigEarthNet ViT.

Measures theoretical complexity and real throughput for each
(batch_size, trace_mode) combination and produces time estimates
for the full training run without touching the real dataset.

Usage:
    uv run python scripts/check_feasibility.py
    uv run python scripts/check_feasibility.py --batch-sizes 16 32 64
    uv run python scripts/check_feasibility.py --batch-sizes 32 64 --epochs 10 30
    uv run python scripts/check_feasibility.py --batch-sizes 32 64 --trace-modes off deep
    uv run python scripts/check_feasibility.py --nfs-factor 1.3   # para Verode con NFS
    uv run python scripts/check_feasibility.py --output logs/feasibility.log
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


# ──────────────────────────────────────────────────────────────
# ModelAnalyzer
# ──────────────────────────────────────────────────────────────

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
# HardwareProbe
# ──────────────────────────────────────────────────────────────

class HardwareProbe:
    def probe(self) -> HardwareInfo:
        if not torch.cuda.is_available():
            return HardwareInfo(device_name="CPU", total_vram_gb=0.0, free_vram_gb=0.0, is_cuda=False)

        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1e9
        reserved_gb = torch.cuda.memory_reserved(0) / 1e9

        return HardwareInfo(
            device_name=props.name,
            total_vram_gb=total_gb,
            free_vram_gb=total_gb - reserved_gb,
            is_cuda=True,
        )


# ──────────────────────────────────────────────────────────────
# Benchmarker — measures real train AND eval throughput
# ──────────────────────────────────────────────────────────────

class Benchmarker:
    """Runs timing and memory benchmarks using synthetic data.

    Benchmarks both train (forward + backward + step) and eval
    (forward only, no_grad) to give accurate total-epoch estimates.
    """

    N_WARMUP = 3
    N_MEASURE = 8

    def __init__(self, model: nn.Module, device: torch.device):
        self._model = model.to(device)
        self._device = device
        self._criterion = nn.BCEWithLogitsLoss()
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-4)

    def run(self, batch_size: int, trace_mode: str) -> BenchmarkResult:
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        batches = self._make_batches(batch_size)
        try:
            sec_train, sec_eval, peak_vram = self._benchmark(batches, trace_mode)
            return BenchmarkResult(
                batch_size=batch_size,
                trace_mode=trace_mode,
                seconds_per_batch_train=sec_train,
                seconds_per_batch_eval=sec_eval,
                images_per_second_train=batch_size / sec_train if sec_train > 0 else 0.0,
                images_per_second_eval=batch_size / sec_eval if sec_eval > 0 else 0.0,
                peak_vram_gb=peak_vram,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return BenchmarkResult(
                batch_size=batch_size,
                trace_mode=trace_mode,
                seconds_per_batch_train=0.0,
                seconds_per_batch_eval=0.0,
                images_per_second_train=0.0,
                images_per_second_eval=0.0,
                peak_vram_gb=0.0,
                oom=True,
            )

    def _make_batches(self, batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        n = self.N_WARMUP + self.N_MEASURE
        return [
            (torch.randn(batch_size, 3, 224, 224), torch.randint(0, 2, (batch_size, 19)).float())
            for _ in range(n)
        ]

    def _benchmark(self, batches: list, trace_mode: str) -> tuple[float, float, float]:
        hooks: list = []
        if trace_mode == "deep":
            hooks = self._register_deep_hooks()

        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Warmup (train)
        self._model.train()
        for images, labels in batches[:self.N_WARMUP]:
            self._train_step(images, labels)

        # Measure train
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for images, labels in batches[self.N_WARMUP:]:
            self._train_step(images, labels)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        sec_train = (time.perf_counter() - t0) / self.N_MEASURE

        # Measure eval (forward only)
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

        return sec_train, sec_eval, peak_vram

    def _train_step(self, images: torch.Tensor, labels: torch.Tensor):
        images = images.to(self._device)
        labels = labels.to(self._device)
        self._optimizer.zero_grad()
        loss = self._criterion(self._model(images), labels)
        loss.backward()
        self._optimizer.step()

    def _eval_step(self, images: torch.Tensor, labels: torch.Tensor):
        with torch.no_grad():
            images = images.to(self._device)
            _ = self._model(images)

    def _register_deep_hooks(self) -> list:
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
# TimeEstimator
# ──────────────────────────────────────────────────────────────

class TimeEstimator:
    """Converts per-batch timings into full training time estimates.

    Takes into account both train and eval phases, plus an optional
    NFS overhead factor for networked filesystems (e.g. Verode cluster).
    """

    def estimate(
        self,
        result: BenchmarkResult,
        dataset_train: int,
        dataset_val: int,
        epochs: int,
        nfs_factor: float = 1.0,
    ) -> Optional[dict]:
        """Returns dict with train/eval/total per epoch and full total, or None if OOM."""
        if result.oom or result.images_per_second_train == 0:
            return None

        train_batches = dataset_train / result.batch_size
        eval_batches = dataset_val / result.batch_size

        sec_train = train_batches * result.seconds_per_batch_train * nfs_factor
        sec_eval = eval_batches * result.seconds_per_batch_eval * nfs_factor
        sec_epoch = sec_train + sec_eval
        sec_total = sec_epoch * epochs

        return {
            "train_per_epoch": sec_train,
            "eval_per_epoch": sec_eval,
            "total_per_epoch": sec_epoch,
            "total": sec_total,
        }

    @staticmethod
    def format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m:02d}m"


# ──────────────────────────────────────────────────────────────
# ReportFormatter
# ──────────────────────────────────────────────────────────────

class ReportFormatter:
    """Formats a FeasibilityReport as a human-readable output."""

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
        self._hardware_section(report.hardware_info)
        self._memory_section(report)
        self._benchmark_section(report)
        self._estimates_section(report)
        self._recommendations_section(report)
        self.flush()

    def _header(self, report: FeasibilityReport):
        self._emit("═" * self.W)
        self._emit("  ANÁLISIS DE VIABILIDAD — BigEarthNet ViT")
        self._emit(f"  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        if report.nfs_factor != 1.0:
            self._emit(f"  Factor NFS aplicado: ×{report.nfs_factor:.2f}  (I/O en red más lento)")
        self._emit("═" * self.W)

    def _model_section(self, m: ModelInfo):
        self._emit(f"\n{'─'*self.W}")
        self._emit("  MODELO")
        self._emit(f"{'─'*self.W}")
        self._emit(f"  Nombre:               {m.name}")
        self._emit(f"  Parámetros totales:   {m.total_params:,} ({m.total_params/1e6:.1f}M)")
        self._emit(f"  Parámetros train.:    {m.trainable_params:,} ({m.trainable_params/1e6:.1f}M)")
        if m.flops_per_image_mflops:
            self._emit(f"  FLOPs/imagen:         {m.flops_per_image_mflops:.1f} MFLOPs")
        self._emit(f"  Memoria pesos:        {m.weight_mb:.0f} MB")
        self._emit(f"  Gradientes:           {m.gradient_mb:.0f} MB")
        self._emit(f"  Estado AdamW:         {m.optimizer_mb:.0f} MB")
        self._emit(f"  Total estático:       {m.total_static_mb/1024:.2f} GB  (sin activaciones)")

    def _hardware_section(self, h: HardwareInfo):
        self._emit(f"\n{'─'*self.W}")
        self._emit("  HARDWARE")
        self._emit(f"{'─'*self.W}")
        if h.is_cuda:
            self._emit(f"  GPU:                  {h.device_name}")
            self._emit(f"  VRAM total:           {h.total_vram_gb:.2f} GB")
            self._emit(f"  VRAM libre:           {h.free_vram_gb:.2f} GB")
        else:
            self._emit("  Dispositivo:          CPU (CUDA no disponible)")

    def _memory_section(self, report: FeasibilityReport):
        m = report.model_info
        h = report.hardware_info
        if not m.activation_mb_per_image:
            return

        self._emit(f"\n{'─'*self.W}")
        self._emit("  MEMORIA POR BATCH SIZE")
        self._emit(f"{'─'*self.W}")
        self._emit(f"  Estática (pesos+grad+AdamW): {m.total_static_mb/1024:.2f} GB")
        self._emit(f"  Activaciones por imagen:     {m.activation_mb_per_image:.1f} MB")
        self._emit()
        self._emit(f"  {'Batch':>5}  {'Estática':>10}  {'Activac.':>9}  {'Total est.':>11}  {'Estado':>8}")
        self._emit(f"  {'─'*5}  {'─'*10}  {'─'*9}  {'─'*11}  {'─'*8}")

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
            self._emit(
                f"  {bs:>5}  {static_gb:>8.2f} GB  {act_gb:>7.2f} GB  "
                f"{total_gb:>9.2f} GB  {estado:>8}"
            )

    def _benchmark_section(self, report: FeasibilityReport):
        self._emit(f"\n{'─'*self.W}")
        self._emit(f"  BENCHMARK  ({Benchmarker.N_MEASURE} batches sintéticos — sin I/O real)")
        self._emit(f"{'─'*self.W}")
        self._emit(
            f"  {'Batch':>5}  {'Modo':<8}  {'s/batch(train)':>14}  "
            f"{'imgs/s(train)':>13}  {'s/batch(eval)':>13}  {'VRAM':>7}"
        )
        self._emit(f"  {'─'*5}  {'─'*8}  {'─'*14}  {'─'*13}  {'─'*13}  {'─'*7}")

        baseline = {
            bs: next((r for r in report.results if r.batch_size == bs and r.trace_mode == "off"), None)
            for bs in report.batch_sizes
        }

        for r in report.results:
            if r.oom:
                self._emit(f"  {r.batch_size:>5}  {r.trace_mode:<8}  {'OOM':>14}  {'—':>13}  {'—':>13}  {'—':>7}")
            else:
                base = baseline.get(r.batch_size)
                overhead = ""
                if base and not base.oom and r.trace_mode != "off":
                    pct = (r.seconds_per_batch_train / base.seconds_per_batch_train - 1) * 100
                    overhead = f"  (+{max(0,round(pct)):.0f}% vs off)"
                self._emit(
                    f"  {r.batch_size:>5}  {r.trace_mode:<8}  "
                    f"{r.seconds_per_batch_train:>12.3f}s  "
                    f"{r.images_per_second_train:>11.1f}  "
                    f"{r.seconds_per_batch_eval:>11.3f}s  "
                    f"{r.peak_vram_gb:>5.2f} GB"
                    f"{overhead}"
                )

    def _estimates_section(self, report: FeasibilityReport):
        estimator = TimeEstimator()
        nfs = report.nfs_factor
        self._emit(f"\n{'─'*self.W}")
        label = f"train={report.dataset_train:,} | val={report.dataset_val:,}"
        if nfs != 1.0:
            label += f" | NFS ×{nfs:.2f}"
        self._emit(f"  ESTIMACIONES  ({label})")
        self._emit(f"{'─'*self.W}")

        for epochs in report.epochs_list:
            self._emit(f"\n  {epochs} epochs:")
            self._emit(
                f"  {'Batch':>5}  {'Modo':<8}  {'Train/epoch':>11}  "
                f"{'Eval/epoch':>10}  {'Total/epoch':>11}  {'TOTAL':>10}"
            )
            self._emit(f"  {'─'*5}  {'─'*8}  {'─'*11}  {'─'*10}  {'─'*11}  {'─'*10}")
            for r in report.results:
                est = estimator.estimate(r, report.dataset_train, report.dataset_val, epochs, nfs)
                if est is None:
                    self._emit(f"  {r.batch_size:>5}  {r.trace_mode:<8}  {'OOM':>11}  {'—':>10}  {'—':>11}  {'—':>10}")
                else:
                    self._emit(
                        f"  {r.batch_size:>5}  {r.trace_mode:<8}  "
                        f"{TimeEstimator.format_time(est['train_per_epoch']):>11}  "
                        f"{TimeEstimator.format_time(est['eval_per_epoch']):>10}  "
                        f"{TimeEstimator.format_time(est['total_per_epoch']):>11}  "
                        f"{TimeEstimator.format_time(est['total']):>10}"
                    )

    def _recommendations_section(self, report: FeasibilityReport):
        self._emit(f"\n{'─'*self.W}")
        self._emit("  RECOMENDACIONES")
        self._emit(f"{'─'*self.W}")
        estimator = TimeEstimator()
        nfs = report.nfs_factor
        target_epochs = max(report.epochs_list)

        viable = [r for r in report.results if not r.oom and r.trace_mode == "off"]
        if not viable:
            self._emit("  ✗ Ningún batch size viable en esta GPU")
            return

        best = max(viable, key=lambda r: r.images_per_second_train)
        self._emit(f"  ✓ Batch size óptimo:  {best.batch_size} ({best.images_per_second_train:.1f} imgs/s en train)")

        for mode in report.trace_modes:
            r = next((x for x in report.results if x.batch_size == best.batch_size and x.trace_mode == mode), None)
            if r and not r.oom:
                est = estimator.estimate(r, report.dataset_train, report.dataset_val, target_epochs, nfs)
                if est:
                    self._emit(
                        f"  → --trace {mode:<8} para {target_epochs} epochs: "
                        f" ~{TimeEstimator.format_time(est['total'])}"
                        f"  (train {TimeEstimator.format_time(est['train_per_epoch'])}/epoch"
                        f" + eval {TimeEstimator.format_time(est['eval_per_epoch'])}/epoch)"
                    )

        off_r = next((r for r in report.results if r.batch_size == best.batch_size and r.trace_mode == "off"), None)
        deep_r = next((r for r in report.results if r.batch_size == best.batch_size and r.trace_mode == "deep"), None)
        if off_r and deep_r and not off_r.oom and not deep_r.oom:
            pct = (deep_r.seconds_per_batch_train / off_r.seconds_per_batch_train - 1) * 100
            self._emit(f"  ⚠ --trace deep añade un {pct:.0f}% de overhead — úsalo solo para análisis puntual")

        if nfs == 1.0:
            self._emit()
            self._emit("  💡 En Verode (NFS), usa --nfs-factor 1.3 para estimaciones más precisas")

        self._emit()

    def write_csv(self, report: FeasibilityReport, env: str = "local"):
        """Write benchmark results to a structured CSV in logs/{env}/feasibility/."""
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        out_dir = Path(f"logs/{env}/feasibility")
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
            # Also store model memory breakdown for web display
            writer.writerow([
                "#model_mem", "weight_mb", "gradient_mb", "optimizer_mb",
                "activation_mb_per_image", "total_static_mb",
            ])
            writer.writerow([
                "#model_mem",
                round(mi.weight_mb, 1),
                round(mi.gradient_mb, 1),
                round(mi.optimizer_mb, 1),
                round(mi.activation_mb_per_image, 1),
                round(mi.total_static_mb, 1),
            ])
            # Benchmark rows — train and eval columns separated
            writer.writerow([
                "batch_size", "trace_mode",
                "s_per_batch_train", "imgs_per_s_train",
                "s_per_batch_eval", "imgs_per_s_eval",
                "peak_vram_gb", "oom",
                "est_train_min_per_epoch", "est_eval_min_per_epoch",
                "est_total_min_per_epoch",
                f"est_total_h_{target_epochs}ep",
            ])
            for r in report.results:
                est = estimator.estimate(r, report.dataset_train, report.dataset_val, target_epochs) if not r.oom else None
                writer.writerow([
                    r.batch_size,
                    r.trace_mode,
                    round(r.seconds_per_batch_train, 4) if not r.oom else "",
                    round(r.images_per_second_train, 1) if not r.oom else "",
                    round(r.seconds_per_batch_eval, 4) if not r.oom else "",
                    round(r.images_per_second_eval, 1) if not r.oom else "",
                    round(r.peak_vram_gb, 2) if not r.oom else "",
                    "yes" if r.oom else "no",
                    round(est["train_per_epoch"] / 60, 1) if est else "",
                    round(est["eval_per_epoch"] / 60, 1) if est else "",
                    round(est["total_per_epoch"] / 60, 1) if est else "",
                    round(est["total"] / 3600, 2) if est else "",
                ])

        print(f"  → CSV guardado: {csv_path}")


# ──────────────────────────────────────────────────────────────
# FeasibilityChecker — Facade
# ──────────────────────────────────────────────────────────────

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
    ):
        self._model_name = model_name
        self._batch_sizes = batch_sizes
        self._epochs_list = epochs_list
        self._trace_modes = trace_modes
        self._dataset_train = dataset_train
        self._dataset_val = dataset_val
        self._nfs_factor = nfs_factor
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
            nfs_factor=self._nfs_factor,
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
    parser.add_argument(
        "--nfs-factor",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Overhead multiplier for NFS storage (e.g. 1.3 for Verode). Default: 1.0 (local SSD)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save report to FILE (default: auto-generate in logs/)",
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

    # Auto-generate output path if not specified
    output_path = args.output
    if output_path is None:
        ts = datetime.now().strftime("%d%m%Y_%H%M%S")
        env = cfg.get("output", {}).get("env", "local")
        output_path = Path(f"logs/{env}/feasibility/feasibility_{ts}.log")

    checker = FeasibilityChecker(
        model_name=model_name,
        batch_sizes=batch_sizes,
        epochs_list=epochs_list,
        trace_modes=args.trace_modes,
        dataset_train=237871,
        dataset_val=122342,
        nfs_factor=args.nfs_factor,
    )

    report = checker.run()
    formatter = ReportFormatter(output_path=output_path)
    formatter.print(report)
    formatter.write_csv(report, env=env)


if __name__ == "__main__":
    main()
