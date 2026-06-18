"""Benchmarker — measures real train/eval throughput on synthetic batches."""
from __future__ import annotations

import time

import torch
import torch.nn as nn

from src.feasibility.value_objects import BenchmarkResult


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
