import logging
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.training.metrics import eta_str, f1_score, accuracy
from src.training.oop_decorators.base import TrainerDecorator


@dataclass
class LayerStats:
    """Activation and gradient statistics for one layer."""
    act_mean: float = 0.0
    act_std: float = 0.0
    act_max: float = 0.0
    dead_ratio: float = 0.0
    grad_norm: float = 0.0
    grad_max: float = 0.0
    vanishing: bool = False
    exploding: bool = False


@dataclass
class ParamStats:
    """Gradient and weight statistics for one parameter tensor."""
    weight_norm: float = 0.0
    grad_norm: float = 0.0
    grad_max: float = 0.0
    update_ratio: float = 0.0
    vanishing: bool = False
    exploding: bool = False


class DeepTracingDecorator(TrainerDecorator):
    """Maximum-depth tracing: forward hooks, backward hooks, param hooks, GPU memory.

    Registers hooks on every module and parameter to capture per-layer
    statistics in real time during training.
    """

    VANISHING_THRESHOLD = 1e-7
    EXPLODING_THRESHOLD = 10.0
    DEAD_NEURON_THRESHOLD = 1e-6
    HEALTHY_UPDATE_RATIO_MIN = 1e-4
    HEALTHY_UPDATE_RATIO_MAX = 1.0

    def __init__(self, trainer: BaseTrainer, logger: logging.Logger, log_every: int = 100):
        super().__init__(trainer)
        self._logger = logger
        self.log_every = log_every
        self._layer_stats: dict[str, LayerStats] = {}
        self._param_stats: dict[str, ParamStats] = {}
        self._forward_hooks: list = []
        self._param_hooks: list = []

    def _register_forward_hooks(self):
        for name, module in self._trainer.model.named_modules():
            if not list(module.children()):
                h = module.register_forward_hook(self._make_forward_hook(name))
                self._forward_hooks.append(h)

    def _make_forward_hook(self, name: str):
        def hook(_module, _input, output):
            if not isinstance(output, torch.Tensor):
                return
            with torch.no_grad():
                flat = output.detach().float()
                s = self._layer_stats.setdefault(name, LayerStats())
                s.act_mean = flat.abs().mean().item()
                s.act_std = flat.std().item()
                s.act_max = flat.abs().max().item()
                s.dead_ratio = (flat.abs() < self.DEAD_NEURON_THRESHOLD).float().mean().item()
        return hook

    def _register_backward_hooks(self):
        for name, module in self._trainer.model.named_modules():
            if not list(module.children()):
                h = module.register_full_backward_hook(self._make_backward_hook(name))
                self._forward_hooks.append(h)

    def _make_backward_hook(self, name: str):
        def hook(_module, _grad_input, grad_output):
            if grad_output[0] is None:
                return
            with torch.no_grad():
                g = grad_output[0].detach().float()
                norm = g.norm().item()
                s = self._layer_stats.setdefault(name, LayerStats())
                s.grad_norm = norm
                s.grad_max = g.abs().max().item()
                s.vanishing = norm < self.VANISHING_THRESHOLD
                s.exploding = norm > self.EXPLODING_THRESHOLD
        return hook

    def _register_param_hooks(self):
        for name, param in self._trainer.model.named_parameters():
            if not param.requires_grad:
                continue

            def hook(grad, n=name, p=param):
                with torch.no_grad():
                    g = grad.detach().float()
                    g_norm = g.norm().item()
                    w_norm = p.data.detach().float().norm().item()
                    ps = self._param_stats.setdefault(n, ParamStats())
                    ps.weight_norm = w_norm
                    ps.grad_norm = g_norm
                    ps.grad_max = g.abs().max().item()
                    ps.update_ratio = g_norm / (w_norm + 1e-8)
                    ps.vanishing = g_norm < self.VANISHING_THRESHOLD
                    ps.exploding = g_norm > self.EXPLODING_THRESHOLD

            self._param_hooks.append(param.register_hook(hook))

    def _remove_all_hooks(self):
        for h in self._forward_hooks:
            h.remove()
        for h in self._param_hooks:
            h.remove()
        self._forward_hooks.clear()
        self._param_hooks.clear()

    def _select_representative_layers(self) -> list[tuple[str, LayerStats]]:
        selected: dict[str, LayerStats] = {}
        for name in sorted(self._layer_stats):
            if "patch_embed" in name and "proj" in name:
                selected[name] = self._layer_stats[name]
                break
        for i in range(12):
            key = f"backbone.blocks.{i}.attn.proj"
            if key in self._layer_stats:
                selected[key] = self._layer_stats[key]
        for name in sorted(self._layer_stats):
            if "head" in name and isinstance(
                dict(self._trainer.model.named_modules()).get(name), nn.Linear
            ):
                selected[name] = self._layer_stats[name]
                break
        return list(selected.items())

    def _gpu_memory_str(self) -> str:
        if not torch.cuda.is_available():
            return "cpu"
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"{alloc:.1f}GB/{reserved:.1f}GB"

    def _lr_str(self) -> str:
        lrs = [f"{g['lr']:.2e}" for g in self._trainer.optimizer.param_groups]
        return " | ".join(lrs)

    def _log_layer_table(self, epoch: int, batch_idx: int, n_batches: int):
        layers = self._select_representative_layers()
        if not layers:
            return
        self._logger.debug(
            f"[E{epoch:03d}/B{batch_idx:05d}/{n_batches}] "
            f"GPU={self._gpu_memory_str()}  LR={self._lr_str()}"
        )
        self._logger.debug(
            f"  {'Layer':<45} {'act_μ':>7} {'act_σ':>7} {'dead%':>6} "
            f"{'grad_n':>8} {'grad_max':>9} {'status':>10}"
        )
        self._logger.debug("  " + "─" * 100)
        for name, s in layers:
            status = "⚠ DEAD" if s.dead_ratio > 0.5 else (
                "⚠ EXPLODE" if s.exploding else (
                    "⚠ VANISH" if s.vanishing and s.grad_norm > 0 else "✓ OK"
                )
            )
            short = name[-43:] if len(name) > 43 else name
            self._logger.debug(
                f"  {short:<45} "
                f"{s.act_mean:>7.4f} {s.act_std:>7.4f} {s.dead_ratio*100:>5.1f}% "
                f"{s.grad_norm:>8.4f} {s.grad_max:>9.4f} {status:>10}"
            )

    def _log_param_anomalies(self, epoch: int):
        dead = [n for n, s in self._layer_stats.items() if s.dead_ratio > 0.5]
        exploding = [n for n, s in self._param_stats.items() if s.exploding]
        vanishing = [n for n, s in self._param_stats.items() if s.vanishing and s.grad_norm > 0]
        bad_ratio = [
            n for n, s in self._param_stats.items()
            if s.update_ratio > 0 and (
                s.update_ratio < self.HEALTHY_UPDATE_RATIO_MIN or
                s.update_ratio > self.HEALTHY_UPDATE_RATIO_MAX
            )
        ]
        if dead:
            self._logger.warning(f"[E{epoch:03d}] Neuronas muertas en {len(dead)} capas: {dead[:3]}")
        if exploding:
            self._logger.warning(f"[E{epoch:03d}] Gradiente explosivo en: {exploding[:3]}")
        if vanishing:
            self._logger.warning(f"[E{epoch:03d}] Gradiente evanescente en: {vanishing[:3]}")
        if bad_ratio:
            self._logger.warning(f"[E{epoch:03d}] Update ratio anómalo en {len(bad_ratio)} params")
        if not any([dead, exploding, vanishing, bad_ratio]):
            self._logger.info(f"[E{epoch:03d}] Flujo de gradientes: sin anomalías ✓")

    def _show_model_summary(self):
        try:
            from torchinfo import summary
            self._logger.info("=== Arquitectura del modelo (torchinfo) ===")
            stats = summary(self._trainer.model, input_size=(1, 3, 224, 224), verbose=0,
                            device=self._trainer.device)
            for line in str(stats).splitlines():
                self._logger.info(line)
        except ImportError:
            self._logger.warning("torchinfo no disponible")

    def train_epoch(self, loader: DataLoader) -> dict:
        model = self._trainer.model
        optimizer = self._trainer.optimizer
        criterion = self._trainer.criterion
        device = self._trainer.device
        scheduler = self._trainer.scheduler

        model.train()
        total_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        start = time.time()

        for batch_idx, (images, labels) in enumerate(loader, 1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                preds = torch.sigmoid(logits) > 0.5
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

            if batch_idx % self.log_every == 0:
                self._log_layer_table(self._current_epoch, batch_idx, len(loader))

        if scheduler:
            scheduler.step()

        all_preds_t = torch.cat(all_preds)
        all_labels_t = torch.cat(all_labels)
        return {
            "loss": total_loss / len(loader),
            "f1": f1_score(all_preds_t, all_labels_t),
            "accuracy": accuracy(all_preds_t, all_labels_t),
            "time": time.time() - start,
        }

    def eval_epoch(self, loader: DataLoader) -> dict:
        return self._trainer.eval_epoch(loader)

    def save_checkpoint(self, epoch: int, metrics: dict):
        self._trainer.save_checkpoint(epoch, metrics)
        self._logger.info(f"[E{epoch:03d}] Checkpoint guardado")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        self._show_model_summary()
        self._register_forward_hooks()
        self._register_backward_hooks()
        self._register_param_hooks()

        self._logger.info(
            f"Iniciando entrenamiento profundo — {epochs} epochs | "
            f"Módulos: {len(list(self._trainer.model.named_modules()))} | "
            f"Params: {sum(1 for p in self._trainer.model.parameters() if p.requires_grad)}"
        )

        best_f1 = 0.0
        epoch_times: list[float] = []

        try:
            for epoch in range(1, epochs + 1):
                t0 = time.time()
                self._current_epoch = epoch
                self._layer_stats.clear()
                self._param_stats.clear()

                self._logger.info(
                    f"[E{epoch:03d}/{epochs}] ── Entrenamiento  "
                    f"GPU={self._gpu_memory_str()}  LR={self._lr_str()}"
                )
                train_m = self.train_epoch(train_loader)
                self._log_param_anomalies(epoch)

                self._logger.info(f"[E{epoch:03d}/{epochs}] ── Evaluación")
                val_m = self.eval_epoch(val_loader)
                epoch_times.append(time.time() - t0)

                if val_m["f1"] > best_f1:
                    best_f1 = val_m["f1"]
                    self.save_checkpoint(epoch, val_m)

                self._logger.info(
                    f"[E{epoch:03d}/{epochs}] ══ RESUMEN  "
                    f"train_loss={train_m['loss']:.4f}  train_f1={train_m['f1']:.4f}  train_acc={train_m['accuracy']:.4f} | "
                    f"val_loss={val_m['loss']:.4f}  val_f1={val_m['f1']:.4f}  best={best_f1:.4f} | "
                    f"val_prec={val_m['precision']:.4f}  val_rec={val_m['recall']:.4f} | "
                    f"time={train_m['time']:.0f}s  ETA={eta_str(epoch_times, epoch, epochs)}  "
                    f"GPU={self._gpu_memory_str()}"
                )
        finally:
            self._remove_all_hooks()

        self._logger.info(f"Entrenamiento completado — mejor Val F1: {best_f1:.4f}")
