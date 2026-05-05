"""DeepTracingDecorator — per-layer, per-neuron real-time tracing.

Extends TracingDecorator: inherits the fit loop (EpochController) and the
logging infrastructure (_emit). Adds hook registration around the loop and
batch-level table output inside train_epoch.
"""

import logging
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.training.metrics import f1_score, accuracy, eta_str
from src.training.decorators.tracing import TracingDecorator


@dataclass
class LayerStats:
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
    weight_norm: float = 0.0
    grad_norm: float = 0.0
    grad_max: float = 0.0
    update_ratio: float = 0.0
    vanishing: bool = False
    exploding: bool = False


class DeepTracingDecorator(TracingDecorator):
    """Maximum-depth tracing: forward hooks, backward hooks, param hooks, GPU memory.

    Inherits from TracingDecorator so the fit loop and logging are reused.
    Adds:
      - Model summary via torchinfo at startup
      - Forward hooks on every leaf module → activation stats
      - Backward hooks on every leaf module → gradient stats
      - Param hooks on every trainable parameter → weight/update ratio
      - Batch-level layer table logged every log_every batches
      - Anomaly detection: dead neurons, vanishing/exploding gradients
    """

    VANISHING_THRESHOLD = 1e-7
    EXPLODING_THRESHOLD = 10.0
    DEAD_NEURON_THRESHOLD = 1e-6
    HEALTHY_RATIO_MIN = 1e-4
    HEALTHY_RATIO_MAX = 1.0

    def __init__(self, trainer: BaseTrainer, logger: logging.Logger, log_every: int = 100):
        super().__init__(trainer, logger=logger)
        self.log_every = log_every
        self._layer_stats: dict[str, LayerStats] = {}
        self._param_stats: dict[str, ParamStats] = {}
        self._forward_hooks: list = []
        self._param_hooks: list = []
        self._current_epoch: int = 0

    # ── Hook registration ────────────────────────────────────────────────────

    def _register_all_hooks(self):
        model = self._trainer.model
        for name, module in model.named_modules():
            if not list(module.children()):
                self._forward_hooks.append(
                    module.register_forward_hook(self._fwd_hook(name))
                )
                self._forward_hooks.append(
                    module.register_full_backward_hook(self._bwd_hook(name))
                )
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._param_hooks.append(param.register_hook(self._param_hook(name, param)))

    def _remove_all_hooks(self):
        for h in self._forward_hooks:
            h.remove()
        for h in self._param_hooks:
            h.remove()
        self._forward_hooks.clear()
        self._param_hooks.clear()

    def _fwd_hook(self, name: str):
        def hook(_m, _i, output):
            if not isinstance(output, torch.Tensor):
                return
            with torch.no_grad():
                f = output.detach().float()
                s = self._layer_stats.setdefault(name, LayerStats())
                s.act_mean = f.abs().mean().item()
                s.act_std = f.std().item()
                s.act_max = f.abs().max().item()
                s.dead_ratio = (f.abs() < self.DEAD_NEURON_THRESHOLD).float().mean().item()
        return hook

    def _bwd_hook(self, name: str):
        def hook(_m, _gi, grad_output):
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

    def _param_hook(self, name: str, param: torch.nn.Parameter):
        def hook(grad):
            with torch.no_grad():
                g = grad.detach().float()
                g_norm = g.norm().item()
                w_norm = param.data.detach().float().norm().item()
                ps = self._param_stats.setdefault(name, ParamStats())
                ps.weight_norm = w_norm
                ps.grad_norm = g_norm
                ps.grad_max = g.abs().max().item()
                ps.update_ratio = g_norm / (w_norm + 1e-8)
                ps.vanishing = g_norm < self.VANISHING_THRESHOLD
                ps.exploding = g_norm > self.EXPLODING_THRESHOLD
        return hook

    # ── EpochController hooks ────────────────────────────────────────────────

    def _on_fit_start(self, epochs: int):
        self._show_model_summary()
        self._register_all_hooks()
        n_mod = len(list(self._trainer.model.named_modules()))
        n_par = sum(1 for p in self._trainer.model.parameters() if p.requires_grad)
        self._emit(
            f"Iniciando entrenamiento profundo — {epochs} epochs | "
            f"módulos: {n_mod} | params: {n_par}"
        )

    def _on_epoch_start(self, epoch: int, epochs: int):
        self._current_epoch = epoch
        self._layer_stats.clear()
        self._param_stats.clear()
        self._emit(
            f"[E{epoch:03d}/{epochs}] ── Entrenamiento  "
            f"GPU={self._gpu_str()}  LR={self._lr_str()}"
        )

    def _on_epoch_end(self, epoch, epochs, train_m, val_m, best_f1, epoch_times):
        self._log_anomalies(epoch)
        self._emit(
            f"[E{epoch:03d}/{epochs}] ══ RESUMEN  "
            f"train_loss={train_m['loss']:.4f}  train_f1={train_m['f1']:.4f}  train_acc={train_m['accuracy']:.4f} | "
            f"val_loss={val_m['loss']:.4f}  val_f1={val_m['f1']:.4f}  best={best_f1:.4f} | "
            f"val_prec={val_m['precision']:.4f}  val_rec={val_m['recall']:.4f} | "
            f"time={train_m['time']:.0f}s  ETA={eta_str(epoch_times, epoch, epochs)}  "
            f"GPU={self._gpu_str()}"
        )

    def _on_fit_end(self, best_f1: float):
        self._remove_all_hooks()
        self._emit(f"Entrenamiento completado — mejor Val F1: {best_f1:.4f}")

    # ── Custom train_epoch with batch-level table ────────────────────────────

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
            images, labels = images.to(device), labels.to(device)
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
        result = {
            "loss": total_loss / len(loader),
            "f1": f1_score(all_preds_t, all_labels_t),
            "accuracy": accuracy(all_preds_t, all_labels_t),
            "time": time.time() - start,
        }
        self._propagate_train_result(result)
        return result

    def _propagate_train_result(self, result: dict):
        """Notify inner decorators of train metrics.

        DeepTracingDecorator owns the training loop directly (for per-batch tables),
        bypassing inner decorators' train_epoch. This method propagates the final
        result so that inner decorators like PlottingDecorator can still record it.
        """
        inner = self._trainer
        while hasattr(inner, "_trainer"):
            if hasattr(inner, "_record_train_result"):
                inner._record_train_result(result)
            inner = inner._trainer

    # ── Logging helpers ──────────────────────────────────────────────────────

    def _show_model_summary(self):
        try:
            from torchinfo import summary
            self._emit("=== Arquitectura del modelo (torchinfo) ===")
            stats = summary(self._trainer.model, input_size=(1, 3, 224, 224),
                            verbose=0, device=self._trainer.device)
            for line in str(stats).splitlines():
                self._emit(line)
        except ImportError:
            self._emit("torchinfo no disponible — omitiendo resumen")

    def _gpu_str(self) -> str:
        if not torch.cuda.is_available():
            return "cpu"
        return f"{torch.cuda.memory_allocated()/1e9:.1f}GB/{torch.cuda.memory_reserved()/1e9:.1f}GB"

    def _lr_str(self) -> str:
        return " | ".join(f"{g['lr']:.2e}" for g in self._trainer.optimizer.param_groups)

    def _layer_status(self, s: LayerStats) -> str:
        if s.dead_ratio > 0.5:
            return "⚠ DEAD"
        if s.exploding:
            return "⚠ EXPLODE"
        if s.vanishing and s.grad_norm > 0:
            return "⚠ VANISH"
        return "✓ OK"

    def _representative_layers(self) -> list[tuple[str, LayerStats]]:
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

    def _log_layer_table(self, epoch: int, batch_idx: int, n_batches: int):
        layers = self._representative_layers()
        if not layers:
            return
        self._emit(
            f"[E{epoch:03d}/B{batch_idx:05d}/{n_batches}] "
            f"GPU={self._gpu_str()}  LR={self._lr_str()}"
        )
        self._emit(
            f"  {'Layer':<45} {'act_μ':>7} {'act_σ':>7} {'dead%':>6} "
            f"{'grad_n':>8} {'grad_max':>9} {'status':>10}"
        )
        self._emit("  " + "─" * 100)
        for name, s in layers:
            short = name[-43:] if len(name) > 43 else name
            self._emit(
                f"  {short:<45} "
                f"{s.act_mean:>7.4f} {s.act_std:>7.4f} {s.dead_ratio*100:>5.1f}% "
                f"{s.grad_norm:>8.4f} {s.grad_max:>9.4f} {self._layer_status(s):>10}"
            )

    def _log_anomalies(self, epoch: int):
        dead = [n for n, s in self._layer_stats.items() if s.dead_ratio > 0.5]
        exploding = [n for n, s in self._param_stats.items() if s.exploding]
        vanishing = [n for n, s in self._param_stats.items() if s.vanishing and s.grad_norm > 0]
        bad = [
            n for n, s in self._param_stats.items()
            if s.update_ratio > 0 and not (self.HEALTHY_RATIO_MIN <= s.update_ratio <= self.HEALTHY_RATIO_MAX)
        ]
        if dead:
            self._emit(f"[E{epoch:03d}] ⚠ Neuronas muertas en {len(dead)} capas: {dead[:3]}")
        if exploding:
            self._emit(f"[E{epoch:03d}] ⚠ Gradiente explosivo en: {exploding[:3]}")
        if vanishing:
            self._emit(f"[E{epoch:03d}] ⚠ Gradiente evanescente en: {vanishing[:3]}")
        if bad:
            self._emit(f"[E{epoch:03d}] ⚠ Update ratio anómalo en {len(bad)} params")
        if not any([dead, exploding, vanishing, bad]):
            self._emit(f"[E{epoch:03d}] ✓ Flujo de gradientes sin anomalías")
