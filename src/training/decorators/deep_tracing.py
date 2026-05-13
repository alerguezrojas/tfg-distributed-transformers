"""DeepTracingDecorator — per-layer, per-neuron real-time tracing.

Extends TracingDecorator: inherits the fit loop (EpochController) and the
logging infrastructure (_emit). Each diagnostic feature is independently
activatable via the `features` parameter.

Features:
  model-summary  — torchinfo architecture summary at fit start
  grad-monitor   — forward/backward/param hooks (activation & gradient stats)
  batch-table    — per-layer table logged every log_every batches (requires hooks)
  anomalies      — dead-neuron / vanishing / exploding gradient alerts at epoch end
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


ALL_INSPECT_FEATURES: frozenset[str] = frozenset(
    {"model-summary", "grad-monitor", "batch-table", "anomalies"}
)


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
    """Granular diagnostic tracing — each feature independently activatable.

    Inherits from TracingDecorator so the fit loop and logging are reused.

    Pass `features` to select which diagnostics to enable (default: all).
    Available features:
      "model-summary"  — torchinfo summary at fit start
      "grad-monitor"   — hooks on every leaf module and param
      "batch-table"    — per-layer table every log_every batches
      "anomalies"      — anomaly alerts at epoch end (needs grad-monitor)

    Examples:
        # Only show model architecture — zero hook overhead:
        DeepTracingDecorator(trainer, logger, features={"model-summary"})

        # Monitor gradients and detect anomalies, no per-batch output:
        DeepTracingDecorator(trainer, logger, features={"grad-monitor", "anomalies"})

        # Full depth (equivalent to legacy --trace deep):
        DeepTracingDecorator(trainer, logger)  # features defaults to all
    """

    VANISHING_THRESHOLD = 1e-7
    EXPLODING_THRESHOLD = 10.0
    DEAD_NEURON_THRESHOLD = 1e-6
    HEALTHY_RATIO_MIN = 1e-4
    HEALTHY_RATIO_MAX = 1.0

    def __init__(
        self,
        trainer: BaseTrainer,
        logger: logging.Logger,
        log_every: int = 100,
        patience: int | None = None,
        features: set[str] | None = None,
    ):
        super().__init__(trainer, logger=logger, patience=patience)
        self._features: frozenset[str] = (
            frozenset(features) if features is not None else ALL_INSPECT_FEATURES
        )
        self.log_every = log_every
        self._layer_stats: dict[str, LayerStats] = {}
        self._param_stats: dict[str, ParamStats] = {}
        self._forward_hooks: list = []
        self._param_hooks: list = []
        self._current_epoch: int = 0

    def _needs_hooks(self) -> bool:
        return bool(self._features & {"grad-monitor", "batch-table", "anomalies"})

    def _needs_own_train_loop(self) -> bool:
        return "batch-table" in self._features

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
                # detach() drops autograd graph; float() avoids precision loss on fp16
                f = output.detach().float()
                s = self._layer_stats.setdefault(name, LayerStats())
                s.act_mean = f.abs().mean().item()
                s.act_std = f.std().item()
                s.act_max = f.abs().max().item()
                s.dead_ratio = (f.abs() < self.DEAD_NEURON_THRESHOLD).float().mean().item()
        return hook

    def _bwd_hook(self, name: str):
        def hook(_m, _gi, grad_output):
            # grad_output is a tuple of tensors (one per output); index [0] is the main one
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
                ps.update_ratio = g_norm / (w_norm + 1e-8)  # +1e-8: avoid div-by-zero on zero-init params
                ps.vanishing = g_norm < self.VANISHING_THRESHOLD
                ps.exploding = g_norm > self.EXPLODING_THRESHOLD
        return hook

    # ── EpochController hooks ────────────────────────────────────────────────

    def _on_fit_start(self, epochs: int):
        if "model-summary" in self._features:
            self._show_model_summary()
        if self._needs_hooks():
            self._register_all_hooks()
            n_mod = len(list(self._trainer.model.named_modules()))
            n_par = sum(1 for p in self._trainer.model.parameters() if p.requires_grad)
            self._emit(
                f"Iniciando entrenamiento profundo — {epochs} epochs | "
                f"módulos: {n_mod} | params: {n_par} | "
                f"features: {', '.join(sorted(self._features))}"
            )
        else:
            self._emit(
                f"Iniciando entrenamiento — {epochs} epochs | "
                f"features: {', '.join(sorted(self._features))}"
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
        if "anomalies" in self._features and self._needs_hooks():
            self._log_anomalies(epoch)
        self._emit(
            f"[E{epoch:03d}/{epochs}] ══ RESUMEN  "
            f"train_loss={train_m['loss']:.4f}  train_f1={train_m['f1']:.4f}  train_acc={train_m['accuracy']:.4f} | "
            f"val_loss={val_m['loss']:.4f}  val_f1={val_m['f1']:.4f}  val_acc={val_m['accuracy']:.4f}  best={best_f1:.4f} | "
            f"val_prec={val_m['precision']:.4f}  val_rec={val_m['recall']:.4f} | "
            f"time={train_m['time']:.0f}s  ETA={eta_str(epoch_times, epoch, epochs)}  "
            f"GPU={self._gpu_str()}"
        )

    def _on_fit_end(self, best_f1: float):
        if self._needs_hooks():
            self._remove_all_hooks()
        self._emit(f"Entrenamiento completado — mejor Val F1: {best_f1:.4f}")

    # ── train_epoch: own loop only when batch-table is active ────────────────
    # When batch-table is inactive, delegates to inner trainer (normal chain).
    # When active, reimplements the loop to fire _log_layer_table every log_every
    # batches. Inner decorators' train_epoch is bypassed; _propagate_train_result
    # notifies them (e.g. PlottingDecorator) at the end.

    def train_epoch(self, loader: DataLoader) -> dict:
        if not self._needs_own_train_loop():
            return self._trainer.train_epoch(loader)
        return self._train_epoch_deep(loader)

    def _train_epoch_deep(self, loader: DataLoader) -> dict:
        model = self._trainer.model
        optimizer = self._trainer.optimizer
        criterion = self._trainer.criterion
        device = self._trainer.device

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
            grad_clip = getattr(self._trainer, "grad_clip", None)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                preds = torch.sigmoid(logits) > 0.5
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

            if batch_idx % self.log_every == 0:
                self._log_layer_table(self._current_epoch, batch_idx, len(loader))

        scheduler = getattr(self._trainer, "scheduler", None)
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
        """Notify inner decorators of train metrics when we own the training loop."""
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
        lrs = [g["lr"] for g in self._trainer.optimizer.param_groups]
        if len(lrs) <= 4:
            return " | ".join(f"{lr:.2e}" for lr in lrs)
        return f"{min(lrs):.2e} … {max(lrs):.2e} ({len(lrs)} groups)"

    def _layer_status(self, s: LayerStats) -> str:
        if s.dead_ratio > 0.5:
            return "DEAD"
        if s.exploding:
            return "EXPLODE"
        if s.vanishing and s.grad_norm > 0:
            return "VANISH"
        return "OK"

    def _representative_layers(self) -> list[tuple[str, LayerStats]]:
        # Selects 14 diagnostic points in ViT-B/16:
        #   1 patch embedding projection + 12 attention output projections (one per block)
        #   + 1 classification head → captures the full depth of the network
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
            f"{'grad_n':>8} {'grad_max':>9} {'status':>8}"
        )
        self._emit("  " + "─" * 98)
        for name, s in layers:
            short = name[-43:] if len(name) > 43 else name
            self._emit(
                f"  {short:<45} "
                f"{s.act_mean:>7.4f} {s.act_std:>7.4f} {s.dead_ratio*100:>5.1f}% "
                f"{s.grad_norm:>8.4f} {s.grad_max:>9.4f} {self._layer_status(s):>8}"
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
            self._emit(f"[E{epoch:03d}] Neuronas muertas en {len(dead)} capas: {dead[:3]}")
        if exploding:
            self._emit(f"[E{epoch:03d}] Gradiente explosivo en: {exploding[:3]}")
        if vanishing:
            self._emit(f"[E{epoch:03d}] Gradiente evanescente en: {vanishing[:3]}")
        if bad:
            self._emit(f"[E{epoch:03d}] Update ratio anómalo en {len(bad)} params")
        if not any([dead, exploding, vanishing, bad]):
            self._emit(f"[E{epoch:03d}] Flujo de gradientes sin anomalías")
