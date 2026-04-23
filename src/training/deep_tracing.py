"""DeepTracingDecorator — per-layer, per-neuron real-time tracing.

Goes as deep as PyTorch allows without modifying the model:
  - Forward hooks  → activation stats per layer (mean, std, dead neurons)
  - Parameter hooks → gradient stats per parameter (norm, max, vanishing/exploding)
  - Weight tracking  → weight norm + update ratio per layer
  - GPU memory       → allocated / reserved per step
  - Learning rate    → per optimizer group
  - torchinfo        → full architecture summary at startup
  - Anomaly alerts   → dead neurons, vanishing and exploding gradients
"""

import logging
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.training.trainer_decorators import TrainerDecorator


def _eta_str(epoch_times: list[float], epochs_done: int, epochs_total: int) -> str:
    if not epoch_times:
        return "?"
    remaining_s = (epochs_total - epochs_done) * (sum(epoch_times) / len(epoch_times))
    h, m = int(remaining_s // 3600), int((remaining_s % 3600) // 60)
    return f"{h}h {m:02d}m"


@dataclass
class LayerStats:
    """Activation and gradient statistics for a single layer."""
    # Forward (activations)
    act_mean: float = 0.0
    act_std: float = 0.0
    act_max: float = 0.0
    dead_ratio: float = 0.0      # fraction of neurons with |act| < 1e-6

    # Backward (gradients w.r.t. output)
    grad_norm: float = 0.0
    grad_max: float = 0.0
    vanishing: bool = False       # grad_norm < 1e-7
    exploding: bool = False       # grad_norm > 10.0


@dataclass
class ParamStats:
    """Gradient and weight statistics for a single parameter tensor."""
    weight_norm: float = 0.0
    grad_norm: float = 0.0
    grad_max: float = 0.0
    update_ratio: float = 0.0    # ||grad|| / (||weight|| + 1e-8) — healthy: 0.001–0.1
    vanishing: bool = False
    exploding: bool = False


class DeepTracingDecorator(TrainerDecorator):
    """Maximum-depth tracing decorator for PyTorch training.

    Registers forward hooks on every nn.Module and gradient hooks on
    every trainable parameter to capture per-layer, per-neuron statistics
    in real time during training.

    Output levels:
      INFO  — epoch start/end, metrics summary, anomaly alerts
      DEBUG — every log_every batches: per-block activation + gradient table

    Usage:
        from src.training.deep_tracing import DeepTracingDecorator
        from src.training.logger_setup import setup_logger

        logger = setup_logger("deep", level=logging.DEBUG, log_file="logs/deep.log")
        trainer = DeepTracingDecorator(Trainer(...), logger=logger, log_every=50)
        trainer.fit(train_loader, val_loader, epochs=30)
    """

    # Thresholds for anomaly detection
    VANISHING_THRESHOLD = 1e-7
    EXPLODING_THRESHOLD = 10.0
    DEAD_NEURON_THRESHOLD = 1e-6
    HEALTHY_UPDATE_RATIO_MIN = 1e-4
    HEALTHY_UPDATE_RATIO_MAX = 1.0

    def __init__(
        self,
        trainer: BaseTrainer,
        logger: logging.Logger,
        log_every: int = 100,
    ):
        """
        Args:
            trainer: Inner trainer (Trainer or another decorator).
            logger: Python logger from setup_logger().
            log_every: Log layer stats every N batches.
        """
        super().__init__(trainer)
        self._logger = logger
        self.log_every = log_every

        self._layer_stats: dict[str, LayerStats] = {}
        self._param_stats: dict[str, ParamStats] = {}
        self._forward_hooks: list = []
        self._param_hooks: list = []

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_forward_hooks(self):
        """Register a forward hook on every module to capture activations."""
        for name, module in self._trainer.model.named_modules():
            if not list(module.children()):  # leaf modules only
                hook = module.register_forward_hook(self._make_forward_hook(name))
                self._forward_hooks.append(hook)

    def _make_forward_hook(self, name: str):
        def hook(_module, _input, output):
            if not isinstance(output, torch.Tensor):
                return
            with torch.no_grad():
                flat = output.detach().float()  # stays on GPU, .item() transfers only the scalar
                stats = self._layer_stats.setdefault(name, LayerStats())
                stats.act_mean = flat.abs().mean().item()
                stats.act_std = flat.std().item()
                stats.act_max = flat.abs().max().item()
                stats.dead_ratio = (flat.abs() < self.DEAD_NEURON_THRESHOLD).float().mean().item()
        return hook

    def _register_backward_hooks(self):
        """Register a backward hook on every module to capture gradient flow."""
        for name, module in self._trainer.model.named_modules():
            if not list(module.children()):
                hook = module.register_full_backward_hook(self._make_backward_hook(name))
                self._forward_hooks.append(hook)  # reuse same list for cleanup

    def _make_backward_hook(self, name: str):
        def hook(_module, _grad_input, grad_output):
            if grad_output[0] is None:
                return
            with torch.no_grad():
                g = grad_output[0].detach().float()  # stays on GPU
                norm = g.norm().item()
                stats = self._layer_stats.setdefault(name, LayerStats())
                stats.grad_norm = norm
                stats.grad_max = g.abs().max().item()
                stats.vanishing = norm < self.VANISHING_THRESHOLD
                stats.exploding = norm > self.EXPLODING_THRESHOLD
        return hook

    def _register_param_hooks(self):
        """Register gradient hooks on each trainable parameter tensor."""
        for name, param in self._trainer.model.named_parameters():
            if not param.requires_grad:
                continue

            def hook(grad, n=name, p=param):
                with torch.no_grad():
                    g = grad.detach().float()  # stays on GPU
                    g_norm = g.norm().item()
                    w_norm = p.data.detach().float().norm().item()
                    ratio = g_norm / (w_norm + 1e-8)
                    ps = self._param_stats.setdefault(n, ParamStats())
                    ps.weight_norm = w_norm
                    ps.grad_norm = g_norm
                    ps.grad_max = g.abs().max().item()
                    ps.update_ratio = ratio
                    ps.vanishing = g_norm < self.VANISHING_THRESHOLD
                    ps.exploding = g_norm > self.EXPLODING_THRESHOLD

            handle = param.register_hook(hook)
            self._param_hooks.append(handle)

    def _remove_all_hooks(self):
        for h in self._forward_hooks:
            h.remove()
        for h in self._param_hooks:
            h.remove()
        self._forward_hooks.clear()
        self._param_hooks.clear()

    # ------------------------------------------------------------------
    # Layer selection — one representative layer per transformer block
    # ------------------------------------------------------------------

    def _select_representative_layers(self) -> list[tuple[str, LayerStats]]:
        """Pick one layer per transformer block + patch embed + head.

        Shows the full depth of the network (block 0 → block 11 → head)
        instead of always showing the first N alphabetically.
        """
        selected: dict[str, LayerStats] = {}

        # Patch embedding (entrada del ViT)
        for name in sorted(self._layer_stats):
            if "patch_embed" in name and "proj" in name:
                selected[name] = self._layer_stats[name]
                break

        # Proyección de atención de cada bloque Transformer
        for i in range(12):
            key = f"backbone.blocks.{i}.attn.proj"
            if key in self._layer_stats:
                selected[key] = self._layer_stats[key]

        # Cabeza clasificadora
        for name in sorted(self._layer_stats):
            if "head" in name and isinstance(
                dict(self._trainer.model.named_modules()).get(name), nn.Linear
            ):
                selected[name] = self._layer_stats[name]
                break

        return list(selected.items())

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

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
        """Log a compact per-block activation + gradient table."""
        layers = self._select_representative_layers()
        if not layers:
            return

        header = (
            f"[E{epoch:03d}/B{batch_idx:05d}/{n_batches}] "
            f"GPU={self._gpu_memory_str()}  LR={self._lr_str()}"
        )
        self._logger.debug(header)
        self._logger.debug(
            f"  {'Layer':<45} {'act_μ':>7} {'act_σ':>7} {'dead%':>6} "
            f"{'grad_n':>8} {'grad_max':>9} {'status':>10}"
        )
        self._logger.debug("  " + "─" * 100)

        for name, s in layers:
            status = self._layer_status(s)
            short_name = name[-43:] if len(name) > 43 else name
            self._logger.debug(
                f"  {short_name:<45} "
                f"{s.act_mean:>7.4f} {s.act_std:>7.4f} {s.dead_ratio*100:>5.1f}% "
                f"{s.grad_norm:>8.4f} {s.grad_max:>9.4f} {status:>10}"
            )

    def _layer_status(self, s: LayerStats) -> str:
        if s.dead_ratio > 0.5:
            return "⚠ DEAD"
        if s.exploding:
            return "⚠ EXPLODE"
        if s.vanishing and s.grad_norm > 0:
            return "⚠ VANISH"
        return "✓ OK"

    def _log_param_anomalies(self, epoch: int):
        """Log any parameter-level anomalies at end of epoch."""
        dead_layers = [n for n, s in self._layer_stats.items() if s.dead_ratio > 0.5]
        exploding = [n for n, s in self._param_stats.items() if s.exploding]
        vanishing = [n for n, s in self._param_stats.items() if s.vanishing and s.grad_norm > 0]
        unhealthy_ratio = [
            n for n, s in self._param_stats.items()
            if s.update_ratio > 0 and (
                s.update_ratio < self.HEALTHY_UPDATE_RATIO_MIN or
                s.update_ratio > self.HEALTHY_UPDATE_RATIO_MAX
            )
        ]

        if dead_layers:
            self._logger.warning(f"[E{epoch:03d}] Neuronas muertas en {len(dead_layers)} capas: {dead_layers[:3]}...")
        if exploding:
            self._logger.warning(f"[E{epoch:03d}] Gradiente explosivo en: {exploding[:3]}...")
        if vanishing:
            self._logger.warning(f"[E{epoch:03d}] Gradiente evanescente en: {vanishing[:3]}...")
        if unhealthy_ratio:
            self._logger.warning(f"[E{epoch:03d}] Update ratio anómalo en {len(unhealthy_ratio)} params")

        if not any([dead_layers, exploding, vanishing, unhealthy_ratio]):
            self._logger.info(f"[E{epoch:03d}] Flujo de gradientes: sin anomalías detectadas ✓")

    def _show_model_summary(self):
        """Print architecture summary using torchinfo."""
        try:
            from torchinfo import summary
            self._logger.info("=== Arquitectura del modelo (torchinfo) ===")
            stats = summary(
                self._trainer.model,
                input_size=(1, 3, 224, 224),
                verbose=0,
                device=self._trainer.device,
            )
            for line in str(stats).splitlines():
                self._logger.info(line)
        except ImportError:
            self._logger.warning("torchinfo no disponible — omitiendo resumen de arquitectura")

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------

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
                self._log_layer_table(
                    epoch=self._current_epoch,
                    batch_idx=batch_idx,
                    n_batches=len(loader),
                )

        if scheduler:
            scheduler.step()

        all_preds_t = torch.cat(all_preds)
        all_labels_t = torch.cat(all_labels)
        inner = self._trainer  # access _f1_score / _accuracy helpers
        return {
            "loss": total_loss / len(loader),
            "f1": inner._f1_score(all_preds_t, all_labels_t),
            "accuracy": inner._accuracy(all_preds_t, all_labels_t),
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

        n_modules = len(list(self._trainer.model.named_modules()))
        n_params = sum(1 for p in self._trainer.model.parameters() if p.requires_grad)
        self._logger.info(
            f"Iniciando entrenamiento profundo — {epochs} epochs | "
            f"Módulos monitorizados: {n_modules} | "
            f"Params con gradiente: {n_params}"
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
                    f"val_loss={val_m['loss']:.4f}  val_f1={val_m['f1']:.4f}  best={best_f1:.4f}  val_acc={val_m['accuracy']:.4f} | "
                    f"val_prec={val_m['precision']:.4f}  val_rec={val_m['recall']:.4f} | "
                    f"time={train_m['time']:.0f}s  ETA={_eta_str(epoch_times, epoch, epochs)}  "
                    f"GPU={self._gpu_memory_str()}"
                )

        finally:
            self._remove_all_hooks()

        self._logger.info(f"Entrenamiento completado — mejor Val F1: {best_f1:.4f}")
