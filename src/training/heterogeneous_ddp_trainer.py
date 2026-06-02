"""HeterogeneousDDPTrainer — DDP para clústeres heterogéneos GPU+CPU.

Extiende DDPTrainer para manejar correctamente batch sizes diferentes por rank.

Problema con DDP estándar + batch sizes distintos
--------------------------------------------------
DDP promedia gradientes entre todos los ranks con igual peso. Si rank 0
tiene batch_size=64 y rank 1 tiene batch_size=4, promediar sus losses
con igual peso sobrepondera al rank 1 (1/2 en lugar del correcto 4/68).

Solución: normalización de gradientes por batch global
-------------------------------------------------------
Cada rank escala su loss por:

    scale = (local_batch_size × world_size) / global_batch_size

donde global_batch_size = Σ local_batch_size_i (suma sobre todos los ranks,
obtenida via all_reduce al inicio de cada batch).

Equivalentemente, cada rank computa la loss como suma (no media) y divide
por el batch global total. Después del all_reduce de DDP, el resultado es
el gradiente que se calcularía sobre el mini-batch global concatenado.

Backend: gloo — soporta CPU y GPU. NCCL requiere CUDA en todos los ranks.

Lanzar con torchrun (gloo backend, nodos mixtos GPU+CPU):
    # Nodo GPU (verode21, rank 0):
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \\
      --master_addr=verode21 --master_port=29500 \\
      scripts/train_heterogeneous_ddp.py \\
      --config configs/train_heterogeneous_ddp.yaml --trace simple

    # Nodo CPU (verode16, rank 1):
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \\
      --master_addr=verode21 --master_port=29500 \\
      scripts/train_heterogeneous_ddp.py \\
      --config configs/train_heterogeneous_ddp.yaml --trace simple
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.ddp_trainer import DDPTrainer
from src.training import metrics as m
from src.training.augmentations import mixup_batch


def _smooth(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    return labels * (1 - smoothing) + smoothing / 2


class HeterogeneousDDPTrainer(DDPTrainer):
    """DDP trainer que maneja batch sizes heterogéneos entre ranks.

    Parámetros
    ----------
    local_batch_size:
        Número de muestras que procesa este rank por step.
        Necesario para la ponderación correcta del gradiente.
    Resto de parámetros: heredados de DDPTrainer / Trainer.
    """

    def __init__(self, *args, local_batch_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_batch_size = local_batch_size
        # Criterio con reducción sum para normalización manual
        self._criterion_sum = nn.BCEWithLogitsLoss(reduction="sum")

    # ── train_epoch con normalización de gradientes ────────────────────────────

    def train_epoch(self, loader: DataLoader) -> dict:
        self._epoch += 1
        self._current_epoch += 1  # requerido para batch hooks y checkpoints

        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(self._epoch)

        self.model.train()
        total_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_labels_list: list[torch.Tensor] = []
        start = time.time()
        n_batches = len(loader)

        for batch_idx, (images, labels) in enumerate(loader, 1):
            images = images.to(self.device)
            labels = labels.to(self.device)
            step_bs = labels.shape[0]

            # All-reduce del batch size real para obtener el global de este step
            bs_tensor = torch.tensor(float(step_bs), device=self.device)
            dist.all_reduce(bs_tensor, op=dist.ReduceOp.SUM)
            global_bs = bs_tensor.item()

            self.optimizer.zero_grad()

            # Mixup: devuelve (mixed_images, mixed_labels) — dos valores
            if self.mixup_alpha > 0 and torch.rand(1).item() < 0.5:
                images, labels_mixed = mixup_batch(images, labels, self.mixup_alpha)
                labels_for_loss = _smooth(labels_mixed, self.label_smoothing)
                labels_for_metrics = labels > 0.5  # usar labels originales para métricas
            else:
                labels_for_loss = (
                    _smooth(labels, self.label_smoothing)
                    if self.label_smoothing > 0 else labels
                )
                labels_for_metrics = labels

            logits = self.model(images)

            # loss_sum / global_bs = gradiente correcto ponderado por batch global
            loss_sum = self._criterion_sum(logits, labels_for_loss)
            loss = loss_sum / global_bs
            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            batch_loss = loss_sum.detach().item() / step_bs  # mean para logging
            total_loss += batch_loss

            with torch.no_grad():
                preds = (torch.sigmoid(logits.detach()) >= 0.5).cpu()
                all_preds.append(preds)
                all_labels_list.append((labels_for_metrics.detach() >= 0.5).cpu())

            # Batch hooks — necesario para BatchMonitorDecorator
            if self._batch_hooks:
                running_loss = total_loss / batch_idx
                lr = self.optimizer.param_groups[0]["lr"]
                for hook in self._batch_hooks:
                    hook(self._current_epoch, batch_idx, n_batches,
                         running_loss, batch_loss, lr)

        if self.scheduler is not None:
            self.scheduler.step()

        preds_t = torch.cat(all_preds)
        labels_t = torch.cat(all_labels_list)

        return {
            "loss": total_loss / n_batches if n_batches > 0 else 0.0,
            "f1": m.f1_score(preds_t, labels_t),
            "accuracy": m.accuracy(preds_t, labels_t),
            "time": time.time() - start,
            "_preds": preds_t,
            "_labels": labels_t,
        }
