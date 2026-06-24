"""DDPTrainer — multi-GPU trainer using PyTorch DistributedDataParallel.

Inherits from Trainer and overrides only the three methods that differ in a
distributed context:
  - train_epoch: sets the DistributedSampler epoch for correct per-epoch shuffling
  - eval_epoch:  all_gathers _preds and _labels from every rank before computing metrics
  - save_checkpoint: only rank 0 writes to disk

The decorator stack (TracingDecorator, PlottingDecorator, etc.) remains entirely
unchanged — it wraps DDPTrainer the same way it wraps Trainer.

Launch with torchrun:
    torchrun --nproc_per_node=N scripts/train_ddp.py --config configs/train_ddp_verode.yaml
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.training.trainer import Trainer
from src.training import metrics as m


class DDPTrainer(Trainer):
    """DDP-aware trainer for multi-GPU training on a single node.

    Parameters
    ----------
    rank:       Global rank of this process (0 … world_size-1).
    world_size: Total number of processes / GPUs.

    The model is wrapped with DDP in __init__, AFTER the optimizer is built
    by the builder (LLRD requires access to model.backbone.blocks before wrapping).
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        criterion=None,
        grad_clip: float | None = None,
        label_smoothing: float = 0.0,
        mixup_alpha: float = 0.0,
        precision: str = "fp32",
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__(model, optimizer, scheduler, device, checkpoint_dir,
                         criterion, grad_clip, label_smoothing, mixup_alpha,
                         precision=precision)
        self.rank = rank
        self.world_size = world_size
        self._epoch = 0
        # device_ids only valid for CUDA; omit for CPU (gloo backend)
        if device.type == "cuda":
            self.model = DDP(self.model, device_ids=[rank])
        else:
            self.model = DDP(self.model)

    # ── Overrides ────────────────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader) -> dict:
        self._epoch += 1
        # DistributedSampler must know the epoch to produce different shuffles
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(self._epoch)
        return super().train_epoch(loader)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> dict:
        result = super().eval_epoch(loader)

        # Gather the raw PROBABILITIES and labels from every rank, so all the
        # global metrics — including the optimal-threshold search — are computed on
        # the FULL validation set, not just this rank's shard. (Gathering binary
        # preds would force the threshold search to stay per-rank.)
        probs_gpu = result["_probs"].float().to(self.device)
        labels_gpu = result["_labels"].float().to(self.device)

        gathered_probs = [torch.zeros_like(probs_gpu) for _ in range(self.world_size)]
        gathered_labels = [torch.zeros_like(labels_gpu) for _ in range(self.world_size)]
        dist.all_gather(gathered_probs, probs_gpu)
        dist.all_gather(gathered_labels, labels_gpu)

        all_probs = torch.cat(gathered_probs).cpu()
        all_labels = torch.cat(gathered_labels).cpu()
        all_preds = (all_probs > 0.5).long()

        # Average loss across all ranks.
        # NOTE: the AVG reduce op is NOT supported by the gloo backend (only
        # NCCL), and Verode runs gloo on torch 2.7.1. Use SUM + manual division,
        # which is portable across both backends and torch versions.
        loss_t = torch.tensor(result["loss"], device=self.device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        loss_t /= self.world_size

        # Optimal-threshold search on the GLOBAL probabilities.
        f1_base = m.f1_score(all_preds, all_labels)
        best_thresh, best_f1_thresh = 0.5, f1_base
        for t in m.THRESHOLD_GRID:
            f1_t = m.f1_score((all_probs > t).long(), all_labels)
            if f1_t > best_f1_thresh:
                best_thresh, best_f1_thresh = t, f1_t

        result["_preds"] = all_preds
        result["_labels"] = all_labels
        result["_probs"] = all_probs
        result["loss"] = loss_t.item()
        result["f1"] = f1_base
        result["accuracy"] = m.accuracy(all_preds, all_labels)
        result["precision"] = m.precision(all_preds, all_labels)
        result["recall"] = m.recall(all_preds, all_labels)
        result["_optimal_threshold"] = best_thresh
        result["_f1_at_optimal_threshold"] = best_f1_thresh

        return result

    def save_checkpoint(self, epoch: int, metrics: dict):
        # Only rank 0 writes to disk; DDP.state_dict() delegates to module
        if self.rank == 0:
            super().save_checkpoint(epoch, metrics)
