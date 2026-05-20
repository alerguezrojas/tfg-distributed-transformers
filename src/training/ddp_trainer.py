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
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__(model, optimizer, scheduler, device, checkpoint_dir, criterion, grad_clip)
        self.rank = rank
        self.world_size = world_size
        self._epoch = 0
        # Wrap AFTER super().__init__ moves model to device
        self.model = DDP(self.model, device_ids=[rank])

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

        # _preds: bool tensor (N, C) on CPU from Trainer.eval_epoch
        # _labels: float tensor (N, C) on CPU
        # Convert to float for NCCL all_gather, then move to CUDA
        preds_gpu = result["_preds"].float().to(self.device)
        labels_gpu = result["_labels"].float().to(self.device)

        gathered_preds = [torch.zeros_like(preds_gpu) for _ in range(self.world_size)]
        gathered_labels = [torch.zeros_like(labels_gpu) for _ in range(self.world_size)]
        dist.all_gather(gathered_preds, preds_gpu)
        dist.all_gather(gathered_labels, labels_gpu)

        all_preds = torch.cat(gathered_preds).cpu().bool()
        all_labels = torch.cat(gathered_labels).cpu()

        # Average loss across all ranks
        loss_t = torch.tensor(result["loss"], device=self.device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)

        result["_preds"] = all_preds
        result["_labels"] = all_labels
        result["loss"] = loss_t.item()
        result["f1"] = m.f1_score(all_preds, all_labels)
        result["accuracy"] = m.accuracy(all_preds, all_labels)
        result["precision"] = m.precision(all_preds, all_labels)
        result["recall"] = m.recall(all_preds, all_labels)

        return result

    def save_checkpoint(self, epoch: int, metrics: dict):
        # Only rank 0 writes to disk; DDP.state_dict() delegates to module
        if self.rank == 0:
            super().save_checkpoint(epoch, metrics)
