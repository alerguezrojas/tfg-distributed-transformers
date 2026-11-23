"""Distributed training script (multi-GPU, multi-node DDP); dataset from data.name.

Launch with torchrun:
    # Single node, 2 GPUs:
    torchrun --nproc_per_node=2 scripts/train_ddp.py --config configs/train_ddp_verode.yaml

    # Multi-node via Slurm (run from login node inside tmux):
    srun --partition=batch --nodes=2 --nodelist=verode16,verode21 \\
      --gres=gpu:tesla:1 --ntasks-per-node=1 --cpus-per-task=8 --time=48:00:00 \\
      bash -c 'cd ~/tfg-distributed-transformers && \\
        .venv/bin/torchrun \\
          --nnodes=2 --nproc_per_node=1 \\
          --node_rank=$SLURM_NODEID \\
          --master_addr=verode16 --master_port=29500 \\
          scripts/train_ddp.py --config configs/train_ddp_verode.yaml --trace simple'

    # Smoke test (1 GPU, validates DDP code path without a second GPU):
    torchrun --nproc_per_node=1 scripts/train_ddp.py \\
        --config configs/train_v3.yaml --model vit_tiny_patch16_224 --epochs 1

torchrun injects RANK, LOCAL_RANK, WORLD_SIZE as environment variables.
Only rank 0 prints to console and writes logs / checkpoints.

The batch_size in the config is PER GPU — the effective global batch size is
batch_size × world_size.  Adjust lr proportionally if needed (linear scaling rule).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.launch import main as launch  # noqa: E402


def main():
    """The flow lives in src/training/launch.py; this strategy is registered in strategies.py."""
    launch("ddp")


if __name__ == "__main__":
    main()
