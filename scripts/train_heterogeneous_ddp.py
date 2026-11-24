"""Entrenamiento DDP heterogéneo — GPU (verode21 V100) + CPU (verode16/18).

Reparte la carga proporcionalmente a la capacidad de cómputo de cada nodo:
  - rank 0 (GPU): procesa la fracción grande del dataset con batch_size alto
  - rank 1 (CPU): procesa la fracción pequeña con batch_size reducido

El backend gloo soporta tanto CUDA como CPU en el mismo grupo de procesos.

Lanzar (dos terminales tmux separadas):

    # Terminal 1 — verode21 (V100, rank 0):
    ssh verode21
    cd ~/tfg-distributed-transformers
    .venv/bin/torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \\
      --master_addr=verode21 --master_port=29500 \\
      scripts/train_heterogeneous_ddp.py \\
      --config configs/train_heterogeneous_ddp.yaml --trace simple

    # Terminal 2 — verode16 o verode18 (CPU, rank 1):
    ssh verode16
    cd ~/tfg-distributed-transformers
    .venv/bin/torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \\
      --master_addr=verode21 --master_port=29500 \\
      scripts/train_heterogeneous_ddp.py \\
      --config configs/train_heterogeneous_ddp.yaml --trace simple
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.launch import main as launch  # noqa: E402


def main():
    """The flow lives in src/training/launch.py; this strategy is registered in strategies.py."""
    launch("heterogeneous")


if __name__ == "__main__":
    main()
