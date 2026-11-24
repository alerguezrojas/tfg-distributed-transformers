"""Pipeline-parallel training with micro-batches (GPipe); the dataset comes from the config.

The model is split across two devices like ``train_model_parallel.py`` (see
``src/models/model_parallel.py``), and each batch is split into ``--micro-batches``
micro-batches that flow through the two stages at the same time: while stage 1 runs
micro-batch i, stage 0 runs micro-batch i+1. The update is the same as with the whole batch.
Same flags as train_model_parallel.py, plus:
  --micro-batches M                  micro-batches per batch (default 4)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.launch import main as launch  # noqa: E402


def main():
    """The flow lives in src/training/launch.py; this strategy is registered in strategies.py."""
    launch("pipeline")


if __name__ == "__main__":
    main()
