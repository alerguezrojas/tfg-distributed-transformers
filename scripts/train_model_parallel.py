"""Model-parallel (pipeline) training; the dataset comes from the config (data.name).

Splits a ViT across two devices (see ``src/models/model_parallel.py``) and trains
it through the SAME stack as every other strategy: the Template-Method loop in
``EpochController`` and the full decorator stack, assembled by
``TrainingSessionBuilder.with_model_parallel(...)``. So a model-parallel run gets
the same flags and the same artifacts (curves, per-class, confusion, batch metrics,
energy) as single/DDP — it is a first-class strategy, not a bespoke loop.

Flags (same meaning as train_single_gpu.py):
  --trace off|simple                 logging controller (deep needs a single-device model)
  --layers plot hooks confusion batch-monitor   stackable aspect decorators
  --fn timing energy                 @ decorators (energy samples BOTH split GPUs)
  --metrics loss f1 accuracy precision_recall   metric reporters
Model-parallel specific:
  --devices cuda:0,cuda:1            where to place the two stages (auto-detected if omitted)
  --split-block N                    block index where stage 0 ends (default: half)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.launch import main as launch  # noqa: E402


def main():
    """The flow lives in src/training/launch.py; this strategy is registered in strategies.py."""
    launch("model_parallel")


if __name__ == "__main__":
    main()
