"""Single-GPU training script; the dataset comes from the config (data.name).

A wrapper of the common launcher (src/training/launch.py): the four strategies share these options.

Flags
-----
--trace off|simple|deep
    Logging controller:
      off    — TracingDecorator, console only
      simple — TracingDecorator, log to logs/train_FECHA.log
      deep   — DeepTracingDecorator with all inspection features active
               (equivalent to --inspect model-summary grad-monitor batch-table anomalies)

--inspect [model-summary] [grad-monitor] [batch-table] [anomalies]
    Activates DeepTracingDecorator with only the selected features:
      model-summary  — torchinfo architecture summary at fit start
      grad-monitor   — forward/backward/param hooks on every layer
      batch-table    — per-layer stats table every log_batch_every batches
      anomalies      — dead-neuron / vanishing / exploding gradient alerts
    If --inspect is used, --trace deep is implied (DeepTracingDecorator becomes controller).

--layers [plot] [hooks] [confusion] [batch-monitor]
    Stackable aspect decorators:
      plot         — PlottingDecorator: keeps the train/val metric history in memory (no files)
      hooks        — LayerHooksDecorator: forward hooks on Linear layers every 5 epochs
      confusion    — ConfusionMatrixDecorator: per-class F1/precision/recall and confusion matrix
                     as perclass_metrics_*.csv and confusion_matrix_*.csv (the web draws the charts)
      batch-monitor — BatchMonitorDecorator: batch-level loss CSV in logs/

--fn [timing] [energy]
    Python @ decorators applied to train_epoch and eval_epoch:
      timing — print execution time
      energy — measure GPU energy consumption (requires nvidia-ml-py)

--metrics [loss] [f1] [accuracy] [precision_recall]
    Metric reporters (active only when --inspect is NOT used):
      loss             — LossReporter: train_loss / val_loss
      f1               — F1Reporter: train_f1 / val_f1
      accuracy         — AccuracyReporter: train_acc / val_acc
      precision_recall — PrecisionRecallReporter: val_precision / val_recall
    Pass --metrics with no args to disable all reporters.

--model <name>
    Override model name from config (any timm model ID):
      vit_tiny_patch16_224    — ~5.7M params, fast for local testing
      vit_small_patch16_224   — ~22M params
      vit_base_patch16_224    — ~85M params (default)
      resnet50                — ~25M params, CNN
      efficientnet_b0         — ~5.3M params, CNN
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.launch import main as launch  # noqa: E402


def main():
    """The flow lives in src/training/launch.py; this strategy is registered in strategies.py."""
    launch("single")


if __name__ == "__main__":
    main()
