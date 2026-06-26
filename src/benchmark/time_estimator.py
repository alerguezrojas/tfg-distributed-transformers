"""TimeEstimator — turns measured throughput into per-epoch / total estimates."""
from __future__ import annotations

import math
from typing import Optional

from src.benchmark.value_objects import ModelInfo, BenchmarkResult


class TimeEstimator:
    def estimate(
        self,
        result: BenchmarkResult,
        dataset_train: int,
        dataset_val: int,
        epochs: int,
        nfs_factor: float = 1.0,
        model_info: ModelInfo | None = None,
    ) -> Optional[dict]:
        if result.oom or result.images_per_second_train == 0:
            return None

        train_batches = math.ceil(dataset_train / result.batch_size)
        eval_batches = math.ceil(dataset_val / result.batch_size)

        sec_train = train_batches * result.seconds_per_batch_train * nfs_factor
        sec_eval = eval_batches * result.seconds_per_batch_eval * nfs_factor  # eval also reads from disk
        sec_epoch = sec_train + sec_eval
        sec_total = sec_epoch * epochs

        energy_train_wh = energy_eval_wh = 0.0
        if result.avg_power_w > 0:
            energy_train_wh = result.avg_power_w * sec_train / 3600
            energy_eval_wh = result.avg_power_w * 0.4 * sec_eval / 3600

        flops_train = flops_eval = 0.0
        if model_info and model_info.flops_per_image_mflops:
            flops_img = model_info.flops_per_image_mflops / 1000
            flops_train = flops_img * 3 * dataset_train
            flops_eval = flops_img * dataset_val

        # Note: distributed scaling is NOT projected here. The flat-efficiency DDP
        # estimate was removed in favour of the compute/IO/sync-aware DDPOptimizer
        # (the #ddp block), which is the single source of truth for DDP scaling.
        return {
            "train_per_epoch": sec_train,
            "eval_per_epoch": sec_eval,
            "total_per_epoch": sec_epoch,
            "total": sec_total,
            "energy_train_wh_per_epoch": energy_train_wh,
            "energy_eval_wh_per_epoch": energy_eval_wh,
            "energy_total_wh": (energy_train_wh + energy_eval_wh) * epochs,
            "flops_train_gflops_per_epoch": flops_train,
            "flops_eval_gflops_per_epoch": flops_eval,
            "optimizer_steps_per_epoch": train_batches,
        }

    @staticmethod
    def format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m:02d}m"
