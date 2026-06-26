"""Unit tests for benchmark checker components."""
import math
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.benchmark import (
    BenchmarkResult, ModelInfo, HardwareInfo, BenchmarkReport,
    TimeEstimator,
)


@pytest.fixture
def sample_result():
    # avg_power_w may not exist on older branches
    try:
        return BenchmarkResult(
            batch_size=32, trace_mode="off",
            seconds_per_batch_train=0.046,
            seconds_per_batch_eval=0.016,
            images_per_second_train=695.0,
            images_per_second_eval=2000.0,
            peak_vram_gb=4.95,
            avg_power_w=120.0,
        )
    except TypeError:
        return BenchmarkResult(
            batch_size=32, trace_mode="off",
            seconds_per_batch_train=0.046,
            seconds_per_batch_eval=0.016,
            images_per_second_train=695.0,
            images_per_second_eval=2000.0,
            peak_vram_gb=4.95,
        )


@pytest.fixture
def sample_model_info():
    return ModelInfo(
        name="vit_base_patch16_224",
        total_params=85_813_267,
        trainable_params=85_813_267,
        flops_per_image_mflops=17_000.0,
        weight_mb=343.0,
        gradient_mb=343.0,
        optimizer_mb=686.0,
        activation_mb_per_image=40.0,
    )


class TestTimeEstimator:

    def test_returns_none_on_oom(self, sample_result):
        oom_result = BenchmarkResult(
            batch_size=128, trace_mode="off",
            seconds_per_batch_train=0.0, seconds_per_batch_eval=0.0,
            images_per_second_train=0.0, images_per_second_eval=0.0,
            peak_vram_gb=0.0, oom=True,
        )
        est = TimeEstimator().estimate(oom_result, 237_871, 122_342, 30)
        assert est is None

    def test_time_keys_present(self, sample_result):
        est = TimeEstimator().estimate(sample_result, 237_871, 122_342, 30)
        assert "train_per_epoch" in est
        assert "eval_per_epoch" in est
        assert "total_per_epoch" in est
        assert "total" in est

    def test_total_is_sum(self, sample_result):
        est = TimeEstimator().estimate(sample_result, 237_871, 122_342, 30)
        assert est["total"] == pytest.approx(
            est["total_per_epoch"] * 30, rel=1e-3
        )

    def test_epoch_is_sum_of_phases(self, sample_result):
        est = TimeEstimator().estimate(sample_result, 237_871, 122_342, 30)
        assert est["total_per_epoch"] == pytest.approx(
            est["train_per_epoch"] + est["eval_per_epoch"], rel=1e-6
        )

    def test_nfs_factor_scales_train(self, sample_result):
        est1 = TimeEstimator().estimate(sample_result, 237_871, 122_342, 1, nfs_factor=1.0)
        est2 = TimeEstimator().estimate(sample_result, 237_871, 122_342, 1, nfs_factor=1.3)
        assert est2["train_per_epoch"] == pytest.approx(
            est1["train_per_epoch"] * 1.3, rel=1e-4
        )

    def test_energy_keys_present_with_power(self, sample_result, sample_model_info):
        import inspect
        sig = inspect.signature(TimeEstimator().estimate)
        if "model_info" not in sig.parameters:
            pytest.skip("Extended estimates not available in this branch")
        est = TimeEstimator().estimate(sample_result, 237_871, 122_342, 30,
                                       model_info=sample_model_info)
        if "energy_train_wh_per_epoch" not in est:
            pytest.skip("Energy estimates not available in this version")
        assert est["energy_train_wh_per_epoch"] >= 0

    def test_energy_zero_without_power(self, sample_model_info):
        import inspect
        sig = inspect.signature(TimeEstimator().estimate)
        if "model_info" not in sig.parameters:
            pytest.skip("Extended estimates not available in this branch")
        try:
            result_no_power = BenchmarkResult(
                batch_size=32, trace_mode="off",
                seconds_per_batch_train=0.046, seconds_per_batch_eval=0.016,
                images_per_second_train=695.0, images_per_second_eval=2000.0,
                peak_vram_gb=4.95, avg_power_w=0.0,
            )
        except TypeError:
            pytest.skip("avg_power_w not available")
        est = TimeEstimator().estimate(result_no_power, 237_871, 122_342, 5)
        if "energy_train_wh_per_epoch" in est:
            assert est["energy_train_wh_per_epoch"] == 0.0

    def test_flops_keys_present(self, sample_result, sample_model_info):
        import inspect
        sig = inspect.signature(TimeEstimator().estimate)
        if "model_info" not in sig.parameters:
            pytest.skip("Extended estimates not available in this branch")
        est = TimeEstimator().estimate(sample_result, 237_871, 122_342, 10,
                                       model_info=sample_model_info)
        if "flops_train_gflops_per_epoch" not in est:
            pytest.skip("FLOPs estimates not in this version")
        assert est["flops_train_gflops_per_epoch"] > 0

    def test_flops_train_3x_forward(self, sample_result, sample_model_info):
        import inspect
        sig = inspect.signature(TimeEstimator().estimate)
        if "model_info" not in sig.parameters:
            pytest.skip("Extended estimates not available in this branch")
        est = TimeEstimator().estimate(sample_result, 237_871, 122_342, 1,
                                       model_info=sample_model_info)
        if "flops_train_gflops_per_epoch" not in est:
            pytest.skip("FLOPs estimates not in this version")
        train_gflops = sample_model_info.flops_per_image_mflops / 1000 * 3 * 237_871
        assert est["flops_train_gflops_per_epoch"] == pytest.approx(train_gflops, rel=1e-4)

    def test_optimizer_steps_correct(self, sample_result):
        est = TimeEstimator().estimate(sample_result, 237_871, 122_342, 1)
        if "optimizer_steps_per_epoch" not in est:
            pytest.skip("optimizer_steps not in this version")
        expected = math.ceil(237_871 / 32)
        assert est["optimizer_steps_per_epoch"] == expected

    def test_no_flat_ddp_projection(self, sample_result):
        # DDP scaling is owned by the compute/IO/sync-aware DDPOptimizer, not by a
        # flat-efficiency projection in the TimeEstimator (which was removed).
        est = TimeEstimator().estimate(sample_result, 237_871, 122_342, 30)
        assert "ddp_total_2gpu_h" not in est
        assert "ddp_total_4gpu_h" not in est

    def test_format_time_hours(self):
        assert TimeEstimator.format_time(7200) == "2h 00m"

    def test_format_time_minutes(self):
        assert TimeEstimator.format_time(90) == "0h 01m"

    def test_format_time_zero(self):
        assert "0h" in TimeEstimator.format_time(0)


class TestModelInfo:
    def test_total_static_mb(self, sample_model_info):
        expected = (sample_model_info.weight_mb +
                    sample_model_info.gradient_mb +
                    sample_model_info.optimizer_mb)
        assert sample_model_info.total_static_mb == expected

    def test_total_mb_increases_with_batch(self, sample_model_info):
        mb_32 = sample_model_info.total_mb(32)
        mb_64 = sample_model_info.total_mb(64)
        assert mb_64 > mb_32
        assert mb_64 - mb_32 == pytest.approx(
            32 * sample_model_info.activation_mb_per_image, rel=1e-4
        )
