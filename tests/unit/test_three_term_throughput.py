"""Three-term throughput model in the predictor (src/performance_model.py, section 8 of
docs/performance_model.md): t = a·F + b·A + d/B, mixed precision only.

The F/A features are the ones the architecture study measured at 512 px (CPU, no GPU), so
these tests run without torch; the throughputs are the versioned measurements."""
import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.performance_model import (  # noqa: E402
    GPU_TABLE, MIXED_PRECISIONS, THROUGHPUT_FITS, THROUGHPUT_MEASURED, THROUGHPUT_MFU,
    THROUGHPUT_REFERENCE_GPU, THROUGHPUT_THREE_TERM, ModelSpec, estimate_rc, gpu_spec,
    model_spec, predict, three_term_ips, throughput_fit, train_throughput,
)

DATA = ROOT / "docs/research/datos/throughput_arquitecturas_amp.csv"
BATCH = 16
# (forward GFLOPs, saved activation GB) per image at 512 px — the study's features
# (scripts/predictor_architecture_study.py, output of 23/11/2026).
FEATURES_512 = {
    "vit_small_patch16_224": (31.7, 0.309),
    "deit3_small_patch16_224.fb_in22k_ft_in1k": (31.7, 0.347),
    "eva02_small_patch14_224.mim_in22k": (43.3, 0.522),
    "swin_tiny_patch4_window7_224": (25.5, 0.591),
    "convnext_tiny.fb_in22k": (23.3, 0.570),
    "resnet50": (21.4, 0.449),
    "efficientnet_b0": (2.0, 0.461),
    "tf_efficientnetv2_s": (14.9, 0.805),
    "vit_base_patch16_224.augreg2_in21k_ft_in1k": (107.0, 0.615),
    "efficientnet_b3": (5.0, 0.890),
    "convnext_small.fb_in22k": (45.4, 0.910),
    "maxvit_tiny_tf_512.in1k": (33.2, 1.877),
    "swinv2_tiny_window8_256": (23.7, 0.727),
}


def _measured(gpu: str) -> dict[str, float]:
    with open(DATA, newline="") as f:
        return {r["model"]: float(r["train_ips"]) for r in csv.DictReader(f)
                if r["gpu"] == gpu and r["img_size"] == "512"}


def _spec(name: str) -> ModelSpec:
    f, a = FEATURES_512[name]
    return ModelSpec(name, 20.0, f, 0.1, calibrated=False, saved_gb_per_img=a)


@pytest.mark.parametrize("gpu", list(THROUGHPUT_FITS))
def test_fit_reproduces_its_own_measurements(gpu):
    """Calibration (in-sample): each GPU's fit against the throughputs it was fitted on."""
    errs = []
    for name, real in _measured(gpu).items():
        ips, calibrated = three_term_ips(_spec(name), gpu_spec(gpu), BATCH)
        assert calibrated
        errs.append(abs(ips / real - 1))
    # 3060 Ti and T4 below 12 %; the V100 (26/11/2026) 12.3 %, with the Swin family and
    # EfficientNet-B0 as its worst cases.
    assert sum(errs) / len(errs) < 0.13
    assert max(errs) < 0.40          # the Swin family is the known worst case


def test_flops_only_is_far_worse_on_the_same_data():
    """Why the term exists: FLOPs only (MFU × π) misses the memory-bound architectures."""
    g = gpu_spec(THROUGHPUT_REFERENCE_GPU)
    errs = [abs(estimate_rc(_spec(n), g, "amp") / real - 1)
            for n, real in _measured(THROUGHPUT_REFERENCE_GPU).items()]
    assert sum(errs) / len(errs) > 1.0


def test_transfer_keeps_the_relative_cost_on_the_t4():
    """Validation (out-of-sample): the 3060 Ti fit moved to a T4 by the datasheets only.
    Absolute speed is off by a common factor (~1.5); the relative cost of architectures
    (error after removing the median ratio) stays small, including 3 unseen models."""
    ref = THROUGHPUT_FITS.pop("Tesla T4")      # pretend the T4 was never measured
    try:
        fit, calibrated = throughput_fit(gpu_spec("Tesla T4"))
        assert not calibrated
        ratios = {n: three_term_ips(_spec(n), gpu_spec("Tesla T4"), BATCH)[0] / real
                  for n, real in _measured("Tesla T4").items()}
    finally:
        THROUGHPUT_FITS["Tesla T4"] = ref
    common = sorted(ratios.values())[len(ratios) // 2]
    shape = [abs(r / common - 1) for r in ratios.values()]
    assert 1.2 < common < 1.8                   # optimistic, as documented (×1.49)
    assert sum(shape) / len(shape) < 0.15


def test_gpu_without_fit_uses_the_transferred_one():
    a100 = gpu_spec("A100 40GB")                     # no measurements of its own
    fit, calibrated = throughput_fit(a100)
    ref = THROUGHPUT_FITS[THROUGHPUT_REFERENCE_GPU]
    assert not calibrated
    assert fit.b_s_per_gb == pytest.approx(ref.b_s_per_gb * 448.0 / a100.mem_bw_gbs)
    assert fit.d_s_per_step == ref.d_s_per_step


def test_every_gpu_has_a_bandwidth():
    assert all(g.mem_bw_gbs > 0 for g in GPU_TABLE.values())


def test_bigger_batch_amortises_the_fixed_cost():
    m, g = _spec("efficientnet_b0"), gpu_spec("RTX 3060 Ti")
    assert three_term_ips(m, g, 64)[0] > three_term_ips(m, g, 8)[0]


def test_model_without_saved_figure_falls_back_to_mfu():
    m, g = ModelSpec("x", 20.0, 10.0, 0.1), gpu_spec("Tesla T4")
    assert three_term_ips(m, g, 16) is None
    ips, how, _ = train_throughput(m, g, "amp", 16)
    assert how == THROUGHPUT_MFU
    assert ips == pytest.approx(estimate_rc(m, g, "amp"))


@pytest.mark.parametrize("precision,expected", [("fp32", THROUGHPUT_MFU),
                                                 *[(p, THROUGHPUT_THREE_TERM) for p in MIXED_PRECISIONS]])
def test_three_term_only_in_mixed_precision(precision, expected):
    _, how, _ = train_throughput(model_spec("vit_base_patch16_224"), gpu_spec("Tesla T4"),
                                 precision, 32)
    assert how == expected


def test_measured_throughput_wins():
    g = gpu_spec("Tesla T4")
    ips, how, calibrated = train_throughput(model_spec("vit_base_patch16_224"), g, "amp", 32,
                                            rc_measured=26.0)
    assert how == THROUGHPUT_MEASURED and calibrated
    assert ips == pytest.approx(26.0 * g.tensor_speedup)


def test_prediction_reports_the_throughput_and_its_source():
    p = predict("single", "vit_base_patch16_224", "Tesla T4", batch=32, precision="amp")
    assert p.throughput_model == THROUGHPUT_THREE_TERM
    assert p.train_ips_per_gpu > 0
    assert p.t_compute_s == pytest.approx(5000 / p.train_ips_per_gpu)
    v = predict("single", "vit_base_patch16_224", "A100 40GB", batch=32, precision="amp")
    assert any("datasheet" in n for n in v.notes)
    v100 = predict("single", "vit_base_patch16_224", "Tesla V100", batch=32, precision="amp")
    assert not any("datasheet" in n for n in v100.notes)    # measured on VERODE (26/11/2026)


def test_vit_base_amp_on_t4_matches_the_measured_speedup():
    """Validation (out-of-sample, BigEarthNet 224 px): the T4 fit was made on synthetic 512 px
    batches; vit_base fp32 → AMP measured 3.80× on Kaggle (section 4)."""
    fp32 = predict("single", "vit_base_patch16_224", "Tesla T4", batch=32, precision="fp32")
    amp = predict("single", "vit_base_patch16_224", "Tesla T4", batch=32, precision="amp")
    assert fp32.train_ips_per_gpu * 3.80 == pytest.approx(amp.train_ips_per_gpu, rel=0.10)


def test_fixed_cost_per_step_matches_the_batch_sweep():
    """Calibration (in-sample) of d: RTX 3060 Ti, 224 px, AMP, batch 8/32/64 (logs/local/benchmark/
    benchmark_23112026_223*). The table models predict those throughputs within ~9 %, and the
    fitted d agrees with the per-step intercepts measured per model (12-20 ms)."""
    measured = {"vit_small_patch16_224": {8: 318.5, 32: 507.8, 64: 573.5},
                "resnet50": {8: 282.6, 32: 409.1, 64: 446.7},
                "efficientnet_b0": {8: 286.6, 32: 657.4, 64: 718.0}}
    g = gpu_spec("RTX 3060 Ti")
    errs = [abs(three_term_ips(model_spec(m), g, b)[0] / ips - 1)
            for m, by_batch in measured.items() for b, ips in by_batch.items()]
    assert sum(errs) / len(errs) < 0.12
    assert 0.013 <= THROUGHPUT_FITS["RTX 3060 Ti"].d_s_per_step <= 0.020
