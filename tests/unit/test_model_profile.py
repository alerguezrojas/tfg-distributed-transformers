"""Unit tests for src/model_profile.py — automatic params/FLOPs for any timm model.

All on CPU (the meta device, no weights) and without network (pretrained=False).
The profile must agree with the curated MODEL_TABLE, so that a model outside the
table is measured in the same units the predictor was calibrated with.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

pytest.importorskip("timm")

from src import model_profile
from src.model_profile import profile_model, resolve_model_spec
from src.performance_model import (
    MODEL_TABLE, CONV_ACTIVATION_FACTOR, _CONF_BAND, estimate_act_gb_per_img, expected_best_f1,
    predict, predict_quality, spec_from_profile,
)

# Profiled vs MODEL_TABLE, both with the 1000-class ImageNet head of timm's tables.
# Measured deviations: params ≤ 0.9 %, FLOPs ≤ 3.6 % (vit_tiny 1.25 vs 1.3 — the
# table rounds to one decimal).
_TOLERANCE = 0.05
_UNKNOWN_MODEL = "convnext_tiny"


@pytest.mark.parametrize("name", list(MODEL_TABLE))
def test_profile_matches_model_table(name):
    prof = profile_model(name, num_classes=1000)
    ref = MODEL_TABLE[name]
    assert prof.device == "meta"
    assert prof.params_m == pytest.approx(ref.params_m, rel=_TOLERANCE)
    assert prof.gflops_fwd == pytest.approx(ref.gflops_fwd, rel=_TOLERANCE)


def test_flops_unit_is_multiply_add_as_one():
    """FlopCounterMode gives vit_base ≈ 35.1 G; the table (and timm) say 17.6."""
    assert profile_model("vit_base_patch16_224").gflops_fwd == pytest.approx(17.6, rel=0.01)


def test_convolutional_flag():
    assert profile_model("resnet50").convolutional
    assert profile_model("efficientnet_b0").convolutional
    assert not profile_model("vit_base_patch16_224").convolutional   # only the patch embedding


def test_in_chans_one_works():
    rgb = profile_model("vit_tiny_patch16_224", in_chans=3)
    gray = profile_model("vit_tiny_patch16_224", in_chans=1)
    assert gray.in_chans == 1
    assert gray.params_m < rgb.params_m                       # smaller patch embedding
    assert gray.gflops_fwd == pytest.approx(rgb.gflops_fwd, rel=0.05)
    assert resolve_model_spec("resnet18", in_chans=1) is not None


def test_fixed_resolution_model_is_rebuilt_for_224():
    """A 384² ViT rejects 224² input as built; the profiler rebuilds it at 224."""
    prof = profile_model("vit_base_patch16_384", img_size=224)
    assert prof.gflops_fwd == pytest.approx(17.6, rel=0.02)


def test_cpu_fallback_when_meta_fails(monkeypatch):
    real_build = model_profile._build

    def build(name, device, *args):
        if device == "meta":
            raise NotImplementedError("no meta kernel")
        return real_build(name, device, *args)

    monkeypatch.setattr(model_profile, "_build", build)
    prof = model_profile.profile_model.__wrapped__("vit_tiny_patch16_224")
    assert prof.device == "cpu"
    # The forced math-attention kernel makes CPU count the same as meta.
    assert prof.gflops_fwd == pytest.approx(profile_model("vit_tiny_patch16_224").gflops_fwd, rel=1e-3)


def test_unknown_timm_model():
    with pytest.raises(ValueError):
        profile_model("definitely_not_a_timm_model")
    assert resolve_model_spec("definitely_not_a_timm_model") is None


# ── Resolver + predictor ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", list(MODEL_TABLE))
def test_table_models_resolve_to_table_entry(name):
    assert resolve_model_spec(name) is MODEL_TABLE[name]


# Pinned from develop's predictor (src/performance_model.py before model profiling), at
# dataset_size=5000, batch=96, epochs=15: (model, gpu, strategy, n_gpus, precision) →
# (time/epoch s, VRAM/GPU GB, energy/epoch Wh). The table path must not move.
_PINNED = [
    (("vit_base_patch16_224", "Tesla T4", "ddp", 2, "amp"), (40.7352, 4.8228, 2.0761)),
    (("resnet50", "Tesla V100", "single", 1, "fp32"), (38.4615, 10.6096, 1.6055)),
    (("vit_large_patch16_224", "Tesla T4", "model_parallel", 2, "fp32"), (673.0313, 30.408, 22.8917)),
    (("vit_tiny_patch16_224", "RTX 3060 Ti", "single", 1, "fp32"), (38.4615, 2.9952, 2.8058)),
    (("efficientnet_b0", "Tesla V100", "heterogeneous", 2, "fp32"), (112.8783, 3.5648, 3.7347)),
]


@pytest.mark.parametrize("combo,expected", _PINNED)
def test_table_predictions_unchanged(combo, expected):
    name, gpu, strategy, n, prec = combo
    kw = {"n_gpus": n, "dataset_size": 5000, "batch": 96, "precision": prec, "epochs": 15}
    for model in (None, resolve_model_spec(name)):            # by name, and via the resolver
        p = predict(strategy, name, gpu, model=model, **kw)
        assert (p.time_per_epoch_train_s, p.vram_per_gpu_gb, p.energy_per_epoch_wh) == \
            pytest.approx(expected, abs=1e-3)
        assert not any("profiled automatically" in note for note in p.notes)


# ── Exact lookup: names that merely CONTAIN a table key are not table models ──────

def test_lookalike_name_is_profiled_not_fuzzy_matched():
    spec = resolve_model_spec("wide_resnet50_2", num_classes=1000)
    assert spec is not None and not spec.calibrated
    assert spec.params_m == pytest.approx(68.9, rel=0.02)          # resnet50 would give 25.6
    assert spec.gflops_fwd == pytest.approx(11.4, rel=0.05)
    assert spec.name == "wide_resnet50_2"


def test_pretrained_tag_is_dropped_for_table_lookup():
    assert resolve_model_spec("resnet50.a1_in1k") is MODEL_TABLE["resnet50"]


# ── Overrides on a table model invalidate its table entry ─────────────────────────

def test_table_model_at_other_resolution_is_profiled():
    base = MODEL_TABLE["vit_base_patch16_224"]
    big = resolve_model_spec("vit_base_patch16_224", img_size=384)
    assert not big.calibrated
    assert big.gflops_fwd / base.gflops_fwd == pytest.approx(3.15, rel=0.05)   # timm quotes 55.5 GMACs at 384² vs 17.6


def test_table_model_with_other_channels_is_profiled():
    spec = resolve_model_spec("vit_base_patch16_224", in_chans=1)
    assert not spec.calibrated
    assert spec.params_m < MODEL_TABLE["vit_base_patch16_224"].params_m


def test_num_classes_alone_keeps_the_table_entry():
    assert resolve_model_spec("vit_base_patch16_224", num_classes=5) is MODEL_TABLE["vit_base_patch16_224"]


# ── Honesty: notes and quality confidence for uncalibrated specs ──────────────────

def test_uncalibrated_note_mentions_mfu():
    p = predict("single", _UNKNOWN_MODEL, "Tesla T4", model=resolve_model_spec(_UNKNOWN_MODEL))
    assert any("MFU" in n and "not on this architecture" in n for n in p.notes)


def test_quality_confidence_forced_low_for_uncalibrated():
    assert predict_quality("vit_base_patch16_224", 5000).confidence == "high"
    q = predict_quality("vit_base_patch16_224", 5000, calibrated=False)
    assert q.confidence == "low" and q.band == _CONF_BAND["low"]
    assert any("confidence forced to low" in n for n in q.notes)
    assert expected_best_f1("vit_base_patch16_224", 5000, calibrated=False)[1] == "low"


@pytest.mark.parametrize("name", [_UNKNOWN_MODEL, "deit_small_patch16_224"])
def test_unknown_model_gets_uncalibrated_prediction(name):
    assert predict("single", name, "Tesla T4") is None        # name alone: not in the table
    spec = resolve_model_spec(name)
    assert spec is not None and not spec.calibrated
    p = predict("single", name, "Tesla T4", model=spec)
    assert p is not None and p.time_per_epoch_train_s > 0 and p.vram_per_gpu_gb > 0
    assert any("profiled automatically" in n and "not calibrated" in n for n in p.notes)


def test_deit_small_profiles_like_vit_small():
    """Same architecture as vit_small → same FLOPs and (by rule) same activations."""
    deit = resolve_model_spec("deit_small_patch16_224")
    vit = MODEL_TABLE["vit_small_patch16_224"]
    assert deit.gflops_fwd == pytest.approx(vit.gflops_fwd, rel=0.02)
    assert deit.act_gb_per_img == pytest.approx(vit.act_gb_per_img, rel=0.05)


# ── Activation-memory rule (pure) ─────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["vit_tiny_patch16_224", "vit_small_patch16_224",
                                  "vit_base_patch16_224", "vit_large_patch16_224"])
def test_act_rule_reproduces_vit_entries(name):
    """Fed with the PROFILED FLOPs (what it gets in practice), the rule lands within
    3 % of the table (vit_tiny +3 %, on the safe side; base/large are its anchors)."""
    ref = MODEL_TABLE[name]
    act = estimate_act_gb_per_img(profile_model(name).gflops_fwd)
    assert act == pytest.approx(ref.act_gb_per_img, rel=0.035)


def test_act_rule_conv_factor_anchored_to_resnet50():
    ref = MODEL_TABLE["resnet50"]
    assert estimate_act_gb_per_img(ref.gflops_fwd, convolutional=True) == pytest.approx(
        ref.act_gb_per_img, rel=1e-6)
    assert 2.0 < CONV_ACTIVATION_FACTOR < 2.5


def test_act_rule_monotonic_and_spec_uncalibrated():
    acts = [estimate_act_gb_per_img(g) for g in (0.5, 2, 8, 17.6, 40, 100)]
    assert acts == sorted(acts)
    spec = spec_from_profile("x", 10.0, 2.0, convolutional=True)
    assert not spec.calibrated
    assert spec.act_gb_per_img == pytest.approx(estimate_act_gb_per_img(2.0, True))


# ── CLI ───────────────────────────────────────────────────────────────────────────

def test_cli_estimate_any_timm_model():
    from typer.testing import CliRunner
    from src.cli import app

    res = CliRunner().invoke(app, ["estimate", "--model", _UNKNOWN_MODEL, "--gpu", "Tesla T4"],
                             env={"COLUMNS": "240"})
    assert res.exit_code == 0, res.output
    assert "Unknown model" not in res.output
    assert "profiled automatically" in res.output
    assert "estimated by rule" in res.output

    res = CliRunner().invoke(app, ["estimate", "--model", "definitely_not_a_timm_model"],
                             env={"COLUMNS": "240"})
    assert "Unknown model" in res.output
