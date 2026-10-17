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
    MODEL_TABLE, CONV_ACTIVATION_FACTOR, estimate_act_gb_per_img, predict, spec_from_profile,
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


@pytest.mark.parametrize("name", list(MODEL_TABLE))
def test_table_predictions_unchanged(name):
    kw = {"n_gpus": 2, "dataset_size": 5000, "batch": 96, "precision": "amp", "epochs": 15}
    assert predict("ddp", name, "Tesla T4", **kw) == predict(
        "ddp", name, "Tesla T4", model=resolve_model_spec(name), **kw)


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
