"""Unit tests for scripts/eval.py — the held-out test-set evaluator.

The script was broken (it called build_model with an unsupported ``dropout``
kwarg) and had never been run. These tests lock in the pure helpers so the
test-set number that feeds the dashboard stays correct.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location("eval_script", ROOT / "scripts" / "eval.py")
eval_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_script)


def test_threshold_search_picks_best_macro_f1():
    # Both classes recoverable (each positive in one row) → macro F1 = 1 at a
    # threshold between the low and high probs.
    probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    labels = torch.tensor([[1, 0], [0, 1]])
    t, f1 = eval_script.threshold_search(probs, labels)
    assert f1 > 0.99
    assert t < 0.9


def test_per_class_metrics_shape_and_perfect_class():
    probs = torch.tensor([[0.9, 0.1], [0.7, 0.2], [0.95, 0.05]])
    labels = torch.tensor([[1, 0], [1, 0], [1, 0]])
    rows = eval_script.per_class_metrics(probs, labels, threshold=0.5)
    assert len(rows) == 2
    # class 0 is always present and always predicted → F1 = 1
    assert rows[0]["f1"] > 0.99
    # class 1 never present nor predicted → F1 = 0
    assert rows[1]["f1"] == 0.0
    assert set(rows[0]) == {"class_idx", "class_name", "f1", "precision", "recall"}


def test_evaluate_respects_max_batches():
    class _Tiny(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 2)

    # 5 batches of 4 available; cap at 2.
    data = [(torch.zeros(4, 3, 8, 8), torch.zeros(4, 2)) for _ in range(5)]
    probs, labels, loss = eval_script.evaluate(_Tiny(), data, torch.device("cpu"), max_batches=2)
    assert probs.shape[0] == 8           # 2 batches × 4
    assert labels.shape[0] == 8
    assert loss == loss                  # not NaN (divided by n_batches, not len)


class TestChooseThreshold:
    """The test number must use a threshold tuned on validation, never on test."""

    probs = torch.tensor([[0.3, 0.1], [0.1, 0.3]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    def _choose(self, source, ckpt_metrics, val_outputs=None):
        from src.training.tasks import MultiLabelTask
        return eval_script.choose_threshold(source, ckpt_metrics, MultiLabelTask(),
                                            self.probs, self.labels, val_outputs)

    def test_uses_the_threshold_stored_in_the_checkpoint(self):
        def fail():
            raise AssertionError("validation should not be evaluated")
        assert self._choose("checkpoint", {"optimal_threshold": 0.35}, fail) == (0.35, "checkpoint")

    def test_old_checkpoint_falls_back_to_validation(self):
        val = (torch.tensor([[0.45, 0.05]]), torch.tensor([[1.0, 0.0]]))
        t, source = self._choose("checkpoint", {}, lambda: val)
        assert source == "val"
        assert t < 0.45                     # tuned on the validation outputs

    def test_test_source_is_explicit_and_labelled(self):
        t, source = self._choose("test", {"optimal_threshold": 0.5})
        assert source == "test"
        assert t < 0.3                      # tuned on the evaluated split


# ── Ensemble of checkpoints ──────────────────────────────────────────────────────

class _Lin(torch.nn.Module):
    def __init__(self, n_out=3):
        super().__init__()
        self.fc = torch.nn.Linear(4, n_out)

    def forward(self, x):
        return self.fc(x)


def _save_member(path, seed, epoch=1, metrics=None, prefix="", n_out=3):
    torch.manual_seed(seed)
    model = _Lin(n_out)
    sd = {prefix + k: v for k, v in model.state_dict().items()}
    torch.save({"model_state_dict": sd, "epoch": epoch, "metrics": metrics or {}}, path)
    return model


def _data():
    g = torch.Generator().manual_seed(0)
    return [(torch.randn(5, 4, generator=g), (torch.rand(5, 3, generator=g) > 0.5).float())
            for _ in range(3)]


def _members(monkeypatch, paths, n_outs=None):
    from src.training.tasks import MultiLabelTask
    outs = iter(n_outs or [3] * len(paths))
    monkeypatch.setattr(eval_script, "build_eval_model",
                        lambda cfg, name, spec: _Lin(next(outs)))
    return eval_script.evaluate_members(paths, {}, "tiny", None, _data(), torch.device("cpu"),
                                        MultiLabelTask())


def test_average_probabilities_is_the_arithmetic_mean():
    a, b = torch.tensor([[0.2, 0.8]]), torch.tensor([[0.6, 0.4]])
    y = torch.tensor([[0.0, 1.0]])
    probs, labels = eval_script.average_probabilities([a, b], [y, y.clone()])
    assert torch.allclose(probs, torch.tensor([[0.4, 0.6]]))
    assert torch.equal(labels, y)


def test_average_probabilities_single_member_is_untouched():
    a, y = torch.tensor([[0.2, 0.8]]), torch.tensor([[0.0, 1.0]])
    probs, _ = eval_script.average_probabilities([a], [y])
    assert probs is a


def test_average_probabilities_rejects_different_shapes_and_targets():
    import pytest
    a, y = torch.zeros(2, 3), torch.zeros(2, 3)
    with pytest.raises(ValueError, match="forma"):
        eval_script.average_probabilities([a, torch.zeros(2, 4)], [y, y])
    with pytest.raises(ValueError, match="objetivos"):
        eval_script.average_probabilities([a, a], [y, torch.ones(2, 3)])


def test_ensemble_is_the_mean_of_the_individual_probabilities(tmp_path, monkeypatch):
    p1, p2 = tmp_path / "a.pt", tmp_path / "b.pt"
    _save_member(p1, 1)
    _save_member(p2, 2, prefix="module.")          # a DDP checkpoint loads too
    pa, _, la, _ = _members(monkeypatch, [p1])
    pb, _, lb, _ = _members(monkeypatch, [p2])
    pe, labels, le, infos = _members(monkeypatch, [p1, p2])
    assert torch.allclose(pe, (pa + pb) / 2)
    assert le == (la + lb) / 2
    assert len(infos) == 2 and labels.shape == (15, 3)


def test_single_checkpoint_matches_plain_evaluation(tmp_path, monkeypatch):
    from src.training.tasks import MultiLabelTask
    p = tmp_path / "a.pt"
    model = _save_member(p, 1)
    probs, labels, loss, infos = _members(monkeypatch, [p])
    ref_p, ref_l, ref_loss = eval_script.evaluate(model, _data(), torch.device("cpu"),
                                                  task=MultiLabelTask())
    assert torch.equal(probs, ref_p) and torch.equal(labels, ref_l) and loss == ref_loss
    assert infos == [{"epoch": 1, "metrics": {}}]


def test_ensemble_with_different_targets_fails_clearly(tmp_path, monkeypatch):
    import pytest
    from src.training.tasks import MultiLabelTask
    p1, p2 = tmp_path / "a.pt", tmp_path / "b.pt"
    _save_member(p1, 1)
    _save_member(p2, 2)
    monkeypatch.setattr(eval_script, "build_eval_model", lambda cfg, name, spec: _Lin())
    loaders = iter([_data(), [(x, 1 - y) for x, y in _data()]])    # the 2nd sees other targets
    real = eval_script.evaluate
    monkeypatch.setattr(eval_script, "evaluate",
                        lambda model, loader, *a, **k: real(model, next(loaders), *a, **k))
    with pytest.raises(ValueError, match="objetivos"):
        eval_script.evaluate_members([p1, p2], {}, "tiny", None, None, torch.device("cpu"),
                                     MultiLabelTask())


def test_stored_threshold_metrics_single_and_ensemble():
    one = [{"epoch": 1, "metrics": {"optimal_threshold": 0.3}}]
    assert eval_script.stored_threshold_metrics(one) == {"optimal_threshold": 0.3}
    two = one + [{"epoch": 2, "metrics": {"optimal_threshold": 0.5}}]
    assert eval_script.stored_threshold_metrics(two)["optimal_threshold"] == 0.4
    assert eval_script.stored_threshold_metrics(one + [{"epoch": 3, "metrics": {}}]) == {}


# ── Modelo efectivo: el checkpoint dice qué modelo contiene ───────────────────

def _save_named(path, name, seed=1):
    torch.manual_seed(seed)
    ckpt = {"model_state_dict": _Lin(3).state_dict(), "epoch": 1, "metrics": {}}
    if name is not None:
        ckpt["model_name"] = name
    torch.save(ckpt, path)


CFG = {"model": {"name": "from_config"}}


def test_eval_infers_the_model_from_the_checkpoint(tmp_path):
    p = tmp_path / "a.pt"
    _save_named(p, "swin_tiny")
    assert eval_script.resolve_model_name(None, CFG, [p]) == "swin_tiny"
    assert eval_script.resolve_model_name("swin_tiny", CFG, [p]) == "swin_tiny"


def test_eval_rejects_a_model_that_contradicts_the_checkpoint(tmp_path):
    p = tmp_path / "a.pt"
    _save_named(p, "swin_tiny")
    with pytest.raises(SystemExit, match="swin_tiny"):
        eval_script.resolve_model_name("vit_small", CFG, [p])
    with pytest.raises(SystemExit, match="swin_tiny"):    # también al cargar el miembro
        eval_script.load_member(p, CFG, "vit_small", None, torch.device("cpu"))


def test_old_checkpoint_without_model_name_uses_the_config(tmp_path, monkeypatch):
    p = tmp_path / "old.pt"
    _save_named(p, None)
    assert eval_script.resolve_model_name(None, CFG, [p]) == "from_config"
    assert eval_script.resolve_model_name("forced", CFG, [p]) == "forced"
    monkeypatch.setattr(eval_script, "build_eval_model", lambda cfg, name, spec: _Lin(3))
    _, ckpt = eval_script.load_member(p, CFG, "from_config", None, torch.device("cpu"))
    assert "model_name" not in ckpt


def test_ensemble_mixing_models_fails_unless_all_agree(tmp_path):
    a, b, c = tmp_path / "a.pt", tmp_path / "b.pt", tmp_path / "c.pt"
    _save_named(a, "swin_tiny")
    _save_named(b, "vit_small")
    _save_named(c, "swin_tiny", seed=2)
    with pytest.raises(SystemExit, match="distintos"):
        eval_script.resolve_model_name(None, CFG, [a, b])
    assert eval_script.resolve_model_name(None, CFG, [a, c]) == "swin_tiny"
