"""Clinical cut-offs, metrics per group and the predictions .npz (src/analysis/clinical.py, io.py).

The fixture is small enough to compute by hand: 4 samples, 3 ordered classes.
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.clinical import clinical_cutoffs, group_summary  # noqa: E402
from src.analysis.io import load_predictions, save_predictions  # noqa: E402
from src.training import metrics as m  # noqa: E402

_spec = importlib.util.spec_from_file_location("eval_script", ROOT / "scripts" / "eval.py")
eval_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_script)

CLASSES = ["g0", "g1", "g2"]
PROBS = np.array([[0.8, 0.1, 0.1],     # grade 0, predicted 0
                  [0.2, 0.6, 0.2],     # grade 0, predicted 1 (false positive at k=1)
                  [0.1, 0.7, 0.2],     # grade 1, predicted 1
                  [0.1, 0.2, 0.7]])    # grade 2, predicted 2
TARGETS = np.eye(3)[[0, 0, 1, 2]]


def test_cutoffs_by_hand():
    k1, k2 = clinical_cutoffs(PROBS, TARGETS, CLASSES)
    assert (k1["k"], k1["label"], k1["n_pos"], k1["n_neg"]) == (1, ">= g1", 2, 2)
    assert (k1["tp"], k1["fp"], k1["fn"], k1["tn"]) == (2, 1, 0, 1)
    assert k1["sensitivity"] == 1.0 and k1["specificity"] == 0.5
    assert k1["ppv"] == pytest.approx(2 / 3) and k1["npv"] == 1.0
    assert k1["auc"] == 1.0          # scores P(>=1) = .2 .8 .9 .9: both positives above negatives
    assert (k2["label"], k2["n_pos"], k2["tp"], k2["fp"], k2["tn"]) == (">= g2", 1, 1, 0, 3)
    assert k2["sensitivity"] == k2["specificity"] == k2["ppv"] == k2["npv"] == 1.0


def test_cutoff_auc_uses_the_score_not_the_argmax():
    # The positive (grade 1) is predicted 0 by argmax, but P(>=1) = 0.45 beats the negative's 0.3.
    probs = np.array([[0.55, 0.40, 0.05], [0.70, 0.20, 0.10]])
    targets = np.eye(3)[[1, 0]]
    (k1, _) = clinical_cutoffs(probs, targets)
    assert k1["tp"] == 0 and k1["fn"] == 1 and k1["sensitivity"] == 0.0
    assert math.isnan(k1["ppv"])          # no predicted positives
    assert k1["auc"] == 1.0


def test_cutoff_without_positives_is_nan():
    (k1, k2) = clinical_cutoffs(PROBS[:2], TARGETS[:2], CLASSES)
    assert math.isnan(k1["sensitivity"]) and math.isnan(k1["auc"])
    assert k1["n_pos"] == 0 and k2["n_pos"] == 0


def test_cutoffs_need_two_classes():
    with pytest.raises(ValueError):
        clinical_cutoffs(np.ones((3, 1)), np.ones((3, 1)))


def test_group_summary_by_hand():
    rows = group_summary(PROBS, TARGETS, "multiclass", {"centre": np.array(["x", "x", "y", "y"])},
                         ordinal=True)
    x, y = rows
    assert (x["variable"], x["group"], x["n"]) == ("centre", "x", 2)
    assert x["accuracy"] == 0.5 and x["qwk"] == pytest.approx(0.0)
    assert math.isnan(x["auc"])           # one grade only: no class has both positives and negatives
    assert x["f1"] == pytest.approx(2 / 3)    # class 0 (P 1, R .5); the other F1 are undefined
    assert (y["group"], y["n"], y["accuracy"]) == ("y", 2, 1.0)
    assert y["qwk"] == pytest.approx(1.0) and y["f1"] == pytest.approx(1.0)


def test_group_summary_qwk_matches_the_torch_metric():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(6), size=200)
    targets = np.eye(6)[rng.integers(0, 6, 200)]
    (row,) = group_summary(probs, targets, "multiclass", {"all": np.zeros(200, int)}, ordinal=True)
    expected = m.quadratic_weighted_kappa(torch.eye(6)[torch.as_tensor(probs.argmax(1))],
                                          torch.as_tensor(targets))
    assert row["qwk"] == pytest.approx(expected)
    assert "qwk" not in group_summary(probs, targets, "multiclass", {"all": np.zeros(200)})[0]


def test_group_summary_multilabel_uses_the_threshold():
    probs = np.array([[0.9, 0.2], [0.6, 0.4], [0.3, 0.8], [0.1, 0.1]])
    targets = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    (row,) = group_summary(probs, targets, "multilabel", {"g": np.array(["a"] * 4)}, threshold=0.5)
    assert row["accuracy"] == pytest.approx(7 / 8)      # one wrong label out of eight
    assert row["f1"] == pytest.approx((1.0 + 2 / 3) / 2)


def test_group_summary_validates_inputs():
    with pytest.raises(ValueError):
        group_summary(PROBS, TARGETS, "multiclass", {"g": np.zeros(3)})
    with pytest.raises(ValueError):
        group_summary(PROBS, TARGETS, "multilabel", {"g": np.zeros(4)}, ordinal=True)


def test_predictions_roundtrip(tmp_path):
    out = save_predictions(tmp_path / "sub" / "p.npz", PROBS, TARGETS, 0.5, CLASSES, "multiclass",
                           True, {"centre": np.array(["x", "x", "y", "y"])})
    p = load_predictions(out)
    assert p.probs.shape == (4, 3) and p.probs.dtype == np.float32
    assert p.targets.dtype == np.uint8 and p.targets.sum() == 4
    assert (p.threshold, p.classes, p.task, p.ordinal) == (0.5, CLASSES, "multiclass", True)
    assert list(p.groups["centre"]) == ["x", "x", "y", "y"]
    # What the file holds is enough to redo the analysis offline.
    assert [c["k"] for c in clinical_cutoffs(p.probs, p.targets, p.classes)] == [1, 2]


def test_predictions_without_groups_and_bad_shapes(tmp_path):
    out = save_predictions(tmp_path / "p.npz", PROBS, TARGETS, 0.3, CLASSES, "multiclass")
    p = load_predictions(out)
    assert p.groups == {} and p.ordinal is False and p.threshold == pytest.approx(0.3)
    with pytest.raises(ValueError):
        save_predictions(tmp_path / "q.npz", PROBS, TARGETS[:3], 0.5, CLASSES, "multiclass")
    with pytest.raises(ValueError):
        save_predictions(tmp_path / "q.npz", PROBS, TARGETS, 0.5, CLASSES, "multiclass",
                         groups={"g": np.zeros(3)})


# ── eval.py: dataset hooks and printed sections ────────────────────────────────

class _DS:
    def groups(self):
        return {"data_provider": np.array(["x", "x", "y", "y", "y"])}


def test_dataset_groups_duck_typing_and_truncation():
    assert eval_script.dataset_groups(object(), 4) == {}
    g = eval_script.dataset_groups(_DS(), 4)             # e.g. --max-batches stopped early
    assert list(g["data_provider"]) == ["x", "x", "y", "y"]


def test_eval_prints_group_and_cutoff_sections(capsys):
    groups = {"centre": np.array(["x", "x", "y", "y"])}
    eval_script.print_group_metrics(PROBS, TARGETS, "multiclass", groups, 0.5, True)
    eval_script.print_clinical_cutoffs(PROBS, TARGETS, CLASSES)
    out = capsys.readouterr().out
    assert "── Metrics by group: centre" in out and "QWK" in out
    assert "── Clinical cut-offs" in out and ">= g1" in out and ">= g2" in out
    eval_script.print_group_metrics(PROBS, TARGETS, "multiclass", {}, 0.5, True)
    assert capsys.readouterr().out == ""                 # no groups -> no section
