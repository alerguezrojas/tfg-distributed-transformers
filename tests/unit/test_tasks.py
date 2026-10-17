"""Unit tests for the task abstraction (src/training/tasks.py)."""
import pytest
import torch
import torch.nn as nn

from src.training.losses import FocalLoss
from src.training.tasks import (
    BinaryTask, MultiClassTask, MultiLabelTask, TASKS, get_task,
)


class TestRegistry:
    def test_known_tasks(self):
        assert set(TASKS) == {"multilabel", "binary", "multiclass"}
        assert isinstance(get_task("multiclass"), MultiClassTask)

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="unknown task"):
            get_task("regression")


class TestMultiLabel:
    task = MultiLabelTask()

    def test_default_loss_is_plain_bce(self):
        crit = self.task.build_criterion()
        assert isinstance(crit, nn.BCEWithLogitsLoss) and crit.pos_weight is None

    def test_focal_and_pos_weight(self):
        assert isinstance(self.task.build_criterion("focal"), FocalLoss)
        pw = torch.ones(3)
        assert self.task.build_criterion("bce", pos_weight=pw).pos_weight is pw

    def test_rejects_cross_entropy(self):
        with pytest.raises(ValueError, match="not valid for a multilabel"):
            self.task.build_criterion("ce")

    def test_smoothing_goes_towards_half(self):
        out = self.task.smooth_targets(torch.tensor([[1.0, 0.0]]), 0.1)
        assert out[0].tolist() == pytest.approx([0.95, 0.05])

    def test_independent_labels(self):
        probs = self.task.probabilities(torch.tensor([[2.0, 2.0, -2.0]]))
        assert self.task.predict(probs).tolist() == [[1, 1, 0]]   # several at once

    def test_accuracy_is_label_averaged(self):
        preds = torch.tensor([[1, 0, 0, 0]])
        hard = torch.tensor([[1, 1, 0, 0]])
        assert self.task.accuracy(preds, hard) == pytest.approx(0.75)

    def test_threshold_search_finds_lower_threshold(self):
        probs = torch.tensor([[0.3, 0.1], [0.1, 0.3]])
        hard = torch.tensor([[1, 0], [0, 1]])
        t, f1 = self.task.search_threshold(probs, hard)
        assert t < 0.3 and f1 == pytest.approx(1.0, abs=1e-6)

    def test_threshold_search_keeps_half_on_ties(self):
        probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])   # perfect at 0.5 already
        hard = torch.tensor([[1, 0], [0, 1]])
        assert self.task.search_threshold(probs, hard)[0] == 0.5


class TestBinary:
    def test_is_single_column_multilabel(self):
        task = BinaryTask()
        assert task.sigmoid_outputs and task.searches_threshold
        probs = task.probabilities(torch.tensor([[3.0], [-3.0]]))
        assert task.predict(probs).tolist() == [[1], [0]]
        m = task.metrics(task.predict(probs), torch.tensor([[1], [0]]))
        assert m["f1"] == pytest.approx(1.0, abs=1e-6)
        assert m["accuracy"] == pytest.approx(1.0)


class TestMultiClass:
    task = MultiClassTask()

    def test_loss_is_cross_entropy(self):
        assert isinstance(self.task.build_criterion(), nn.CrossEntropyLoss)

    @pytest.mark.parametrize("loss", ["bce", "focal"])
    def test_rejects_sigmoid_losses(self, loss):
        with pytest.raises(ValueError, match="use 'ce'"):
            self.task.build_criterion(loss)

    def test_rejects_pos_weight(self):
        with pytest.raises(ValueError, match="pos_weight"):
            self.task.build_criterion(pos_weight=torch.ones(3))

    def test_cross_entropy_accepts_soft_targets(self):
        crit = self.task.build_criterion()
        logits = torch.tensor([[2.0, 0.0, 0.0]])
        soft = self.task.smooth_targets(torch.tensor([[1.0, 0.0, 0.0]]), 0.3)
        assert torch.isfinite(crit(logits, soft))

    def test_smoothing_keeps_a_distribution(self):
        out = self.task.smooth_targets(torch.tensor([[0.0, 1.0, 0.0, 0.0]]), 0.2)
        assert out.sum().item() == pytest.approx(1.0)
        assert out[0].tolist() == pytest.approx([0.05, 0.85, 0.05, 0.05])

    def test_exactly_one_prediction(self):
        probs = self.task.probabilities(torch.tensor([[0.1, 2.0, 0.3], [5.0, 0.0, 0.0]]))
        assert probs.sum(dim=1).tolist() == pytest.approx([1.0, 1.0])
        assert self.task.predict(probs).tolist() == [[0, 1, 0], [1, 0, 0]]

    def test_accuracy_is_exact_match(self):
        preds = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]])
        hard = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0]])
        # multi-label averaging would give 0.875; exact match is 1 of 2
        assert self.task.accuracy(preds, hard) == pytest.approx(0.5)

    def test_hard_targets_from_soft(self):
        assert self.task.hard_targets(torch.tensor([[0.2, 0.7, 0.1]])).tolist() == [[0, 1, 0]]

    def test_no_threshold_search(self):
        assert self.task.search_threshold(torch.rand(4, 3), torch.eye(3)[[0, 1, 2, 0]]) is None


@pytest.mark.parametrize("task, row_sum", [(MultiLabelTask(), None), (MultiClassTask(), 1.0)])
def test_random_targets_have_the_task_format(task, row_sum):
    y = task.random_targets(16, 5)
    assert y.shape == (16, 5) and y.dtype == torch.float32
    assert set(y.unique().tolist()) <= {0.0, 1.0}
    if row_sum is not None:
        assert (y.sum(dim=1) == row_sum).all()
