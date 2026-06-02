"""Unit tests for src/training/metrics.py"""
import pytest
import torch
from src.training import metrics as m


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _preds(data):
    return torch.tensor(data, dtype=torch.long)

def _labels(data):
    return torch.tensor(data, dtype=torch.float)


# ── f1_score ──────────────────────────────────────────────────────────────────

class TestF1Score:
    def test_perfect(self):
        p = _preds([[1, 0], [0, 1]])
        l = _labels([[1, 0], [0, 1]])
        assert m.f1_score(p, l) == pytest.approx(1.0, abs=1e-4)

    def test_all_zero_preds(self):
        p = _preds([[0, 0], [0, 0]])
        l = _labels([[1, 0], [0, 1]])
        # TP=0, FP=0, FN=2 → F1=0 for each class → macro=0
        assert m.f1_score(p, l) == pytest.approx(0.0, abs=1e-4)

    def test_partial(self):
        # Class 0: TP=1,FP=0,FN=1 → P=1,R=0.5,F1=2/3
        # Class 1: TP=1,FP=1,FN=0 → P=0.5,R=1,F1=2/3
        p = _preds([[1, 1], [0, 1]])
        l = _labels([[1, 0], [1, 1]])
        score = m.f1_score(p, l)
        assert 0.0 < score <= 1.0

    def test_multi_label(self):
        p = _preds([[1, 1, 0], [0, 1, 1]])
        l = _labels([[1, 0, 0], [0, 1, 1]])
        score = m.f1_score(p, l)
        assert 0.0 < score <= 1.0

    def test_single_class(self):
        p = _preds([[1], [0], [1]])
        l = _labels([[1], [1], [0]])
        score = m.f1_score(p, l)
        assert 0.0 <= score <= 1.0


# ── accuracy ─────────────────────────────────────────────────────────────────

class TestAccuracy:
    def test_perfect(self):
        p = _preds([[1, 0], [0, 1]])
        l = _labels([[1, 0], [0, 1]])
        assert m.accuracy(p, l) == pytest.approx(1.0, abs=1e-4)

    def test_zero(self):
        p = _preds([[0, 1], [1, 0]])
        l = _labels([[1, 0], [0, 1]])
        assert m.accuracy(p, l) == pytest.approx(0.0, abs=1e-4)

    def test_partial(self):
        # Sample 0: correct=[1,1] → exact match. Sample 1: correct=[1,0] → no match
        p = _preds([[1, 0], [1, 0]])
        l = _labels([[1, 0], [0, 1]])
        assert m.accuracy(p, l) == pytest.approx(0.5, abs=1e-4)

    def test_returns_float(self):
        p = _preds([[1, 0]])
        l = _labels([[1, 0]])
        result = m.accuracy(p, l)
        assert isinstance(result, float)


# ── precision ─────────────────────────────────────────────────────────────────

class TestPrecision:
    def test_perfect(self):
        p = _preds([[1, 0], [0, 1]])
        l = _labels([[1, 0], [0, 1]])
        assert m.precision(p, l) == pytest.approx(1.0, abs=1e-4)

    def test_all_false_positives(self):
        p = _preds([[1, 1], [1, 1]])
        l = _labels([[0, 0], [0, 0]])
        assert m.precision(p, l) == pytest.approx(0.0, abs=1e-4)

    def test_no_predictions(self):
        p = _preds([[0, 0], [0, 0]])
        l = _labels([[1, 1], [1, 1]])
        # TP=0, FP=0 → precision = 0 / (0+eps) ≈ 0
        result = m.precision(p, l)
        assert 0.0 <= result <= 1.0


# ── recall ────────────────────────────────────────────────────────────────────

class TestRecall:
    def test_perfect(self):
        p = _preds([[1, 0], [0, 1]])
        l = _labels([[1, 0], [0, 1]])
        assert m.recall(p, l) == pytest.approx(1.0, abs=1e-4)

    def test_all_missed(self):
        p = _preds([[0, 0], [0, 0]])
        l = _labels([[1, 1], [1, 1]])
        assert m.recall(p, l) == pytest.approx(0.0, abs=1e-4)

    def test_symmetry_with_precision_on_perfect(self):
        p = _preds([[1, 1], [0, 0]])
        l = _labels([[1, 1], [0, 0]])
        assert m.precision(p, l) == pytest.approx(m.recall(p, l), abs=1e-4)


# ── eta_str ───────────────────────────────────────────────────────────────────

class TestEtaStr:
    def test_returns_string(self):
        # eta_str(epoch_times: list, epochs_done: int, epochs_total: int)
        result = m.eta_str([60.0, 70.0], 2, 10)
        assert isinstance(result, str)

    def test_contains_time_unit(self):
        result = m.eta_str([3600.0], 1, 5)
        assert "h" in result

    def test_empty_times_returns_question_mark(self):
        result = m.eta_str([], 0, 10)
        assert result == "?"

    def test_completed_returns_zero_time(self):
        result = m.eta_str([60.0], 10, 10)
        assert "0h" in result
