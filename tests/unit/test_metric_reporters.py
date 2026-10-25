"""Metric reporters: cachean la métrica de train y emiten train/val tras eval_epoch,
al logger si se les da uno y a stdout si no."""
import logging

from src.training.base_trainer import BaseTrainer
from src.training.decorators.metric_reporters import (
    AccuracyReporter,
    F1Reporter,
    LossReporter,
    PrecisionRecallReporter,
)


class _Fake(BaseTrainer):
    def __init__(self, train=None, val=None):
        self.train = train or {}
        self.val = val or {}

    def train_epoch(self, loader):
        return dict(self.train)

    def eval_epoch(self, loader):
        return dict(self.val)

    def save_checkpoint(self, epoch, metrics):
        pass

    def fit(self, train_loader, val_loader, epochs):
        pass


def _epoch(reporter):
    return reporter.train_epoch(None), reporter.eval_epoch(None)


def test_loss_reporter_prints_train_and_val(capsys):
    rep = LossReporter(_Fake({"loss": 0.5}, {"loss": 0.25}))
    _epoch(rep)
    assert capsys.readouterr().out == f"  {'loss':<14} train=0.5000  val=0.2500\n"


def test_each_train_val_reporter_reads_its_own_key(capsys):
    fake = _Fake({"f1": 0.1, "accuracy": 0.2, "loss": 0.9},
                 {"f1": 0.3, "accuracy": 0.4, "loss": 0.8})
    _epoch(F1Reporter(fake))
    _epoch(AccuracyReporter(fake))
    out = capsys.readouterr().out
    assert f"  {'f1':<14} train=0.1000  val=0.3000" in out
    assert f"  {'accuracy':<14} train=0.2000  val=0.4000" in out
    assert "loss" not in out


def test_val_only_when_no_train_value_cached(capsys):
    rep = F1Reporter(_Fake(val={"f1": 0.7}))
    rep.eval_epoch(None)                       # sin train_epoch previo
    assert capsys.readouterr().out == f"  {'f1':<14} val=0.7000\n"


def test_silent_when_val_lacks_the_key(capsys):
    rep = LossReporter(_Fake({"loss": 1.0}, {"f1": 0.5}))
    _, val = _epoch(rep)
    assert capsys.readouterr().out == ""
    assert val == {"f1": 0.5}


def test_reporters_do_not_alter_results():
    rep = AccuracyReporter(_Fake({"accuracy": 0.2, "x": 1}, {"accuracy": 0.4, "y": 2}))
    train, val = _epoch(rep)
    assert train == {"accuracy": 0.2, "x": 1}
    assert val == {"accuracy": 0.4, "y": 2}


def test_precision_recall_reporter(capsys):
    rep = PrecisionRecallReporter(_Fake(val={"precision": 0.6, "recall": 0.8}))
    rep.eval_epoch(None)
    assert capsys.readouterr().out == (
        f"  {'precision':<14} val=0.6000\n  {'recall':<14} val=0.8000\n")


def test_precision_recall_skips_missing_values(capsys):
    PrecisionRecallReporter(_Fake(val={"recall": 0.8})).eval_epoch(None)
    assert capsys.readouterr().out == f"  {'recall':<14} val=0.8000\n"


def test_reporter_uses_logger_instead_of_stdout(capsys, caplog):
    logger = logging.getLogger("test-reporters")
    rep = LossReporter(_Fake({"loss": 0.5}, {"loss": 0.25}), logger=logger)
    with caplog.at_level(logging.INFO, logger="test-reporters"):
        _epoch(rep)
    assert capsys.readouterr().out == ""
    assert "train=0.5000  val=0.2500" in caplog.text
