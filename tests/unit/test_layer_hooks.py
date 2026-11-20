"""LayerHooksDecorator: hooks solo durante train_epoch, informe cada N epochs, salida
al logger "trainer" si tiene handlers y a stdout si no."""
import logging

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.decorators.layer_hooks import LayerHooksDecorator
from src.training.trainer import Trainer


@pytest.fixture(autouse=True)
def _no_trainer_handlers(monkeypatch):
    """Otros tests pueden dejar handlers en el logger global "trainer"; partimos sin ellos."""
    monkeypatch.setattr(logging.getLogger("trainer"), "handlers", [])


def _trainer() -> Trainer:
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    return Trainer(model, opt, None, torch.device("cpu"))


def _loader() -> DataLoader:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 3, 2, 2, generator=g)
    y = (torch.rand(8, 3, generator=g) < 0.5).float()
    return DataLoader(TensorDataset(x, y), batch_size=4)


def _n_forward_hooks(model: nn.Module) -> int:
    return sum(len(m._forward_hooks) for m in model.modules())


def test_hooks_are_removed_after_each_epoch():
    dec = LayerHooksDecorator(_trainer(), log_every_n_epochs=99)
    dec.train_epoch(_loader())
    assert _n_forward_hooks(dec.model) == 0
    assert dec._hooks == []


def test_hooks_capture_only_linear_layers():
    dec = LayerHooksDecorator(_trainer(), log_every_n_epochs=99)
    dec.train_epoch(_loader())
    assert set(dec._activations) == {"1", "3"}          # las dos nn.Linear
    assert all(v >= 0.0 for v in dec._activations.values())


def test_hooks_are_removed_if_train_epoch_raises():
    trainer = _trainer()

    def boom(_loader):
        raise RuntimeError("fallo simulado")
    trainer.train_epoch = boom
    dec = LayerHooksDecorator(trainer)
    with pytest.raises(RuntimeError):
        dec.train_epoch(_loader())
    assert _n_forward_hooks(dec.model) == 0


def test_result_is_the_inner_result_untouched():
    dec = LayerHooksDecorator(_trainer(), log_every_n_epochs=99)
    result = dec.train_epoch(_loader())
    assert {"loss", "f1", "accuracy", "time"} <= set(result)


def test_report_only_every_n_epochs(capsys):
    dec = LayerHooksDecorator(_trainer(), log_every_n_epochs=2)
    dec.train_epoch(_loader())
    assert capsys.readouterr().out == ""
    dec.train_epoch(_loader())
    out = capsys.readouterr().out
    assert "[hooks] Activaciones medias — epoch 002:" in out
    assert out.count("\n  ") == 2                       # una línea por Linear


def test_report_goes_to_trainer_logger_when_it_has_handlers(capsys):
    records: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("trainer")
    handler = _Collect()
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        LayerHooksDecorator(_trainer(), log_every_n_epochs=1).train_epoch(_loader())
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    assert capsys.readouterr().out == ""
    assert any("[hooks] Activaciones medias — epoch 001:" in r for r in records)
