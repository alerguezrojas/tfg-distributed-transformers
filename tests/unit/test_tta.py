"""Test-time augmentation by the symmetries of the square (src/training/tta.py, eval --tta)."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.tta import TTA_VIEWS, dihedral_views  # noqa: E402

_spec = importlib.util.spec_from_file_location("eval_script_tta", ROOT / "scripts" / "eval.py")
eval_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_script)


def test_eight_views_are_the_eight_distinct_symmetries():
    x = torch.arange(9.0).reshape(1, 1, 3, 3)
    views = dihedral_views(x, 8)
    assert torch.equal(views[0], x)
    assert len({tuple(v.flatten().tolist()) for v in views}) == 8


@pytest.mark.parametrize("n", TTA_VIEWS)
def test_first_n_views_keep_shape_and_identity_first(n):
    x = torch.randn(2, 3, 4, 4)
    views = dihedral_views(x, n)
    assert len(views) == n and torch.equal(views[0], x)
    assert all(v.shape == x.shape for v in views)


def test_rotations_need_square_images_and_n_is_checked():
    x = torch.zeros(1, 3, 4, 6)
    assert len(dihedral_views(x, 4)) == 4            # flips keep any shape
    with pytest.raises(ValueError, match="square"):
        dihedral_views(x, 8)
    with pytest.raises(ValueError, match="tta"):
        dihedral_views(x, 3)


class _Corner(torch.nn.Module):
    """Logit = value of the top-left pixel: changes with every flip or rotation."""
    def forward(self, x):
        v = x[:, 0, 0, 0]
        return torch.stack([v, -v], dim=1)


def test_evaluate_averages_the_views_and_keeps_the_plain_loss():
    from src.training.tasks import MultiClassTask
    task = MultiClassTask()
    x = torch.zeros(1, 3, 2, 2)
    x[0, 0] = torch.tensor([[2.0, 0.0], [0.0, 0.0]])  # only the top-left corner is set
    y = torch.tensor([[1.0, 0.0]])
    plain, _, loss1 = eval_script.evaluate(_Corner(), [(x, y)], torch.device("cpu"), task=task)
    tta, _, loss8 = eval_script.evaluate(_Corner(), [(x, y)], torch.device("cpu"), task=task, tta=8)
    expected = sum(task.probabilities(_Corner()(v)) for v in dihedral_views(x, 8)) / 8
    assert torch.allclose(tta, expected)
    assert not torch.allclose(tta, plain)
    assert loss1 == pytest.approx(loss8)             # the loss is the one of the original images
