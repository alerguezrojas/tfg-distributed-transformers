"""What the web shows about a non-BigEarthNet dataset: only what the run's inspection
files hold (class mix of the validation set, thumbnails of the fixed panel)."""
from pathlib import Path

import numpy as np

from src.web.dataset_view import inspect_dir_of, panel_examples, val_class_counts
from src.web.run_registry import RunInfo


def _inspect(tmp_path, with_thumbs=True, with_val_targets=True):
    d = tmp_path / "inspect_19112026_191255"
    d.mkdir()
    targets = np.eye(3, dtype=np.uint8)[[0, 0, 1, 2]]
    extra = {"thumbs": np.arange(4 * 2 * 2 * 3, dtype=np.uint8).reshape(4, 2, 2, 3),
             "panel": np.array([1, 3])} if with_thumbs else {}
    np.savez(d / "monitor_set.npz", indices=np.arange(4), targets=targets,
             classes=np.array(["a", "b", "c"]), task=np.array("multiclass"), **extra)
    probs = np.zeros((4, 3), dtype=np.float16)
    for epoch in (1, 2):
        arrays = {"probs": probs, "loss": np.zeros(4, dtype=np.float32)}
        if with_val_targets and epoch == 2:               # only the last epoch carries them
            arrays["val_targets"] = np.eye(3, dtype=np.uint8)[[0, 0, 0, 1, 2, 2]]
        np.savez(d / f"epoch_{epoch:03d}.npz", **arrays)
    return d


def _run(tmp_path):
    return RunInfo(timestamp="19112026_191255", log_path=tmp_path / "train_19112026_191255.log",
                   trace_mode="simple")


def test_inspect_dir_of_only_when_the_monitor_set_is_there(tmp_path):
    assert inspect_dir_of(_run(tmp_path)) is None
    d = _inspect(tmp_path)
    assert inspect_dir_of(_run(tmp_path)) == d


def test_val_class_counts_come_from_the_last_epoch_that_has_them(tmp_path):
    classes, counts = val_class_counts(_inspect(tmp_path))
    assert classes == ["a", "b", "c"] and counts.tolist() == [3, 1, 2]
    assert val_class_counts(None) is None


def test_val_class_counts_none_without_validation_targets(tmp_path):
    assert val_class_counts(_inspect(tmp_path, with_val_targets=False)) is None


def test_panel_examples_are_the_fixed_panel_with_their_dominant_class(tmp_path):
    ex = panel_examples(_inspect(tmp_path))
    assert [e.class_name for e in ex] == ["a", "c"]          # positions 1 and 3
    assert ex[0].thumb.shape == (2, 2, 3)
    assert len(panel_examples(tmp_path / "inspect_19112026_191255", limit=1)) == 1


def test_no_thumbnails_or_no_directory_means_no_examples(tmp_path):
    assert panel_examples(None) == []
    assert panel_examples(Path(tmp_path / "missing")) == []
