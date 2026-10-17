"""MedMNIST reader, specs, download checks and the per-dataset output layout.
No network: tiny .npz files with the official layout are written to tmp_path."""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import medmnist as mm
from src.data.registry import get_dataset_spec


def _fake_npz(root: Path, subset: mm.Subset, n: int = 6, size: int = 28) -> Path:
    rng = np.random.default_rng(0)
    shape = (n, size, size) if subset.channels == 1 else (n, size, size, 3)
    arrays = {}
    for split in ("train", "val", "test"):
        arrays[f"{split}_images"] = rng.integers(0, 256, shape, dtype=np.uint8)
        if subset.task == "multilabel":
            labels = rng.integers(0, 2, (n, len(subset.labels)))
        else:
            labels = np.arange(n).reshape(-1, 1) % len(subset.labels)
        arrays[f"{split}_labels"] = labels.astype(np.uint8)
    root.mkdir(parents=True, exist_ok=True)
    path = root / mm.npz_filename(subset, size)
    np.savez(path, **arrays)
    return path


class TestSpecs:
    @pytest.mark.parametrize("name, task, channels, n_classes", [
        ("medmnist_breast", "binary", 1, 1),
        ("medmnist_derma", "multiclass", 3, 7),
        ("medmnist_path", "multiclass", 3, 9),
        ("medmnist_chest", "multilabel", 1, 14),
    ])
    def test_registered(self, name, task, channels, n_classes):
        spec = get_dataset_spec(name)
        assert spec.task == task
        assert spec.in_channels == channels
        assert spec.num_classes == n_classes
        assert spec.required_keys == ("root",)

    def test_breast_positive_class_is_malignant(self):
        assert get_dataset_spec("medmnist_breast").classes == ("malignant",)

    def test_every_size_has_a_checksum(self):
        for subset in mm.SUBSETS.values():
            assert set(subset.md5) == set(mm.SIZES)


class TestFiles:
    def test_filenames_follow_the_official_scheme(self):
        breast = mm.SUBSETS["medmnist_breast"]
        assert mm.npz_filename(breast) == "breastmnist.npz"
        assert mm.npz_filename(breast, 224) == "breastmnist_224.npz"
        assert mm.download_url(breast, 64).endswith("/breastmnist_64.npz?download=1")

    def test_unknown_size_rejected(self):
        with pytest.raises(ValueError, match="size"):
            mm.npz_filename(mm.SUBSETS["medmnist_breast"], 32)

    def test_download_rejects_a_corrupt_file(self, tmp_path, monkeypatch):
        def fake_retrieve(url, dest):
            Path(dest).write_bytes(b"not the real file")
        monkeypatch.setattr(mm.urllib.request, "urlretrieve", fake_retrieve)
        with pytest.raises(IOError, match="MD5 mismatch"):
            mm.download(mm.SUBSETS["medmnist_breast"], tmp_path)
        assert list(tmp_path.iterdir()) == []          # nothing half-written left

    def test_download_skips_a_valid_file(self, tmp_path, monkeypatch):
        subset = mm.SUBSETS["medmnist_breast"]
        path = tmp_path / mm.npz_filename(subset)
        path.write_bytes(b"x")
        monkeypatch.setattr(mm, "_md5", lambda p: subset.md5[28])
        monkeypatch.setattr(mm.urllib.request, "urlretrieve",
                            lambda *a: pytest.fail("should not download"))
        assert mm.download(subset, tmp_path) == path


class TestTargets:
    def test_binary_marks_the_positive_label(self):
        breast = mm.SUBSETS["medmnist_breast"]                 # 0 = malignant
        out = mm.encode_targets(np.array([[0], [1], [0]]), breast)
        assert out.tolist() == [[1.0], [0.0], [1.0]]

    def test_multiclass_is_one_hot(self):
        derma = mm.SUBSETS["medmnist_derma"]
        out = mm.encode_targets(np.array([[4], [0]]), derma)
        assert out.shape == (2, 7)
        assert out.argmax(dim=1).tolist() == [4, 0]
        assert out.sum(dim=1).tolist() == [1.0, 1.0]

    def test_multilabel_kept_as_is(self):
        chest = mm.SUBSETS["medmnist_chest"]
        raw = np.zeros((1, 14), dtype=np.uint8)
        raw[0, [2, 5]] = 1
        assert mm.encode_targets(raw, chest).nonzero()[:, 1].tolist() == [2, 5]


class TestDataset:
    @pytest.mark.parametrize("name", sorted(mm.SUBSETS))
    def test_items_have_the_model_input_shape(self, tmp_path, name):
        subset = mm.SUBSETS[name]
        _fake_npz(tmp_path, subset)
        spec = get_dataset_spec(name)
        ds = spec.dataset({"root": str(tmp_path)}, "val")
        image, target = ds[0]
        assert image.shape == (subset.channels, 224, 224)
        assert image.dtype == torch.float32
        assert target.shape == (spec.num_classes,)
        assert len(ds) == 6

    def test_size_selects_the_file(self, tmp_path):
        subset = mm.SUBSETS["medmnist_derma"]
        _fake_npz(tmp_path, subset, size=64)
        ds = get_dataset_spec("medmnist_derma").dataset({"root": str(tmp_path), "size": 64}, "test")
        assert ds.images.shape[1:3] == (64, 64)

    def test_class_counts_for_pos_weight(self, tmp_path):
        subset = mm.SUBSETS["medmnist_breast"]
        _fake_npz(tmp_path, subset, n=6)                       # labels 0,1,0,1,0,1
        pos, n = get_dataset_spec("medmnist_breast").class_counts({"root": str(tmp_path)}, "train")
        assert n == 6
        assert pos.tolist() == [3.0]


class TestOutputLayout:
    def test_bigearthnet_keeps_the_historical_layout(self):
        from src.training.builder import TrainingSessionBuilder
        spec = get_dataset_spec("bigearthnet")
        assert TrainingSessionBuilder.run_subdir(spec, "ddp", "vit_tiny") == Path("ddp/vit_tiny")

    def test_other_datasets_get_their_own_folder(self):
        from src.training.builder import TrainingSessionBuilder
        spec = get_dataset_spec("medmnist_derma")
        assert (TrainingSessionBuilder.run_subdir(spec, "single", "vit_tiny")
                == Path("medmnist_derma/single/vit_tiny"))

    def test_web_reads_the_dataset_from_the_path(self):
        from src.web.run_registry import _DEFAULT_DATASET, _env_mode_model_from_path
        from src.data.registry import DEFAULT_DATASET
        assert _DEFAULT_DATASET == DEFAULT_DATASET              # web keeps its own literal
        root = Path("logs")
        assert _env_mode_model_from_path(root / "local/ddp/vit/train_1.log", root) == (
            "local", "ddp", "vit", "bigearthnet")
        assert _env_mode_model_from_path(
            root / "kaggle/medmnist_chest/single/vit/train_1.log", root) == (
            "kaggle", "single", "vit", "medmnist_chest")
