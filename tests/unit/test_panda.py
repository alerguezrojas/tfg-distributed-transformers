"""PANDA: mosaic maths, synthetic pyramidal TIFF, fixed splits, spec and reader.
No dataset and no GPU: tiny synthetic images with the real layout are written to tmp_path."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import torch
from PIL import Image

from src.data import panda, wsi
from src.data.registry import get_dataset_spec

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
import prepare_panda  # noqa: E402


def _slide(h=70, w=90, dark_tiles=()):
    """White slide where each (row, col) of ``dark_tiles`` (tile = 16) is tissue-like."""
    img = np.full((h, w, 3), 255, np.uint8)
    for r, c in dark_tiles:
        img[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16] = 100
    return img


class TestMosaic:
    def test_pad_to_multiple_with_white(self):
        out = wsi.pad_to_multiple(np.zeros((10, 17, 3), np.uint8), 8)
        assert out.shape == (16, 24, 3)
        assert out[:10, :17].max() == 0 and (out[10:] == 255).all() and (out[:, 17:] == 255).all()

    def test_pad_noop_when_already_multiple(self):
        img = np.zeros((16, 16, 3), np.uint8)
        assert wsi.pad_to_multiple(img, 8) is img

    def test_scores_rank_tissue_first(self):
        img = wsi.pad_to_multiple(_slide(dark_tiles=[(1, 2)]), 16)    # 5x6 grid
        scores = wsi.tissue_scores(img, 16)
        assert len(scores) == 5 * 6
        assert wsi.select_tiles(scores, 1)[0] == 1 * 6 + 2

    def test_select_is_deterministic_on_ties(self):
        assert list(wsi.select_tiles(np.zeros(10), 3)) == [0, 1, 2]

    def test_mosaic_shape_and_selected_tiles(self):
        dark = [(0, 0), (1, 1), (2, 3), (3, 2)]
        mos = wsi.make_mosaic(_slide(dark_tiles=dark), tile=16, n_tiles=4)
        assert mos.shape == (32, 32, 3)
        assert (mos == 100).all()          # exactly the four dark tiles, no white

    def test_fewer_tiles_than_requested_pads_with_white(self):
        img = _slide(32, 32, dark_tiles=[(0, 0), (1, 1)])              # 2x2 grid = 4 tiles
        mos = wsi.make_mosaic(img, tile=16, n_tiles=9)                 # 3x3 mosaic
        assert mos.shape == (48, 48, 3)
        assert (mos[:16, :16] == 100).all() and (mos[:16, 16:32] == 100).all()
        assert (mos[32:] == 255).all()                                 # last row all blank
        assert wsi.tissue_fraction(mos) == pytest.approx(2 * 256 / 48 ** 2)

    def test_n_tiles_must_be_a_square(self):
        with pytest.raises(ValueError):
            wsi.make_mosaic(_slide(), tile=16, n_tiles=6)


def _write_pyramid(path: Path, base: np.ndarray, levels: int = 3) -> None:
    """Pyramidal RGB TIFF with SubIFD levels reduced 1x, 4x, 16x like PANDA's."""
    with tifffile.TiffWriter(path) as tif:
        tif.write(base, photometric="rgb", subifds=levels - 1)
        for k in range(1, levels):
            tif.write(base[::4 ** k, ::4 ** k], photometric="rgb", subfiletype=1)


class TestTiff:
    def test_read_level_picks_the_pyramid_level(self, tmp_path):
        base = np.random.default_rng(0).integers(0, 255, (160, 192, 3), dtype=np.uint8)
        _write_pyramid(tmp_path / "s.tiff", base)
        assert wsi.read_level(tmp_path / "s.tiff", 0).shape == (160, 192, 3)
        assert wsi.read_level(tmp_path / "s.tiff", 1).shape == (40, 48, 3)
        with pytest.raises(ValueError):
            wsi.read_level(tmp_path / "s.tiff", 5)

    def test_process_one_builds_mosaic_skips_and_survives_errors(self, tmp_path):
        base = np.full((256, 256, 3), 255, np.uint8)
        base[:64, :64] = 90                       # one tissue tile at level 1 (16 px tiles)
        _write_pyramid(tmp_path / "a.tiff", base)
        out = tmp_path / "out"
        out.mkdir()
        job = ("a", str(tmp_path / "a.tiff"), str(out), 1, 16, 4)
        assert prepare_panda.process_one(job)[1] == "done"
        mosaic = np.asarray(Image.open(out / "a.jpg"))
        assert mosaic.shape == (32, 32, 3)
        assert mosaic[:16, :16].mean() < 110      # the tissue tile comes first
        assert prepare_panda.process_one(job)[1] == "skipped"
        missing = prepare_panda.process_one(("b", str(tmp_path / "b.tiff"), str(out), 1, 16, 4))
        assert missing[1] == "error" and not (out / "b.jpg").exists()


def _labels(per_stratum=40):
    rows = [(f"{p[0]}{g}_{i:03d}", p, g)
            for p in ("karolinska", "radboud") for g in range(6) for i in range(per_stratum)]
    return pd.DataFrame(rows, columns=["image_id", "data_provider", "isup_grade"])


class TestSplits:
    def test_partition_without_overlap_and_complete(self):
        labels = _labels()
        s = panda.make_splits(labels)
        assert len(s) == len(labels) and s["image_id"].is_unique
        assert set(s["image_id"]) == set(labels["image_id"])
        assert set(s["split"]) == {"train", "val", "test"}

    def test_proportions_per_stratum(self):
        labels = _labels(40)
        m = labels.merge(panda.make_splits(labels), on="image_id")
        counts = m.groupby(["data_provider", "isup_grade", "split"]).size().unstack()
        assert (counts["train"] == 28).all() and (counts["val"] == 6).all()
        assert (counts["test"] == 6).all()

    def test_deterministic_and_independent_of_row_order(self):
        labels = _labels()
        a = panda.make_splits(labels)
        b = panda.make_splits(labels.sample(frac=1, random_state=3))
        pd.testing.assert_frame_equal(a, b)
        assert not a.equals(panda.make_splits(labels, seed=7))

    def test_tiny_strata_do_not_crash(self):
        labels = pd.DataFrame({"image_id": ["a", "b"], "data_provider": ["radboud", "radboud"],
                               "isup_grade": [0, 5]})
        assert list(panda.make_splits(labels)["split"]) == ["train", "train"]

    def test_fractions_must_add_up(self):
        with pytest.raises(ValueError):
            panda.make_splits(_labels(), fractions=(0.5, 0.2, 0.2))


@pytest.fixture
def panda_root(tmp_path):
    """train.csv + splits.csv + 3 synthetic 64 px mosaics (two train, one val)."""
    rows = [("a1", "radboud", 4, "train"), ("b2", "karolinska", 1, "train"),
            ("c3", "karolinska", 0, "val")]
    meta = pd.DataFrame(rows, columns=["image_id", "data_provider", "isup_grade", "split"])
    meta[["image_id", "data_provider", "isup_grade"]].assign(gleason_score="x").to_csv(
        tmp_path / "train.csv", index=False)
    meta[["image_id", "split"]].to_csv(tmp_path / "splits.csv", index=False)
    mosaics = tmp_path / panda.DEFAULT_MOSAIC_DIR
    mosaics.mkdir()
    for i in meta["image_id"]:
        Image.fromarray(np.full((64, 64, 3), 128, np.uint8)).save(mosaics / f"{i}.jpg")
    return tmp_path


class TestSpecAndDataset:
    def test_registered(self):
        spec = get_dataset_spec("panda")
        assert spec.task == "multiclass" and spec.in_channels == 3
        assert spec.classes == tuple(f"ISUP {g}" for g in range(6))
        assert spec.required_keys == ("root",)

    def test_samples_resized_one_hot_and_provider(self, panda_root):
        cfg = {"root": str(panda_root), "image_size": 32}
        ds = get_dataset_spec("panda").dataset(cfg, "train")
        assert len(ds) == 2
        img, target = ds[0]
        assert img.shape == (3, 32, 32) and img.dtype == torch.float32
        assert target.shape == (6,) and target.sum() == 1 and target[4] == 1   # a1: ISUP 4
        assert ds.data_provider == ["radboud", "karolinska"]
        assert list(ds.groups()["data_provider"]) == ["radboud", "karolinska"]
        assert ds.targets.shape == (2, 6)

    def test_val_split_and_eval_transform_is_deterministic(self, panda_root):
        spec = get_dataset_spec("panda")
        ds = spec.dataset({"root": str(panda_root), "image_size": 16}, "val")
        assert len(ds) == 1 and ds.image_ids == ["c3"]
        assert torch.equal(ds[0][0], ds[0][0])

    def test_train_transform_augments_and_keeps_shape(self, panda_root):
        ds = get_dataset_spec("panda").dataset({"root": str(panda_root), "image_size": 16}, "train")
        assert ds[0][0].shape == (3, 16, 16)
        assert panda.RandomRot90()(torch.arange(6.).reshape(1, 2, 3)).numel() == 6

    def test_class_counts(self, panda_root):
        counts, n = get_dataset_spec("panda").class_counts({"root": str(panda_root)}, "train")
        assert n == 2 and counts.tolist() == [0, 1, 0, 0, 1, 0]

    def test_missing_mosaic_fails_early(self, panda_root):
        (panda_root / panda.DEFAULT_MOSAIC_DIR / "a1.jpg").unlink()
        with pytest.raises(FileNotFoundError, match="prepare_panda"):
            get_dataset_spec("panda").dataset({"root": str(panda_root)}, "train")
