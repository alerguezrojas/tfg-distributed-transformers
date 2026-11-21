"""Out-of-fold label cleaning: folds, scores, cleaned partition and the PANDA script."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.label_noise import (
    EXCLUDED,
    assign_folds,
    cleaned_splits,
    fold_splits,
    oof_scores,
)

ROOT = Path(__file__).resolve().parents[2]


def _toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"id{i:04d}" for i in range(n)]
    labels = pd.DataFrame({"image_id": ids,
                           "data_provider": rng.choice(["karolinska", "radboud"], n),
                           "isup_grade": rng.integers(0, 6, n)})
    split = np.array(["train"] * n, dtype=object)
    split[: n // 5] = "val"
    split[n // 5: 2 * n // 5] = "test"
    splits = pd.DataFrame({"image_id": ids, "split": rng.permutation(split)})
    return splits, labels


class TestFolds:
    def test_partition_of_train_with_val_and_test_excluded(self):
        splits, labels = _toy()
        folds = fold_splits(splits, labels, n_folds=4)
        train_ids = set(splits.loc[splits["split"] == "train", "image_id"])
        held_out = [set(f.loc[f["split"] == "val", "image_id"]) for f in folds]
        assert set().union(*held_out) == train_ids
        assert sum(len(h) for h in held_out) == len(train_ids)          # disjoint
        for f, h in zip(folds, held_out):
            assert set(f.loc[f["split"] == "train", "image_id"]) == train_ids - h
            assert set(f.loc[f["split"] == EXCLUDED, "image_id"]) == set(splits["image_id"]) - train_ids
            assert list(f["image_id"]) == list(splits["image_id"])

    def test_stratified_and_deterministic(self):
        splits, labels = _toy(n=600)
        a = fold_splits(splits, labels, n_folds=5)
        b = fold_splits(splits.sample(frac=1, random_state=1).reset_index(drop=True), labels, 5)
        for fa, fb in zip(a, b):     # same folds whatever the row order
            assert set(fa.loc[fa["split"] == "val", "image_id"]) == set(fb.loc[fb["split"] == "val", "image_id"])
        train = splits[splits["split"] == "train"].merge(labels, on="image_id")
        for f in a:
            held = train[train["image_id"].isin(f.loc[f["split"] == "val", "image_id"])]
            per_stratum = held.groupby(["data_provider", "isup_grade"]).size()
            total = train.groupby(["data_provider", "isup_grade"]).size()
            assert ((per_stratum - total / 5).abs() <= 1).all()

    def test_seed_changes_the_folds(self):
        ids = pd.Series([f"x{i}" for i in range(50)])
        strata = pd.Series(["a"] * 50)
        assert not assign_folds(ids, strata, 5, seed=1).equals(assign_folds(ids, strata, 5, seed=2))

    def test_needs_two_folds_and_labels(self):
        splits, labels = _toy()
        with pytest.raises(ValueError, match="n_folds"):
            fold_splits(splits, labels, n_folds=1)
        with pytest.raises(ValueError, match="label"):
            fold_splits(splits, labels.iloc[:10], n_folds=3)


class TestScores:
    def test_expected_grade_and_gap(self):
        probs = np.array([[1, 0, 0], [0, 0.5, 0.5], [0, 0, 1.0]])
        s = oof_scores(probs, np.array([0, 2, 0]))
        assert s["expected"].tolist() == pytest.approx([0.0, 1.5, 2.0])
        assert s["gap"].tolist() == pytest.approx([0.0, 0.5, 2.0])
        assert s["p_label"].tolist() == pytest.approx([1.0, 0.5, 0.0])
        assert s["pred"].tolist()[2] == 2

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            oof_scores(np.zeros((3, 6)), np.zeros(4, dtype=int))


class TestCleaned:
    def test_only_flagged_train_rows_move(self):
        splits, _ = _toy()
        train_ids = splits.loc[splits["split"] == "train", "image_id"].tolist()
        out = cleaned_splits(splits, train_ids[:3])
        assert (out.set_index("image_id").loc[train_ids[:3], "split"] == EXCLUDED).all()
        changed = (out["split"] != splits["split"]).sum()
        assert changed == 3

    def test_rejects_val_test_or_unknown_ids(self):
        splits, _ = _toy()
        val_id = splits.loc[splits["split"] == "val", "image_id"].iloc[0]
        with pytest.raises(ValueError, match="training"):
            cleaned_splits(splits, [val_id])
        with pytest.raises(ValueError, match="not in"):
            cleaned_splits(splits, ["nope"])


def _script():
    spec = importlib.util.spec_from_file_location("_panda_oof", ROOT / "scripts" / "panda_oof.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_folds_then_clean(tmp_path):
    """End to end on a toy root: folds + configs, fake predictions, cleaned partition."""
    splits, labels = _toy(n=100)
    root = tmp_path / "panda"
    root.mkdir()
    splits.to_csv(root / "splits.csv", index=False)
    labels.to_csv(root / "train.csv", index=False)
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"data": {"name": "panda", "root": str(root)},
                                    "training": {"epochs": 1}}))
    out = tmp_path / "oof"
    mod = _script()
    mod.main(["folds", "--config", str(base), "--out", str(out), "--n-folds", "3"])
    cfg = yaml.safe_load((out / "train_fold1.yaml").read_text())
    assert cfg["data"]["splits"] == str(out / "splits_fold1.csv")
    bad = None
    for k in range(3):                      # predictions in the reader's order (sorted ids)
        f = pd.read_csv(out / f"splits_fold{k}.csv")
        ids = sorted(f.loc[f["split"] == "val", "image_id"])
        grades = labels.set_index("image_id").loc[ids, "isup_grade"].to_numpy()
        pred = grades.copy()
        if k == 0:                          # one confidently wrong prediction
            bad = ids[0]
            pred[0] = 5 if grades[0] < 3 else 0
        np.savez(out / f"fold{k}_val_preds.npz", probs=np.eye(6)[pred], targets=np.eye(6)[grades])
    mod.main(["clean", "--out", str(out), "--n-folds", "3", "--max-gap", "1.5"])
    clean = pd.read_csv(out / "splits_clean.csv").set_index("image_id")["split"]
    assert clean[bad] == EXCLUDED
    assert (clean != splits.set_index("image_id")["split"]).sum() == 1
    oof = pd.read_csv(out / "oof.csv")
    assert len(oof) == (splits["split"] == "train").sum() and oof["flagged"].sum() == 1
    assert yaml.safe_load((out / "train_clean.yaml").read_text())["data"]["splits"] == str(out / "splits_clean.csv")


def test_script_detects_misaligned_predictions(tmp_path):
    splits, labels = _toy(n=60)
    root = tmp_path / "panda"
    root.mkdir()
    splits.to_csv(root / "splits.csv", index=False)
    labels.to_csv(root / "train.csv", index=False)
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"data": {"root": str(root)}}))
    out = tmp_path / "oof"
    mod = _script()
    mod.main(["folds", "--config", str(base), "--out", str(out), "--n-folds", "2"])
    for k in range(2):
        f = pd.read_csv(out / f"splits_fold{k}.csv")
        n = (f["split"] == "val").sum()
        np.savez(out / f"fold{k}_val_preds.npz", probs=np.eye(6)[np.zeros(n, int)],
                 targets=np.eye(6)[np.full(n, 5)])
    with pytest.raises(ValueError, match="targets"):
        mod.main(["clean", "--out", str(out), "--n-folds", "2"])
