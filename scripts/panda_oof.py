"""PANDA: out-of-fold label cleaning (folds -> train/eval per fold -> clean).

Step 1, ``folds``: writes in OUT the K fold partitions (``splits_fold<k>.csv``: the held-out
fold is ``val``, the rest of train is ``train``) and one training config per fold
(``train_fold<k>.yaml``, the base config with ``data.splits`` pointing at its fold).

Step 2 (outside this script): train each fold and save its predictions on the held-out fold:

    paravit.py train --strategy single --config OUT/train_fold0.yaml --precision amp
    paravit.py eval --checkpoint <run>/best.pt --config OUT/train_fold0.yaml --split val \\
        --save-predictions OUT/fold0_val_preds.npz

Step 3, ``clean``: joins the K out-of-fold predictions (``oof.csv``), flags the training
biopsies whose predicted mean grade is more than ``--max-gap`` grades from the label and
writes ``splits_clean.csv`` (flagged -> ``excluded``; val and test untouched) plus
``train_clean.yaml``. Logic in :mod:`src.data.label_noise`.

    uv run python scripts/panda_oof.py folds --config configs/train_panda.yaml --out DIR
    uv run python scripts/panda_oof.py clean --out DIR --max-gap 1.5
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.label_noise import cleaned_splits, fold_splits, oof_scores  # noqa: E402


def _paths(cfg: dict) -> tuple[Path, Path]:
    root = Path(cfg["data"]["root"]).expanduser()
    splits = Path(cfg["data"].get("splits") or root / "splits.csv").expanduser()
    return root / "train.csv", splits


def _write_config(base: dict, splits: Path, path: Path) -> None:
    cfg = copy.deepcopy(base)
    cfg["data"]["splits"] = str(splits)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


def cmd_folds(args) -> None:
    base = yaml.safe_load(Path(args.config).read_text())
    labels_csv, splits_csv = _paths(base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    folds = fold_splits(pd.read_csv(splits_csv), pd.read_csv(labels_csv), args.n_folds, args.seed)
    for k, table in enumerate(folds):
        fold_csv = out / f"splits_fold{k}.csv"
        table.to_csv(fold_csv, index=False)
        _write_config(base, fold_csv, out / f"train_fold{k}.yaml")
        counts = table["split"].value_counts().to_dict()
        print(f"fold {k}: train={counts.get('train', 0)} val={counts.get('val', 0)}")
    (out / "base_config.yaml").write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=True))
    print(f"written to {out}")


def collect_oof(out: Path, n_folds: int, labels: pd.DataFrame) -> pd.DataFrame:
    """The out-of-fold rows of every training id: the held-out fold of each partition, in the
    reader's order (sorted ``image_id``), checked against the saved targets."""
    parts = []
    for k in range(n_folds):
        table = pd.read_csv(out / f"splits_fold{k}.csv")
        ids = sorted(table.loc[table["split"] == "val", "image_id"])
        z = np.load(out / f"fold{k}_val_preds.npz", allow_pickle=False)
        probs, targets = z["probs"], z["targets"].argmax(axis=1)
        if len(ids) != len(probs):
            raise ValueError(f"fold {k}: {len(probs)} predictions for {len(ids)} held-out ids")
        meta = labels.set_index("image_id").loc[ids]
        if not (meta["isup_grade"].to_numpy() == targets).all():
            raise ValueError(f"fold {k}: the saved targets do not match train.csv")
        if "group__data_provider" in z.files and not (
                z["group__data_provider"].astype(str) == meta["data_provider"].to_numpy()).all():
            raise ValueError(f"fold {k}: the saved centres do not match train.csv")
        scores = oof_scores(probs, targets)
        scores.insert(0, "image_id", ids)
        scores.insert(1, "fold", k)
        scores.insert(2, "data_provider", meta["data_provider"].to_numpy())
        scores.insert(3, "isup_grade", targets)
        parts.append(scores)
    return pd.concat(parts, ignore_index=True)


def cmd_clean(args) -> None:
    out = Path(args.out)
    base = yaml.safe_load((out / "base_config.yaml").read_text())
    labels_csv, splits_csv = _paths(base)
    oof = collect_oof(out, args.n_folds, pd.read_csv(labels_csv))
    oof["flagged"] = oof["gap"] > args.max_gap
    oof.to_csv(out / "oof.csv", index=False)
    flagged = oof.loc[oof["flagged"], "image_id"]
    clean = cleaned_splits(pd.read_csv(splits_csv), flagged)
    clean.to_csv(out / "splits_clean.csv", index=False)
    _write_config(base, out / "splits_clean.csv", out / "train_clean.yaml")
    print(f"{len(oof)} out-of-fold predictions; flagged (gap > {args.max_gap}): "
          f"{len(flagged)} ({len(flagged) / len(oof):.1%})")
    print(oof.groupby("data_provider")["flagged"].agg(["sum", "mean"]).to_string())
    print(oof.groupby("isup_grade")["flagged"].agg(["sum", "mean"]).to_string())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("folds", help="write the fold partitions and configs")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_folds)
    c = sub.add_parser("clean", help="join the out-of-fold predictions and clean train")
    c.add_argument("--out", required=True)
    c.add_argument("--n-folds", type=int, default=5)
    c.add_argument("--max-gap", type=float, default=1.5,
                   help="flag when |predicted mean grade - label| exceeds this")
    c.set_defaults(func=cmd_clean)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
