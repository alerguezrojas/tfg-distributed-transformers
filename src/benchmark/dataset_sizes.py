"""Dataset-dependent inputs of the benchmark report: split sizes and whether the
BigEarthNet-only parts (I/O profile of ``*.tif`` patches, empirical F1 prior) apply."""
from __future__ import annotations

from pathlib import Path

from src.data.registry import DEFAULT_DATASET, dataset_name_from_config, dataset_spec_from_config

# BigEarthNet-S2 full splits (fallback when the metadata cannot be read).
BIGEARTHNET_TRAIN, BIGEARTHNET_VAL = 237871, 122342


def is_bigearthnet(cfg: dict) -> bool:
    return dataset_name_from_config(cfg or {}) == DEFAULT_DATASET


def _bigearthnet_sizes(cfg: dict) -> tuple[int, int]:
    """Train/val sizes from ``data.metadata`` (a subset config gives the subset)."""
    n_train, n_val = BIGEARTHNET_TRAIN, BIGEARTHNET_VAL
    meta_path = (cfg.get("data") or {}).get("metadata")
    if meta_path and Path(meta_path).exists():
        try:
            import pandas as pd
            counts = pd.read_parquet(meta_path, columns=["split"])["split"].value_counts()
            n_train = int(counts.get("train", n_train))
            n_val = int(counts.get("validation", n_val))
            print(f"Dataset (metadata): train={n_train:,}  val={n_val:,}")
        except Exception as exc:
            print(f"[aviso] no se pudo leer el tamaño del metadata ({exc}); "
                  f"usando full set {n_train:,}/{n_val:,}")
    return n_train, n_val


def resolve_split_sizes(cfg: dict, train_size: int | None = None,
                        val_size: int | None = None) -> tuple[int, int]:
    """(n_train, n_val) of the dataset in the config.

    Order: explicit ``--train-size/--val-size``; then what the dataset says (BigEarthNet:
    its metadata, falling back to the full set; the others: the spec's ``class_counts``,
    which reads only the CSV/npz, honours the config filters and opens no image).
    A dataset whose sizes cannot be read is an error: guessing would print times for
    another dataset.
    """
    cfg = cfg or {}
    if train_size is not None and val_size is not None:
        return int(train_size), int(val_size)
    if is_bigearthnet(cfg):
        n_train, n_val = _bigearthnet_sizes(cfg)
    else:
        spec = dataset_spec_from_config(cfg)
        data_cfg = cfg.get("data") or {}
        try:
            if spec.class_counts is None:
                raise ValueError("la ficha no sabe contar muestras")
            n_train = spec.class_counts(data_cfg, "train")[1]
            n_val = spec.class_counts(data_cfg, "val")[1]
            print(f"Dataset ({spec.name}): train={n_train:,}  val={n_val:,}")
        except Exception as exc:
            raise ValueError(
                f"no se pudo leer el tamaño de los splits de {spec.name!r} ({exc}); "
                f"indíquelos con --train-size y --val-size") from exc
    return (int(train_size) if train_size is not None else n_train,
            int(val_size) if val_size is not None else n_val)
