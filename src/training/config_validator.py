"""Config validation for training YAML files.

Validates the config dict loaded from YAML before any training starts,
providing clear error messages instead of cryptic crashes deep in the code.

Usage (called automatically by TrainingSessionBuilder.build()):
    from src.training.config_validator import validate_config
    validate_config(cfg)   # raises ConfigError on invalid config
"""

from __future__ import annotations


class ConfigError(ValueError):
    """Raised when a training config is invalid."""


# ── Field descriptors ─────────────────────────────────────────────────────────

_REQUIRED_SECTIONS = ["data", "model", "training"]

_DATA_REQUIRED   = ["root", "metadata"]
_MODEL_REQUIRED  = ["name", "num_classes"]
_TRAIN_REQUIRED  = ["epochs", "batch_size", "lr"]

_TRAIN_POSITIVE_INT  = ["epochs", "batch_size"]
_TRAIN_POSITIVE_FLOAT = ["lr", "weight_decay", "grad_clip", "warmup_epochs",
                          "lr_min", "llrd_decay", "label_smoothing", "mixup_alpha"]
_TRAIN_NON_NEGATIVE  = ["label_smoothing", "mixup_alpha", "warmup_epochs",
                         "llrd_decay", "weight_decay"]


def validate_config(cfg: dict) -> None:
    """Validate a training config dict.  Raises ConfigError with a human-readable message."""
    errors: list[str] = []

    # Top-level sections
    for section in _REQUIRED_SECTIONS:
        if section not in cfg:
            errors.append(f"Missing required section: '{section}'")

    if errors:
        _raise(errors)

    _check_data(cfg["data"], errors)
    _check_model(cfg["model"], errors)
    _check_training(cfg["training"], errors)
    _check_distributed(cfg.get("distributed", {}), errors)

    if errors:
        _raise(errors)


def _check_data(data: dict, errors: list) -> None:
    for key in _DATA_REQUIRED:
        if key not in data:
            errors.append(f"data.{key} is required")
    if "num_workers" in data:
        nw = data["num_workers"]
        if not isinstance(nw, int) or nw < 0:
            errors.append(f"data.num_workers must be a non-negative integer, got {nw!r}")


def _check_model(model: dict, errors: list) -> None:
    for key in _MODEL_REQUIRED:
        if key not in model:
            errors.append(f"model.{key} is required")
    if "num_classes" in model:
        nc = model["num_classes"]
        if not isinstance(nc, int) or nc <= 0:
            errors.append(f"model.num_classes must be a positive integer, got {nc!r}")
    if "dropout" in model:
        d = model["dropout"]
        if not isinstance(d, (int, float)) or not (0.0 <= float(d) < 1.0):
            errors.append(f"model.dropout must be in [0, 1), got {d!r}")


def _check_training(train: dict, errors: list) -> None:
    for key in _TRAIN_REQUIRED:
        if key not in train:
            errors.append(f"training.{key} is required")

    for key in _TRAIN_POSITIVE_INT:
        if key in train:
            v = train[key]
            if not isinstance(v, int) or v <= 0:
                errors.append(
                    f"training.{key} must be a positive integer, got {v!r}. "
                    f"(Hint: YAML parses '1e-4' as a string — use '0.0001' instead)"
                )

    for key in _TRAIN_POSITIVE_FLOAT:
        if key not in train:
            continue
        v = train[key]
        if isinstance(v, str):
            errors.append(
                f"training.{key}={v!r} is a string, not a number. "
                f"YAML parses scientific notation as strings — write '0.0001' not '1e-4'"
            )
            continue
        if not isinstance(v, (int, float)):
            errors.append(f"training.{key} must be numeric, got {type(v).__name__}: {v!r}")
            continue
        fv = float(v)
        if key not in _TRAIN_NON_NEGATIVE and fv <= 0:
            errors.append(f"training.{key} must be positive, got {fv}")
        elif key in _TRAIN_NON_NEGATIVE and fv < 0:
            errors.append(f"training.{key} must be >= 0, got {fv}")

    # Specific range checks
    if "label_smoothing" in train:
        ls = train["label_smoothing"]
        if isinstance(ls, (int, float)) and not (0.0 <= float(ls) <= 0.5):
            errors.append(f"training.label_smoothing should be in [0, 0.5], got {ls}")

    if "loss" in train:
        lk = str(train["loss"]).lower()
        if lk not in ("bce", "bcewithlogits", "bce_with_logits", "focal"):
            errors.append(f"training.loss must be 'bce' or 'focal', got {train['loss']!r}")

    if "select_by" in train:
        sb = str(train["select_by"]).lower()
        if sb not in ("f1", "f1_optimal"):
            errors.append(f"training.select_by must be 'f1' or 'f1_optimal', got {train['select_by']!r}")

    if "focal_gamma" in train:
        g = train["focal_gamma"]
        if not isinstance(g, (int, float)) or float(g) < 0:
            errors.append(f"training.focal_gamma must be a number >= 0, got {g!r}")

    if "pos_weight" in train:
        pw = train["pos_weight"]
        ok = (isinstance(pw, str) and pw.lower() == "auto") or (
            isinstance(pw, (list, tuple))
            and all(isinstance(x, (int, float)) and x > 0 for x in pw)
        )
        if not ok:
            errors.append(
                "training.pos_weight must be 'auto' or a list of positive numbers, "
                f"got {pw!r}"
            )

    if "early_stopping_patience" in train:
        p = train["early_stopping_patience"]
        if p is not None and (not isinstance(p, int) or p <= 0):
            errors.append(f"training.early_stopping_patience must be a positive int or null, got {p!r}")

    if "warmup_epochs" in train and "epochs" in train:
        we = train["warmup_epochs"]
        ep = train["epochs"]
        if isinstance(we, int) and isinstance(ep, int) and we >= ep:
            errors.append(
                f"training.warmup_epochs ({we}) >= training.epochs ({ep}). "
                "The cosine phase would have 0 epochs. Reduce warmup_epochs or increase epochs."
            )


def _check_distributed(dist: dict, errors: list) -> None:
    if not dist:
        return
    backend = dist.get("backend", "nccl")
    if backend not in ("nccl", "gloo", "mpi"):
        errors.append(f"distributed.backend must be 'nccl', 'gloo', or 'mpi', got {backend!r}")

    ranks = dist.get("ranks", [])
    for i, rank in enumerate(ranks):
        if "device" not in rank:
            errors.append(f"distributed.ranks[{i}].device is required")
        elif rank["device"] not in ("cuda", "cpu"):
            errors.append(f"distributed.ranks[{i}].device must be 'cuda' or 'cpu'")
        if "batch_size" in rank:
            bs = rank["batch_size"]
            if not isinstance(bs, int) or bs <= 0:
                errors.append(f"distributed.ranks[{i}].batch_size must be positive int, got {bs!r}")
        if "compute_weight" in rank:
            w = rank["compute_weight"]
            if not isinstance(w, (int, float)) or float(w) <= 0:
                errors.append(f"distributed.ranks[{i}].compute_weight must be positive, got {w!r}")


def _raise(errors: list) -> None:
    msg = "\n".join(f"  • {e}" for e in errors)
    raise ConfigError(f"Config validation failed with {len(errors)} error(s):\n{msg}")
