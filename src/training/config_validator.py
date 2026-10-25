"""Config validation for training YAML files.

Validates the config dict loaded from YAML before any training starts,
providing clear error messages instead of cryptic crashes deep in the code.

Usage (called automatically by TrainingSessionBuilder.build()):
    from src.training.config_validator import validate_config
    validate_config(cfg)   # raises ConfigError on invalid config
"""

from __future__ import annotations

from src.data.registry import available_datasets, dataset_name_from_config, get_dataset_spec
from src.data.spec import DatasetSpec


class ConfigError(ValueError):
    """Raised when a training config is invalid."""


# ── Field descriptors ─────────────────────────────────────────────────────────

_REQUIRED_SECTIONS = ["data", "model", "training"]

# data.* required keys depend on the dataset (DatasetSpec.required_keys); the
# number of classes comes from the dataset too (model.num_classes is optional).
_MODEL_REQUIRED  = ["name"]
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

    spec = _check_data(cfg, errors)
    _check_model(cfg["model"], spec, errors)
    _check_training(cfg["training"], spec, errors)
    _check_distributed(cfg.get("distributed", {}), errors)
    _check_monitor(cfg.get("monitor") or {}, errors)

    if errors:
        _raise(errors)


def _check_data(cfg: dict, errors: list) -> DatasetSpec | None:
    """Check the data section against its dataset spec; return the spec if known."""
    data = cfg["data"]
    name = dataset_name_from_config(cfg)
    if name not in available_datasets():
        errors.append(f"data.name {name!r} is not a registered dataset; "
                      f"available: {available_datasets()}")
        spec = None
    else:
        spec = get_dataset_spec(name)
        for key in spec.required_keys:
            if key not in data:
                errors.append(f"data.{key} is required for dataset {name!r}")
    if "num_workers" in data:
        nw = data["num_workers"]
        if not isinstance(nw, int) or nw < 0:
            errors.append(f"data.num_workers must be a non-negative integer, got {nw!r}")
    return spec


def _check_model(model: dict, spec: DatasetSpec | None, errors: list) -> None:
    for key in _MODEL_REQUIRED:
        if key not in model:
            errors.append(f"model.{key} is required")
    if "num_classes" in model:
        nc = model["num_classes"]
        if not isinstance(nc, int) or nc <= 0:
            errors.append(f"model.num_classes must be a positive integer, got {nc!r}")
        elif spec is not None and nc != spec.num_classes:
            errors.append(f"model.num_classes={nc} does not match dataset {spec.name!r} "
                          f"({spec.num_classes} classes); omit it to use the dataset's")
    if "dropout" in model:
        d = model["dropout"]
        if not isinstance(d, (int, float)) or not (0.0 <= float(d) < 1.0):
            errors.append(f"model.dropout must be in [0, 1), got {d!r}")


def _check_training(train: dict, spec: DatasetSpec | None, errors: list) -> None:
    for key in _TRAIN_REQUIRED:
        if key not in train:
            errors.append(f"training.{key} is required")
    _check_numeric_fields(train, errors)
    _check_ranges(train, errors)
    _check_model_selection(train, spec, errors)
    _check_loss(train, spec, errors)


def _check_numeric_fields(train: dict, errors: list) -> None:
    for key in _TRAIN_POSITIVE_INT:
        v = train.get(key)
        if key in train and (not isinstance(v, int) or v <= 0):
            errors.append(
                f"training.{key} must be a positive integer, got {v!r}. "
                f"(Hint: YAML parses '1e-4' as a string — use '0.0001' instead)"
            )
    for key in _TRAIN_POSITIVE_FLOAT:
        if key in train:
            _check_float(key, train[key], errors)


def _check_float(key: str, v, errors: list) -> None:
    if isinstance(v, str):
        errors.append(
            f"training.{key}={v!r} is a string, not a number. "
            f"YAML parses scientific notation as strings — write '0.0001' not '1e-4'"
        )
        return
    if not isinstance(v, (int, float)):
        errors.append(f"training.{key} must be numeric, got {type(v).__name__}: {v!r}")
        return
    fv = float(v)
    if key in _TRAIN_NON_NEGATIVE:
        if fv < 0:
            errors.append(f"training.{key} must be >= 0, got {fv}")
    elif fv <= 0:
        errors.append(f"training.{key} must be positive, got {fv}")


def _check_ranges(train: dict, errors: list) -> None:
    ls = train.get("label_smoothing")
    if isinstance(ls, (int, float)) and not (0.0 <= float(ls) <= 0.5):
        errors.append(f"training.label_smoothing should be in [0, 0.5], got {ls}")

    s = train.get("seed")
    if s is not None and (not isinstance(s, int) or s < 0):
        errors.append(f"training.seed must be a non-negative integer or null, got {s!r}")

    if "focal_gamma" in train:
        g = train["focal_gamma"]
        if not isinstance(g, (int, float)) or float(g) < 0:
            errors.append(f"training.focal_gamma must be a number >= 0, got {g!r}")

    p = train.get("early_stopping_patience")
    if p is not None and (not isinstance(p, int) or p <= 0):
        errors.append(f"training.early_stopping_patience must be a positive int or null, got {p!r}")

    we, ep = train.get("warmup_epochs"), train.get("epochs")
    if isinstance(we, int) and isinstance(ep, int) and we >= ep:
        errors.append(
            f"training.warmup_epochs ({we}) >= training.epochs ({ep}). "
            "The cosine phase would have 0 epochs. Reduce warmup_epochs or increase epochs."
        )


def _check_model_selection(train: dict, spec: DatasetSpec | None, errors: list) -> None:
    if "select_by" not in train:
        return
    sb = str(train["select_by"]).lower()
    if sb not in ("f1", "f1_optimal", "auc"):
        errors.append(f"training.select_by must be 'f1', 'f1_optimal' or 'auc', "
                      f"got {train['select_by']!r}")
    elif sb == "f1_optimal" and spec is not None and not spec.make_task().searches_threshold:
        errors.append(f"training.select_by: f1_optimal needs a tunable threshold, which "
                      f"the {spec.task} task of {spec.name!r} does not have; use 'f1'")


def _check_loss(train: dict, spec: DatasetSpec | None, errors: list) -> None:
    if spec is not None and train.get("loss") is not None:
        try:   # the task is the single source of truth for which losses are valid
            spec.make_task().build_criterion(str(train["loss"]))
        except ValueError as exc:
            errors.append(f"training.loss: {exc}")

    if "pos_weight" not in train:
        return
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
    elif spec is not None:
        _check_pos_weight_against(spec, pw, errors)


def _check_pos_weight_against(spec: DatasetSpec, pw, errors: list) -> None:
    if not spec.make_task().sigmoid_outputs:
        errors.append(f"training.pos_weight only applies to multi-label/binary tasks; "
                      f"dataset {spec.name!r} is {spec.task}")
    elif isinstance(pw, str) and spec.class_counts is None:
        errors.append(f"training.pos_weight: auto is not available for dataset "
                      f"{spec.name!r} (it exposes no class counts)")
    elif isinstance(pw, (list, tuple)) and len(pw) != spec.num_classes:
        errors.append(f"training.pos_weight has {len(pw)} values but dataset "
                      f"{spec.name!r} has {spec.num_classes} classes")


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


# monitor.* (docs/inspeccion.md): key → (type, minimum)
_MONITOR_KEYS: dict[str, tuple[type, float | None]] = {
    "enabled": (bool, None), "embeddings": (bool, None), "layers": (bool, None),
    "samples": (int, 1), "every": (int, 1), "seed": (int, 0), "thumbnail": (int, 8),
    "gradcam": (int, 0), "full_val_max": (int, 0), "system_interval_s": (float, 0),
}


def _check_monitor(monitor: dict, errors: list) -> None:
    if not isinstance(monitor, dict):
        errors.append(f"monitor must be a mapping, got {type(monitor).__name__}")
        return
    for key, value in monitor.items():
        if key not in _MONITOR_KEYS:
            errors.append(f"monitor.{key} is not a known key; "
                          f"available: {sorted(_MONITOR_KEYS)}")
            continue
        kind, minimum = _MONITOR_KEYS[key]
        ok = (isinstance(value, bool) if kind is bool
              else isinstance(value, (int, float)) and not isinstance(value, bool)
              and (kind is float or isinstance(value, int)))
        if not ok:
            errors.append(f"monitor.{key} must be {kind.__name__}, got {value!r}")
        elif minimum is not None and value < minimum:
            errors.append(f"monitor.{key} must be >= {minimum}, got {value!r}")
