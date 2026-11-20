"""Reader of an inspection directory (``logs/<env>/<subdir>/inspect_<ts>/``).

Layout and keys in ``docs/inspeccion.md``. Arrays keep the dtype they were stored with
(float16 probabilities, uint8 targets); the analysis functions upcast as needed. Optional keys
that are absent come back as ``None``.
"""
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

_EPOCH_FILE = re.compile(r"^epoch_(\d+)\.npz$")


@dataclass
class MonitorSet:
    """Contents of ``monitor_set.npz``: the fixed subset the monitor follows."""
    indices: np.ndarray          # (M,) int64 positions in the validation partition
    targets: np.ndarray          # (M, C) uint8
    classes: list[str]
    task: str                    # multilabel | binary | multiclass
    thumbs: np.ndarray | None = None   # (M, t, t, 3) uint8
    panel: np.ndarray | None = None    # (P,) int64 positions with Grad-CAM


@dataclass
class EpochData:
    """Contents of one ``epoch_<EEE>.npz``."""
    epoch: int
    probs: np.ndarray            # (M, C) float16
    loss: np.ndarray             # (M,) float32
    embeddings: np.ndarray | None = None    # (M, D) float16
    cam: np.ndarray | None = None           # (P, h, w) uint8
    cam_class: np.ndarray | None = None     # (P,) int64
    val_probs: np.ndarray | None = None     # (N, C) float16
    val_targets: np.ndarray | None = None   # (N, C) uint8
    threshold: float | None = None


def _read(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    """All the arrays of an ``.npz`` (read eagerly so the file is closed)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{path.name}: missing required keys {missing}")
    return data


def load_monitor_set(directory) -> MonitorSet:
    """Read ``monitor_set.npz`` of an inspection directory."""
    d = _read(Path(directory) / "monitor_set.npz", ("indices", "targets", "classes", "task"))
    return MonitorSet(indices=d["indices"], targets=d["targets"],
                      classes=[str(c) for c in d["classes"]], task=str(d["task"]),
                      thumbs=d.get("thumbs"), panel=d.get("panel"))


def list_epochs(directory) -> list[int]:
    """Epochs with a file ``epoch_<EEE>.npz``, ascending (empty if none or no directory)."""
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(int(m.group(1)) for f in d.iterdir() if (m := _EPOCH_FILE.match(f.name)))


def load_epoch(directory, epoch: int) -> EpochData:
    """Read ``epoch_<EEE>.npz`` (``FileNotFoundError`` if that epoch was not inspected)."""
    d = _read(Path(directory) / f"epoch_{int(epoch):03d}.npz", ("probs", "loss"))
    threshold = d.get("threshold")
    return EpochData(epoch=int(epoch), probs=d["probs"], loss=d["loss"],
                     embeddings=d.get("embeddings"), cam=d.get("cam"),
                     cam_class=d.get("cam_class"), val_probs=d.get("val_probs"),
                     val_targets=d.get("val_targets"),
                     threshold=None if threshold is None else float(threshold))


def load_groups(directory, n: int | None = None) -> dict[str, np.ndarray] | None:
    """Read the optional ``groups.npz``: ``{variable: (M,) array}`` aligned with the monitor set.

    One 1-D array per group variable (sex, age band, centre...). Returns ``None`` if the file does
    not exist (the monitor only writes it when the dataset exposes metadata). With ``n`` (the
    monitor-set size), arrays that do not have exactly ``n`` entries raise ``ValueError``: a
    misaligned variable would silently give wrong subgroup numbers.
    """
    path = Path(directory) / "groups.npz"
    if not path.is_file():
        return None
    out = {}
    for name, arr in _read(path, ()).items():
        if arr.ndim != 1 or (n is not None and arr.shape[0] != n):
            raise ValueError(f"groups.npz: {name!r} has shape {arr.shape}; expected ({n},)")
        out[name] = arr
    return out


# ── Test-set predictions (``scripts/eval.py --save-predictions``) ──────────────

_GROUP_PREFIX = "group__"


@dataclass
class Predictions:
    """Contents of the ``.npz`` written by ``eval.py --save-predictions``."""
    probs: np.ndarray            # (N, C) float32
    targets: np.ndarray          # (N, C) uint8
    threshold: float
    classes: list[str]
    task: str                    # multilabel | binary | multiclass
    ordinal: bool
    groups: dict[str, np.ndarray]    # {variable: (N,)}; empty if the dataset has none


def save_predictions(path, probs, targets, threshold: float, classes, task: str,
                     ordinal: bool = False, groups: dict | None = None) -> Path:
    """Write the per-sample test predictions; group variables go under ``group__<name>``."""
    probs, targets = np.asarray(probs, dtype=np.float32), np.asarray(targets)
    if probs.shape != targets.shape or probs.ndim != 2:
        raise ValueError(f"probs {probs.shape} and targets {targets.shape} must be (N, C) and equal")
    arrays = {}
    for name, arr in (groups or {}).items():
        arr = np.asarray(arr)
        if arr.shape != (probs.shape[0],):
            raise ValueError(f"group {name!r} has shape {arr.shape}; expected ({probs.shape[0]},)")
        arrays[_GROUP_PREFIX + name] = arr
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probs=probs, targets=(targets > 0.5).astype(np.uint8),
                        threshold=np.float32(threshold), classes=np.asarray(list(classes), dtype=str),
                        task=np.asarray(task), ordinal=np.asarray(bool(ordinal)), **arrays)
    return path


def load_predictions(path) -> Predictions:
    """Read an ``.npz`` written by :func:`save_predictions`."""
    d = _read(Path(path), ("probs", "targets", "threshold", "classes", "task"))
    return Predictions(
        probs=d["probs"], targets=d["targets"], threshold=float(d["threshold"]),
        classes=[str(c) for c in d["classes"]], task=str(d["task"]),
        ordinal=bool(d["ordinal"]) if "ordinal" in d else False,
        groups={k[len(_GROUP_PREFIX):]: v for k, v in d.items() if k.startswith(_GROUP_PREFIX)})


# ── Where the test predictions live (``test_preds_<ts>.npz`` in the run's log folder) ──────────

#: Stamp format of the run artefacts (``tipo_DDMMYYYY_HHMMSS.{log,csv}``).
TS_FORMAT = "%d%m%Y_%H%M%S"
_TRAIN_LOG = re.compile(r"^train_(\d{8}_\d{6})\.log$")
_TEST_PREDS = re.compile(r"^test_preds_(\d{8}_\d{6})\.npz$")


def _parse_ts(ts: str) -> datetime | None:
    """Run stamp as a datetime (DDMMYYYY_HHMMSS, or the legacy YYYYMMDD_HHMMSS)."""
    for fmt in (TS_FORMAT, "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def run_dir_of_checkpoint(checkpoint) -> Path | None:
    """Log folder of the run a checkpoint belongs to, or ``None`` if it cannot be deduced.

    Checkpoints live in ``<checkpoint.dir>/<env>/<run_subdir>/`` and the logs of the same run in
    ``logs/<env>/<run_subdir>/`` (``TrainingSessionBuilder``), so the folder is the checkpoint's
    own with the ``checkpoints`` component swapped for ``logs``.
    """
    parts = Path(checkpoint).parent.parts
    if "checkpoints" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("checkpoints")      # the innermost one
    return Path(*parts[:i], "logs", *parts[i + 1:])


def run_stamp_for(run_dir, checkpoint_mtime: float | None = None) -> str | None:
    """Stamp of the run in ``run_dir`` that wrote a checkpoint: the latest ``train_<ts>.log`` that
    started before the checkpoint was saved (the latest overall without a modification time).

    Checkpoint names repeat from run to run in the same folder, so the modification time is what
    tells which run wrote it. ``None`` if the folder has no training log.
    """
    d = Path(run_dir)
    runs = []
    if d.is_dir():
        runs = sorted((t, m.group(1)) for f in d.iterdir()
                      if (m := _TRAIN_LOG.match(f.name)) and (t := _parse_ts(m.group(1))))
    if checkpoint_mtime is not None:
        before = [r for r in runs if r[0].timestamp() <= checkpoint_mtime]
        runs = before or runs
    return runs[-1][1] if runs else None


def default_predictions_path(checkpoint, now: datetime | None = None) -> Path:
    """Where ``eval --save-predictions`` (no path) writes: ``<run log folder>/test_preds_<ts>.npz``.

    The folder and the stamp are those of the run that trained the checkpoint (see
    :func:`run_dir_of_checkpoint` and :func:`run_stamp_for`). If either cannot be deduced (a
    checkpoint outside ``checkpoints/``, a folder without training log) the file goes to
    ``logs/`` (next to the checkpoint's own folder name when known) with the stamp of the
    evaluation.
    """
    ckpt = Path(checkpoint)
    run_dir = run_dir_of_checkpoint(ckpt)
    stamp = None
    if run_dir is not None:
        try:
            mtime = ckpt.stat().st_mtime
        except OSError:
            mtime = None
        stamp = run_stamp_for(run_dir, mtime)
    return (run_dir or Path("logs")) / (
        f"test_preds_{stamp or (now or datetime.now()).strftime(TS_FORMAT)}.npz")


def list_test_predictions(root) -> list[tuple[str, Path]]:
    """Every ``test_preds_<ts>.npz`` under ``<root>/logs`` as ``(stamp, path)``."""
    logs = Path(root) / "logs"
    if not logs.is_dir():
        return []
    return [(m.group(1), f) for f in sorted(logs.rglob("test_preds_*.npz"))
            if (m := _TEST_PREDS.match(f.name))]
