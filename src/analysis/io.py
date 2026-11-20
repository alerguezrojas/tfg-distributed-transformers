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
    threshold_source: str | None = None    # checkpoint | val | test (None: unknown / no threshold)


def save_predictions(path, probs, targets, threshold: float, classes, task: str,
                     ordinal: bool = False, groups: dict | None = None,
                     threshold_source: str | None = None) -> Path:
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
                        task=np.asarray(task), ordinal=np.asarray(bool(ordinal)), **arrays,
                        **({"threshold_source": np.asarray(threshold_source)}
                           if threshold_source else {}))
    return path


def load_predictions(path) -> Predictions:
    """Read an ``.npz`` written by :func:`save_predictions`."""
    d = _read(Path(path), ("probs", "targets", "threshold", "classes", "task"))
    return Predictions(
        probs=d["probs"], targets=d["targets"], threshold=float(d["threshold"]),
        classes=[str(c) for c in d["classes"]], task=str(d["task"]),
        ordinal=bool(d["ordinal"]) if "ordinal" in d else False,
        threshold_source=str(d["threshold_source"]) if "threshold_source" in d else None,
        groups={k[len(_GROUP_PREFIX):]: v for k, v in d.items() if k.startswith(_GROUP_PREFIX)})


# ── Where the test predictions live (``test_preds_<ts>.npz`` in the run's log folder) ──────────

#: Stamp format of the run artefacts (``tipo_DDMMYYYY_HHMMSS.{log,csv}``).
TS_FORMAT = "%d%m%Y_%H%M%S"
_STAMP = re.compile(r"^\d{8}_\d{6}$")
# Files that mark a run in its log folder: train_<ts>.log, train_deep_<ts>.log, events_<ts>.jsonl.
_RUN_FILE = re.compile(r"^(?:train_(?:deep_)?|events_)(\d{8}_\d{6})\.(?:log|jsonl)$")
_TEST_PREDS = re.compile(r"^test_preds_(\d{8}_\d{6})\.npz$")


def _parse_ts(ts: str) -> datetime | None:
    """Run stamp as a datetime (DDMMYYYY_HHMMSS, or the legacy YYYYMMDD_HHMMSS)."""
    for fmt in (TS_FORMAT, "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def run_location(checkpoint, checkpoint_dir=None, env: str | None = None
                 ) -> tuple[Path | None, str | None]:
    """``(log folder, stamp)`` of the run a checkpoint belongs to; ``None`` for what is unknown.

    The builder writes the checkpoints to ``<checkpoint.dir>/<run_subdir>/<ts>/{best,last}.pt``
    (older runs: ``checkpoint_epoch_NNN.pt`` straight in ``<run_subdir>/``) and the logs to
    ``logs/<output.env>/<run_subdir>/``. A checkpoint folder named like a stamp *is* the run's
    stamp and its parent is the ``run_subdir``. With the config's ``checkpoint_dir`` and ``env``
    the log folder is built the way the builder does; without them it is guessed by swapping a
    ``checkpoints`` path component for ``logs`` (right only if ``checkpoint.dir`` is
    ``checkpoints/<env>``).
    """
    folder = Path(checkpoint).parent
    stamp = None
    if _STAMP.match(folder.name):
        stamp, folder = folder.name, folder.parent
    if checkpoint_dir is not None and env:
        for a, b in ((folder, Path(checkpoint_dir)), (folder.resolve(), Path(checkpoint_dir).resolve())):
            if a.is_relative_to(b):
                return Path("logs") / env / a.relative_to(b), stamp
    parts = folder.parts
    if "checkpoints" not in parts:
        return None, stamp
    i = len(parts) - 1 - parts[::-1].index("checkpoints")      # the innermost one
    return Path(*parts[:i], "logs", *parts[i + 1:]), stamp


def run_stamp_for(run_dir, checkpoint_mtime: float | None = None) -> tuple[str | None, bool]:
    """``(stamp, ambiguous)`` of the run in ``run_dir`` that wrote a checkpoint of the old shared
    folder: the latest run (``train_``/``train_deep_`` log or events file) that started before the
    checkpoint was saved, or the latest overall without a modification time.

    Checkpoint names repeat from run to run, so the modification time is what tells which run
    wrote it. ``ambiguous`` is True when several runs were possible or none started before the
    file (e.g. copied with scp). ``(None, False)`` if the folder has no run.
    """
    d = Path(run_dir)
    runs = []
    if d.is_dir():
        found = {m.group(1) for f in d.iterdir() if (m := _RUN_FILE.match(f.name))}
        runs = sorted((t, s) for s in found if (t := _parse_ts(s)))
    if not runs:
        return None, False
    before = runs
    if checkpoint_mtime is not None:
        before = [r for r in runs if r[0].timestamp() <= checkpoint_mtime]
    return (before or runs)[-1][1], len(runs) > 1 or not before


def default_predictions_path(checkpoint, now: datetime | None = None, checkpoint_dir=None,
                             env: str | None = None, warn=None) -> Path:
    """Where ``eval --save-predictions`` (no path) writes: ``<run log folder>/test_preds_<ts>.npz``.

    Folder and stamp come from :func:`run_location` (stamp folder of the checkpoint) or, for the
    old shared folder, from :func:`run_stamp_for`. If either cannot be deduced the file goes to
    ``logs/`` with the stamp of the evaluation. ``warn(message)`` is called when the stamp is a
    guess (several possible runs) or the fallback is used.
    """
    ckpt = Path(checkpoint)
    run_dir, stamp = run_location(ckpt, checkpoint_dir, env)
    if run_dir is not None and stamp is None:
        try:
            mtime = ckpt.stat().st_mtime
        except OSError:
            mtime = None
        stamp, ambiguous = run_stamp_for(run_dir, mtime)
        if stamp and ambiguous and warn:
            warn(f"several runs in {run_dir}: assuming {stamp} by the checkpoint's date")
    if (run_dir is None or stamp is None) and warn:
        warn("could not deduce the run of the checkpoint: using the evaluation's stamp"
             + ("" if run_dir else " in logs/"))
    return (run_dir or Path("logs")) / (
        f"test_preds_{stamp or (now or datetime.now()).strftime(TS_FORMAT)}.npz")


def list_test_predictions(root) -> list[tuple[str, Path]]:
    """Every ``test_preds_<ts>.npz`` under ``<root>/logs`` as ``(stamp, path)``."""
    logs = Path(root) / "logs"
    if not logs.is_dir():
        return []
    return [(m.group(1), f) for f in sorted(logs.rglob("test_preds_*.npz"))
            if (m := _TEST_PREDS.match(f.name))]
