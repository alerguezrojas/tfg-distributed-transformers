"""Reader of an inspection directory (``logs/<env>/<subdir>/inspect_<ts>/``).

Layout and keys in ``docs/inspeccion.md``. Arrays keep the dtype they were stored with
(float16 probabilities, uint8 targets); the analysis functions upcast as needed. Optional keys
that are absent come back as ``None``.
"""
import re
from dataclasses import dataclass
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
