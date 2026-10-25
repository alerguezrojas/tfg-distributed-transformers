"""Synthetic ``inspect_<ts>/`` directory that follows the contract of docs/inspeccion.md.

Writes ``monitor_set.npz`` and one ``epoch_EEE.npz`` per epoch with **every** key of the
contract (dtypes and shapes included), so the web and the tests can develop against it without
training anything. The probabilities improve with the epoch (some samples never do), so the
curves, the calibration and the cartography have something to show.

    python tests/fixtures/make_inspect_fixture.py OUT_DIR [multilabel|binary|multiclass]
"""
import sys
from pathlib import Path

import numpy as np

TASKS = ("multilabel", "binary", "multiclass")


def _targets(rng, task: str, m: int, c: int) -> np.ndarray:
    if task == "multiclass":
        return np.eye(c, dtype=np.uint8)[rng.integers(0, c, m)]
    if task == "binary":
        return rng.integers(0, 2, (m, 1)).astype(np.uint8)
    y = (rng.random((m, c)) < 0.3).astype(np.uint8)
    y[np.arange(m), rng.integers(0, c, m)] = 1          # at least one label per sample
    return y


def _probs(rng, task: str, y: np.ndarray, quality: float) -> np.ndarray:
    """Probabilities that agree with the labels more as ``quality`` (0..1) grows."""
    noise = rng.normal(0.0, 1.0, y.shape)
    logits = (2 * y.astype(np.float64) - 1) * 4.0 * quality + noise
    if task == "multiclass":
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    return 1.0 / (1.0 + np.exp(-logits))


def _loss(task: str, p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    if task == "multiclass":
        return -(y * np.log(p)).sum(axis=1)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(axis=1)


def make_inspect_fixture(out_dir, ts: str = "25102026_120000", task: str = "multilabel",
                         m: int = 48, c: int = 5, n_val: int = 200, epochs=(1, 2, 3),
                         dim: int = 16, thumb: int = 8, n_panel: int = 4, cam_hw: int = 4,
                         seed: int = 0) -> Path:
    """Create ``<out_dir>/inspect_<ts>/`` and return its path."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; available: {list(TASKS)}")
    c = 1 if task == "binary" else c
    rng = np.random.default_rng(seed)
    d = Path(out_dir) / f"inspect_{ts}"
    d.mkdir(parents=True, exist_ok=True)

    y = _targets(rng, task, m, c)
    panel = np.sort(rng.choice(m, size=min(n_panel, m), replace=False)).astype(np.int64)
    hard = rng.random(m) < 0.15                          # samples that never improve
    np.savez_compressed(
        d / "monitor_set.npz",
        indices=np.sort(rng.choice(10 * m, size=m, replace=False)).astype(np.int64),
        targets=y,
        thumbs=rng.integers(0, 256, (m, thumb, thumb, 3), dtype=np.uint8),
        panel=panel,
        classes=np.array([f"class_{i}" for i in range(c)]),
        task=np.array(task),
    )

    y_val = _targets(rng, task, n_val, c)
    centres = rng.normal(0.0, 1.0, (c if task != "binary" else 2, dim))
    cls = y.argmax(axis=1) if task == "multiclass" else y.sum(axis=1) % len(centres)
    for k, epoch in enumerate(epochs):
        quality = (k + 1) / len(epochs)
        q = np.where(hard, 0.05, quality)[:, None]       # per-sample quality
        p = np.concatenate([_probs(rng, task, y[i:i + 1], float(q[i, 0])) for i in range(m)])
        p_val = _probs(rng, task, y_val, quality)
        np.savez_compressed(
            d / f"epoch_{epoch:03d}.npz",
            probs=p.astype(np.float16),
            loss=_loss(task, p, y).astype(np.float32),
            embeddings=(centres[cls] * quality + rng.normal(0, 0.5, (m, dim))).astype(np.float16),
            cam=rng.integers(0, 256, (len(panel), cam_hw, cam_hw), dtype=np.uint8),
            cam_class=p[panel].argmax(axis=1).astype(np.int64),
            val_probs=p_val.astype(np.float16),
            val_targets=y_val,
            threshold=np.float32(0.5),
        )
    return d


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print(make_inspect_fixture(sys.argv[1], task=sys.argv[2] if len(sys.argv) > 2 else "multilabel"))
