"""2-D projection of the embeddings."""
import numpy as np

from src.analysis._common import as_matrix


def project_2d(embeddings, method: str = "pca") -> np.ndarray:
    """Project (M, D) embeddings to (M, 2).

    ``"pca"``: NumPy SVD of the centred data, first two principal components. Deterministic:
    the sign of every component is fixed so that its largest-magnitude loading is positive
    (SVD signs are arbitrary), so the same data always gives the same plot. If the data have
    fewer than two components (M < 2 or D < 2) the missing columns are zero.

    ``"umap"``: only if ``umap-learn`` is installed (``random_state=0``); otherwise ``ValueError``.
    Any other name is also a ``ValueError``.
    """
    x = as_matrix(embeddings, "embeddings")
    if method == "pca":
        return _pca(x)
    if method == "umap":
        try:
            import umap
        except ImportError:
            raise ValueError("method='umap' needs the optional package umap-learn "
                             "(not installed); use method='pca'") from None
        return np.asarray(umap.UMAP(n_components=2, random_state=0).fit_transform(x),
                          dtype=np.float64)
    raise ValueError(f"unknown projection method {method!r}; available: 'pca', 'umap'")


def _pca(x: np.ndarray) -> np.ndarray:
    centred = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    comps = vt[:2]
    # Fix the sign: largest |loading| positive.
    pivot = np.abs(comps).argmax(axis=1)
    comps = comps * np.sign(comps[np.arange(len(comps)), pivot])[:, None]
    out = np.zeros((x.shape[0], 2))
    out[:, :len(comps)] = centred @ comps.T
    return out
