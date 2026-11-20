"""project_2d: PCA por SVD determinista y UMAP opcional."""
import importlib.util

import numpy as np
import pytest

from src.analysis.projection import project_2d


def test_points_on_a_line_project_onto_the_first_axis():
    t = np.linspace(-3, 5, 20)
    x = np.stack([t, 2 * t, -t], axis=1)                 # recta en R^3
    out = project_2d(x)
    assert out.shape == (20, 2)
    assert np.abs(out[:, 1]).max() < 1e-9                # sin varianza en la 2.ª componente
    # La distancia entre puntos se conserva en el primer eje.
    assert np.abs(out[:, 0]).max() == pytest.approx(np.linalg.norm([1, 2, -1]) * 4.0)
    assert out[:, 0].mean() == pytest.approx(0.0, abs=1e-9)


def test_pca_is_centred_and_orders_by_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 6)) * np.array([5, 3, 1, 0.5, 0.2, 0.1]) + 100
    out = project_2d(x)
    assert out.mean(axis=0) == pytest.approx([0, 0], abs=1e-9)
    assert out[:, 0].var() > out[:, 1].var() > 0
    assert out[:, 0].var() == pytest.approx(x[:, 0].var(), rel=0.2)


def test_pca_is_deterministic_and_sign_fixed():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 8))
    a, b = project_2d(x), project_2d(x.copy())
    assert np.array_equal(a, b)
    # Invertir los datos solo invierte el signo de las coordenadas, pero el signo del eje es fijo:
    # el componente de mayor |carga| es positivo, así que x y -x dan la MISMA proyección salvo
    # el reflejo de los puntos.
    assert np.allclose(project_2d(-x), -a)
    # La carga de mayor magnitud de cada componente es positiva.
    centred = x - x.mean(axis=0)
    for k in range(2):
        comp = np.linalg.lstsq(centred, a[:, k], rcond=None)[0]
        assert comp[np.abs(comp).argmax()] > 0


def test_float16_input_and_degenerate_shapes():
    x16 = np.random.default_rng(2).normal(size=(10, 4)).astype(np.float16)
    assert project_2d(x16).shape == (10, 2)
    one_dim = project_2d(np.arange(6, dtype=float)[:, None])
    assert one_dim.shape == (6, 2) and np.all(one_dim[:, 1] == 0)
    assert project_2d(np.ones((1, 5))).shape == (1, 2)
    assert np.all(project_2d(np.ones((4, 3))) == 0)       # sin varianza


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="unknown projection method"):
        project_2d(np.zeros((3, 3)), method="tsne")


@pytest.mark.skipif(importlib.util.find_spec("umap") is not None, reason="umap instalado")
def test_umap_without_package_gives_clear_error():
    with pytest.raises(ValueError, match="umap-learn"):
        project_2d(np.zeros((5, 3)), method="umap")
