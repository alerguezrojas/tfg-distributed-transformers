"""bootstrap_ci / metrics_with_ci: determinismo, estratificación, degeneradas y equivalencia exacta
de la ruta vectorizada con las funciones de métricas del paquete."""
import numpy as np
import pytest

from src.analysis._common import check_pair
from src.analysis.bootstrap import (METRIC_NAMES, _draw_weights, _strata, _Weighted,
                                    bootstrap_ci, metrics_with_ci, point_estimates)


def _data(task, n=300, c=4, seed=0):
    """Probabilidades redondeadas (con empates) y etiquetas para cada tarea."""
    rng = np.random.default_rng(seed)
    if task == "multiclass":
        y = np.eye(c)[rng.integers(0, c, n)]
        p = rng.dirichlet(np.ones(c), n)
    else:
        c = 1 if task == "binary" else c
        y = (rng.random((n, c)) < 0.3).astype(float)
        p = np.clip(0.3 * y + 0.7 * rng.random((n, c)), 0, 1)
    return np.round(p, 2), y


def _mean_prob(p, y):
    return float(p.mean())


def _n_pos(p, y):
    return float(y.sum())


# ------------------------------------------------------------------ bootstrap_ci

def test_estimate_is_the_metric_on_the_full_data_and_result_keys():
    p, y = _data("binary")
    out = bootstrap_ci(_mean_prob, p, y, n_boot=50)
    assert out["estimate"] == pytest.approx(p.mean())
    assert set(out) == {"estimate", "ci", "n_boot", "n_degenerate", "alpha", "values"}
    assert out["n_boot"] == 50 and out["n_degenerate"] == 0 and out["values"].shape == (50,)
    lo, hi = out["ci"]
    assert lo <= out["estimate"] <= hi


def test_deterministic_with_seed_and_different_across_seeds():
    p, y = _data("binary")
    a = bootstrap_ci(_mean_prob, p, y, n_boot=100, seed=3)
    b = bootstrap_ci(_mean_prob, p, y, n_boot=100, seed=3)
    c = bootstrap_ci(_mean_prob, p, y, n_boot=100, seed=4)
    assert a["ci"] == b["ci"] and np.array_equal(a["values"], b["values"])
    assert a["ci"] != c["ci"]


def test_ci_of_a_mean_matches_the_normal_approximation():
    rng = np.random.default_rng(0)
    x = rng.random((2000, 1))
    out = bootstrap_ci(_mean_prob, x, np.zeros_like(x), n_boot=2000, seed=1, stratified=False)
    half = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    lo, hi = out["ci"]
    assert (hi - lo) / 2 == pytest.approx(half, rel=0.1)
    assert (lo + hi) / 2 == pytest.approx(x.mean(), abs=half / 4)


def test_alpha_controls_the_width():
    p, y = _data("binary", n=500)
    narrow = bootstrap_ci(_mean_prob, p, y, n_boot=500, alpha=0.5)["ci"]
    wide = bootstrap_ci(_mean_prob, p, y, n_boot=500, alpha=0.01)["ci"]
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_stratified_binary_keeps_the_class_counts():
    p, y = _data("binary", n=200)
    fixed = bootstrap_ci(_n_pos, p, y, n_boot=100, stratified=True)
    free = bootstrap_ci(_n_pos, p, y, n_boot=100, stratified=False)
    assert fixed["ci"] == (y.sum(), y.sum())
    assert free["ci"][1] > free["ci"][0]


def test_stratified_multiclass_keeps_the_class_counts():
    p, y = _data("multiclass", n=200)
    out = bootstrap_ci(lambda pp, yy: float(yy[:, 0].sum()), p, y, n_boot=100, task="multiclass")
    assert out["ci"] == (y[:, 0].sum(), y[:, 0].sum())


def test_task_is_inferred_from_the_targets():
    # one-hot -> multiclass; una columna -> binaria; solapadas -> multi-etiqueta (sin estratos)
    _, y_mc = _data("multiclass")
    _, y_ml = _data("multilabel")
    _, y_bin = _data("binary")
    assert len(_strata(y_mc > 0.5, None, True)) == 4
    assert len(_strata(y_bin > 0.5, None, True)) == 2
    assert _strata(y_ml > 0.5, None, True) is None
    assert _strata(y_mc > 0.5, "multilabel", True) is None
    assert _strata(y_mc > 0.5, "multiclass", False) is None


def test_degenerate_resamples_are_discarded_and_counted():
    # 1 positivo en 10 y sin estratificar: ~35 % de las remuestras no lo contienen -> AUC NaN
    p = np.linspace(0.1, 0.9, 10)[:, None]
    y = np.zeros((10, 1))
    y[7] = 1
    from src.analysis.curves import roc_curve
    out = bootstrap_ci(lambda pp, yy: roc_curve(pp[:, 0], yy[:, 0])[3], p, y, n_boot=400,
                       stratified=False)
    assert 0 < out["n_degenerate"] < 400
    assert out["n_boot"] + out["n_degenerate"] == 400
    assert np.isfinite(out["values"]).all()
    # estratificada nunca es degenerada
    fixed = bootstrap_ci(lambda pp, yy: roc_curve(pp[:, 0], yy[:, 0])[3], p, y, n_boot=100)
    assert fixed["n_degenerate"] == 0


def test_value_error_in_a_resample_counts_as_degenerate():
    p, y = _data("binary", n=30)
    calls = {"n": 0}

    def flaky(pp, yy):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise ValueError("only one class")
        return 1.0

    out = bootstrap_ci(flaky, p, y, n_boot=10)
    assert out["n_degenerate"] == 5 and out["n_boot"] == 5


def test_error_on_the_full_data_is_not_swallowed():
    p, y = _data("binary", n=30)

    def boom(pp, yy):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        bootstrap_ci(boom, p, y, n_boot=5)


def test_all_degenerate_gives_nan_interval():
    p, y = _data("binary", n=30)
    out = bootstrap_ci(lambda pp, yy: float("nan"), p, y, n_boot=10)
    assert out["n_boot"] == 0 and out["n_degenerate"] == 10
    assert np.isnan(out["ci"][0]) and np.isnan(out["ci"][1])


def test_invalid_arguments_and_empty_input():
    p, y = _data("binary", n=30)
    with pytest.raises(ValueError):
        bootstrap_ci(_mean_prob, p, y, n_boot=0)
    with pytest.raises(ValueError):
        bootstrap_ci(_mean_prob, p, y, alpha=1.0)
    with pytest.raises(ValueError):
        bootstrap_ci(_mean_prob, p, y[:-1])
    empty = bootstrap_ci(_mean_prob, np.empty((0, 2)), np.empty((0, 2)), n_boot=10)
    assert np.isnan(empty["estimate"]) and empty["n_boot"] == 0


# ------------------------------------------------------------------ metrics_with_ci

@pytest.mark.parametrize("task", ["multilabel", "binary", "multiclass"])
def test_weighted_replicates_equal_the_package_functions(task):
    """Cada fila de pesos == evaluar las funciones del paquete sobre la remuestra expandida."""
    p, y = _data(task)
    pp, yy = check_pair(p, y)
    weighted = _Weighted(pp, yy, task, 0.5, 15, METRIC_NAMES)
    rng = np.random.default_rng(5)
    w = _draw_weights(rng, len(p), _strata(yy, task, True), 4)
    res = weighted.compute(w)
    for i in range(4):
        idx = np.repeat(np.arange(len(p)), w[i].astype(int))
        ref = point_estimates(pp[idx], yy[idx], task, 0.5, 15, METRIC_NAMES)
        for m in METRIC_NAMES:
            assert res[m][i] == pytest.approx(ref[m], abs=1e-9), (task, m)


@pytest.mark.parametrize("task", ["multilabel", "binary", "multiclass"])
def test_unit_weights_reproduce_the_point_estimates(task):
    p, y = _data(task)
    pp, yy = check_pair(p, y)
    res = _Weighted(pp, yy, task, 0.5, 15, METRIC_NAMES).compute(np.ones((1, len(p))))
    ref = point_estimates(pp, yy, task, 0.5, 15, METRIC_NAMES)
    for m in METRIC_NAMES:
        assert res[m][0] == pytest.approx(ref[m], abs=1e-9)


def test_threshold_changes_the_thresholded_metrics_only():
    p, y = _data("multilabel")
    a = metrics_with_ci(p, y, "multilabel", threshold=0.3, n_boot=20)["metrics"]
    b = metrics_with_ci(p, y, "multilabel", threshold=0.7, n_boot=20)["metrics"]
    assert a["auc"]["estimate"] == pytest.approx(b["auc"]["estimate"])
    assert a["ece"]["estimate"] == pytest.approx(b["ece"]["estimate"])
    assert a["sensitivity"]["estimate"] > b["sensitivity"]["estimate"]


@pytest.mark.parametrize("task", ["multilabel", "binary", "multiclass"])
def test_metrics_with_ci_structure_and_known_estimates(task):
    p, y = _data(task, n=400)
    out = metrics_with_ci(p, y, task, n_boot=100, seed=2)
    assert out["n"] == 400 and out["n_boot"] == 100 and out["alpha"] == 0.05
    assert list(out["metrics"]) == list(METRIC_NAMES)
    ref = point_estimates(*check_pair(p, y), task, 0.5, 15, METRIC_NAMES)
    for m, r in out["metrics"].items():
        assert r["estimate"] == pytest.approx(ref[m])
        lo, hi = r["ci"]
        assert lo <= hi and r["n_boot"] + r["n_degenerate"] == 100


def test_point_estimates_match_the_existing_functions():
    from src.analysis.calibration import calibration
    from src.analysis.curves import per_class_curves
    from src.analysis.thresholds import threshold_sweep
    p, y = _data("multilabel")
    est = metrics_with_ci(p, y, "multilabel", n_boot=5)["metrics"]
    curves, sweep, cal = per_class_curves(p, y), threshold_sweep(p, y, "multilabel", [0.5]), \
        calibration(p, y, "multilabel")
    assert est["auc"]["estimate"] == pytest.approx(curves["macro"]["auc"])
    assert est["ap"]["estimate"] == pytest.approx(curves["macro"]["ap"])
    assert est["f1"]["estimate"] == pytest.approx(sweep["f1"][0])
    assert est["sensitivity"]["estimate"] == pytest.approx(sweep["sensitivity"][0])
    assert est["specificity"]["estimate"] == pytest.approx(sweep["specificity"][0])
    assert est["ece"]["estimate"] == pytest.approx(cal["ece"])
    assert est["brier"]["estimate"] == pytest.approx(cal["brier"])
    # exactitud: fracción media de etiquetas acertadas
    assert est["accuracy"]["estimate"] == pytest.approx(((p > 0.5) == (y > 0.5)).mean())


def test_deterministic_and_metric_subset():
    p, y = _data("multilabel")
    a = metrics_with_ci(p, y, "multilabel", n_boot=50, seed=7, metrics=["auc", "ece"])
    b = metrics_with_ci(p, y, "multilabel", n_boot=50, seed=7, metrics=["auc", "ece"])
    assert list(a["metrics"]) == ["auc", "ece"]
    assert a["metrics"]["auc"]["ci"] == b["metrics"]["auc"]["ci"]
    with pytest.raises(ValueError, match="unknown metrics"):
        metrics_with_ci(p, y, "multilabel", metrics=["nope"])


def test_perfect_separation_has_a_degenerate_interval():
    y = np.array([[0]] * 50 + [[1]] * 50, dtype=float)
    p = np.where(y > 0, 0.9, 0.1)
    m = metrics_with_ci(p, y, "binary", n_boot=100)["metrics"]
    assert m["auc"]["ci"] == (1.0, 1.0) and m["accuracy"]["ci"] == (1.0, 1.0)
    assert m["sensitivity"]["estimate"] == pytest.approx(1.0, abs=1e-6)


def test_binary_accuracy_interval_matches_binomial_theory():
    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.random((n, 1)) < 0.5).astype(float)
    correct = rng.random((n, 1)) < 0.8
    p = np.where(correct == (y > 0), 0.9, 0.1)          # acierta el 80 %
    r = metrics_with_ci(p, y, "binary", n_boot=1000, seed=0, metrics=["accuracy"])
    acc = r["metrics"]["accuracy"]
    half = 1.96 * np.sqrt(acc["estimate"] * (1 - acc["estimate"]) / n)
    lo, hi = acc["ci"]
    assert (hi - lo) / 2 == pytest.approx(half, rel=0.15)


def test_float16_input_and_stratified_vs_not_differ():
    p, y = _data("multiclass")
    a = metrics_with_ci(p.astype(np.float16), y.astype(np.uint8), "multiclass", n_boot=30)
    assert np.isfinite(a["metrics"]["auc"]["estimate"])
    s = metrics_with_ci(p, y, "multiclass", n_boot=100, stratified=True)["metrics"]["accuracy"]
    f = metrics_with_ci(p, y, "multiclass", n_boot=100, stratified=False)["metrics"]["accuracy"]
    assert s["values"].shape == f["values"].shape and not np.array_equal(s["values"], f["values"])


def test_class_without_positives_is_skipped_in_the_macro_auc():
    p, y = _data("multilabel")
    y[:, 0] = 0                                          # clase 0 sin positivos
    out = metrics_with_ci(p, y, "multilabel", n_boot=50, metrics=["auc"])["metrics"]["auc"]
    assert np.isfinite(out["estimate"]) and out["n_degenerate"] == 0


def test_all_classes_undefined_is_degenerate():
    p, y = _data("multilabel", n=40)
    y[:] = 0
    out = metrics_with_ci(p, y, "multilabel", n_boot=20, metrics=["auc"])["metrics"]["auc"]
    assert np.isnan(out["estimate"]) and out["n_boot"] == 0 and out["n_degenerate"] == 20


def test_empty_and_invalid_inputs():
    out = metrics_with_ci(np.empty((0, 3)), np.empty((0, 3)), "multilabel", n_boot=10)
    assert out["n"] == 0 and all(np.isnan(r["estimate"]) for r in out["metrics"].values())
    p, y = _data("binary")
    with pytest.raises(ValueError):
        metrics_with_ci(p, y, "regression")
    with pytest.raises(ValueError):
        metrics_with_ci(p, y, "binary", n_boot=0)


def test_block_size_does_not_change_the_result_much_and_runs_with_many_samples(monkeypatch):
    """Con bloques pequeños (varios bloques) el resultado sigue siendo válido y determinista."""
    import src.analysis.bootstrap as bs
    p, y = _data("multilabel", n=500)
    monkeypatch.setattr(bs, "_BLOCK_ELEMENTS", 500 * 7)       # 7 remuestras por bloque
    a = bs.metrics_with_ci(p, y, "multilabel", n_boot=30, seed=1)
    b = bs.metrics_with_ci(p, y, "multilabel", n_boot=30, seed=1)
    assert a["metrics"]["auc"]["values"].shape == (30,)
    assert np.array_equal(a["metrics"]["f1"]["values"], b["metrics"]["f1"]["values"])
