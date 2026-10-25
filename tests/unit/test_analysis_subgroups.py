"""subgroup_metrics: métricas por grupo, grupos pequeños, brecha e intervalos."""
import numpy as np
import pytest

from src.analysis.curves import per_class_curves
from src.analysis.subgroups import SUBGROUP_METRICS, subgroup_metrics
from src.analysis.thresholds import threshold_sweep


def _two_groups(seed=0, n=200):
    """Grupo 'A' con predicciones casi perfectas y grupo 'B' casi al azar (binaria)."""
    rng = np.random.default_rng(seed)
    y = (rng.random((2 * n, 1)) < 0.4).astype(float)
    p = np.empty_like(y)
    p[:n] = np.clip(0.15 + 0.7 * y[:n] + rng.normal(0, 0.05, (n, 1)), 0, 1)
    p[n:] = rng.random((n, 1))
    g = np.array(["A"] * n + ["B"] * n)
    return p, y, g


def test_per_group_metrics_equal_the_metrics_on_that_subset():
    p, y, g = _two_groups()
    out = subgroup_metrics(p, y, "binary", g)
    assert [e["group"] for e in out["groups"]] == ["A", "B"]
    for e in out["groups"]:
        rows = g == e["group"]
        sweep = threshold_sweep(p[rows], y[rows], "binary", [0.5])
        assert e["n"] == rows.sum() and not e["small"] and e["ci"] is None
        assert e["metrics"]["auc"] == pytest.approx(per_class_curves(p[rows], y[rows])["macro"]["auc"])
        assert e["metrics"]["f1"] == pytest.approx(sweep["f1"][0])
        assert e["metrics"]["sensitivity"] == pytest.approx(sweep["sensitivity"][0])
        assert e["metrics"]["specificity"] == pytest.approx(sweep["specificity"][0])
        assert set(e["metrics"]) == set(SUBGROUP_METRICS)
    overall = out["overall"]
    assert overall["n"] == len(p) and overall["group"] is None
    assert overall["metrics"]["f1"] == pytest.approx(threshold_sweep(p, y, "binary", [0.5])["f1"][0])


def test_gap_is_max_minus_min_with_group_names():
    p, y, g = _two_groups()
    out = subgroup_metrics(p, y, "binary", g)
    auc = {e["group"]: e["metrics"]["auc"] for e in out["groups"]}
    gap = out["gap"]["auc"]
    assert gap["max_group"] == "A" and gap["min_group"] == "B"
    assert gap["gap"] == pytest.approx(auc["A"] - auc["B"]) and gap["gap"] > 0.3
    assert gap["max"] == pytest.approx(auc["A"]) and gap["min"] == pytest.approx(auc["B"])


def test_small_groups_are_flagged_skip_ci_and_are_left_out_of_the_gap():
    p, y, g = _two_groups(n=100)
    g = g.astype(object)
    g[:5] = "C"                                     # 5 muestras de 'A' pasan a un grupo pequeño
    out = subgroup_metrics(p, y, "binary", g, min_size=20, n_boot=20)
    by = {e["group"]: e for e in out["groups"]}
    assert by["C"]["small"] and by["C"]["n"] == 5 and by["C"]["ci"] is None
    assert not by["A"]["small"] and by["A"]["ci"] is not None
    assert out["gap"]["auc"]["max_group"] != "C" and out["gap"]["auc"]["min_group"] != "C"
    assert by["C"]["metrics"]["auc"] is not None      # el valor sí se calcula


def test_fewer_than_two_reliable_groups_gives_nan_gap():
    p, y, g = _two_groups(n=15)
    out = subgroup_metrics(p, y, "binary", g, min_size=20)
    assert all(e["small"] for e in out["groups"])
    assert np.isnan(out["gap"]["f1"]["gap"]) and out["gap"]["f1"]["max_group"] is None


def test_min_size_boundary_is_inclusive():
    p, y, g = _two_groups(n=20)
    assert not any(e["small"] for e in subgroup_metrics(p, y, "binary", g, min_size=20)["groups"])
    assert all(e["small"] for e in subgroup_metrics(p, y, "binary", g, min_size=21)["groups"])


def test_bootstrap_intervals_present_deterministic_and_wider_for_the_smaller_group():
    p, y, g = _two_groups(n=300)
    keep = np.r_[np.arange(300), 300 + np.arange(60)]            # B queda con 60 muestras
    p, y, g = p[keep], y[keep], g[keep]
    a = subgroup_metrics(p, y, "binary", g, n_boot=200, seed=1)
    b = subgroup_metrics(p, y, "binary", g, n_boot=200, seed=1)
    assert a["n_boot"] == 200 and a["alpha"] == 0.05
    by = {e["group"]: e for e in a["groups"]}
    assert set(by["A"]["ci"]) == set(SUBGROUP_METRICS)
    assert by["A"]["ci"] == {e["group"]: e for e in b["groups"]}["A"]["ci"]
    width = lambda e: e["ci"]["f1"][1] - e["ci"]["f1"][0]       # noqa: E731
    assert width(by["B"]) > width(by["A"])
    assert a["overall"]["ci"] is not None


def test_group_labels_can_be_numbers_and_multiclass_multilabel_work():
    rng = np.random.default_rng(3)
    n, c = 240, 4
    y = np.eye(c)[rng.integers(0, c, n)]
    p = rng.dirichlet(np.ones(c), n)
    ages = np.repeat([30, 50, 70], n // 3)
    out = subgroup_metrics(p, y, "multiclass", ages, n_boot=10)
    assert [e["group"] for e in out["groups"]] == [30, 50, 70]
    assert all(isinstance(e["group"], int) for e in out["groups"])
    y2 = (rng.random((n, c)) < 0.3).astype(float)
    out2 = subgroup_metrics(rng.random((n, c)), y2, "multilabel", ages > 40)
    assert [e["group"] for e in out2["groups"]] == [False, True]


def test_groups_cover_every_sample_once():
    p, y, g = _two_groups()
    out = subgroup_metrics(p, y, "binary", g)
    assert sum(e["n"] for e in out["groups"]) == len(p)


def test_invalid_groups_and_arguments():
    p, y, g = _two_groups(n=30)
    with pytest.raises(ValueError, match="one label per sample"):
        subgroup_metrics(p, y, "binary", g[:-1])
    with pytest.raises(ValueError, match="one label per sample"):
        subgroup_metrics(p, y, "binary", np.zeros((len(p), 2)))
    bad = np.array([0.0] * (len(p) - 1) + [np.nan])
    with pytest.raises(ValueError, match="NaN"):
        subgroup_metrics(p, y, "binary", bad)
    with pytest.raises(ValueError):
        subgroup_metrics(p, y, "binary", g, min_size=0)
    with pytest.raises(ValueError):
        subgroup_metrics(p, y, "regression", g)


def test_group_with_a_single_class_gives_nan_auc_not_an_error():
    p, y, g = _two_groups(n=40)
    y[g == "B"] = 0
    out = subgroup_metrics(p, y, "binary", g, n_boot=10)
    b = {e["group"]: e for e in out["groups"]}["B"]
    assert np.isnan(b["metrics"]["auc"]) and np.isfinite(b["metrics"]["specificity"])
    assert out["gap"]["auc"]["max_group"] is None            # solo A tiene AUC definida
