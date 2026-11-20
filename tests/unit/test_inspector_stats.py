"""Helpers puros de las vistas clínicas del Inspector (src/web/inspector_stats.py)."""
import numpy as np
import pytest

from src.analysis import (calibration_curve_stats, load_groups, metrics_with_ci,
                          operating_point, per_class_calibration, subgroup_metrics)
from src.web import inspector_stats as s
from tests.fixtures.make_inspect_fixture import make_inspect_fixture


def _binary(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 1)) < 0.3).astype(np.uint8)
    p = 1 / (1 + np.exp(-((2 * y.astype(int) - 1) * 1.2 + rng.normal(0, 1, y.shape))))
    return p, y


def _multiclass(n=300, c=4, seed=1):
    rng = np.random.default_rng(seed)
    y = np.eye(c, dtype=np.uint8)[rng.integers(0, c, n)]
    z = 2.0 * y + rng.normal(0, 1, y.shape)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True), y


def test_fmt_ci_and_note():
    assert s.fmt_ci(0.9123, 0.8871, 0.9345) == "0.912 [0.887, 0.934]"
    assert s.fmt_ci(0.5, float("nan"), float("nan")) == "0.500"
    assert s.fmt_ci(float("nan"), 0.1, 0.2) == "—"
    assert s.ci_note(1000) == "95% CI, percentile bootstrap, n=1,000 resamples (seed 0)."


@pytest.mark.parametrize("task", ["multilabel", "binary", "multiclass"])
def test_macro_ci_matches_metrics_with_ci_and_is_deterministic(task):
    p, y = _multiclass() if task == "multiclass" else _binary()
    a = s.macro_ci(p, y, task, 0.5, 60, seed=3)
    b = s.macro_ci(p, y, task, 0.5, 60, seed=3)
    full = metrics_with_ci(p, y, task, 0.5, 60, seed=3, metrics=s.MACRO_METRICS)["metrics"]
    assert a == b
    for m in s.MACRO_METRICS:
        assert a[m]["estimate"] == pytest.approx(full[m]["estimate"])
        assert (a[m]["lo"], a[m]["hi"]) == pytest.approx(full[m]["ci"])
        assert "values" not in a[m]                    # ligero para cachear


def test_class_ci_multilabel_and_multiclass():
    p, y = _multiclass()
    cis = s.class_ci(p, y, "multiclass", 0.5, 40)
    assert len(cis) == 4 and set(cis[0]) == {"auc", "ap"}          # sin F1/sens/spec por argmax
    rng = np.random.default_rng(0)
    yl = (rng.random((200, 3)) < 0.4).astype(np.uint8)
    pl = np.clip(yl * 0.5 + rng.random((200, 3)) * 0.5, 0, 1)
    cl = s.class_ci(pl, yl, "multilabel", 0.5, 40)
    assert set(cl[0]) == {"auc", "ap", "f1", "sensitivity", "specificity"}
    col = s.ci_column(cl, "auc")
    assert len(col) == 3 and all("[" in v for v in col)
    assert s.ci_column(cis, "f1") == ["—"] * 4


def test_clinical_report_fixed_matches_operating_point():
    p, y = _binary()
    r = s.clinical_report(p, y, 0, "fixed", threshold=0.4, n_boot=0)
    op = operating_point(p, y, "binary", 0.4)
    assert r["ok"] and r["counts"] == {k: int(op[k][0]) for k in ("tp", "fp", "fn", "tn")}
    assert r["metrics"]["sensitivity"]["estimate"] == pytest.approx(op["sensitivity"][0])
    assert r["metrics"]["ppv"]["estimate"] == pytest.approx(op["ppv"][0])
    assert np.isnan(r["metrics"]["ppv"]["lo"])                      # sin n_boot no hay IC
    tp, fp, fn = (r["counts"][k] for k in ("tp", "fp", "fn"))
    assert r["metrics"]["f1"]["estimate"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
    assert r["point"] == pytest.approx((1 - op["specificity"][0], op["sensitivity"][0]))
    assert r["prevalence"]["observed"] == pytest.approx(y.mean())


@pytest.mark.parametrize("criterion,target", [("sensitivity", 0.9), ("sensitivity", 0.95),
                                              ("youden", 0.9)])
def test_clinical_report_criteria(criterion, target):
    p, y = _binary()
    r = s.clinical_report(p, y, 0, criterion, sens_target=target, n_boot=0)
    if criterion == "sensitivity":
        assert r["metrics"]["sensitivity"]["estimate"] >= target - 1e-9
    else:                                       # Youden: ningún otro umbral de la rejilla gana
        j = r["metrics"]["sensitivity"]["estimate"] + r["metrics"]["specificity"]["estimate"] - 1
        for t in np.linspace(0.02, 0.98, 49):
            o = operating_point(p, y, "binary", t)
            assert j >= o["sensitivity"][0] + o["specificity"][0] - 1 - 1e-9


def test_clinical_report_intervals_contain_the_estimate_and_are_deterministic():
    p, y = _binary(600)
    a = s.clinical_report(p, y, 0, "fixed", threshold=0.5, n_boot=300, seed=4)
    b = s.clinical_report(p, y, 0, "fixed", threshold=0.5, n_boot=300, seed=4)
    assert a["metrics"] == b["metrics"]
    for m, v in a["metrics"].items():
        assert v["lo"] <= v["estimate"] <= v["hi"], m
        assert 0 <= v["lo"] and v["hi"] <= 1
    # La sensibilidad es una proporción binomial: su IC debe parecerse al de Wald
    se = a["metrics"]["sensitivity"]
    n_pos = a["counts"]["tp"] + a["counts"]["fn"]
    half = 1.96 * np.sqrt(se["estimate"] * (1 - se["estimate"]) / n_pos)
    assert (se["hi"] - se["lo"]) / 2 == pytest.approx(half, rel=0.35)


def test_clinical_report_prevalence_changes_predictive_values_only():
    p, y = _binary()
    obs = s.clinical_report(p, y, 0, "fixed", threshold=0.5)
    low = s.clinical_report(p, y, 0, "fixed", threshold=0.5, prevalence=0.01, n_boot=100)
    assert low["metrics"]["sensitivity"]["estimate"] == obs["metrics"]["sensitivity"]["estimate"]
    assert low["metrics"]["ppv"]["estimate"] < obs["metrics"]["ppv"]["estimate"]
    assert low["metrics"]["npv"]["estimate"] > obs["metrics"]["npv"]["estimate"]
    assert low["prevalence"] == {"observed": pytest.approx(y.mean()), "used": 0.01}
    ppv = low["metrics"]["ppv"]
    assert ppv["lo"] <= ppv["estimate"] <= ppv["hi"]
    with pytest.raises(ValueError):
        s.clinical_report(p, y, 0, prevalence=1.5)
    with pytest.raises(ValueError):
        s.clinical_report(p, y, 0, criterion="nope")


def test_clinical_report_multiclass_is_one_vs_rest_and_degenerate_class():
    p, y = _multiclass()
    r = s.clinical_report(p, y, 2, "youden", n_boot=50)
    assert r["ok"] and 0 < r["threshold"] < 1
    yy = y.copy()
    yy[:, 3] = 0
    out = s.clinical_report(p, yy, 3)
    assert out["ok"] is False and "no positive" in out["reason"]


def test_confusion_and_point_frames():
    p, y = _binary()
    r = s.clinical_report(p, y, 0, "fixed", threshold=0.5, n_boot=50)
    cm = s.confusion_2x2(r["counts"])
    assert cm.to_numpy().sum() == len(y)
    assert cm.loc["Actual positive", "Predicted positive"] == r["counts"]["tp"]
    assert cm.loc["Actual negative", "Predicted positive"] == r["counts"]["fp"]
    tab = s.point_frame(r)
    assert list(tab["Metric"])[:2] == ["Sensitivity (recall)", "Specificity"]
    assert all("[" in v for v in tab["Value"])


def test_interpret_slope_and_citl():
    assert s.interpret_slope(0.6, 0.05)[0] == "over"
    assert "overconfident" in s.interpret_slope(0.6, 0.05)[1]
    assert s.interpret_slope(1.5, 0.05)[0] == "under"
    assert s.interpret_slope(0.98, 0.05)[0] == "ok"
    assert s.interpret_slope(0.7, 0.3)[0] == "ok"            # IC ancho: compatible con 1
    assert s.interpret_slope(0.7, None)[0] == "over"         # sin error estándar, banda ±0,1
    assert s.interpret_slope(float("nan"))[0] == "none"
    assert "underestimated" in s.interpret_citl(0.5)
    assert "overestimated" in s.interpret_citl(-0.5)
    assert "close to 0" in s.interpret_citl(0.02)
    assert s.interpret_citl(float("nan")) == "Intercept undefined."


def test_slope_interpretation_on_synthetic_overconfident_model():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.5).astype(np.uint8)[:, None]
    z = (2 * y.astype(int) - 1) * 0.5 + rng.normal(0, 1, y.shape)   # logit verdadero
    over = 1 / (1 + np.exp(-3 * z))                         # logits inflados x3
    st = calibration_curve_stats(over, y, "binary")
    assert st["slope"] < 0.6
    assert s.interpret_slope(st["slope"], st["slope_se"])[0] == "over"


def test_calibration_class_table():
    p, y = _multiclass()
    pc = per_class_calibration(p, y, "multiclass", 10)
    st = calibration_curve_stats(p, y, "multiclass")
    df = s.calibration_class_table([f"c{i}" for i in range(4)], pc, st, [0.1, 0.2, 0.3, 0.4])
    assert list(df.columns) == ["Class", "Positives", "ECE", "MCE", "Brier", "ACE (equal mass)",
                                "Slope", "Intercept (CITL)"]
    assert len(df) == 4 and df["Positives"].sum() == len(y)
    assert "ACE (equal mass)" not in s.calibration_class_table(["a"] * 4, pc, st).columns


def test_bin_groups():
    assert s.bin_groups(np.array(["F", "M", "F"])).tolist() == ["F", "M", "F"]
    few = s.bin_groups(np.array([1, 2, 1, 3]))
    assert sorted(set(few)) == ["1", "2", "3"]                     # pocos valores: categórica
    b = s.bin_groups(np.arange(20, 100), 4)
    assert len(set(b)) == 4 and b[0].startswith("Q1") and b[-1].startswith("Q4")
    withnan = s.bin_groups(np.array([np.nan, 1.0, 2.0] + list(range(20))), 3)
    assert withnan[0] == "missing"
    assert s.bin_groups(np.array([1.0, np.nan, 2.0]))[1] == "missing"


def test_subgroup_frames():
    p, y = _binary(300)
    groups = np.where(np.arange(300) % 3 == 0, "F", "M").astype(object)
    groups[:5] = "rare"
    res = subgroup_metrics(p, y, "binary", groups, min_size=20, n_boot=30)
    df = s.subgroup_frame(res)
    assert df["Group"].tolist() == ["All samples", "F", "M", "rare"]
    assert df.loc[3, "n"] == "5 (small)" and "[" not in df.loc[3, "AUC"]   # pequeño: sin IC
    assert "[" in df.loc[1, "AUC"]
    gap = s.gap_frame(res)
    assert list(gap["Metric"]) == ["AUC", "F1", "Sensitivity", "Specificity", "ECE"]
    assert (gap["Gap (max - min)"] >= 0).all()
    assert set(gap["Best group"]) <= {"F", "M"}
    one = s.gap_frame(subgroup_metrics(p, y, "binary", np.zeros(300, int), min_size=20))
    assert one["Gap (max - min)"].isna().all() and (one["Best group"] == "—").all()


def test_load_groups(tmp_path):
    d = make_inspect_fixture(tmp_path, m=32, with_groups=True)
    g = load_groups(d, 32)
    assert set(g) == {"sex", "age", "centre"} and all(v.shape == (32,) for v in g.values())
    assert g["sex"].dtype.kind == "U" and g["age"].dtype.kind == "i"
    with pytest.raises(ValueError, match="expected"):
        load_groups(d, 31)
    assert load_groups(d) is not None
    plain = make_inspect_fixture(tmp_path, ts="24102026_100000", m=32)
    assert load_groups(plain, 32) is None
    np.savez(plain / "groups.npz", bad=np.zeros((32, 2)))
    with pytest.raises(ValueError):
        load_groups(plain, 32)
