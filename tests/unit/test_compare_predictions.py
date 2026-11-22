"""Tests of the paired comparison of runs: src/analysis/compare.py + scripts/compare_predictions.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.analysis.compare import (check_aligned, compare_groups, mean_std,  # noqa: E402
                                  paired_bootstrap, rank_auc, run_metrics)
from src.analysis.io import save_predictions  # noqa: E402
from src.cli import build_compare_cmd  # noqa: E402
import compare_predictions as script  # noqa: E402


def _onehot(y, c):
    return np.eye(c)[y]


def _synthetic(n=120, c=4, seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, c, n)
    logits = rng.normal(0, noise, (n, c)) + 2.0 * _onehot(y, c)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True), _onehot(y, c), rng.choice(["x", "y"], n)


def test_metrics_by_hand():
    # grades 0,1,2,3 -> predictions 0,2,2,0: exact 2/4, within one grade 3/4
    y = np.array([0, 1, 2, 3])
    pred = np.array([0, 2, 2, 0])
    r = run_metrics(_onehot(pred, 4), _onehot(y, 4), groups=np.array(["a", "a", "b", "b"]))
    assert r["accuracy"] == pytest.approx(0.5)
    assert r["acc_pm1"] == pytest.approx(0.75)
    # QWK: weights (i-j)^2/9; observed sum of weights = 1/9 + 9/9 = 10/9
    # expected = outer(row, col)/4: row=col counts -> compute directly
    row, col = np.ones(4), np.array([2.0, 0, 2, 0])
    g = np.arange(4)
    w = (g[:, None] - g[None, :]) ** 2 / 9
    expected = (np.outer(row, col) / 4 * w).sum()
    assert r["qwk"] == pytest.approx(1 - (10 / 9) / expected)
    # primary metric per subgroup: group a = grades 0,1 -> preds 0,2 ; b = 2,3 -> 2,0
    assert set(r["groups"]) == {"a", "b"}
    # auc_ge1: score 1 - p0; positives (grades 1,2,3) score 1,1,0, the negative (grade 0) scores 0:
    # two pairs well ordered and one tie -> (2 + 0.5) / 3
    assert r["auc_ge1"] == pytest.approx(2.5 / 3)


def test_non_ordinal_has_no_qwk():
    p, y, _ = _synthetic()
    r = run_metrics(p, y, ordinal=False)
    assert "qwk" not in r and "acc_pm1" not in r and "accuracy" in r
    with pytest.raises(ValueError, match="ordinal"):
        run_metrics(p, y, metrics=["qwk"], ordinal=False)


def test_rank_auc_known_case():
    # 2 positives (0.8, 0.4) vs 2 negatives (0.6, 0.2): 3 of the 4 pairs are well ordered
    assert rank_auc(np.array([0.8, 0.4, 0.6, 0.2]), np.array([1, 1, 0, 0])) == pytest.approx(0.75)
    # a tie between a positive and a negative counts 1/2
    assert rank_auc(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(0.5)
    assert np.isnan(rank_auc(np.array([0.1, 0.9]), np.array([0, 0])))     # no positives


def test_mean_std_uses_ddof1():
    m, s = mean_std([1.0, 2.0, 3.0])
    assert m == pytest.approx(2.0) and s == pytest.approx(1.0)
    assert np.isnan(mean_std([1.0])[1])


def test_bootstrap_deterministic_and_seeded():
    p, y, _ = _synthetic()
    q = [_synthetic(seed=s)[0] for s in (1, 2)]
    a = compare_groups([p], q[:1], y, n_boot=50, seed=3)
    b = compare_groups([p], q[:1], y, n_boot=50, seed=3)
    c = compare_groups([p], q[:1], y, n_boot=50, seed=4)
    assert a["diff"]["mean"]["qwk"] == b["diff"]["mean"]["qwk"]
    assert a["diff"]["mean"]["qwk"]["lo"] != c["diff"]["mean"]["qwk"]["lo"]


def test_identical_groups_zero_difference():
    p, y, g = _synthetic()
    res = compare_groups([p, p], [p, p], y, g, n_boot=40)
    for part in (res["diff"]["mean"], res["diff"]["ensemble"], *res["diff"]["pairs"]):
        for r in part.values():
            assert (r["diff"], r["lo"], r["hi"]) == (0.0, 0.0, 0.0)
            assert r["p"] == 1.0


def test_pairs_only_if_same_size_and_summary():
    ps = [_synthetic(seed=s)[0] for s in (1, 2, 3)]
    y = _synthetic()[1]
    res = compare_groups(ps[:2], ps[2:], y, n_boot=20)
    assert res["diff"]["pairs"] == []
    res = compare_groups(ps[:1], ps[1:2], y, n_boot=20)
    assert len(res["diff"]["pairs"]) == 1
    assert np.isnan(res["summary"]["a"]["qwk"][1])      # one run: no std


def test_paired_bootstrap_same_indices():
    # with a shared statistic the difference is exactly 0 in every resample
    r = paired_bootstrap(lambda i: float(i.mean()), lambda i: float(i.mean()), 30, n_boot=10)
    assert r["lo"] == r["hi"] == 0.0


def test_misaligned_raises():
    _, y, g = _synthetic()
    with pytest.raises(ValueError, match="targets"):
        check_aligned([y, np.roll(y, 1, axis=0)])
    with pytest.raises(ValueError, match="subgroups"):
        check_aligned([y, y], [g, np.roll(g, 1)])
    check_aligned([y, y.copy()], [g, g.copy()])


# ── script / CLI ───────────────────────────────────────────────────────────────

def _write(tmp_path, name, seed, y=None, groups=None):
    p, yy, g = _synthetic(seed=seed)
    save_predictions(tmp_path / name / "test_preds.npz", p, yy if y is None else y, 0.5,
                     ["0", "1", "2", "3"], "multiclass", ordinal=True,
                     groups={"centre": g if groups is None else groups})
    return str(tmp_path / name / "test_preds.npz")


def _targets_groups():
    _, y, g = _synthetic(seed=0)
    return y, g


def test_script_text_and_markdown(tmp_path, capsys):
    y, g = _targets_groups()
    a = [_write(tmp_path, f"a{i}", i + 1, y, g) for i in range(2)]
    b = [_write(tmp_path, f"b{i}", i + 10, y, g) for i in range(2)]
    base = ["--a", *a, "--b", *b, "--label-a", "Var", "--label-b", "Base", "--group", "centre",
            "--n-boot", "30"]
    script.main(base)
    text = capsys.readouterr().out
    assert "Var − Base" in text and "QWK centre" not in text and "QWK x" in text
    assert "|" not in text and "0." in text
    script.main(base + ["--markdown"])
    md = capsys.readouterr().out
    assert "| Run |" in md and "|---|" in md
    assert "0," in md and "−" in md
    assert "Par 2" in md


def test_run_names_across_groups():
    """``<run>/test_preds.npz`` on both sides: the folder names the run, not the file."""
    assert script.run_names(["x/base/test_preds.npz", "x/swin/test_preds.npz"]) == ["base", "swin"]
    assert script.run_names(["s42.npz", "s43.npz"]) == ["s42", "s43"]
    assert script.run_names(["a/p.npz", "a/p.npz"]) == ["a #1", "a #2"]


def test_script_one_run_per_group_has_no_nan(tmp_path, capsys):
    y, g = _targets_groups()
    a, b = _write(tmp_path, "base", 1, y, g), _write(tmp_path, "swin", 2, y, g)
    script.main(["--a", a, "--b", b, "--n-boot", "20"])
    text = capsys.readouterr().out
    assert "nan" not in text
    assert "A: base" in text and "B: swin" in text and "(base − swin)" in text


def test_script_misaligned_files(tmp_path):
    y, g = _targets_groups()
    a = [_write(tmp_path, "a0", 1, y, g)]
    b = [_write(tmp_path, "b0", 2, np.roll(y, 1, axis=0), g)]
    with pytest.raises(SystemExit, match="targets"):
        script.main(["--a", *a, "--b", *b, "--n-boot", "10"])


def test_script_missing_group(tmp_path):
    y, g = _targets_groups()
    a = [_write(tmp_path, "a0", 1, y, g)]
    with pytest.raises(SystemExit, match="--group"):
        script.main(["--a", *a, "--b", *a, "--group", "nope", "--n-boot", "10"])


def test_build_compare_cmd():
    cmd = build_compare_cmd(["a1.npz", "a2.npz"], ["b.npz"], label_a="V", group="data_provider",
                            n_boot=100, markdown=True)
    s = " ".join(cmd)
    assert "scripts/compare_predictions.py" in s
    assert "--a a1.npz a2.npz --b b.npz" in s
    assert "--group data_provider" in s and "--n-boot 100" in s and "--markdown" in s
    assert "--group" not in " ".join(build_compare_cmd(["a"], ["b"]))


def test_cli_compare_preds_dry_run():
    from typer.testing import CliRunner
    from src.cli import app
    r = CliRunner().invoke(app, ["compare-preds", "--a", "x.npz", "--a", "y.npz", "--b", "z.npz",
                                 "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "compare_predictions.py" in r.output.replace("\n", "")
