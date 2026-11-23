"""Tests of the ensemble choice on validation: src/analysis/ensemble.py + scripts/select_ensemble.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.analysis.compare import METRICS  # noqa: E402
from src.analysis.ensemble import (average, best_per_group, choose_k,  # noqa: E402
                                   greedy_selection, topk_curve)
from src.analysis.io import save_predictions  # noqa: E402
from src.cli import build_select_ensemble_cmd  # noqa: E402
import select_ensemble as script  # noqa: E402

ACC = METRICS["accuracy"][0]


def _probs(y, c, strength, seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.0, (len(y), c)) + strength * np.eye(c)[y]
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def _data(n=200, c=4, seed=0):
    y = np.random.default_rng(seed).integers(0, c, n)
    return y, np.eye(c)[y]


def test_average_weights():
    a, b = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    assert np.allclose(average([a, b]), [[0.5, 0.5]])
    assert np.allclose(average([a, b], weights=[3, 1]), [[0.75, 0.25]])
    with pytest.raises(ValueError):
        average([])
    with pytest.raises(ValueError):
        average([a, b], weights=[1])


def test_greedy_picks_the_good_run_and_stops():
    y, t = _data()
    # "bad" is chance level; "good" is far better: adding anything else never helps
    vp = {"good": _probs(y, 4, 6.0, 1), "bad": _probs(y, 4, 0.0, 2)}
    chosen, trace = greedy_selection(vp, t, ACC, max_size=8, with_replacement=False)
    assert chosen[0] == "good" and "bad" not in chosen
    assert len(chosen) == len(trace) == 1
    scores = [s for _, s in trace]
    assert scores == sorted(scores)


def test_greedy_with_replacement_can_repeat():
    y, t = _data()
    vp = {"a": _probs(y, 4, 2.0, 1), "b": _probs(y, 4, 0.5, 2)}
    calls = []

    def rising(p, _t):                     # every evaluation scores higher than the last
        calls.append(1)
        return float(len(calls))
    chosen, trace = greedy_selection(vp, t, rising, max_size=3, with_replacement=True)
    assert chosen == ["b", "b", "b"] and len(trace) == 3       # last candidate wins every step
    calls.clear()
    chosen, _ = greedy_selection(vp, t, rising, max_size=3, with_replacement=False)
    assert chosen == ["b", "a"]            # without replacement a run is used once


def test_greedy_ties_are_deterministic_by_name():
    y, t = _data()
    p = _probs(y, 4, 3.0, 1)
    vp = {"zeta": p, "alpha": p.copy(), "mid": p.copy()}
    chosen, _ = greedy_selection(vp, t, ACC, max_size=4)
    assert chosen == ["alpha"]             # identical scores: the smallest name, then no gain
    again, _ = greedy_selection(dict(reversed(list(vp.items()))), t, ACC, max_size=4)
    assert again == chosen


def test_greedy_validates_input():
    y, t = _data()
    with pytest.raises(ValueError):
        greedy_selection({}, t, ACC)
    with pytest.raises(ValueError):
        greedy_selection({"a": _probs(y, 4, 1, 1)}, t, ACC, max_size=0)


def test_best_per_group_orders_by_validation():
    scores = {"vit_a": 0.70, "vit_b": 0.75, "swin_a": 0.74, "deit_a": 0.60}
    groups = {"vit_a": "vit", "vit_b": "vit", "swin_a": "swin", "deit_a": "deit"}
    assert best_per_group(scores, groups) == [("vit", "vit_b", 0.75), ("swin", "swin_a", 0.74),
                                              ("deit", "deit_a", 0.60)]
    with pytest.raises(ValueError, match="without a group"):
        best_per_group(scores, {"vit_a": "vit"})
    # tie inside a group: smaller run name
    assert best_per_group({"b": 0.5, "a": 0.5}, {"a": "g", "b": "g"}) == [("g", "a", 0.5)]


def test_choose_k_and_curve():
    assert choose_k([(1, 0.70), (2, 0.74), (3, 0.72)]) == 2
    assert choose_k([(1, 0.70), (2, 0.70), (3, 0.69)]) == 1       # tie: the smallest k
    with pytest.raises(ValueError):
        choose_k([])
    y, t = _data()
    vp = {"a": _probs(y, 4, 3.0, 1), "b": _probs(y, 4, 3.0, 2)}
    curve = topk_curve(vp, t, ["a", "b"], ACC)
    assert [k for k, _ in curve] == [1, 2]
    assert curve[0][1] == pytest.approx(ACC(vp["a"], t))


# ── script, end to end ────────────────────────────────────────────────────────────────────────

def _make_run(tmp_path, name, strength, seed, y_val, y_test, test_targets=None):
    c = 4
    for fname, y, s in (("val_preds.npz", y_val, seed), ("test_preds.npz", y_test, seed + 100)):
        t = np.eye(c)[y]
        if fname == "test_preds.npz" and test_targets is not None:
            t = test_targets
        save_predictions(tmp_path / name / fname, _probs(y, c, strength, s), t, 0.5,
                         ["0", "1", "2", "3"], "multiclass", ordinal=True)
    return str(tmp_path / name)


@pytest.fixture
def runs(tmp_path):
    yv, _ = _data(150, seed=1)
    yt, _ = _data(150, seed=2)
    return {n: _make_run(tmp_path, n, s, i, yv, yt)
            for i, (n, s) in enumerate([("vit_s42", 3.0), ("vit_s43", 2.5), ("swin_s42", 3.5),
                                        ("junk", 0.2)], start=1)}, (yv, yt)


def test_script_text_and_markdown(runs, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r, _ = runs
    groups = tmp_path / "groups.csv"
    groups.write_text("run,group\nvit_s42,vit\nvit_s43,vit\nswin_s42,swin\njunk,junk\n")
    base = ["--runs", *r.values(), "--group-by", str(groups), "--reference",
            r["vit_s42"], r["vit_s43"], "--n-boot", "30"]
    script.main(base)
    text = capsys.readouterr().out
    assert "solo validación" in text and "Evaluación única en test" in text
    assert "Conjunto elegido:" in text and "k = 1" in text and "|" not in text
    assert "elegido − referencia" in text and "P(d ≤ 0)" in text
    script.main(base + ["--markdown", "--method", "greedy"])
    md = capsys.readouterr().out
    assert "| Run |" in md and "|---|" in md and "0," in md and "+ " in md


def test_script_without_reference_or_groups(runs, capsys):
    r, _ = runs
    script.main(["--runs", *r.values()])
    text = capsys.readouterr().out
    assert "referencia" not in text and "nan" not in text


def test_script_missing_file(runs, tmp_path):
    r, _ = runs
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="Missing val predictions"):
        script.main(["--runs", *r.values(), str(tmp_path / "empty")])


def test_script_misaligned_test(runs, tmp_path):
    r, (yv, yt) = runs
    bad = _make_run(tmp_path, "bad", 3.0, 9, yv, yt, test_targets=np.roll(np.eye(4)[yt], 1, axis=0))
    with pytest.raises(SystemExit, match="test predictions"):
        script.main(["--runs", *r.values(), bad])


def test_script_unknown_group_name(runs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r, _ = runs
    g = tmp_path / "g.yaml"
    g.write_text("nope: x\n")
    with pytest.raises(SystemExit, match="unknown runs"):
        script.main(["--runs", *r.values(), "--group-by", str(g)])


def test_read_group_map_yaml_and_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    y = tmp_path / "g.yaml"
    y.write_text("a: vit\nb: swin\n")
    assert script.read_group_map(str(y)) == {"a": "vit", "b": "swin"}
    c = tmp_path / "g.csv"
    c.write_text("run,group\n# comentario\na,vit\n")
    assert script.read_group_map(str(c)) == {"a": "vit"}
    with pytest.raises(SystemExit, match="not found"):
        script.read_group_map(str(tmp_path / "x.csv"))


def test_build_select_ensemble_cmd_and_cli_dry_run():
    s = " ".join(build_select_ensemble_cmd(["r1", "r2"], group_by="g.csv", method="greedy",
                                           reference=["b1"], markdown=True))
    assert "scripts/select_ensemble.py --runs r1 r2" in s
    assert "--group-by g.csv" in s and "--method greedy" in s and "--reference b1" in s
    assert "--markdown" in s
    assert "--reference" not in " ".join(build_select_ensemble_cmd(["r"]))
    from typer.testing import CliRunner
    from src.cli import app
    r = CliRunner().invoke(app, ["select-ensemble", "--runs", "x", "--runs", "y", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "select_ensemble.py" in r.output.replace("\n", "")
    r = CliRunner().invoke(app, ["select-ensemble", "--runs", "x", "--method", "nope", "--dry-run"])
    assert r.exit_code != 0


def test_group_map_must_be_inside_the_working_directory(tmp_path, monkeypatch):
    work, outside = tmp_path / "work", tmp_path / "outside"
    work.mkdir()
    outside.mkdir()
    (outside / "g.csv").write_text("a,vit\n")
    monkeypatch.chdir(work)
    with pytest.raises(SystemExit, match="working directory"):
        script.read_group_map(str(outside / "g.csv"))
    with pytest.raises(SystemExit, match="working directory"):
        script.read_group_map("../outside/g.csv")


def test_group_map_symlink_cannot_escape(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "g.csv").write_text("a,vit\n")
    (work / "link.csv").symlink_to(tmp_path / "g.csv")
    monkeypatch.chdir(work)
    with pytest.raises(SystemExit, match="working directory"):
        script.read_group_map("link.csv")
